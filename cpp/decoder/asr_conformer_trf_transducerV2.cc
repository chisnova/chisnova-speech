#include "decoder/asr_conformer_trf_transducerV2.h"

namespace MERCURY {
    template<typename T>
    AsrConformerTrfTransducerV2<T>::AsrConformerTrfTransducerV2(JsonObject asr_conformer_path, std::string base_path) {
        num_layer = int(asr_conformer_path.parserValue("num_layer"));
        conformer_stream_encoder = std::make_shared<ConformerStreamEncoderV2<T>>(asr_conformer_path.parseObject("encoder"), base_path);
        transformer_transducer = std::make_shared<TransformerTransducer<T>>(asr_conformer_path.parseObject("decoder"), base_path);
        conformer_ctc = std::make_shared<ConformerCTC<T>>(asr_conformer_path.parseObject("ctc_encoder"), base_path);
        sos=2;
        eos=3;
        position = 0;
    }

    template<typename T>
    AsrConformerTrfTransducerV2<T>::~AsrConformerTrfTransducerV2() {
        conformer_stream_encoder = nullptr;
        transformer_transducer = nullptr;
        conformer_ctc = nullptr;
    }

    template<typename T>
    int AsrConformerTrfTransducerV2<T>::getNumDecoderLayer() {
        return num_layer;
    }

    template<typename T>
    void AsrConformerTrfTransducerV2<T>::setDecoderCache(std::shared_ptr<DecoderCache> _decoder_cache) {
        transformer_transducer->setDecoderCache(_decoder_cache);
    }

    template<typename T>
    std::vector<Beam> AsrConformerTrfTransducerV2<T>::makeInitBeams() {
        std::vector<Beam> ret;
        Beam init_beam = Beam(sos, eos);
        init_beam.makeInitBeam(num_layer);
        ret.push_back(init_beam);
        //ret.push_back(init_beam);
        //ret.push_back(init_beam);
        return ret;
    }

    template<typename T>
    void AsrConformerTrfTransducerV2<T>::clear(){
        enc_arr.clear();
        conformer_stream_encoder->clear();
        transformer_transducer->clear();
        position = 0;
    }

    template<typename T>
    std::shared_ptr<Tensor<float>> AsrConformerTrfTransducerV2<T>::getCtcOut() {
        std::shared_ptr<Tensor<T>> out = std::make_shared<Tensor<T>>();
        for(int i=0;i<enc_arr.size();++i){
            out->appendChunkRow(enc_arr[i]);
        }
        out = conformer_ctc->forward(out);
        out = Softmax().forward(out);
        return out;
    }
    
    template<typename T>
    std::shared_ptr<Tensor<T>> AsrConformerTrfTransducerV2<T>::encoding(std::shared_ptr<Tensor<T>> enc) {
        return conformer_stream_encoder->forward(enc);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrConformerTrfTransducerV2<T>::decoding(std::shared_ptr<Tensor<int>> dec, int position) {
        // deprecated
        std::shared_ptr<Tensor<T>> out;
        return out;
    }

    template<typename T>
    std::vector<Score> AsrConformerTrfTransducerV2<T>::getNbest(std::shared_ptr<Tensor<float>> logits, int num_nbest){
        std::priority_queue<Score> nbest;
        std::vector<Score> ret;

        for(int i=2; i<logits->shape(1); ++i) {
            Score new_score;
            float val = std::log(logits->value(0, i));
            new_score.idx = i;
            new_score.logit = val;
            new_score.org_logit = logits->value(0, i);
            nbest.push(new_score);
        }

        while(ret.size() < num_nbest){
            Score cur_score = nbest.top();
            nbest.pop();
            ret.push_back(cur_score);
        }
        return ret;
    }

    template<typename T>    
    bool AsrConformerTrfTransducerV2<T>::isBreak(std::priority_queue<Beam> stop, std::priority_queue<Beam> running, int num_nbest){
        Beam runnning_best = running.top();
        int runing_len = runnning_best.ids.size();
        int count=0;
        while(!stop.empty()){
            Beam each_stop = stop.top();
            stop.pop();
            int stop_len = each_stop.ids.size();

            if(each_stop.normscore > runnning_best.normscore || stop_len + 3 < runing_len){
                count++;
            }
        }
        if(count < num_nbest) {
            return false;
        }
        return true;
    }    

    template<typename T>
    std::vector<Beam> AsrConformerTrfTransducerV2<T>::run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams){
        // clear();
        struct timespec begin, end ;
        struct timespec begin2, end2;

        float decoder_only=0.0;
        std::vector<Beam> ret;
        std::priority_queue<Beam> beamsearch;
        std::priority_queue<Beam> stop;
        std::priority_queue<Beam> running;

        clock_gettime(CLOCK_MONOTONIC, &begin);
        std::shared_ptr<Tensor<T>> enc_val = encoding(feature);
        clock_gettime(CLOCK_MONOTONIC, &end);
        float time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        LoggerManager::Log("INFO", " encoder time: " + std::to_string(time));
        enc_arr.push_back(enc_val);

        clock_gettime(CLOCK_MONOTONIC, &begin);
        for(int i=0;i<beams.size();++i){
            beamsearch.push(beams[i]);
        }

        int TimeLen = enc_val->shape(0);
        for(int i=0; i < TimeLen; ++i) {
            std::shared_ptr<Tensor<T>> each_enc = std::make_shared<Tensor<T>>();
            each_enc->value = enc_val->value.row(i);
            // each_enc->printValue();
            while(true) {
                while(!beamsearch.empty()) {
                    Beam org_beam = Beam(sos, eos);
                    Beam cur_beam = beamsearch.top();
                    org_beam.clone(cur_beam);
                    // cur_beam.print();
                    beamsearch.pop();
                    transformer_transducer->setDecoderCache(cur_beam.cache);
                    std::shared_ptr<Tensor<int>> token_idx = std::make_shared<Tensor<int>>();
                    // std::cout << "cur beam" << std::endl;
                    // cur_beam.print();
                    // std::cout << "token_idx" << cur_beam.ids.back() << std::endl;
                    token_idx->setValue({cur_beam.ids.back()}, {1,1});
                    
                    // std::cout << "enc_in" << std::endl;
                    // each_enc->printValue();

                    clock_gettime(CLOCK_MONOTONIC, &begin2);
                    std::shared_ptr<Tensor<T>> logits = transformer_transducer->forwardContext(token_idx, each_enc, cur_beam.position);
                    clock_gettime(CLOCK_MONOTONIC, &end2);
                    time = (end2.tv_sec - begin2.tv_sec) + (end2.tv_nsec - begin2.tv_nsec) / 1000000000.0;
                    decoder_only+=time;
                    // std::cout << "logit" << std::endl;
                    // logits->printValue();
                    logits = Softmax().forward(logits);

                    // blank
                    Beam blank_beam = Beam(sos, eos);
                    blank_beam.clone(org_beam);
                    blank_beam.score += std::log(logits->value(0, 1));
                    blank_beam.normScore();
                    stop.push(blank_beam);

                    // wo blank
                    std::vector<Score> scores = getNbest(logits, num_nbest);
                    for(int i=0;i<num_nbest;++i){
                        Beam run_beam = Beam(sos, eos);
                        // std::cout << scores[i].idx << std::endl;
                        run_beam.incremental(cur_beam, scores[i].idx, scores[i].logit, scores[i].org_logit);
                        // std::cout << "gen beam" << std::endl;
                        // run_beam.print();
                        running.push(run_beam);
                    }
                }
                // std::cout << "beamserach out" << std::endl;
                // std::cout << "=============================" << std::endl;
                if(isBreak(stop, running, num_nbest)) {
                    // std::cout << "break!!" << std::endl;
                    break;
                }
                int cnt=0;
                while(!running.empty()){
                    cnt++;
                    beamsearch.push(running.top());
                    running.pop();
                    if(cnt > num_nbest) break;
                }
            }
            while(!beamsearch.empty()) beamsearch.pop();
            int cnt=0;
            while(!stop.empty()){
                cnt++;
                // auto tmp = stop.top();
                // tmp.print();
                beamsearch.push(stop.top());
                stop.pop();
                if(cnt >= num_nbest) break;
            }
            while(!stop.empty()) stop.pop();
        }
        int cnt=0;
        while(!beamsearch.empty()){
            cnt++;
            ret.push_back(beamsearch.top());
            beamsearch.pop();
            if(cnt > num_nbest) break;
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        LoggerManager::Log("INFO", " decoder only time: " + std::to_string(decoder_only));
        LoggerManager::Log("INFO", " beam-search time: " + std::to_string(time));

        return ret;
    }

    template class AsrConformerTrfTransducerV2<float>;
    // template class AsrConformer<int>;
};

