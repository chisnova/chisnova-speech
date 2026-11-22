#include "decoder/asr_stream_conformer_trf_transducerV2.h"

namespace MERCURY {
    template<typename T>
    AsrStreamConformerTrfTransducerV2<T>::AsrStreamConformerTrfTransducerV2(JsonObject asr_conformer_path, std::string base_path) {
        conformer_stream_encoder = std::make_shared<ConformerStreamEncoderV2<T>>(asr_conformer_path.parseObject("stream_encoder"), base_path);
        conformer_ctc = std::make_shared<ConformerCTC<T>>(asr_conformer_path.parseObject("ctc_encoder"), base_path);
        word_decoder = std::make_shared<ConformerDecoder<T>>(asr_conformer_path.parseObject("wrd_decoder"), base_path);
        phone_decoder = std::make_shared<ConformerDecoder<T>>(asr_conformer_path.parseObject("phn_decoder"), base_path);
        
        ctc_linear2 = std::make_shared<Linear<T>>(asr_conformer_path.parseObject("ctc_linear2"), base_path);

        transformer_config = asr_conformer_path.parseObject("wrd_decoder");
        num_layer = int(transformer_config.parserValue("num_layer"));
        sos=2;
        eos=3;
        position = 0;
    }

    template<typename T>
    AsrStreamConformerTrfTransducerV2<T>::~AsrStreamConformerTrfTransducerV2() {
        conformer_stream_encoder = nullptr;
        conformer_ctc = nullptr;
        enc_out_val = nullptr;
        ctc_out_val = nullptr;
        phone_decoder = nullptr;
        word_decoder = nullptr;
        ctc_linear2 = nullptr;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformerTrfTransducerV2<T>::forward(std::shared_ptr<Tensor<int>> dec, std::shared_ptr<Tensor<T>> enc) {
        std::shared_ptr<Tensor<T>> out;
        out = conformer_stream_encoder->forward(enc);
        return out;
    }

    template<typename T>
    std::vector<int> AsrStreamConformerTrfTransducerV2<T>::getCtcAsrIds() {
        if(ctc_out_val == nullptr){
            ctc_out_val = conformer_ctc->forward(enc_out_val);
        }

        std::shared_ptr<Tensor<T>> ctc_out = ctc_linear2->forward(ctc_out_val);
        ctc_out = Softmax().forward(ctc_out);
        std::vector<int> ctc_asr_ids;
        for (auto row : ctc_out->value.rowwise()){
            int max_idx;
            float max_val = row.maxCoeff(&max_idx);
            ctc_asr_ids.push_back(max_idx);
        }
        return ctc_asr_ids;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformerTrfTransducerV2<T>::getPhoneScore(std::vector<int> phone_dec) {
        std::shared_ptr<Tensor<int>> phone_dec_tensor = std::make_shared<Tensor<int>>();
        phone_dec_tensor->setValue(phone_dec, {1, int(phone_dec.size())});
        if(ctc_out_val == nullptr){
            ctc_out_val = conformer_ctc->forward(enc_out_val);
        }
        std::shared_ptr<Tensor<T>> ret = phone_decoder->forward(phone_dec_tensor, ctc_out_val);
        ret = Softmax().forward(ret);
        return ret;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformerTrfTransducerV2<T>::getWordScore(std::vector<int> word_dec) {
        std::shared_ptr<Tensor<int>> word_dec_tensor = std::make_shared<Tensor<int>>();
        word_dec_tensor->setValue(word_dec, {1, int(word_dec.size())});
        if(ctc_out_val == nullptr){
            ctc_out_val = conformer_ctc->forward(enc_out_val);
        }
        std::shared_ptr<Tensor<T>> ret = word_decoder->forward(word_dec_tensor, ctc_out_val);
        ret = Softmax().forward(ret);
        return ret;
    }

    template<typename T>
    std::vector<Beam> AsrStreamConformerTrfTransducerV2<T>::makeInitBeams() {
        std::vector<Beam> ret;
        Beam init_beam = Beam(sos, eos);
        init_beam.makeInitBeam(num_layer);
        ret.push_back(init_beam);
        return ret;
    }

    template<typename T>
    void AsrStreamConformerTrfTransducerV2<T>::clear(){
        enc_out_val = nullptr;
        ctc_out_val = nullptr;
        conformer_stream_encoder->clear();
        position = 0;
        word_decoder->clear();
        phone_decoder->clear();
    }


    template<typename T>
    std::vector<Score> AsrStreamConformerTrfTransducerV2<T>::getNbest(std::shared_ptr<Tensor<T>> logits, int num_nbest){
        std::priority_queue<Score> nbest;
        std::vector<Score> ret;

        for(int i=2; i<logits->shape(1); ++i) {
            Score new_score;
            float val = std::log(logits->value(0, i));
            new_score.idx = i;
            new_score.logit = val;
            new_score.org_logit=logits->value(0, i);
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
    void AsrStreamConformerTrfTransducerV2<T>::warmUpInference(std::shared_ptr<Tensor<T>> feature){
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformerTrfTransducerV2<T>::encoding(std::shared_ptr<Tensor<T>> feature){
        std::shared_ptr<Tensor<T>> enc = conformer_stream_encoder->forward(feature);
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " encoding forward done");
        LoggerManager::Log("DEBUG", "incremental start");
        #endif
        if(enc_out_val == nullptr){
            enc_out_val = std::make_shared<Tensor<T>>();
            enc_out_val->value = enc->value;
        }
        else{
            enc_out_val->appendChunkRow(enc);
        }
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " incremental done");
        #endif
        return enc;
    }


    template<typename T>
    std::vector<Beam> AsrStreamConformerTrfTransducerV2<T>::run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams){
        struct timespec begin, end ;
        std::vector<Beam> beam_worker;
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " print feature");
        feature->printValue();
        #endif

        clock_gettime(CLOCK_MONOTONIC, &begin);
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " encoding start");
        #endif
        std::shared_ptr<Tensor<T>> enc = encoding(feature);
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " encoding done");
        LoggerManager::Log("DEBUG", " encoding print");
        enc->printValue();
        #endif
        clock_gettime(CLOCK_MONOTONIC, &end);
        float time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        LoggerManager::Log("INFO", " encoder time: " + std::to_string(time));
        return beam_worker;
    }

    template<typename T>
    std::vector<int> AsrStreamConformerTrfTransducerV2<T>::secondPassDecoding(int num_nbest){
        if (ctc_out_val == nullptr){
            ctc_out_val = conformer_ctc->forward(enc_out_val);
        }

        std::vector<Beam> beams = makeInitBeams();
        struct timespec begin, end ;
        std::vector<Beam> ret;
        std::priority_queue<Beam> beamsearch;
        std::priority_queue<Beam> nbest;
        std::priority_queue<Beam> running;

        float time = 0;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        for(int i=0;i<beams.size();++i){
            beamsearch.push(beams[i]);
        }

        while(nbest.size() < num_nbest) {
            while(!beamsearch.empty()) {
                Beam cur_beam = beamsearch.top();
                beamsearch.pop();
                std::shared_ptr<Tensor<int>> token = std::make_shared<Tensor<int>>();
                token->setValue({cur_beam.word_ids.back()}, {1,1});
                word_decoder->setDecoderCache(cur_beam.cache);
                std::shared_ptr<Tensor<T>> logits = word_decoder->forwardContext(token, ctc_out_val, cur_beam.position);
                logits = Softmax().forward(logits);
                std::vector<Score> scores = getNbest(logits, num_nbest);
                for(int i=0;i<scores.size();++i){
                    Beam run_beam(sos, eos);
                    run_beam.workerIncremental(cur_beam, scores[i].idx, scores[i].org_logit);
                    running.push(run_beam);
                }
            }
            int count=0;
            while(count < num_nbest){
                Beam beam = running.top();
                running.pop();
                if(beam.isFinal) {
                    nbest.push(beam);
                }
                else {
                    beamsearch.push(beam);
                }
                count++;
            }
            while(!running.empty()) running.pop();
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        // std::cout <<" beam search time: "  << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0<<std::endl;
        time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        LoggerManager::Log("INFO", " second pass beam search time:" + std::to_string(time));
        while(!nbest.empty()) {
            Beam cur_beam = nbest.top();
            ret.push_back(cur_beam);
            nbest.pop();
        }
        while(!beamsearch.empty()) beamsearch.pop();
        return ret[0].word_ids;
    }

    template class AsrStreamConformerTrfTransducerV2<float>;
    // template class AsrStreamConformerTrfTransducerV2<int>;
};

