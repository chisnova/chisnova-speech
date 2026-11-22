#include "decoder/asr_conformer_trf_transducer.h"
#include <algorithm>

namespace MERCURY {
    template<typename T>
    AsrConformerTrfTransducer<T>::AsrConformerTrfTransducer(JsonObject asr_conformer_path, std::string base_path) {
        num_layer = int(asr_conformer_path.parserValue("num_layer"));
        conformer_stream_encoder = std::make_shared<ConformerStreamEncoder<T>>(asr_conformer_path.parseObject("encoder"), base_path);
        transformer_transducer = std::make_shared<TransformerTransducer<T>>(asr_conformer_path.parseObject("decoder"), base_path);
        sos=2;
        eos=3;
        position = 0;
    }

    template<typename T>
    AsrConformerTrfTransducer<T>::~AsrConformerTrfTransducer() {
        conformer_stream_encoder = nullptr;
        transformer_transducer = nullptr;
        enc_out_val = nullptr;
        ctc_out_val = nullptr;
    }

    template<typename T>
    int AsrConformerTrfTransducer<T>::getNumDecoderLayer() {
        return num_layer;
    }

    template<typename T>
    void AsrConformerTrfTransducer<T>::setDecoderCache(std::shared_ptr<DecoderCache> _decoder_cache) {
        transformer_transducer->setDecoderCache(_decoder_cache);
    }

    template<typename T>
    std::vector<Beam> AsrConformerTrfTransducer<T>::makeInitBeams() {
        std::vector<Beam> ret;
        ret.reserve(1);
        Beam init_beam = Beam(sos, eos);
        init_beam.makeInitBeam(num_layer);
        ret.push_back(std::move(init_beam));
        return ret;
    }

    template<typename T>
    void AsrConformerTrfTransducer<T>::clear(){
        enc_out_val = nullptr;
        ctc_out_val = nullptr;
        conformer_stream_encoder->clear();
        transformer_transducer->clear();
        position = 0;
    }

    template<typename T>
    std::shared_ptr<Tensor<float>> AsrConformerTrfTransducer<T>::getCtcOut() {
        return ctc_out_val;
    }
    
    template<typename T>
    std::shared_ptr<Tensor<T>> AsrConformerTrfTransducer<T>::encoding(std::shared_ptr<Tensor<T>> enc) {
        enc_out_val = conformer_stream_encoder->forward(enc);
        ctc_out_val = conformer_stream_encoder->getCTCOut();
        Softmax softmax;
        ctc_out_val = softmax.forward(ctc_out_val);
        return enc_out_val;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrConformerTrfTransducer<T>::decoding(std::shared_ptr<Tensor<int>> dec, int position) {
        std::shared_ptr<Tensor<T>> ret;
        if (enc_out_val == nullptr) {
            std::cout << "enc_output is nullptr" << std::endl;
            assert(false);
        }
        ret = transformer_transducer->forwardContext(dec, enc_out_val, position);
        return ret;
    }

    template<typename T>
    std::vector<Score> AsrConformerTrfTransducer<T>::getNbest(std::shared_ptr<Tensor<float>> logits, int num_nbest){
        // Use partial_sort instead of full priority_queue - O(N + K*logK) vs O(N*logN)
        const int vocab_size = logits->shape(1);
        const int start_idx = 3;
        const int num_candidates = vocab_size - start_idx;

        std::vector<Score> all_scores;
        all_scores.reserve(num_candidates);

        for(int i = start_idx; i < vocab_size; ++i) {
            Score new_score;
            new_score.org_logit = logits->value(0, i);
            new_score.logit = std::log(new_score.org_logit);
            new_score.idx = i;
            all_scores.push_back(new_score);
        }

        const int k = std::min(static_cast<int>(num_nbest), static_cast<int>(all_scores.size()));
        std::partial_sort(all_scores.begin(), all_scores.begin() + k, all_scores.end(),
                         [](const Score& a, const Score& b) { return a.logit > b.logit; });

        all_scores.resize(k);
        return all_scores;
    }
    // Note: parameters passed by value intentionally (caller's queues must not be consumed)
    // The copies happen at call site; inside we just drain the local copies
    template<typename T>
    bool AsrConformerTrfTransducer<T>::isBreak(std::priority_queue<Beam> stop, std::priority_queue<Beam> running, int num_nbest){
        if(running.empty()) return true;
        const float best_score = running.top().normscore;
        int count=0;
        while(!stop.empty()){
            if(stop.top().normscore > best_score){
                count++;
            }
            stop.pop();
        }
        return count > num_nbest;
    }    

    template<typename T>
    std::vector<Beam> AsrConformerTrfTransducer<T>::run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams){
        // clear();
        struct timespec begin, end ;
        std::vector<Beam> ret;
        std::priority_queue<Beam> beamsearch;
        std::priority_queue<Beam> stop;
        std::priority_queue<Beam> running;

        clock_gettime(CLOCK_MONOTONIC, &begin);
        std::shared_ptr<Tensor<T>> enc = encoding(feature);
        clock_gettime(CLOCK_MONOTONIC, &end);
        float time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        LoggerManager::Log("INFO", " encoder time: " + std::to_string(time));
        
        clock_gettime(CLOCK_MONOTONIC, &begin);
        int numDecLayer = getNumDecoderLayer();
        for(int i=0;i<beams.size();++i){
            beamsearch.push(beams[i]);
        }

        // Reuse enc tensor and token across time steps
        std::shared_ptr<Tensor<T>> each_enc = std::make_shared<Tensor<T>>();
        std::shared_ptr<Tensor<int>> token_idx = std::make_shared<Tensor<int>>();
        Softmax softmax;

        int TimeLen = enc_out_val->shape(0);
        for(int i=0; i < TimeLen; ++i) {
            each_enc->value = enc->value.row(i);
            while(true) {
                while(!beamsearch.empty()) {
                    Beam cur_beam = beamsearch.top();
                    beamsearch.pop();
                    transformer_transducer->setDecoderCache(cur_beam.cache);
                    token_idx->setValue({cur_beam.ids.back()}, {1,1});
                    std::shared_ptr<Tensor<T>> logits = transformer_transducer->forwardContext(token_idx, each_enc, cur_beam.position);
                    logits = softmax.forward(logits);

                    // blank
                    Beam blank_beam = Beam(sos, eos);
                    blank_beam.clone(cur_beam);
                    blank_beam.score += std::log(logits->value(0, 1));
                    blank_beam.normScore();
                    stop.push(blank_beam);

                    // wo blank
                    std::vector<Score> scores = getNbest(logits, num_nbest);
                    for(int i=0;i<num_nbest;++i){
                        Beam run_beam = Beam(sos, eos);
                        run_beam.incremental(cur_beam, scores[i].idx, scores[i].logit, scores[i].org_logit);
                        running.push(run_beam);
                    }
                }
                if(isBreak(stop, running, num_nbest)) break;
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
                beamsearch.push(stop.top());
                stop.pop();
                if(cnt > num_nbest) break;
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
        LoggerManager::Log("INFO", " beam-search time: " + std::to_string(time));

        return ret;
    }

    template class AsrConformerTrfTransducer<float>;
    // template class AsrConformer<int>;
};

