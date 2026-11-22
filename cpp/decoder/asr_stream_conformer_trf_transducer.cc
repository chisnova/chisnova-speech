#include "decoder/asr_stream_conformer_trf_transducer.h"
#include <algorithm>

namespace MERCURY {
    template<typename T>
    AsrStreamConformerTrfTransducer<T>::AsrStreamConformerTrfTransducer(JsonObject asr_conformer_path, std::string base_path) {
        conformer_stream_encoder = std::make_shared<ConformerStreamEncoderV2<T>>(asr_conformer_path.parseObject("stream_encoder"), base_path);
        conformer_ctc = std::make_shared<ConformerCTC<T>>(asr_conformer_path.parseObject("ctc_encoder"), base_path);
        word_decoder = std::make_shared<ConformerDecoder<T>>(asr_conformer_path.parseObject("wrd_decoder"), base_path);
        phone_decoder = std::make_shared<ConformerDecoder<T>>(asr_conformer_path.parseObject("phn_decoder"), base_path);
        
        ctc_linear1 = std::make_shared<Linear<T>>(asr_conformer_path.parseObject("ctc_linear1"), base_path);
        ctc_linear2 = std::make_shared<Linear<T>>(asr_conformer_path.parseObject("ctc_linear2"), base_path);

        sos=2;
        eos=3;
        position = 0;
        transformer_config = asr_conformer_path.parseObject("decoder");
        num_layer = int(transformer_config.parserValue("num_layer"));
        LoggerManager::Log("INFO", " num_layer: " + std::to_string(num_layer));

        #ifdef USE_TRANSFORMER_POOL
        config_base_path = base_path;
        pool_initialized = false;
        getTransformerInstance();
        #else
        transformer_transducer = std::make_shared<TransformerTransducer<T>>(transformer_config, base_path);
        #endif
    }

    template<typename T>
    AsrStreamConformerTrfTransducer<T>::~AsrStreamConformerTrfTransducer() {
        conformer_stream_encoder = nullptr;
        conformer_ctc = nullptr;
        enc_out_val = nullptr;
        ctc_out_val = nullptr;
        phone_decoder = nullptr;
        word_decoder = nullptr;
        ctc_linear1 = nullptr;
        ctc_linear2 = nullptr;
        #ifdef USE_TRANSFORMER_POOL
        for(int i=0; i<transformer_pool.size(); ++i){
            transformer_pool[i] = nullptr;
        }
        transformer_pool.clear();
        #else
        transformer_transducer = nullptr;
        #endif
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformerTrfTransducer<T>::forward(std::shared_ptr<Tensor<int>> dec, std::shared_ptr<Tensor<T>> enc) {
        std::shared_ptr<Tensor<T>> out;
        out = conformer_stream_encoder->forward(enc);
        #ifdef USE_TRANSFORMER_POOL
        auto transformer_instance = getTransformerInstance();
        out = transformer_instance->forwardContext(dec, out, 0);
        #else
        out = transformer_transducer->forwardContext(dec, out, 0);
        #endif
        return out;
    }

    template<typename T>
    std::vector<int> AsrStreamConformerTrfTransducer<T>::getCtcAsrIds() {
        if(ctc_out_val == nullptr){
            ctc_out_val = conformer_ctc->forward(enc_out_val);
        }

        std::shared_ptr<Tensor<T>> ctc_out = ctc_linear2->forward(ctc_out_val);
        // Reuse a single Softmax instance instead of constructing per call
        Softmax softmax;
        ctc_out = softmax.forward(ctc_out);
        std::vector<int> ctc_asr_ids;
        ctc_asr_ids.reserve(ctc_out->value.rows());
        for (auto row : ctc_out->value.rowwise()){
            int max_idx;
            row.maxCoeff(&max_idx);
            ctc_asr_ids.push_back(max_idx);
        }
        return ctc_asr_ids;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformerTrfTransducer<T>::getPhoneScore(std::vector<int> phone_dec) {
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
    std::shared_ptr<Tensor<T>> AsrStreamConformerTrfTransducer<T>::getWordScore(std::vector<int> word_dec) {
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
    std::vector<Beam> AsrStreamConformerTrfTransducer<T>::makeInitBeams() {
        std::vector<Beam> ret;
        ret.reserve(1);
        Beam init_beam = Beam(sos, eos);
        init_beam.makeInitBeam(num_layer);
        ret.push_back(std::move(init_beam));
        return ret;
    }

    template<typename T>
    void AsrStreamConformerTrfTransducer<T>::clear(){
        enc_out_val = nullptr;
        ctc_out_val = nullptr;
        conformer_stream_encoder->clear();
        position = 0;
        word_decoder->clear();
        phone_decoder->clear();
        #ifdef USE_TRANSFORMER_POOL
        for(int i=0; i<transformer_pool.size(); ++i){
            transformer_pool[i]->clear();
        }
        #else
        transformer_transducer->clear();
        #endif
    }


    template<typename T>
    std::vector<Score> AsrStreamConformerTrfTransducer<T>::getNbest(std::shared_ptr<Tensor<T>> logits, int num_nbest){
        // Use partial_sort instead of full priority_queue - O(N + K*logK) vs O(N*logN)
        const int vocab_size = logits->shape(1);
        const int start_idx = 2;
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
        // partial_sort puts top-K elements at front in descending order
        std::partial_sort(all_scores.begin(), all_scores.begin() + k, all_scores.end(),
                         [](const Score& a, const Score& b) { return a.logit > b.logit; });

        all_scores.resize(k);
        return all_scores;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformerTrfTransducer<T>::encoding(std::shared_ptr<Tensor<T>> feature){
        std::shared_ptr<Tensor<T>> enc = conformer_stream_encoder->forward(feature);
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " encoding forward done");
        LoggerManager::Log("DEBUG", "incremental start");
        #endif
        if(enc_out_val == nullptr){
            LoggerManager::Log("INFO", " enc_out_val is nullptr");
            enc_out_val = std::make_shared<Tensor<T>>();
            enc_out_val->value = enc->value;
        }
        else{
            LoggerManager::Log("INFO", " enc_out_val is not nullptr");
            LoggerManager::Log("INFO", " enc_out_val shape: " + std::to_string(enc_out_val->shape(0)));
            LoggerManager::Log("INFO", " enc shape: " + std::to_string(enc->shape(0)));
            enc_out_val->appendChunkRow(enc);
        }
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " incremental done");
        #endif
        return enc;
    }

    template<typename T>
    void AsrStreamConformerTrfTransducer<T>::warmUpInference(std::shared_ptr<Tensor<T>> feature){
        struct timespec warmup_start, warmup_end;

        clock_gettime(CLOCK_MONOTONIC, &warmup_start);
        std::shared_ptr<Tensor<T>> enc_out = encoding(feature);
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " enc_out.shape(0): " + std::to_string(enc_out->shape(0)));
        #endif

        // 2. Decoder warmup - transformer transducer 실행
        std::shared_ptr<Tensor<int>> dummy_token = std::make_shared<Tensor<int>>();
        dummy_token->setValue({sos}, {1, 1}); // SOS token
        
        // 각 time step에 대해 decoder 실행
        std::shared_ptr<Tensor<T>> each_enc = std::make_shared<Tensor<T>>();
        each_enc->value.resize(1, enc_out->shape(1));
        each_enc->value = enc_out->value.row(0);

        #ifdef USE_TRANSFORMER_POOL
        auto transformer_instance = getTransformerInstance();
        std::shared_ptr<Tensor<T>> logits = transformer_instance->forwardContext(dummy_token, each_enc, 0);
        #else
        std::shared_ptr<Tensor<T>> logits = transformer_transducer->forwardContext(dummy_token, each_enc, 0);
        #endif
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " logits.shape(0): " + std::to_string(logits->shape(0)));
        #endif

        // 3. Phone/Word decoder warmup
        // std::vector<int> dummy_phone_dec = {2}; // dummy phone sequence
        // std::vector<int> dummy_word_dec = {2}; // dummy word sequence
        
        // std::shared_ptr<Tensor<float>> p_out = getPhoneScore(dummy_phone_dec);
        // std::shared_ptr<Tensor<float>> w_out = getWordScore(dummy_word_dec);
        
        // 4. Clear state after warmup
        clear();
        
        clock_gettime(CLOCK_MONOTONIC, &warmup_end);
        float warmup_time = (warmup_end.tv_sec - warmup_start.tv_sec) + 
                           (warmup_end.tv_nsec - warmup_start.tv_nsec) / 1000000000.0;
        
        LoggerManager::Log("INFO", "Complete warmup finished in " + std::to_string(warmup_time) + "s");
    }

    template<typename T>
    bool AsrStreamConformerTrfTransducer<T>::isBreak(std::vector<Beam> stop, std::vector<Beam> worker, int beam_size){
        if(worker.empty()) return true;
        int count=0;
        float best_score = worker[0].normscore;

        for(size_t i=0; i<stop.size(); ++i){
            if(stop[i].normscore > best_score){
                count++;
            }
        }
        return count >= beam_size;
    }        


    template<typename T>
    void AsrStreamConformerTrfTransducer<T>::initializeTransformerPool(int num_instance) {
        std::unique_lock<std::mutex> lock(pool_mutex);
        if(pool_initialized) return;
        
        for(int i=0; i<num_instance; ++i) {
            auto copy_transformer_config = transformer_config.clone();
            auto instance = std::make_shared<TransformerTransducer<T>>(copy_transformer_config, config_base_path);
            if (num_layer < 0) {
                num_layer = instance->getNumLayer();
            }
            transformer_pool.push_back(instance);
        }
        
        pool_initialized = true;
        LoggerManager::Log("INFO", "TransformerTransducer pool initialized with " + std::to_string(num_instance) + " instances");
    }

    template<typename T>
    std::shared_ptr<TransformerTransducer<T>> AsrStreamConformerTrfTransducer<T>::getTransformerInstance() {
        std::unique_lock<std::mutex> lock(pool_mutex);
        
        // 풀이 초기화되지 않았으면 초기화
        if(!pool_initialized) {
            lock.unlock();
            initializeTransformerPool(5);  // 8개 인스턴스로 초기화
            lock.lock();
        }
        
        if(!transformer_pool.empty()) {
            auto instance = transformer_pool.front();
            transformer_pool.erase(transformer_pool.begin());
            return instance;
        }
            
        // 풀이 비어있으면 새로운 인스턴스 생성 (동적 확장)
        LoggerManager::Log("WARNING", "TransformerTransducer pool empty, creating new instance");
        return std::make_shared<TransformerTransducer<T>>(transformer_config, config_base_path);
    }

    template<typename T>
    void AsrStreamConformerTrfTransducer<T>::returnTransformerInstance(std::shared_ptr<TransformerTransducer<T>> instance) {
        std::unique_lock<std::mutex> lock(pool_mutex);
        
        // 풀 크기 제한 (최대 16개)
        if(transformer_pool.size() < 5) {
            // 인스턴스 상태 초기화
            instance->clear();
            transformer_pool.push_back(instance);
        }
        // 풀이 가득 차면 인스턴스는 자동으로 소멸됨
    }

    template<typename T>
    std::pair<Beam, std::vector<Beam>> AsrStreamConformerTrfTransducer<T>::eachBeamSearch(Beam& beam, std::shared_ptr<Tensor<T>> enc_in, int beam_size){
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", "current beam print");
        beam.print();
        #endif

        // Save stop_cache before forwardContext mutates beam.cache
        // Clone once, share via shared_ptr for stopIncremental
        auto stop_cache = std::make_shared<DecoderCache>();
        stop_cache->clone(beam.cache);

        std::shared_ptr<Tensor<int>> dec_in = std::make_shared<Tensor<int>>();
        dec_in->setValue({beam.word_ids.back()}, {1,1});

        std::shared_ptr<Tensor<T>> prd;
        #ifdef USE_TRANSFORMER_POOL
        auto transformer_instance = getTransformerInstance();
        transformer_instance->setDecoderCache(beam.cache);
        prd = transformer_instance->forwardContext(dec_in, enc_in, beam.position);
        returnTransformerInstance(transformer_instance);
        #else
        transformer_transducer->setDecoderCache(beam.cache);
        prd = transformer_transducer->forwardContext(dec_in, enc_in, beam.position);
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " prd print");
        prd->printValue();
        #endif
        #endif
        Softmax softmax;
        prd = softmax.forward(prd);
        Beam stop;
        stop.stopIncremental(beam, std::log(prd->value(0, 1)), stop_cache);
        #ifdef DEBUG
        LoggerManager::Log("DEBUG", " stop beam print");
        stop.print();
        #endif

        std::vector<Score> scores = getNbest(prd, beam_size);
        std::vector<Beam> worker;
        worker.reserve(scores.size());
        for(size_t i=0;i<scores.size();++i){
            Beam tmp;
            tmp.workerIncremental(beam, scores[i].idx, scores[i].logit);
            #ifdef DEBUG
            LoggerManager::Log("DEBUG", " new beam print");
            tmp.print();
            #endif
            worker.push_back(std::move(tmp));
        }
        return std::make_pair(std::move(stop), std::move(worker));
    }


    template<typename T>
    std::vector<Beam> AsrStreamConformerTrfTransducer<T>::run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams){
        struct timespec begin, end ;
        std::vector<Beam> beam_worker;
        std::vector<Beam> stop_worker;
        std::vector<Beam> running_worker;
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
        
        clock_gettime(CLOCK_MONOTONIC, &begin);
        beam_worker.swap(beams);

        int t = enc->shape(0);
        LoggerManager::Log("INFO", " encoding size: " + std::to_string(t));

        // Reuse enc_in tensor across time steps to avoid per-step allocation
        std::shared_ptr<Tensor<T>> enc_in = std::make_shared<Tensor<T>>();
        enc_in->value.resize(1, enc->shape(1));

        for(int i=0; i < t; ++i) {
            enc_in->value = enc->value.row(i);
            #ifdef DEBUG
            LoggerManager::Log("DEBUG", "decoding start :" + std::to_string(i) + " / " + std::to_string(t));
            #endif

            #ifdef USE_TRANSFORMER_POOL
            std::vector<std::future<std::pair<Beam, std::vector<Beam>>>> future_vector;
            #endif

            while(true) {
                #ifdef DEBUG
                LoggerManager::Log("DEBUG", "make stop & worker start");
                #endif
                // Use const reference to avoid copying each beam
                for (const auto& beam : beam_worker){
                    #ifdef USE_TRANSFORMER_POOL
                    // Lambda must copy beam for thread safety, but use explicit copy
                    Beam beam_copy(beam);
                    future_vector.push_back(MERCURY::ThreadPool::submit([this, beam_copy = std::move(beam_copy), enc_in, num_nbest]() mutable {
                        return this->eachBeamSearch(beam_copy, enc_in, num_nbest);
                    }));
                    #else
                    auto result = eachBeamSearch(const_cast<Beam&>(beam), enc_in, num_nbest);
                    stop_worker.push_back(std::move(result.first));
                    // Move beams from result instead of copying
                    for(auto& new_beam : result.second){
                        running_worker.push_back(std::move(new_beam));
                    }
                    #endif
                }
                #ifdef USE_TRANSFORMER_POOL
                for(size_t j=0; j < future_vector.size(); ++j){
                    auto result = future_vector[j].get();
                    stop_worker.push_back(std::move(result.first));
                    for(auto& new_beam : result.second){
                        running_worker.push_back(std::move(new_beam));
                    }
                }
                future_vector.clear();
                #endif
                #ifdef DEBUG
                LoggerManager::Log("DEBUG", "make stop & worker done");
                #endif

                // Use partial_sort for top-N instead of full sort
                std::size_t num_splice = std::min(running_worker.size(), static_cast<std::size_t>(num_nbest));
                if (num_splice < running_worker.size()) {
                    std::partial_sort(running_worker.begin(), running_worker.begin() + num_splice, running_worker.end(),
                                    [](const Beam& a, const Beam& b) { return a.normscore > b.normscore; });
                }
                // Truncate in-place instead of allocating a new vector
                running_worker.resize(num_splice);

                // stop_worker only needs partial sort too
                {
                    std::size_t stop_splice = std::min(stop_worker.size(), static_cast<std::size_t>(num_nbest));
                    if (stop_splice < stop_worker.size()) {
                        std::partial_sort(stop_worker.begin(), stop_worker.begin() + stop_splice, stop_worker.end(),
                                        [](const Beam& a, const Beam& b) { return a.normscore > b.normscore; });
                    }
                }

                if(isBreak(stop_worker, running_worker, num_nbest)) break;
                beam_worker.swap(running_worker);
                running_worker.clear();
            }
            //enc element process end

            //beam process start
            std::size_t num_splice = std::min(stop_worker.size(), static_cast<std::size_t>(num_nbest));
            // Truncate in-place instead of creating new vector
            stop_worker.resize(num_splice);
            beam_worker.swap(stop_worker);
            stop_worker.clear();
            #ifdef DEBUG
            LoggerManager::Log("DEBUG",  std::to_string(i) + "th beam result");
            for (const auto& beam : beam_worker){
                const_cast<Beam&>(beam).print();
            }
            #endif
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        LoggerManager::Log("INFO", " beam-search time: " + std::to_string(time));
        // #endif
        
        return beam_worker;
    }

    template<typename T>
    std::vector<int> AsrStreamConformerTrfTransducer<T>::secondPassDecoding(int num_nbest){
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

        Softmax softmax;
        std::shared_ptr<Tensor<int>> token = std::make_shared<Tensor<int>>();
        while(nbest.size() < static_cast<size_t>(num_nbest)) {
            while(!beamsearch.empty()) {
                Beam cur_beam = beamsearch.top();
                beamsearch.pop();
                token->setValue({cur_beam.word_ids.back()}, {1,1});
                word_decoder->setDecoderCache(cur_beam.cache);
                std::shared_ptr<Tensor<T>> logits = word_decoder->forwardContext(token, ctc_out_val, cur_beam.position);
                logits = softmax.forward(logits);
                std::vector<Score> scores = getNbest(logits, num_nbest);
                for(size_t i=0;i<scores.size();++i){
                    Beam run_beam(sos, eos);
                    run_beam.workerIncremental(cur_beam, scores[i].idx, scores[i].org_logit);
                    running.push(std::move(run_beam));
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

    template class AsrStreamConformerTrfTransducer<float>;
    // template class AsrStreamConformerTrfTransducer<int>;
};

