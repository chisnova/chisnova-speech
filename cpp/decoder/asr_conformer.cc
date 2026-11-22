#include "decoder/asr_conformer.h"

namespace MERCURY {
    template<typename T>
    AsrConformer<T>::AsrConformer(JsonObject asr_conformer_path, std::string base_path) {
        conformer_encoder = std::make_shared<ConformerEncoder<T>>(asr_conformer_path.parseObject("conformer_encoder"), base_path);
        conformer_decoder = std::make_shared<ConformerDecoder<T>>(asr_conformer_path.parseObject("conformer_decoder"), base_path);
        sos=1;
        eos=2;
    }

    template<typename T>
    AsrConformer<T>::~AsrConformer() {
        conformer_encoder = nullptr;
        conformer_decoder = nullptr;
        enc_out_val = nullptr;
        ctc_out_val = nullptr;
    }

    template<typename T>
    int AsrConformer<T>::getNumDecoderLayer() {
        return conformer_decoder->getNumLayer();
    }

    template<typename T>
    std::vector<Score> AsrConformer<T>::getNbest(std::shared_ptr<Tensor<float>> logits, int num_nbest) {
        std::priority_queue<Score> nbest;
        std::vector<Score> ret;

        for(int i=0; i<logits->shape(1); ++i) {
            Score new_score;
            float val = std::log(logits->value(0, i));
            new_score.idx = i;
            new_score.logit = val;
            new_score.org_logit =logits->value(0, i);
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
    void AsrConformer<T>::setDecoderCache(std::shared_ptr<DecoderCache> _decoder_cache) {
        conformer_decoder->setDecoderCache(_decoder_cache);
    }

    template<typename T>
    std::vector<Beam> AsrConformer<T>::makeInitBeams() {
        std::vector<Beam> ret;
        Beam init_beam(sos, eos);
        init_beam.makeInitBeam(conformer_decoder->getNumLayer());
        ret.push_back(init_beam);
        return ret;
    }

    template<typename T>
    void AsrConformer<T>::clear(){
        enc_out_val = nullptr;
        ctc_out_val = nullptr;
        conformer_decoder->clear();
    }

    template<typename T>
    std::shared_ptr<Tensor<float>> AsrConformer<T>::getCtcOut() {
        return ctc_out_val;
    }
    
    template<typename T>
    std::shared_ptr<Tensor<T>> AsrConformer<T>::encoding(std::shared_ptr<Tensor<T>> enc) {
        enc_out_val = conformer_encoder->forward(enc);
        ctc_out_val = conformer_encoder->getCTCOut();
        ctc_out_val = Softmax().forward(ctc_out_val);
        return enc_out_val;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrConformer<T>::decoding(std::shared_ptr<Tensor<int>> dec, int position) {
        std::shared_ptr<Tensor<T>> ret;
        if (enc_out_val == nullptr) {
            std::cout << "enc_output is nullptr" << std::endl;
            assert(false);
        }
        ret = conformer_decoder->forwardContext(dec, enc_out_val, position);
        return ret;
    }

    template<typename T>
    std::vector<Beam> AsrConformer<T>::run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams) {
        clear();
        struct timespec begin, end ;
        std::vector<Beam> ret;
        std::priority_queue<Beam> beamsearch;
        std::priority_queue<Beam> nbest;
        std::priority_queue<Beam> running;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        encoding(feature);
        clock_gettime(CLOCK_MONOTONIC, &end);
        float time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        LoggerManager::Log("INFO", " encoder time: " + std::to_string(time));
        clock_gettime(CLOCK_MONOTONIC, &begin);
        // Beam initBeam(sos, eos);
        // int numDecLayer = getNumDecoderLayer();
        // initBeam.makeInitBeam(numDecLayer);
        // beamsearch.push(initBeam);
        for(int i=0;i<beams.size();++i){
            beamsearch.push(beams[i]);
        }

        while(nbest.size() < num_nbest) {
            while(!beamsearch.empty()) {
                Beam cur_beam = beamsearch.top();
                // cur_beam.print();
                beamsearch.pop();
                std::shared_ptr<Tensor<int>> token = std::make_shared<Tensor<int>>();
                token->setValue({cur_beam.ids.back()}, {1,1});
                setDecoderCache(cur_beam.cache);
                std::shared_ptr<Tensor<float>> logits = decoding(token, cur_beam.position);
                std::vector<Score> scores = getNbest(logits, num_nbest);
                for(int i=0;i<scores.size();++i){
                    Beam run_beam(sos, eos);
                    run_beam.incremental(cur_beam, scores[i].idx, scores[i].logit, scores[i].org_logit);
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
        LoggerManager::Log("INFO", " beam search time:" + std::to_string(time));
        Beam result = nbest.top();
        while(!nbest.empty()) {
            ret.push_back(nbest.top());
            nbest.pop();
        }
        while(!beamsearch.empty()) beamsearch.pop();
        // std::cout << "score: " << result.score << std::endl;
        score = result.score;
        return ret;
    }

    template<typename T>
    std::vector<int> AsrConformer<T>::getCtcAsrIds() {
        std::vector<int> ctc_asr_ids;
        return ctc_asr_ids;
    }

    template class AsrConformer<float>;
    // template class AsrConformer<int>;
};

