#include "decoder/asr_stream_conformer.h"
#include "logger/logger_manager.h"

namespace MERCURY {
    template<typename T>
    AsrStreamConformer<T>::AsrStreamConformer(JsonObject asr_stream_conformer_config, std::string base_path) {
        stream_encoder = std::make_shared<ConformerStreamEncoder<T>>(asr_stream_conformer_config.parseObject("stream_encoder"), base_path);
        second_encoder = std::make_shared<ConformerCTC<T>>(asr_stream_conformer_config.parseObject("second_encoder"), base_path);
        ctc_linear1 = std::make_shared<Linear<T>>(asr_stream_conformer_config.parseObject("ctc_linear1"), base_path);
        ctc_linear2 = std::make_shared<Linear<T>>(asr_stream_conformer_config.parseObject("ctc_linear2"), base_path);

        // Python TransformerDecoder = tin_emb -> pos/layers -> layer_norm -> tout_emb.
        // C++ TransformerDecoder 모듈은 pos/layers 만 담당하므로 나머지는 여기서 잡는다.
        JsonObject word_decoder_config = asr_stream_conformer_config.parseObject("wrd_decoder");
        word_token_embedding = std::make_shared<Embedding>(word_decoder_config.parseObject("tin_emb"), base_path);
        word_decoder = std::make_shared<TransformerDecoder<T>>(word_decoder_config, base_path);
        word_decoder_norm = std::make_shared<LayerNorm<T>>(word_decoder_config.parseObject("layer_norm"), base_path);
        word_output_linear = std::make_shared<Linear<T>>(word_decoder_config.parseObject("tout_emb"), base_path);

        JsonObject phone_decoder_config = asr_stream_conformer_config.parseObject("phn_decoder");
        phone_token_embedding = std::make_shared<Embedding>(phone_decoder_config.parseObject("tin_emb"), base_path);
        phone_decoder = std::make_shared<TransformerDecoder<T>>(phone_decoder_config, base_path);
        phone_decoder_norm = std::make_shared<LayerNorm<T>>(phone_decoder_config.parseObject("layer_norm"), base_path);
        phone_output_linear = std::make_shared<Linear<T>>(phone_decoder_config.parseObject("tout_emb"), base_path);

        ctc_time_reduction = int(asr_stream_conformer_config.parserValue("ctc_time_reduction"));
        stream_encoder->setStreamLeftContext(
            int(asr_stream_conformer_config.parserValue("stream_left_context")));
        num_layer = int(word_decoder_config.parserValue("num_layer"));
        sos = int(asr_stream_conformer_config.parserValue("sos_id"));
        eos = int(asr_stream_conformer_config.parserValue("eos_id"));
    }

    template<typename T>
    AsrStreamConformer<T>::~AsrStreamConformer() {
        stream_encoder = nullptr;
        second_encoder = nullptr;
        ctc_linear1 = nullptr;
        ctc_linear2 = nullptr;
        word_token_embedding = nullptr;
        word_decoder = nullptr;
        word_decoder_norm = nullptr;
        word_output_linear = nullptr;
        phone_token_embedding = nullptr;
        phone_decoder = nullptr;
        phone_decoder_norm = nullptr;
        phone_output_linear = nullptr;
        enc_out_val = nullptr;
        enc2_out_val = nullptr;
    }

    template<typename T>
    void AsrStreamConformer<T>::clear() {
        enc_out_val = nullptr;
        enc2_out_val = nullptr;
        stream_encoder->clear();
        word_decoder->clear();
        phone_decoder->clear();
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformer<T>::forward(std::shared_ptr<Tensor<int>> dec, std::shared_ptr<Tensor<T>> enc) {
        return stream_encoder->forward(enc);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformer<T>::encoding(std::shared_ptr<Tensor<T>> feature) {
        std::shared_ptr<Tensor<T>> encoded_chunk = stream_encoder->forward(feature);
        if (encoded_chunk->value.rows() == 0) {
            return encoded_chunk;   // subsample 커널을 못 채운 짧은 청크 — 다음 청크에서 이어진다
        }
        if (enc_out_val == nullptr) {
            enc_out_val = std::make_shared<Tensor<T>>();
            enc_out_val->value = encoded_chunk->value;
        }
        else {
            enc_out_val->appendChunkRow(encoded_chunk);
        }
        // enc1 이 자랐으므로 이전 2패스 인코딩은 더 이상 유효하지 않다
        enc2_out_val = nullptr;
        return encoded_chunk;
    }

    template<typename T>
    std::vector<Beam> AsrStreamConformer<T>::run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams) {
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        encoding(feature);
        clock_gettime(CLOCK_MONOTONIC, &end);
        float time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        LoggerManager::Log("INFO", " encoder time: " + std::to_string(time));
        return beams;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformer<T>::reduceEncodingTimeResolution(std::shared_ptr<Tensor<T>> encoded_frames) {
        if (ctc_time_reduction <= 1) {
            return encoded_frames;
        }
        // Python: F.avg_pool1d(kernel=stride=ctc_time_reduction, ceil_mode=True)
        int input_frame_count = encoded_frames->value.rows();
        int pooled_frame_count = (input_frame_count + ctc_time_reduction - 1) / ctc_time_reduction;
        std::shared_ptr<Tensor<T>> pooled_frames = std::make_shared<Tensor<T>>();
        pooled_frames->value.resize(pooled_frame_count, encoded_frames->value.cols());
        for (int pooled_index = 0; pooled_index < pooled_frame_count; ++pooled_index) {
            int segment_start = pooled_index * ctc_time_reduction;
            int segment_length = std::min(ctc_time_reduction, input_frame_count - segment_start);
            pooled_frames->value.row(pooled_index) =
                encoded_frames->value.middleRows(segment_start, segment_length).colwise().mean();
        }
        return pooled_frames;
    }

    template<typename T>
    void AsrStreamConformer<T>::encodeSecondPassIfNeeded() {
        if (enc2_out_val != nullptr) {
            return;
        }
        if (enc_out_val == nullptr) {
            // 오디오가 한 청크도 인코딩되지 않았다(무음 종료·즉시 취소 등).
            // 여기서 막지 않으면 아래 forward 들이 널을 그대로 역참조한다.
            LoggerManager::Log("WARN", " second pass skipped: no encoded audio ");
            return;
        }
        enc2_out_val = second_encoder->forward(reduceEncodingTimeResolution(enc_out_val));
    }

    template<typename T>
    std::vector<int> AsrStreamConformer<T>::getFirstPassCtcAsrIds() {
        if (enc_out_val == nullptr) {
            return std::vector<int>();   // 아직 인코딩된 청크가 없다
        }
        std::shared_ptr<Tensor<T>> ctc_out = ctc_linear1->forward(enc_out_val);
        ctc_out = Softmax().forward(ctc_out);
        std::vector<int> ctc_asr_ids;
        for (auto row : ctc_out->value.rowwise()) {
            int max_idx;
            row.maxCoeff(&max_idx);
            ctc_asr_ids.push_back(max_idx);
        }
        return ctc_asr_ids;
    }

    template<typename T>
    std::vector<int> AsrStreamConformer<T>::getCtcAsrIds() {
        encodeSecondPassIfNeeded();
        if (enc2_out_val == nullptr) {
            return std::vector<int>();
        }
        std::shared_ptr<Tensor<T>> ctc_out = ctc_linear2->forward(enc2_out_val);
        ctc_out = Softmax().forward(ctc_out);
        std::vector<int> ctc_asr_ids;
        for (auto row : ctc_out->value.rowwise()) {
            int max_idx;
            row.maxCoeff(&max_idx);
            ctc_asr_ids.push_back(max_idx);
        }
        return ctc_asr_ids;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformer<T>::getWordScore(std::vector<int> word_dec) {
        std::shared_ptr<Tensor<int>> word_dec_tensor = std::make_shared<Tensor<int>>();
        word_dec_tensor->setValue(word_dec, {1, int(word_dec.size())});
        encodeSecondPassIfNeeded();
        if (enc2_out_val == nullptr) {
            return std::make_shared<Tensor<T>>();
        }
        std::shared_ptr<Tensor<T>> ret = word_decoder->forward(word_token_embedding->forward(word_dec_tensor), enc2_out_val);
        ret = word_decoder_norm->forward(ret);
        ret = word_output_linear->forward(ret);
        ret = Softmax().forward(ret);
        return ret;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrStreamConformer<T>::getPhoneScore(std::vector<int> phone_dec) {
        std::shared_ptr<Tensor<int>> phone_dec_tensor = std::make_shared<Tensor<int>>();
        phone_dec_tensor->setValue(phone_dec, {1, int(phone_dec.size())});
        encodeSecondPassIfNeeded();
        if (enc2_out_val == nullptr) {
            return std::make_shared<Tensor<T>>();
        }
        std::shared_ptr<Tensor<T>> ret = phone_decoder->forward(phone_token_embedding->forward(phone_dec_tensor), enc2_out_val);
        ret = phone_decoder_norm->forward(ret);
        ret = phone_output_linear->forward(ret);
        ret = Softmax().forward(ret);
        return ret;
    }

    template<typename T>
    std::vector<Beam> AsrStreamConformer<T>::makeInitBeams() {
        std::vector<Beam> ret;
        Beam init_beam = Beam(sos, eos);
        init_beam.makeInitBeam(num_layer);
        ret.push_back(init_beam);
        return ret;
    }

    template<typename T>
    std::vector<Score> AsrStreamConformer<T>::getNbest(std::shared_ptr<Tensor<T>> logits, int num_nbest) {
        std::priority_queue<Score> nbest;
        std::vector<Score> ret;

        for (int i = 2; i < logits->shape(1); ++i) {
            Score new_score;
            float val = std::log(logits->value(0, i));
            new_score.idx = i;
            new_score.logit = val;
            new_score.org_logit = logits->value(0, i);
            nbest.push(new_score);
        }

        while (!nbest.empty() && ret.size() < num_nbest) {
            Score cur_score = nbest.top();
            nbest.pop();
            ret.push_back(cur_score);
        }
        return ret;
    }

    template<typename T>
    void AsrStreamConformer<T>::warmUpInference(std::shared_ptr<Tensor<T>> feature) {
    }

    template<typename T>
    std::vector<int> AsrStreamConformer<T>::secondPassDecoding(int num_nbest) {
        encodeSecondPassIfNeeded();
        if (enc2_out_val == nullptr) {
            return std::vector<int>();
        }

        if (num_nbest < 1) throw std::invalid_argument("beam size must be positive");
        auto beams = makeInitBeams();
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        // Match Python attention_beam_search: finished hypotheses still occupy
        // slots in the top-k, and the encoder length bounds the number of steps.
        for (int step = 0; step < enc2_out_val->value.rows(); ++step) {
            std::vector<Beam> candidates;
            for (auto& beam : beams) {
                if (beam.isFinal) {
                    candidates.push_back(std::move(beam));
                    continue;
                }
                auto token = std::make_shared<Tensor<int>>();
                token->setValue({beam.word_ids.back()}, {1, 1});
                word_decoder->setDecoderCache(beam.cache);
                auto hidden = word_decoder->forwardContext(
                    word_token_embedding->forward(token), enc2_out_val, beam.position);
                auto logits = word_output_linear->forward(word_decoder_norm->forward(hidden));
                auto probabilities = Softmax().forward(logits);
                for (const auto& score : getNbest(probabilities, num_nbest)) {
                    Beam next(sos, eos);
                    next.workerIncremental(beam, score.idx, score.logit);
                    candidates.push_back(std::move(next));
                }
            }
            std::stable_sort(candidates.begin(), candidates.end(), [](const Beam& a, const Beam& b) {
                return a.score > b.score;
            });
            if (candidates.size() > num_nbest) candidates.resize(num_nbest);
            beams = std::move(candidates);
            if (std::all_of(beams.begin(), beams.end(), [](const Beam& beam) { return beam.isFinal; }))
                break;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        const double elapsed = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        LoggerManager::Log("INFO", " second pass beam search time:" + std::to_string(elapsed));
        return beams.empty() ? std::vector<int>() : beams.front().word_ids;
    }

    template class AsrStreamConformer<float>;
};
