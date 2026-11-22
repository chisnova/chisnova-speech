#include "decoder/asr_async_decoder.h"

#include <cmath>
#include <cstdio>


namespace MERCURY {
    template<typename T>
    AsrAsyncDecoder<T>::AsrAsyncDecoder(JsonObject &asr_model_path, std::string base_path) {
        std::string type = asr_model_path.parserString("model_type");
        if (type.compare("stream-conformer-transformer-transducer") == 0) {
            asr_model = std::make_shared<AsrStreamConformerTrfTransducer<T>>(asr_model_path.parseObject("model"), base_path);
            LoggerManager::Log("INFO", " stream conformer transducer init done ");
        }
        else if (type.compare("conformer_stream") == 0) {
            JsonObject model_config = asr_model_path.parseObject("model");
            asr_model = std::make_shared<AsrStreamConformer<T>>(model_config, base_path);
            ctc_blank_id = int(model_config.parserValue("blank_id"));
            LoggerManager::Log("INFO", " stream conformer (2-pass cascade) init done, blank_id: "
                                       + std::to_string(ctc_blank_id));
        }
        else {
            throw std::invalid_argument("unsupported model_type: " + type);
        }
        word_vocab_size = int(asr_model_path.parserValue("vocab_size"));
        phone_vocab_size = int(asr_model_path.parserValue("phone_size"));
        feature_extractor = std::make_shared<LibrosaFeature>(asr_model_path, base_path);
        beams.clear();
        beams = asr_model->makeInitBeams();
        chunk_audio.clear();
        // 파일 이름들은 manifest(model.json) 가 기록한 값을 따른다 — vocab 이 바뀌면 함께 바뀐다
        scorer = std::make_shared<Scorer>(base_path,
                                          asr_model_path.parserString("vocab_model"),
                                          asr_model_path.parserString("vocab_name"),
                                          asr_model_path.parserString("phone_name"),
                                          asr_model_path.parserString("dict_name"),
                                          int(asr_model_path.parserValue("phone_size")),
                                          asr_model_path.parserValue("phone_weight",
                                                                     Scorer::DEFAULT_PHONE_WEIGHT),
                                          asr_model_path.parserStringOr("function_word_list", ""),
                                          asr_model_path.parserValue("function_word_floor",
                                                                     Scorer::DEFAULT_FUNCTION_WORD_FLOOR),
                                          asr_model_path.parserValue("recognition_weight",
                                                                     Scorer::DEFAULT_RECOGNITION_WEIGHT));
        // 고유명사 하한: 목록 파일과 점수를 manifest 가 정한다. 없으면 쓰지 않는다.
        scorer->setProperNounFloor(base_path,
                                   asr_model_path.parserStringOr("proper_noun_list", ""),
                                   asr_model_path.parserValue("proper_noun_floor",
                                                              Scorer::DEFAULT_PROPER_NOUN_FLOOR));
        // 채점 제외 낱말 목록(마지막 수단). 없으면 모든 낱말을 채점한다.
        scorer->setUngradedWords(base_path, asr_model_path.parserStringOr("ungraded_word_list", ""));
        #ifdef THREAD_POOL_ENABLED
        LoggerManager::Log("INFO", "asr_async_decoder thread pool enabled ");
        (void)MERCURY::ThreadPool::getInstance();
        #endif
    }

    template<typename T>
    AsrAsyncDecoder<T>::~AsrAsyncDecoder() {
        stopEngineThread();
        asr_model = nullptr;
        feature_extractor = nullptr;
        #ifdef THREAD_POOL_ENABLED
        LoggerManager::Log("INFO", "asr_async_decoder thread pool cleanup ");
        (void)MERCURY::ThreadPool::getInstance().cleanupDynamicThreads();
        #endif
    }

    template<typename T>
    std::string AsrAsyncDecoder<T>::getSecondPassDecodingResult(int num_nbest){
        if (engine_error) std::rethrow_exception(engine_error);
        if (num_nbest <= 0) throw std::invalid_argument("beam size must be positive");
        std::string asr_result;
        std::vector<int> second_pass_decoding_ids = asr_model->secondPassDecoding(num_nbest);
        asr_result.reserve(second_pass_decoding_ids.size() * 4);
        for(size_t i=0;i<second_pass_decoding_ids.size();++i){
            asr_result += vocab.at(second_pass_decoding_ids[i]);
        }
        replaceAll(asr_result, "▁"," ");
        replaceAll(asr_result, "<s>","");
        replaceAll(asr_result, "</s>","");
        return asr_result;
    }


    template<typename T>
    void AsrAsyncDecoder<T>::clear() {
        // Model caches must outlive every use by the decoding thread.
        stopEngineThread();
        // chunk_audio 는 엔진 스레드도 만진다 — 잠금 없이 비우면 자료가 깨진다
        {
            std::lock_guard<std::mutex> lock(mtx);
            chunk_audio.clear();
            isFinalAudio = false;
            data_ready = false;
        }
        // 문장별 점수 비중이 다음 문장으로 새지 않게 한다 — 호출자가 문장마다 지정한다
        sentence_score_weights.reset();
        asr_model->clear();
        std::vector<Beam> inits = asr_model->makeInitBeams();
        beams.swap(inits);
        engine_error = nullptr;
        ctc_trailing_silence_ms.store(-1);
        feature_extractor->clear();
        is_canceled = false;
    }

    // 엔진 스레드를 확실히 세운다.
    // is_canceled 만 세워 두면 조건변수에서 자고 있는 스레드가 깨어나지 못하므로 함께 깨운다.
    template<typename T>
    void AsrAsyncDecoder<T>::stopEngineThread() {
        is_canceled = true;
        {
            std::lock_guard<std::mutex> lock(mtx);
            data_ready = true;
        }
        data_cond.notify_all();
        if (engineThread.joinable()) {
            engineThread.join();
        }
        is_canceled = false;
    }

    template<typename T>
    void AsrAsyncDecoder<T>::cancelDecoding() {
        // 예전에는 플래그만 세우고 스레드를 거두지 않았다. 그러면 다음 startBackGroundEngine 의
        // std::thread 이동 대입이 joinable 한 객체를 덮어써 std::terminate 로 프로세스가 죽는다.
        stopEngineThread();
    }

    template<typename T>
    void AsrAsyncDecoder<T>::startBackGroundEngine(int num_nbest) {
        if (num_nbest <= 0) throw std::invalid_argument("beam size must be positive");
        // 이전 세션의 스레드가 남아 있으면 먼저 거둔다.
        // joinable 한 std::thread 에 이동 대입하면 그 자리에서 std::terminate 다.
        if (engineThread.joinable()) {
            LoggerManager::Log("WARN", " previous engine thread still alive — stopping it first ");
            stopEngineThread();
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            data_ready = false;
            chunk_audio.clear();
            isFinalAudio = false;
        }
        engine_error = nullptr;
        engineThread = std::thread(&AsrAsyncDecoder::engine, this, num_nbest);
    }

    template<typename T>
    void AsrAsyncDecoder<T>::finishDecoding() {
        LoggerManager::Log("INFO", " finishDecoding: join 시작 (joinable="
                                   + std::to_string(int(engineThread.joinable())) + ")");
        if(engineThread.joinable()){
            // finish is also an end-of-input signal; never wait forever for a final chunk.
            {
                std::lock_guard<std::mutex> lock(mtx);
                isFinalAudio = true;
                data_ready = true;
            }
            data_cond.notify_all();
            engineThread.join();
        }
        if (engine_error) std::rethrow_exception(engine_error);
        LoggerManager::Log("INFO", " finishDecoding: join 완료");
    }

    template<typename T>
    void AsrAsyncDecoder<T>::setCallBack(DecoderCallbackPtr callback) {
        decoder_callback = callback;
    }

    template<typename T>
    void AsrAsyncDecoder<T>::setVocab(std::vector<std::string> _vocab) {
        vocab = _vocab;
    }

    template<typename T>
    std::string AsrAsyncDecoder<T>::getAsrResult(std::vector<int> word_ids) {
        std::string asr_result;
        // Reserve estimated space to reduce reallocations
        asr_result.reserve(word_ids.size() * 4);
        for(size_t i=0;i<word_ids.size();++i){
            asr_result += vocab.at(word_ids[i]);
        }
        replaceAll(asr_result, "▁"," ");
        replaceAll(asr_result, "<s>","");
        replaceAll(asr_result, "</s>","");
        return asr_result;
    }

    template<typename T>
    void AsrAsyncDecoder<T>::replaceAll(std::string& str, const std::string& from, const std::string& to) {
        if (from.empty())
            return;
        size_t start_pos = 0;
        while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length(); // 계속해서 찾기 위해 to.length()만큼 이동
        }
    }

    template<typename T>
    void AsrAsyncDecoder<T>::sendAudio(float audio[], int size, bool _isFinal) {
        if (size < 0 || (size > 0 && audio == nullptr))
            throw std::invalid_argument("invalid audio buffer");
        std::lock_guard<std::mutex> lock(mtx);
        // A late microphone callback must not reopen a finished stream.
        if (isFinalAudio || is_canceled.load()) return;
        isFinalAudio = _isFinal;
        // 종료 신호는 (nullptr, 0) 으로 온다. 널 포인터로 insert 범위를 만드는 것은
        // 크기가 0이어도 정의되지 않은 동작이므로 넣을 것이 있을 때만 부른다.
        if (audio != nullptr && size > 0) {
            // Bulk insert instead of element-by-element push_back
            chunk_audio.insert(chunk_audio.end(), audio, audio + size);
        }
        if (isFinalAudio) {
            LoggerManager::Log("INFO", " sendAudio FINAL — queued: " + std::to_string(chunk_audio.size()));
        }
        // 이만큼 쌓이면 엔진을 깨운다 — 스트리밍 중간 결과가 나오는 주기가 곧 이 값이다.
        //
        // 값을 고른 근거는 헤더의 WAKE_THRESHOLD_SAMPLES 주석에 적어 두었다.
        if(chunk_audio.size() >= WAKE_THRESHOLD_SAMPLES || isFinalAudio){
            // 예전에는 !data_ready 일 때만 깨웠다. 그 사이 엔진이 data_ready 를 내리면
            // 마지막 알림이 통째로 삼켜져 엔진이 영원히 대기했고(특히 오디오를 빠르게 밀어넣을 때),
            // 뒤쪽 오디오가 처리되지 않은 채 남아 finishDecoding 도 돌아오지 않았다.
            // 깨우는 비용은 무시할 만하므로 조건 없이 항상 알린다.
            data_ready = true;
            data_cond.notify_all();
        }
    }

    template<typename T>
    void AsrAsyncDecoder<T>::warmUpInference(std::vector<float> audio){
        std::shared_ptr<Tensor<T>> dummy_feature = feature_extractor->getFeature(audio.data(), audio.size());
        LoggerManager::Log("INFO", " dummy_feature.shape(0): " + std::to_string(dummy_feature->shape(0)));
        asr_model->warmUpInference(dummy_feature);
        LoggerManager::Log("INFO", " warmup inference done ");
    }

    template<typename T>
    void AsrAsyncDecoder<T>::engine(int num_nbest) {
        try {
        LoggerManager::Log("INFO", " start engine backgroud ");
        while(true){
            if(is_canceled){
                LoggerManager::Log("INFO", " cancel decoding ");
                break;
            }
            std::vector<float> audio_to_process;
            bool final_chunk = false;            
            {
                std::unique_lock<std::mutex> lock(mtx);
                data_cond.wait(lock, [&]{ return data_ready; });
                
                audio_to_process.swap(chunk_audio);
                LoggerManager::Log("INFO", " audio_to_process.size(): " + std::to_string(audio_to_process.size()));
                final_chunk = isFinalAudio;

                // 2. 공유 데이터에 대한 모든 조작을 마친 후 상태를 리셋
                data_ready = false;
            // 3. 여기서 뮤텍스 락이 해제됨. 이제 다른 스레드가 sendAudio를 호출해도 안전  .
            }
            LoggerManager::Log("INFO", " engine woke — got " + std::to_string(audio_to_process.size())
                                       + " samples, final=" + std::to_string(int(final_chunk)));
            if (final_chunk && audio_to_process.empty()) {
                // 종료 신호는 왔지만 처리할 데이터가 없으면 루프 종료
                break;
            }
            if(is_canceled){
                LoggerManager::Log("INFO", " cancel decoding ");
                break;
            }

            if (!audio_to_process.empty()) {
                if (final_chunk) {
                    LoggerManager::Log("INFO", " final chunk padding silence ");
                    // Bulk resize+fill instead of element-by-element push_back
                    audio_to_process.resize(audio_to_process.size() + 1600 * 5, 0.0f);
                }
                if(is_canceled){
                    LoggerManager::Log("INFO", " cancel decoding ");
                    break;
                }

                std::shared_ptr<Tensor<T>> feature = feature_extractor->getFeature(audio_to_process.data(), audio_to_process.size());
                // 어느 단계에서 플랫폼 간 결과가 갈리는지 보려고 특징값 통계를 남긴다.
                // 여기가 다르면 특징 추출(FFT/mel), 여기는 같은데 뒤가 다르면 인코더 연산이 원인이다.
                {
                    const auto& fv = feature->value;
                    double sum = 0.0, absSum = 0.0, minV = 1e30, maxV = -1e30;
                    int nonFinite = 0;
                    for (int r = 0; r < fv.rows(); ++r) {
                        for (int c = 0; c < fv.cols(); ++c) {
                            double v = double(fv(r, c));
                            if (!std::isfinite(v)) { ++nonFinite; continue; }
                            sum += v; absSum += std::fabs(v);
                            minV = std::min(minV, v); maxV = std::max(maxV, v);
                        }
                    }
                    const double n = double(std::max<long long>(1, (long long)fv.rows() * fv.cols()));
                    char buf[320];
                    std::snprintf(buf, sizeof(buf),
                        " feature %lldx%lld mean=%.4f absmean=%.4f min=%.3f max=%.3f nonfinite=%d first=[%.3f %.3f %.3f %.3f]",
                        (long long)fv.rows(), (long long)fv.cols(), sum / n, absSum / n, minV, maxV, nonFinite,
                        fv.size() > 3 ? double(fv(0,0)) : 0.0, fv.size() > 3 ? double(fv(0,1)) : 0.0,
                        fv.size() > 3 ? double(fv(0,2)) : 0.0, fv.size() > 3 ? double(fv(0,3)) : 0.0);
                    LoggerManager::Log("INFO", buf);
                }
                #ifdef DEBUG
                LoggerManager::Log("DEBUG", " getFeature done ");
                #endif
                if(is_canceled){
                    LoggerManager::Log("INFO", " cancel decoding ");
                    break;
                }
                beams = asr_model->run(feature, num_nbest, beams);
                {
                    auto enc = asr_model->getEncoderOutputForDiag();
                    if (enc != nullptr && enc->value.size() > 0) {
                        const auto& ev = enc->value;
                        double absSum = 0.0; int nonFinite = 0;
                        for (int r = 0; r < ev.rows(); ++r)
                            for (int c = 0; c < ev.cols(); ++c) {
                                double v = double(ev(r, c));
                                if (!std::isfinite(v)) { ++nonFinite; continue; }
                                absSum += std::fabs(v);
                            }
                        char buf[220];
                        std::snprintf(buf, sizeof(buf),
                            " enc1 %lldx%lld absmean=%.4f nonfinite=%d first=[%.3f %.3f %.3f]",
                            (long long)ev.rows(), (long long)ev.cols(),
                            absSum / double(std::max<long long>(1, (long long)ev.rows() * ev.cols())), nonFinite,
                            double(ev(0,0)), double(ev(0,1)), double(ev(0,2)));
                        LoggerManager::Log("INFO", buf);
                    }
                }
                #ifdef DEBUG
                LoggerManager::Log("DEBUG", " run done  beam size: " + std::to_string(beams.size()));
                LoggerManager::Log("DEBUG", " partial beam print");
                for(auto beam : beams){
                    beam.print();
                }
                #endif
                // 스트리밍 중에는 1패스 CTC(ctc_linear1), 발화가 끝나면 2패스 CTC(ctc_linear2).
                // 최종 결과는 빈 문자열이라도 내보내야 화면에 남은 중간 결과가 정리된다.
                std::string asr_result;
                if (final_chunk) {
                    asr_result = getCtcAsrResult();
                } else {
                    // id 열을 한 번만 얻어 텍스트와 무음 길이 둘 다에 쓴다.
                    // getFirstPassCtcAsrResult() 를 따로 부르면 ctc_linear1 + softmax 를
                    // 한 번 더 돌게 되는데, 누적 프레임 전체에 대한 연산이라 싸지 않다.
                    std::vector<int> ctc_ids = asr_model->getFirstPassCtcAsrIds();
                    updateCtcTrailingSilence(ctc_ids);
                    asr_result = ctcIdsToText(ctc_ids);
                }
                LoggerManager::Log("INFO", std::string(final_chunk ? " ctc2 result: " : " ctc1 result: ") + asr_result);
                if(!is_canceled && decoder_callback != nullptr && (final_chunk || !asr_result.empty())){
                    decoder_callback->onResult(asr_result);
                }
            }

            if (final_chunk) {
                LoggerManager::Log("INFO", " engine: final 처리 완료, 루프 종료");
                break; // 최종 청크 처리 후 루프 종료
            }
        }
        LoggerManager::Log("INFO", " finish engine background ");
        } catch (...) {
            // Exceptions must not escape a std::thread entry point (std::terminate).
            engine_error = std::current_exception();
            LoggerManager::Log("ERROR", "decoder worker failed; finishDecoding will report the error");
        }
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrAsyncDecoder<T>::getPhoneScore(std::vector<int> phone_dec){
        return asr_model->getPhoneScore(phone_dec);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrAsyncDecoder<T>::getWordScore(std::vector<int> word_dec){
        return asr_model->getWordScore(word_dec);
    }

    template<typename T>
    std::string AsrAsyncDecoder<T>::getPronounceScore(std::string label, int word_idx[], int word_idx_group[], int word_size, int phn_idx[], int phn_idx_group[], int phn_size){
        if (engine_error) std::rethrow_exception(engine_error);
        size_t label_words = 0;
        std::istringstream words(label);
        for (std::string word; words >> word;) ++label_words;
        auto validate = [label_words](const int* ids, const int* groups, int size, int vocab_size) {
            if (!ids || !groups || size < 2 || label_words == 0)
                throw std::invalid_argument("empty or incomplete pronunciation indices");
            for (int i = 0; i < size; ++i) {
                if (ids[i] < 0 || ids[i] >= vocab_size || groups[i] < -1 ||
                    (groups[i] >= 0 && size_t(groups[i]) >= label_words))
                    throw std::invalid_argument("pronunciation index out of range");
            }
        };
        validate(word_idx, word_idx_group, word_size, word_vocab_size);
        validate(phn_idx, phn_idx_group, phn_size, phone_vocab_size);
        // LoggerManager::Log("INFO", "getPronounceScore label: " + label);
        // std::vector<int> word_dec = scorer->makeWordDec(label);
        // LoggerManager::Log("INFO", "getPronounceScore word_dec: " + std::to_string(word_dec.size()));
        // std::vector<int> phone_dec = scorer->makePhoneDec(label);
        // LoggerManager::Log("INFO", "getPronounceScore phone_dec: " + std::to_string(phone_dec.size()));
        
        // Use range constructor instead of push_back loop
        std::vector<int> word_dec_infer(word_idx, word_idx + word_size - 1);
        std::vector<int> phone_dec_infer(phn_idx, phn_idx + phn_size - 1);

        std::shared_ptr<Tensor<T>> w_out = asr_model->getWordScore(word_dec_infer);
        // 인코딩된 오디오가 없으면 디코더가 빈 텐서를 돌려준다. 그대로 채점에 넘기면
        // Scorer 가 존재하지 않는 행을 읽어 죽는다(무음 종료·즉시 취소 경로).
        if (w_out == nullptr || w_out->value.rows() == 0) {
            LoggerManager::Log("WARN", "getPronounceScore skipped: no encoded audio");
            return "";
        }
        LoggerManager::Log("INFO", "getPronounceScore w_out: " + std::to_string(w_out->shape(0)));
        LoggerManager::Log("INFO", "getPronounceScore w_out: " + std::to_string(w_out->shape(1)));
        std::shared_ptr<Tensor<T>> p_out = asr_model->getPhoneScore(phone_dec_infer);
        if (p_out == nullptr || p_out->value.rows() == 0) {
            LoggerManager::Log("WARN", "getPronounceScore skipped: no phone posterior");
            return "";
        }
        LoggerManager::Log("INFO", "getPronounceScore p_out: " + std::to_string(p_out->shape(0)));
        LoggerManager::Log("INFO", "getPronounceScore p_out: " + std::to_string(p_out->shape(1)));

        std::vector<int> word_dec_post;
        std::vector<int> word_idx_group_post;
        std::vector<int> phone_dec_post;
        std::vector<int> phone_idx_group_post;
        for (int i = 1; i < word_size; ++i) {
            word_dec_post.push_back(word_idx[i]);
            word_idx_group_post.push_back(word_idx_group[i]);
        }
        for (int i = 1; i < phn_size; ++i) {
            phone_dec_post.push_back(phn_idx[i]);
            phone_idx_group_post.push_back(phn_idx_group[i]);
        }
        LoggerManager::Log("INFO", "getPronounceScore word_dec_post: " + std::to_string(word_dec_post.size()));
        LoggerManager::Log("INFO", "getPronounceScore phone_dec_post: " + std::to_string(phone_dec_post.size()));
        FinalScore final_score = scorer->getScore(label, w_out, word_dec_post, word_idx_group_post, p_out, phone_dec_post, phone_idx_group_post);

        // 발음 점수만으로는 "무엇을 말했는지" 를 알 수 없다. teacher forcing 이라
        // 전혀 다른 문장을 읽어도 겹치는 낱말에서 점수가 남는다. 2패스 인식 결과와
        // 견줘 총점을 맞춘다 — manifest 의 recognition_weight 가 0 이면 그대로 둔다.
        const std::string recognized = getCtcAsrResult();
        const float before = final_score.avg_score;
        if (sentence_score_weights) {
            // 콘텐츠 .jsonl 의 score_weights 가 있는 문장 — 검수(content_audit)와 같은 비중으로 섞는다.
            final_score = scorer->applyRecognitionBlend(final_score, label, recognized,
                                                        *sentence_score_weights);
            LoggerManager::Log("INFO", "recognition blend("
                                       + std::to_string(sentence_score_weights->pronunciation) + "/"
                                       + std::to_string(sentence_score_weights->recognition) + "): "
                                       + std::to_string(before) + " -> "
                                       + std::to_string(final_score.avg_score)
                                       + " | 인식: " + recognized);
        } else {
            final_score = scorer->applyRecognitionPenalty(final_score, label, recognized);
            if (final_score.avg_score != before) {
                LoggerManager::Log("INFO", "recognition penalty: " + std::to_string(before)
                                           + " -> " + std::to_string(final_score.avg_score)
                                           + " | 인식: " + recognized);
            }
        }
        LoggerManager::Log("INFO", "getPronounceScore final_score: " + std::to_string(final_score.avg_score));
        return FinalScoreToJson(final_score);
    }

    template<typename T>
    std::string AsrAsyncDecoder<T>::ctcIdsToText(const std::vector<int>& ctc_asr_ids){
        std::string asr_result;
        // Skip intermediate vector; collapse repeats directly into result string
        asr_result.reserve(ctc_asr_ids.size() * 4);
        int pre_id = -1;
        for(size_t i=0;i<ctc_asr_ids.size();++i){
            int token_id = ctc_asr_ids[i];
            if(token_id == pre_id) continue;          // CTC 반복 축약
            pre_id = token_id;
            if(token_id == ctc_blank_id) continue;    // blank 는 출력하지 않는다
            if(token_id < 0 || token_id >= int(vocab.size())) continue;
            asr_result += vocab[token_id];
        }
        replaceAll(asr_result, "▁"," ");
        if(ctc_blank_id < 0){
            // blank id 를 선언하지 않는 구형 모델은 vocab[0] 이 "-" 인 관례를 따랐다
            replaceAll(asr_result, "-","");
        }
        replaceAll(asr_result, "<pad>","");
        replaceAll(asr_result, "<unk>","");
        replaceAll(asr_result, "<s>","");
        replaceAll(asr_result, "</s>","");
        return asr_result;
    }

    // 스트리밍 중간 결과: 누적된 enc1 에 대한 ctc_linear1 greedy (2패스 인코딩을 돌리지 않는다)
    template<typename T>
    std::string AsrAsyncDecoder<T>::getFirstPassCtcAsrResult(){
        return ctcIdsToText(asr_model->getFirstPassCtcAsrIds());
    }

    /// 뒤에서부터 연속된 blank 프레임을 세어 "말이 멎은 뒤 흐른 시간" 으로 바꾼다.
    ///
    /// CTC 는 소리가 없는 구간에 blank 를 낸다. 다만 blank 는 무음일 때만 나오는 것이 아니라
    /// 한 토큰이 여러 프레임에 걸칠 때 그 사이에도 나온다 — 그래서 "몇 개나 연속인지" 를 본다.
    /// 발화 도중의 blank 는 길어야 몇 프레임이지만, 말이 끝나면 계속 쌓인다.
    template<typename T>
    void AsrAsyncDecoder<T>::updateCtcTrailingSilence(const std::vector<int>& ctc_asr_ids) {
        if (ctc_blank_id < 0 || ctc_asr_ids.empty()) {
            // blank id 를 모르면 셀 수 없다. 모른다는 뜻으로 -1 을 둔다.
            ctc_trailing_silence_ms.store(-1);
            return;
        }
        int trailing = 0;
        for (auto it = ctc_asr_ids.rbegin(); it != ctc_asr_ids.rend(); ++it) {
            if (*it != ctc_blank_id) break;
            ++trailing;
        }
        if (size_t(trailing) == ctc_asr_ids.size()) {
            // 처음부터 끝까지 blank — 아직 발화를 한 번도 잡지 못했다.
            // 이때의 침묵은 "말이 끝난 뒤" 가 아니므로 구분해서 알린다.
            ctc_trailing_silence_ms.store(-1);
            return;
        }
        ctc_trailing_silence_ms.store(trailing * CTC_FRAME_MS);
    }

    // 최종 결과: second_encoder 를 거친 ctc_linear2 greedy
    template<typename T>
    std::string AsrAsyncDecoder<T>::getCtcAsrResult(){
        if (engine_error) std::rethrow_exception(engine_error);
        return ctcIdsToText(asr_model->getCtcAsrIds());
    }

    template class AsrAsyncDecoder<float>;
    // template class AsrAsyncDecoder<int>;
};
