#ifndef MECURY_ASR_ASYNC_DECODER
#define MECURY_ASR_ASYNC_DECODER

#include <queue>
#include <numeric> 
#include "tensor/tensor.h"
#include "beam/beam.h"
#include "feature/librosa_feature.h"
#include "decoder/asr_transformer.h"
#include "decoder/asr_model.h"
#include "decoder/asr_conformer.h"
#include "decoder/asr_conformer_trf_transducer.h"
#include "decoder/asr_conformer_trf_transducerV2.h"
#include "decoder/asr_stream_conformer_trf_transducer.h"
#include "decoder/asr_stream_conformer.h"
#include "score/scorer.h"
#include <thread>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include "interface/callback_factory.h"
#include "logger/logger.h"
#include "time.h"

namespace MERCURY {
    template<typename T>
    class AsrAsyncDecoder {
    public:
        /// 엔진을 깨우기 전에 모아 두는 오디오 양(16kHz 기준 1.5초).
        ///
        /// 스트리밍 중간 결과가 나오는 주기이자, 발화가 끝났을 때 finishDecoding 이
        /// 마저 처리해야 하는 잔량의 상한이다.
        ///
        /// 3초였다. 화면의 인식 결과가 3초에 한 번씩만 갱신됐고, 말이 끝난 뒤에도
        /// 마지막 덩어리를 처리하느라 결과가 늦게 나왔다.
        ///
        /// 값을 정할 때 두 가지를 봤다.
        ///
        /// 1. 연산량 — 콘텐츠 314문장으로 잰 상대 비용(2패스 기준):
        ///      3000ms 1.00x / 1500ms 1.12x / 1000ms 1.24x / 500ms 1.72x
        ///    Galaxy S9 실측 RTF 는 1초 청크에서 0.3 — 여유가 있다.
        ///
        /// 2. 스트리밍(1패스 CTC) 인식 — full 테스트셋 34,169발화 전수 측정에서
        ///    ctc1 WRR 이 500ms 76.65% / 1초 80.58% / 2.56초 83.01% 로, 500ms 는
        ///    학습 분포(chunk_min 1초) 밖이라 -3.9pp 를 잃는다. 분포를 맞춘 재학습도
        ///    이 격차를 회수하지 못했다(정보량 한계) — report/stream_chunk_1s_benchmark.html.
        ///    청크가 클수록 좋다는 방향이 분명해 1.5초(24000)까지 올린다. 1.5초 자체는
        ///    전수 측정한 지점이 아니라 1초와 2.56초 사이의 보간이다 — 필요하면 같은
        ///    조건으로 재서 확인할 것.
        ///
        /// 3. 발화 종료 반응 — 예전에는 "EndpointDetector 는 이 값과 무관하다" 고 적혀
        ///    있었지만 더는 사실이 아니다. CTC 가 종료를 판정하도록 바꾼 뒤로
        ///    getCtcTrailingSilenceMs() 가 청크마다만 갱신되므로, 이 값이 곧 종료 판정
        ///    지연의 상한이 된다. 키우면 인식은 좋아지고 끊는 반응은 그만큼 느려진다.
        ///    에너지 기반 판정만 이 값과 무관하게 80~100ms 단위로 돈다.
        static constexpr size_t WAKE_THRESHOLD_SAMPLES = 24000;

        AsrAsyncDecoder(JsonObject &asr_model_path, std::string base_path);
        ~AsrAsyncDecoder();
        std::vector<Score> getNbest(std::shared_ptr<Tensor<float>> logits, int num_nbest);
        void engine(int num_nbest);
        std::shared_ptr<Tensor<T>> getPhoneScore(std::vector<int> phone_dec);
        std::shared_ptr<Tensor<T>> getWordScore(std::vector<int> word_dec);
        std::string getCtcAsrResult();
        // 스트리밍 중간 결과: 지금까지 누적된 enc1 에 대한 ctc_linear1 greedy
        std::string getFirstPassCtcAsrResult();

        /// 인코더 출력 한 프레임이 덮는 시간(ms).
        ///
        /// 특징 추출 hop 이 160 샘플(=10ms @16kHz)이고, 인코더 앞단의 conv 두 층이
        /// stride 2 로 4배 줄인다. 그래서 CTC 한 프레임은 40ms 다.
        static constexpr int CTC_FRAME_MS = 40;

        /// 스트리밍 CTC 가 본 "말이 멎은 뒤 흐른 시간"(ms).
        ///
        /// 마지막 청크의 CTC 출력에서 뒤쪽 blank 프레임이 몇 개 연속인지 세어 환산한 값이다.
        /// 아직 발화를 한 번도 잡지 못했으면 -1.
        ///
        /// 음향 에너지가 아니라 모델이 판단하므로 잡음에 강하고, 무성 자음(/s/, /f/, 파열음
        /// 앞 폐쇄)을 무음으로 오해하지 않는다. 대신 값이 갱신되는 시점은 청크가 도착할
        /// 때뿐이라 최대 WAKE_THRESHOLD_SAMPLES 만큼 늦다 — 에너지 기반 판정을 대신하기보다
        /// 그쪽 판정을 확인·보정하는 데 쓰는 것이 맞다.
        int getCtcTrailingSilenceMs() const { return ctc_trailing_silence_ms.load(); }
        std::string getSecondPassDecodingResult(int num_nbest);
        void clear();
        void setCallBack(DecoderCallbackPtr callback);
        void startBackGroundEngine(int num_nbest);
        void sendAudio(float audio[], int size, bool _isFinal);
        void finishDecoding();
        void cancelDecoding();
        /// 엔진 스레드를 깨워서 세우고 거둔다. 스레드를 남긴 채 다시 시작하면 std::terminate 다.
        void stopEngineThread();
        std::string getPronounceScore(std::string label, int word_idx[], int word_idx_group[], int word_size, int phn_idx[], int phn_idx_group[], int phn_size);
        /// 문장별 최종 점수 비중(발음/인식). 콘텐츠 .jsonl 의 score_weights 를 앱이 채점
        /// 직전에 넘긴다 — 지정되면 recognition penalty 대신 이 비중으로 섞는다.
        /// 음수(예: -1,-1)를 주면 지운다. 잘못된 비중(합이 1이 아님)은 무시하고 기본으로 둔다.
        void setSentenceScoreWeights(float pronunciation, float recognition) {
            if (pronunciation < 0 || recognition < 0) {
                sentence_score_weights.reset();
                return;
            }
            try {
                sentence_score_weights = std::make_unique<SentenceScoreWeights>(pronunciation, recognition);
            } catch (const std::exception& e) {
                LoggerManager::Log("WARN", std::string("sentence score weights rejected: ") + e.what());
                sentence_score_weights.reset();
            }
        }
        std::unique_ptr<SentenceScoreWeights> sentence_score_weights;
        std::shared_ptr<AsrStreamConformerTrfTransducer<T>> asr_stream_conformer_trf_transducer;
        std::shared_ptr<AsrModel<T>> asr_model;
        void warmUpInference(std::vector<float> audio);
        std::shared_ptr<LibrosaFeature> feature_extractor;
        int num_dec_layer;
        std::vector<float> chunk_audio;
        std::vector<Beam> beams;
        bool isFinalAudio = false; // mtx protects the final flag and audio queue together
        std::mutex mtx;
        std::condition_variable data_cond;
        bool data_ready = false;
        std::atomic<bool> is_canceled{false};
        std::exception_ptr engine_error; // written by worker, read only after join
        int word_vocab_size = 0;
        int phone_vocab_size = 0;
        std::thread engineThread;
        DecoderCallbackPtr decoder_callback;
        std::vector<std::string> vocab;
        // CTC blank id — manifest(model.json) 가 선언한 값. 선언이 없는 구형 모델은 -1.
        int ctc_blank_id = -1;
        // 엔진 스레드가 쓰고 앱 스레드가 읽는다 — getCtcTrailingSilenceMs() 참고.
        std::atomic<int> ctc_trailing_silence_ms{-1};
        // 방금 받은 CTC id 열로 위 값을 갱신한다.
        void updateCtcTrailingSilence(const std::vector<int>& ctc_asr_ids);
        void setVocab(std::vector<std::string> vocab);
        std::string getAsrResult(std::vector<int> word_ids);
        // CTC greedy id 열을 텍스트로 — 반복 축약과 특수 토큰 제거를 1패스/2패스가 공유한다
        std::string ctcIdsToText(const std::vector<int>& ctc_asr_ids);
        void replaceAll(std::string& str, const std::string& from, const std::string& to);
        std::shared_ptr<Scorer> scorer;


        /// 낱말 표기를 JSON 문자열에 넣기 전에 이스케이프한다. 원문에 큰따옴표가 든
        /// 문장("City of Light")이 그대로 들어가면 깨진 JSON 이 되어 앱이 채점 결과를
        /// 통째로 읽지 못한다 — 콘텐츠 전수검사에서 18문장이 그랬다.
        static std::string EscapeJson(const std::string& text) {
            std::string out;
            out.reserve(text.size() + 8);
            for (unsigned char c : text) {
                switch (c) {
                    case '"':  out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\n': out += "\\n";  break;
                    case '\r': out += "\\r";  break;
                    case '\t': out += "\\t";  break;
                    default:
                        if (c < 0x20) {
                            char buf[8];
                            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                            out += buf;
                        } else {
                            out += static_cast<char>(c);
                        }
                }
            }
            return out;
        }

        std::string FinalScoreToJson(FinalScore final_score) {
            std::string json = "{";
            json += "\"avg_score\":" + std::to_string(final_score.avg_score) + ",";
            json += "\"final_score\":[";
            for (size_t i = 0; i < final_score.final_score.size(); ++i) {
                const PhoneScore& ps = final_score.final_score[i];
                json += "{";
                json += "\"word\":\"" + EscapeJson(ps.word) + "\",";
                json += "\"score\":" + std::to_string(ps.score) + ",";
                json += std::string("\"graded\":") + (ps.graded ? "true" : "false") + ",";
                json += "\"phone_score\":[";
                for (size_t j = 0; j < ps.phone_each_score.size(); ++j) {
                    const WordScore& ws = ps.phone_each_score[j];
                    json += "{";
                    json += "\"word\":\"" + EscapeJson(ws.word) + "\",";
                    json += "\"score\":" + std::to_string(ws.score);
                    json += "}";
                    if (j + 1 < ps.phone_each_score.size()) json += ",";
                }
                json += "]";
                json += "}";
                if (i + 1 < final_score.final_score.size()) json += ",";
            }
            json += "]";
            json += "}";
            return json;
        }
    };
};
#endif
