#ifndef ASR_API_H
#define ASR_API_H

#include "decoder/asr_async_decoder.h"
#include "interface/decoder_callback.h"
#include "score/scorer.h"
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "json_loader/json_object.h"
#include "state/context_manager.h"

namespace MERCURY {
    class AsrAPI{
        public:
            AsrAPI(std::string path);
            AsrAPI();
            ~AsrAPI();
            void clear();
            void setCallBack(DecoderCallbackPtr callback);
            void startBackGroundEngine(int num_nbest);
            void sendAudio(float audio[], int size, bool _isFinal);
            // Close input and wait for the worker. Also works without a separate final chunk.
            void finishDecoding();
            void cancelDecoding();
            std::string getPronounceScore(std::string label, int word_idx[], int word_idx_group[], int word_size, int phn_idx[], int phn_idx_group[], int phn_size);
            /// 문장별 최종 점수 비중(발음/인식, 합 1). 콘텐츠 .jsonl 의 score_weights 를
            /// getPronounceScore 직전에 넘긴다. 음수(예: -1,-1)면 모델 기본 채점으로 되돌린다.
            /// clear() 시에도 초기화된다.
            void setSentenceScoreWeights(float pronunciation, float recognition);
            std::string getCtcAsrResult();
            /// 스트리밍 CTC 로 본 발화 종료 판정용 값 — 말이 멎은 뒤 흐른 시간(ms).
            /// 아직 발화를 잡지 못했으면 -1. 청크가 도착할 때마다 갱신된다.
            int getCtcTrailingSilenceMs();
            std::string getSecondPassDecodingResult(int num_nbest);
            std::shared_ptr<ContextManager> context_manager;
    };
}

#endif
