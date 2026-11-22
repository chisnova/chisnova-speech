#ifndef CONTEXT_MANAGER_H
#define CONTEXT_MANAGER_H

#include <memory>

#include "logger/logger_manager.h"
#include "state/asr_state.h"
#include "decoder/asr_async_decoder.h"
#include "score/scorer.h"
#include "state/idle_state.h"
#include "state/init_state.h"
#include "state/running_state.h"
#include "state/finishing_state.h"

namespace MERCURY {
class ContextManager {
public:
    ContextManager();
    ~ContextManager();

    // IDLE STATE
    void init(const std::string &path);

    // INIT STATE
    void clear();
    void setCallBack(DecoderCallbackPtr callback);

    // RUNNING STATE
    void startBackGroundEngine(int num_nbest);
    void sendAudio(float audio[], int size, bool _isFinal);
    void finishDecoding();

    // FINISHING STATE
    std::string getPronounceScore(const std::string &label, int word_idx[], int word_idx_group[], int word_size, int phn_idx[], int phn_idx_group[], int phn_size);
    /// 문장별 최종 점수 비중(발음/인식). 음수를 주면 모델 기본으로 되돌린다.
    void setSentenceScoreWeights(float pronunciation, float recognition);
    std::string getCtcAsrResult();
    /// 스트리밍 CTC 가 본 "말이 멎은 뒤 흐른 시간"(ms). 아직 발화가 없으면 -1.
    /// 자세한 것은 AsrAsyncDecoder::getCtcTrailingSilenceMs 참고.
    int getCtcTrailingSilenceMs();
    std::string getSecondPassDecodingResult(int num_nbest);

    // SHARED METHODS
    void cancelDecoding();

protected:


    void vocab_parser(std::string file_name, std::string base_path, int vocab_size){
        std::string vocab_path = base_path + "/" + file_name ;
        std::ifstream ifs(vocab_path);
        if (vocab_size <= 0 || !ifs) throw std::runtime_error("invalid vocabulary: " + vocab_path);
        vocab.clear();
        for(int i=0;i<vocab_size;++i) {
            std::string word;
            std::string score;
            if (!(ifs >> word >> score)) throw std::runtime_error("truncated vocabulary: " + vocab_path);
            vocab.push_back(word);
        }
        // 인덱스가 한 칸이라도 밀리면 같은 CTC id 가 다른 토큰으로 풀린다.
        // 플랫폼 간 대조를 위해 대표 인덱스와 총 길이를 남긴다.
        {
            std::string sample;
            for (int idx : {0, 1, 2, 3, 4, 100, 500, 1023}) {
                if (idx < int(vocab.size())) {
                    sample += "[" + std::to_string(idx) + "]='" + vocab[idx] + "' ";
                }
            }
            LoggerManager::Log("INFO", " vocab loaded: " + std::to_string(vocab.size()) + " — " + sample);
        }
    }

    // 읽을 개수는 manifest(model.json)의 phone_size 를 따른다 — 상수로 고정하면
    // 음소 집합이 바뀔 때 뒤쪽 음소가 잘리거나 빈 토큰이 섞인다.
    void phone_parser(std::string file_name, std::string base_path, int phone_size){
        std::string phone_path = base_path + "/" + file_name ;
        std::ifstream ifs(phone_path);
        if (phone_size <= 0) throw std::runtime_error("invalid phone_size");
        phone.clear();
        phone.reserve(phone_size);
        for(int i=0;i<phone_size;++i) {
            std::string word;
            if (!(ifs >> word)) {
                LoggerManager::Log("ERROR", "phone file has only " + std::to_string(i)
                                            + " entries but manifest declares "
                                            + std::to_string(phone_size) + ": " + phone_path);
                throw std::runtime_error("truncated phone vocabulary: " + phone_path);
            }
            phone.push_back(word);
        }
    }

private:
    // Serialize app/control calls (including finish vs cancel); callbacks must dispatch
    // to the app thread instead of re-entering this API on the decoding thread.
    std::mutex api_mutex;
    std::shared_ptr<IdleState> idle_state;
    std::shared_ptr<InitState> init_state;
    std::shared_ptr<RunningState> running_state;
    std::shared_ptr<FinishingState> finishing_state;
    std::shared_ptr<AsrState> asr_state;
    std::map<std::string, std::shared_ptr<AsrState>> state_map;

    std::shared_ptr<AsrAsyncDecoder<float>> asr_async_decoder;
    std::shared_ptr<Scorer> scorer;
    std::vector<std::string> vocab;
    std::vector<std::string> phone;
};

} // namespace MERCURY

#endif // CONTEXT_MANAGER_H
