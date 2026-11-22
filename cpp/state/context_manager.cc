#include "state/context_manager.h"

namespace MERCURY {
ContextManager::ContextManager() {
    idle_state = std::make_shared<IdleState>();
    init_state = std::make_shared<InitState>();
    running_state = std::make_shared<RunningState>();
    finishing_state = std::make_shared<FinishingState>();
    state_map["IdleState"] = idle_state;
    state_map["InitState"] = init_state;
    state_map["RunningState"] = running_state;
    state_map["FinishingState"] = finishing_state;
    asr_state = idle_state;
}

ContextManager::~ContextManager() {
    asr_async_decoder = nullptr; // joins the worker before releasing callback/state data
    state_map.clear();
    idle_state = nullptr;
    init_state = nullptr;
    running_state = nullptr;
    finishing_state = nullptr;
    asr_state = nullptr;

    asr_async_decoder = nullptr;
    scorer = nullptr;
    vocab.clear();
    phone.clear();
}

void ContextManager::init(const std::string &path) {
    std::lock_guard<std::mutex> lock(api_mutex);
    if(asr_state->init()){
        std::string config_path = path + "/" +std::string("model.json");
        JsonObject config_parser = JsonObject(config_path);
        JsonObject model_json = config_parser.parseObject("model_path");
        int vocab_size = int(model_json.parserValue("vocab_size"));
        int phone_size = int(model_json.parserValue("phone_size"));
        std::string model_path = model_json.parserString("vocab_model");
        std::string vocab_name = model_json.parserString("vocab_name");
        std::string phone_name = model_json.parserString("phone_name");
        std::string dict_name = model_json.parserString("dict_name");
        vocab_parser(vocab_name, path, vocab_size);
        phone_parser(phone_name, path, phone_size);
        LoggerManager::Log("INFO", "vocab_size: " + std::to_string(vocab_size)
                                   + ", phone_size: " + std::to_string(phone_size));
        // phone_weight 는 manifest 가 선언했으면 그것을, 없으면 기본값을 쓴다.
        // 옛 model_state 에는 이 항목이 없으므로 기본값 경로가 반드시 있어야 한다.
        float phone_weight = model_json.parserValue("phone_weight", Scorer::DEFAULT_PHONE_WEIGHT);
        // 기능어 하한: 목록 파일과 점수를 manifest 가 정한다. 둘 중 하나라도 없으면 쓰지 않는다.
        std::string function_word_name = model_json.parserStringOr("function_word_list", "");
        float function_word_floor = model_json.parserValue("function_word_floor",
                                                           Scorer::DEFAULT_FUNCTION_WORD_FLOOR);
        scorer = std::make_shared<Scorer>(path, model_path, vocab_name, phone_name, dict_name, phone_size,
                                          phone_weight, function_word_name, function_word_floor,
                                          model_json.parserValue("recognition_weight",
                                                                 Scorer::DEFAULT_RECOGNITION_WEIGHT));
        // 고유명사 하한: 목록 파일과 점수를 manifest 가 정한다. 없으면 쓰지 않는다.
        scorer->setProperNounFloor(path, model_json.parserStringOr("proper_noun_list", ""),
                                   model_json.parserValue("proper_noun_floor",
                                                          Scorer::DEFAULT_PROPER_NOUN_FLOOR));
        // 채점 제외 낱말 목록(마지막 수단). 없으면 모든 낱말을 채점한다.
        scorer->setUngradedWords(path, model_json.parserStringOr("ungraded_word_list", ""));
        LoggerManager::Log("INFO", "scorer init done");
        asr_async_decoder = std::make_shared<AsrAsyncDecoder<float>>(model_json, path);
        asr_async_decoder->setVocab(vocab);
        LoggerManager::Log("INFO", "asr_async_decoder init done");
        std::string next_state = asr_state->getNextState();
        asr_state = state_map[next_state];
    }
    else{
        LoggerManager::Log("ERROR", "ASRState init failed");
    }
}

void ContextManager::clear() {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (!asr_async_decoder) return;
    if(asr_state->clear()){
        asr_async_decoder->clear();
        asr_state = init_state;
    }
    else{
        LoggerManager::Log("ERROR", "Cannot clear failed in " + asr_state->getStateName());
    }
}

void ContextManager::setCallBack(DecoderCallbackPtr callback) {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (asr_state->setCallBack()){
        asr_async_decoder->setCallBack(callback);
    }
    else{
        LoggerManager::Log("ERROR", "Cannot setCallBack failed in " + asr_state->getStateName());
    }
}


void ContextManager::startBackGroundEngine(int num_nbest) {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (asr_state->startBackGroundEngine()){
        asr_async_decoder->startBackGroundEngine(num_nbest);
        std::string next_state = asr_state->getNextState();
        asr_state = state_map[next_state];
    }
    else{
        LoggerManager::Log("ERROR", "Cannot startBackGroundEngine failed in " + asr_state->getStateName());
    }
}

void ContextManager::sendAudio(float audio[], int size, bool _isFinal) {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (asr_state->sendAudio()){
        asr_async_decoder->sendAudio(audio, size, _isFinal);
    }
    else{
        LoggerManager::Log("ERROR", "Cannot sendAudio failed in " + asr_state->getStateName());
    }
}

void ContextManager::finishDecoding() {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (asr_state->finishDecoding()){
        asr_async_decoder->finishDecoding();
        std::string next_state = asr_state->getNextState();
        asr_state = state_map[next_state];
    }
    else{
        LoggerManager::Log("ERROR", "Cannot finishDecoding failed in " + asr_state->getStateName());
    }
}

std::string ContextManager::getPronounceScore(const std::string &label, int word_idx[], int word_idx_group[], int word_size, int phn_idx[], int phn_idx_group[], int phn_size) {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (asr_state->getPronounceScore()){
        return asr_async_decoder->getPronounceScore(label, word_idx, word_idx_group, word_size, phn_idx, phn_idx_group, phn_size);
    }
    else{
        LoggerManager::Log("ERROR", "Cannot getPronounceScore failed in " + asr_state->getStateName());
    }
    return "";
}

int ContextManager::getCtcTrailingSilenceMs() {
    std::lock_guard<std::mutex> lock(api_mutex);
    // 세션이 시작되기 전에는 디코더가 없다. 모른다는 뜻으로 -1.
    if (asr_async_decoder == nullptr) return -1;
    return asr_async_decoder->getCtcTrailingSilenceMs();
}

void ContextManager::setSentenceScoreWeights(float pronunciation, float recognition) {
    std::lock_guard<std::mutex> lock(api_mutex);
    // 상태와 무관한 채점 속성이라 상태 머신을 거치지 않는다. 디코더가 아직 없으면 무시한다.
    if (asr_async_decoder == nullptr) return;
    asr_async_decoder->setSentenceScoreWeights(pronunciation, recognition);
}

std::string ContextManager::getCtcAsrResult() {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (asr_state->getCtcAsrResult()){
        return asr_async_decoder->getCtcAsrResult();
    }
    else{
        LoggerManager::Log("ERROR", "Cannot getCtcAsrResult failed in " + asr_state->getStateName());
    }
    return "";
}

std::string ContextManager::getSecondPassDecodingResult(int num_nbest) {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (asr_state->getSecondPassDecodingResult()){
        return asr_async_decoder->getSecondPassDecodingResult(num_nbest);
    }
    else{
        LoggerManager::Log("ERROR", "Cannot getSecondPassDecodingResult failed in " + asr_state->getStateName());
    }
    return "";
}

void ContextManager::cancelDecoding() {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (asr_state->cancelDecoding()){
        asr_async_decoder->cancelDecoding();
        asr_state = init_state;
    }
    else{
        LoggerManager::Log("ERROR", "Cannot cancelDecoding failed in " + asr_state->getStateName());
    }
}

} // namespace MERCURY
