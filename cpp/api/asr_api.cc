#include "api/asr_api.h"
#include <fstream>

namespace MERCURY {
    AsrAPI::AsrAPI(std::string path) {
        context_manager = std::make_shared<ContextManager>();
        context_manager->init(path);
    }

    AsrAPI::AsrAPI() {
        context_manager = std::make_shared<ContextManager>();
    }   

    AsrAPI::~AsrAPI() {
        context_manager = nullptr;
    }

    void AsrAPI::clear() {
        context_manager->clear();
    }

    void AsrAPI::setCallBack(DecoderCallbackPtr callback) {
        context_manager->setCallBack(callback);
    }

    void AsrAPI::startBackGroundEngine(int num_nbest) {
        context_manager->startBackGroundEngine(num_nbest);
    }

    void AsrAPI::sendAudio(float audio[], int size, bool _isFinal) {
        context_manager->sendAudio(audio, size, _isFinal);
    }

    void AsrAPI::finishDecoding() {
        context_manager->finishDecoding();
    }

    void AsrAPI::cancelDecoding() {
        context_manager->cancelDecoding();
    }

    std::string AsrAPI::getPronounceScore(std::string label, int word_idx[], int word_idx_group[], int word_size, int phn_idx[], int phn_idx_group[], int phn_size) {
        std::string json = context_manager->getPronounceScore(label, word_idx, word_idx_group, word_size, phn_idx, phn_idx_group, phn_size);
        return json;
    }

    void AsrAPI::setSentenceScoreWeights(float pronunciation, float recognition) {
        context_manager->setSentenceScoreWeights(pronunciation, recognition);
    }

    std::string AsrAPI::getCtcAsrResult() {
        return context_manager->getCtcAsrResult();
    }

    int AsrAPI::getCtcTrailingSilenceMs() {
        return context_manager->getCtcTrailingSilenceMs();
    }

    std::string AsrAPI::getSecondPassDecodingResult(int num_nbest) {
        return context_manager->getSecondPassDecodingResult(num_nbest);
    }
};