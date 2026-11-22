#include "state/finishing_state.h"

namespace MERCURY {

bool FinishingState::init() {
    return false;
}

bool FinishingState::clear() {
    return true;
}

bool FinishingState::setCallBack() {
    return false;
}

bool FinishingState::startBackGroundEngine() {
    return false;
}

bool FinishingState::sendAudio() {
    return false;
}

bool FinishingState::finishDecoding() {
    return false;
}

bool FinishingState::cancelDecoding() {
    return true;
}

bool FinishingState::getPronounceScore() {
    return true;
}

bool FinishingState::getCtcAsrResult() {
    return true;
}

bool FinishingState::getSecondPassDecodingResult() {
    return true;
}

std::string FinishingState::getNextState() {
    return "InitState";
}

std::string FinishingState::getStateName() {
    return "FinishingState";
}

} // namespace MERCURY 