#include "state/running_state.h"
#include "state/finishing_state.h"

namespace MERCURY {

bool RunningState::init() {
    return false;
}

bool RunningState::clear() {
    return true;
}

bool RunningState::setCallBack() {
    return false;
}

bool RunningState::startBackGroundEngine() {
    return false;
}

bool RunningState::sendAudio() {
    return true;
}

bool RunningState::finishDecoding() {
    return true;
}

bool RunningState::getCtcAsrResult() {
    return false;
}

bool RunningState::getSecondPassDecodingResult() {
    return false;
}

bool RunningState::getPronounceScore() {
    return false;
}

bool RunningState::cancelDecoding() {
    return true;
}

std::string RunningState::getNextState() {
    return "FinishingState";
}

std::string RunningState::getStateName() {
    return "RunningState";
}

} // namespace MERCURY 