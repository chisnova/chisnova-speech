#include "state/init_state.h"

namespace MERCURY {

bool InitState::init() {
    return false;
}

bool InitState::clear() {
    return true;
}

bool InitState::setCallBack() {
    return true;
}

bool InitState::startBackGroundEngine() {
    return true;
}

bool InitState::sendAudio() {
    return false;
}

bool InitState::finishDecoding() {
    return false;
}

bool InitState::cancelDecoding() {
    return false;
}

bool InitState::getPronounceScore() {
    return false;
}

bool InitState::getCtcAsrResult() {
    return false;
}

bool InitState::getSecondPassDecodingResult() {
    return false;
}

std::string InitState::getNextState() {
    return "RunningState";
}

std::string InitState::getStateName() {
    return "InitState";
}

} // namespace MERCURY  