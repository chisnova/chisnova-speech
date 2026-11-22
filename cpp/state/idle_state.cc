#include "state/idle_state.h"

namespace MERCURY {

bool IdleState::init() {
    return true;
}

bool IdleState::clear() {
    return true;
}

bool IdleState::setCallBack() {
    return false;
}

bool IdleState::startBackGroundEngine() {
    return false;
}

bool IdleState::sendAudio() {
    return false;
}

bool IdleState::finishDecoding() {
    return false;
}

bool IdleState::cancelDecoding() {
    return false;
}

bool IdleState::getPronounceScore() {
    return false;
}

bool IdleState::getCtcAsrResult() {
    return false;
}

bool IdleState::getSecondPassDecodingResult() {
    return false;
}

std::string IdleState::getNextState() {
    return "InitState";
}

std::string IdleState::getStateName() {
    return "IdleState";
}

} // namespace MERCURY  