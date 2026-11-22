#ifndef IDLE_STATE_H
#define IDLE_STATE_H

#include "state/asr_state.h"
#include "logger/logger.h"
#include <string>

namespace MERCURY {
class IdleState : public AsrState {
public:
    IdleState() = default;
    ~IdleState() = default;

    bool init() override;
    bool clear() override;
    bool setCallBack() override;
    bool startBackGroundEngine() override;
    bool sendAudio() override;
    bool finishDecoding() override;
    bool cancelDecoding() override;
    bool getPronounceScore() override;
    bool getCtcAsrResult() override;
    bool getSecondPassDecodingResult() override;
    std::string getNextState() override;
    std::string getStateName() override;
}; // namespace MERCURY 
}
#endif