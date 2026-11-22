#ifndef ASR_STATE_H
#define ASR_STATE_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "logger/logger.h"

namespace MERCURY {
    class AsrState{
        public:
            AsrState() = default;
            virtual ~AsrState() = default;

            // IDLE STATE
            virtual bool init() = 0;

            // INIT STATE
            virtual bool clear() = 0;
            virtual bool setCallBack() = 0;

            // RUNNING STATE
            virtual bool startBackGroundEngine() = 0;
            virtual bool sendAudio() = 0;
            virtual bool finishDecoding() = 0;

            // FINISHING STATE
            virtual bool getPronounceScore() = 0;
            virtual bool getCtcAsrResult() = 0;
            virtual bool getSecondPassDecodingResult() = 0;
            // SHARED METHODS
            virtual bool cancelDecoding() = 0;

            virtual std::string getNextState() = 0;

            virtual std::string getStateName() = 0;
    };
}

#endif