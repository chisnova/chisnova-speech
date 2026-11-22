#ifndef MERCURY_LOGGER_MANAGER_H
#define MERCURY_LOGGER_MANAGER_H

#include "logger/logger.h"
#include <memory>
#include <sstream>
// #define MERCURY_LOG(level, ...)                                             \
//     do {                                                                    \
//         std::ostringstream _merc_log_ss;                                    \
//         _merc_log_ss << __FILE__ << ":" << __LINE__ << " " ;             \
//         _merc_log_ss << __VA_ARGS__;                                        \
//         MERCURY::LoggerManager::Log(level, _merc_log_ss.str());             \
//     } while (0)

// #define MERCURY_LOG_INFO(...)   MERCURY_LOG("INFO", __VA_ARGS__)
// #define MERCURY_LOG_WARN(...)   MERCURY_LOG("WARNING", __VA_ARGS__)
// #define MERCURY_LOG_ERROR(...)  MERCURY_LOG("ERROR", __VA_ARGS__)


namespace MERCURY {
    class LoggerManager {
        public:
        static void BasicLog(const std::string& level, const std::string& message);
        // Enhanced version that receives caller file & line.
        static void LogWithLocation(const std::string& level, const std::string& message,
                                    const char* file, int line);
        static std::shared_ptr<Logger> instance;
    };
};

// Macro to transparently add file:line to existing LoggerManager::Log(...) calls.
#define Log(level, message) LogWithLocation(level, message, __FILE__, __LINE__)

#endif
