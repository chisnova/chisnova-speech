#include "logger/logger_manager.h"
#include <memory>
#include <sstream> // Added for std::ostringstream

namespace MERCURY {
    std::shared_ptr<Logger> LoggerManager::instance = std::make_shared<Logger>();
    
    void LoggerManager::BasicLog(const std::string& level, const std::string& message) {
        if(instance == nullptr){
            instance = std::make_shared<Logger>();
        }

        instance->log(level, message);
    }

    void LoggerManager::LogWithLocation(const std::string& level, const std::string& message,
                                        const char* file, int line) {
        std::ostringstream oss;
        oss << file << ":" << line << " " << message;
        LoggerManager::BasicLog(level, oss.str());
    }
};
