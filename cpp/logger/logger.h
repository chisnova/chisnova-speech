#ifndef MERCURY_LOGGER_H
#define MERCURY_LOGGER_H

#include <string>
#include <iostream>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <sstream>

#ifdef __ANDROID__
#include <android/log.h>
#elif __IOS__
#include <Foundation/Foundation.h>
#endif

#if defined(__APPLE__)
#include <os/log.h>
#include <cstdio>
#include <cstdlib>
#endif

namespace MERCURY {
    // enum LogLevel {
    //     INFO,
    //     WARNING,
    //     ERROR,
    //     DEBUG
    // };

    // Logger class
    class Logger {
    public:
        Logger() {}

        void log(const std::string& logLevelStr, const std::string& message) {
            // std::string logLevelStr = getLogLevelString(level);

    #ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_INFO, "[AsrAPI]", "[%s] %s", logLevelStr.c_str(), message.c_str());
    #elif defined(__APPLE__)
            // 이전에는 __IOS__ 로 갈랐는데 그런 표준 매크로가 없어 늘 거짓이었다.
            // 그 결과 iOS 에서는 아래 std::cout 으로 떨어졌고, 앱의 표준 출력은
            // 어디에도 보이지 않아 모델 로딩 실패 같은 오류가 통째로 묻혔다.
            os_log(OS_LOG_DEFAULT, "[%{public}s] %{public}s", logLevelStr.c_str(), message.c_str());

            // 기기에서 콘솔을 붙이기 어려운 경우가 많아 파일로도 남긴다.
            // iOS 앱의 TMPDIR 은 샌드박스 안이라 devicectl 로 꺼낼 수 있다.
            {
                static FILE* engineLog = []() -> FILE* {
                    const char* tmpDir = std::getenv("TMPDIR");
                    std::string path = std::string(tmpDir ? tmpDir : "/tmp/") + "engine.log";
                    return std::fopen(path.c_str(), "a");
                }();
                if (engineLog != nullptr) {
                    std::fprintf(engineLog, "[%s] %s\n", logLevelStr.c_str(), message.c_str());
                    std::fflush(engineLog);
                }
            }
    #else
            auto now = std::chrono::system_clock::now();
            auto ms_part = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);
            struct tm tm_buf;
#if defined(_WIN32)
            localtime_s(&tm_buf, &now_c);
#else
            localtime_r(&now_c, &tm_buf);
#endif
            std::ostringstream ts;
            ts << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << '.'
               << std::setfill('0') << std::setw(3) << ms_part.count();

            std::ostringstream logEntry;
            logEntry << "[" << ts.str() << "] [" << logLevelStr << "] " << message;

            std::cout << logEntry.str() << std::endl;
    #endif
        }

    // private:
    //     std::string getLogLevelString(LogLevel level) {
    //         switch (level) {
    //             case INFO:
    //                 return "INFO";
    //             case WARNING:
    //                 return "WARNING";
    //             case ERROR:
    //                 return "ERROR";
    //             case DEBUG:
    //                 return "DEBUG";
    //             default:
    //                 return "UNKNOWN";
    //         }
    //     }
    };
};

#endif
