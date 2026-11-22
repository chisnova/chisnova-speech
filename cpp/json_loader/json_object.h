#ifndef CPP_JSON_OBJECT_H
#define CPP_JSON_OBJECT_H

// Use nlohmann/json from third_party directory
#include "third_party/json/single_include/nlohmann/json.hpp"

#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include "logger/logger_manager.h"

namespace MERCURY {
    class JsonObject {
    public:
        JsonObject();
        JsonObject(const std::string& config_path);  // Constructor with config path
        JsonObject parseObject(std::string key);
        float parserValue(std::string key);
        /// 키가 없으면 default_value 를 돌려준다.
        ///
        /// parserValue(key) 는 필수 키가 없으면 예외를 던진다. 나중에 더한 설정은 예전
        /// manifest 에 없을 수밖에 없으므로, 선택적 항목은 이쪽으로 읽어야 구버전
        /// model_state 로도 앱이 뜬다.
        float parserValue(std::string key, float default_value);
        void print();
        std::string parserString(std::string key);
        /// 키가 없으면 default_value 를 돌려준다. parserValue 의 문자열판이다.
        std::string parserStringOr(std::string key, std::string default_value);
        std::vector<std::string> parserStringArray(std::string key);
        std::vector<int> parserIntArray(std::string key);
        std::vector<float> parserValueArray(std::string key);
        void readJson(std::string config_path);
        JsonObject clone();
        
        nlohmann::json obj;
        // No need for complex memory management with nlohmann!
    };
}

#endif
