#include "json_loader/json_object.h"

namespace MERCURY {
    
    JsonObject::JsonObject() {}
    
    JsonObject::JsonObject(const std::string& config_path) {
        readJson(config_path);
    }

    void JsonObject::print() {
        std::cout << obj.dump(4) << std::endl;  // Pretty print with 4 space indentation
    }

    void JsonObject::readJson(std::string config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("cannot open model config: " + config_path);
        }
        
        try {
            file >> obj;  // nlohmann/json automatically parses the JSON
        }
        catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("invalid model config " + config_path + ": " + e.what());
        }
        
        file.close();
    }

    JsonObject JsonObject::parseObject(std::string key) {
        JsonObject ret_object;
        if (obj.contains(key) && obj[key].is_object()) {
            ret_object.obj = obj[key];
        }
        else {
            throw std::invalid_argument("missing or invalid model config field: " + key);
        }
        return ret_object;
    }

    float JsonObject::parserValue(std::string key) {
        float ret = 0;
        if (obj.contains(key) && obj[key].is_number()) {
            ret = obj[key].get<float>();
        }
        else {
            throw std::invalid_argument("missing or invalid model config field: " + key);
        }
        return ret;
    }

    float JsonObject::parserValue(std::string key, float default_value) {
        if (obj.contains(key) && obj[key].is_number()) {
            return obj[key].get<float>();
        }
        return default_value;
    }

    std::string JsonObject::parserString(std::string key) {
        std::string ret;
        if (obj.contains(key) && obj[key].is_string()) {
            ret = obj[key].get<std::string>();
        }
        else {
            throw std::invalid_argument("missing or invalid model config field: " + key);
        }
        return ret;
    }

    std::string JsonObject::parserStringOr(std::string key, std::string default_value) {
        if (obj.contains(key) && obj[key].is_string()) {
            return obj[key].get<std::string>();
        }
        return default_value;
    }

    std::vector<std::string> JsonObject::parserStringArray(std::string key) {
        std::vector<std::string> ret;
        if (obj.contains(key) && obj[key].is_array()) {
            for (const auto& item : obj[key]) {
                if (item.is_string()) {
                    ret.push_back(item.get<std::string>());
                }
            }
        }
        else {
            throw std::invalid_argument("missing or invalid model config field: " + key);
        }
        return ret;
    }

    std::vector<int> JsonObject::parserIntArray(std::string key) {
        std::vector<int> ret;
        if (obj.contains(key) && obj[key].is_array()) {
            for (const auto& item : obj[key]) {
                if (item.is_number_integer()) {
                    ret.push_back(item.get<int>());
                }
            }
        }
        else {
            throw std::invalid_argument("missing or invalid model config field: " + key);
        }
        return ret;
    }

    std::vector<float> JsonObject::parserValueArray(std::string key) {
        std::vector<float> ret;
        if (obj.contains(key) && obj[key].is_array()) {
            for (const auto& item : obj[key]) {
                if (item.is_number()) {
                    ret.push_back(item.get<float>());
                }
            }
        }
        else {
            throw std::invalid_argument("missing or invalid model config field: " + key);
        }
        return ret;
    }



    JsonObject JsonObject::clone() {
        JsonObject ret_object;
        ret_object.obj = obj;  // nlohmann/json automatically does deep copy!
        return ret_object;
    }
}


