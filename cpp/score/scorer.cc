#include "scorer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cassert>
#include <iostream>
#include <numeric>
#include <cmath>
#include <map>
#include <stdexcept>

namespace MERCURY {
    namespace {
        void validateScoreInput(const std::shared_ptr<Tensor<float>>& posterior,
                                const std::vector<int>& target,
                                const std::vector<int>& groups, size_t word_count) {
            if (!posterior || target.size() != groups.size() ||
                target.size() > size_t(posterior->value.rows()))
                throw std::invalid_argument("score tensor/index lengths do not match");
            for (size_t i = 0; i < target.size(); ++i) {
                if (target[i] < 0 || target[i] >= posterior->value.cols() || groups[i] < -1 ||
                    (groups[i] >= 0 && size_t(groups[i]) >= word_count))
                    throw std::invalid_argument("score token/group index out of range");
            }
        }
    }
    PronunciationScoreWeights::PronunciationScoreWeights(float p, float w)
        : phone(p), word(w) {
        if (!std::isfinite(p) || !std::isfinite(w) || p < 0 || p > 1 || w < 0 || w > 1
            || std::fabs(p + w - 1.0f) > 1e-6f)
            throw std::invalid_argument("Pronunciation score weights must be in [0,1] and sum to 1");
    }
    SentenceScoreWeights::SentenceScoreWeights(float p, float r)
        : pronunciation(p), recognition(r) {
        if (!std::isfinite(p) || !std::isfinite(r) || p < 0 || p > 1 || r < 0 || r > 1
            || std::fabs(p + r - 1.0f) > 1e-6f)
            throw std::invalid_argument("Sentence score weights must be in [0,1] and sum to 1");
    }
    Scorer::Scorer(std::string base_path, std::string model_path, std::string vocab_path, std::string phone_path, std::string dict_path, int phone_size, float phone_weight,
                   const std::string& function_word_path, float function_word_floor,
                   float recognition_weight)
        : phone_weight_(phone_weight), function_word_floor_(function_word_floor),
          function_word_floor_min_(function_word_floor * 0.5f),
          recognition_weight_(std::isfinite(recognition_weight) ? std::clamp(recognition_weight, 0.0f, 1.0f) : 0.0f) {
        loadFunctionWords(base_path, function_word_path);
        // Initialize SentencePiece
        sp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
        const auto status = sp_->Load(base_path + "/" + model_path);
        if (!status.ok()) throw std::runtime_error("cannot load tokenizer: " + status.ToString());

        // Load vocabulary
        std::ifstream vocab_file(base_path + "/" + vocab_path);
        if (!vocab_file) throw std::runtime_error("cannot open vocabulary: " + vocab_path);
        std::string line;
        while (std::getline(vocab_file, line)) {
            std::istringstream iss(line);
            std::string token;
            if (std::getline(iss, token, '\t')) {
                vocab_.push_back(token);
            }
        }

        std::vector<std::string> phones;
        std::ifstream ifs(base_path + "/" + phone_path);
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open phone file: " + phone_path);
        }
        phones.reserve(phone_size);
        for(int i=0;i<phone_size;++i) {
            std::string word;
            if (!(ifs >> word)) {
                // 선언보다 항목이 적으면 남은 자리가 빈 토큰으로 채워져 음소 인덱스가 어긋난다
                LoggerManager::Log("ERROR", "phone file has only " + std::to_string(i)
                                            + " entries but manifest declares "
                                            + std::to_string(phone_size) + ": " + phone_path);
                throw std::runtime_error("phone file shorter than declared phone_size: " + phone_path);
            }
            phones.push_back(word);
        }
        std::string surplus_phone;
        if (ifs >> surplus_phone) {
            // 뒤쪽 여분은 인덱스에 영향을 주지 않지만 manifest 와 자산이 어긋난 신호다
            LoggerManager::Log("WARN", "phone file has more entries than manifest declares ("
                                       + std::to_string(phone_size) + "): " + phone_path);
        }

        // Initialize CTC dictionaries
        for (size_t i = 0; i < phones.size(); ++i) {
            ctc_dict_[i] = phones[i];
            ctc_index_[phones[i]] = i;
        }

        // Load phone dictionary
        std::ifstream dict_file(base_path + "/" + dict_path);
        if (!dict_file.is_open()) {
            LoggerManager::Log("ERROR", "Failed to open dict file: " + base_path + "/" + dict_path);
            throw std::runtime_error("Failed to open dict file: " + dict_path);
        }
        int line_num = 0;
        while (std::getline(dict_file, line)) {
            ++line_num;
            // Skip empty lines
            if (line.empty()) continue;
            size_t tab_pos = line.find('\t');
            if (tab_pos == std::string::npos) {
                LoggerManager::Log("ERROR", "No tab separator found in dict at line " + std::to_string(line_num) + ": " + base_path + "/" + dict_path);
                throw std::runtime_error("invalid dictionary entry at line " + std::to_string(line_num));
            }
            std::string word = line.substr(0, tab_pos);
            std::string phone_seq = line.substr(tab_pos + 1);
            // Remove leading/trailing whitespace
            // word.erase(std::remove_if(word.begin(), word.end(), ::isspace), word.end());
            // phone_seq.erase(std::remove_if(phone_seq.begin(), phone_seq.end(), ::isspace), phone_seq.end());

            if (word.empty() || phone_seq.empty()) {
                LoggerManager::Log("ERROR", "Empty word or phone_seq in dict at line " + std::to_string(line_num) + ": " + base_path + "/" + dict_path);
                throw std::runtime_error("empty dictionary entry at line " + std::to_string(line_num));
            }

            phone_dict_[to_lower(word)] = phone_seq;
        }
        LoggerManager::Log("INFO", "scorer init done, phone_weight: " + std::to_string(phone_weight_));
    }

    
    std::string Scorer::to_lower(const std::string& input) {
        std::string result = input;
        for (auto& c : result) {
            if (c >= 'A' && c <= 'Z') {
                c = c - 'A' + 'a';
            }
        }
        return result;
    }

    std::string Scorer::refineLabel(const std::string& input) {
        std::string result;
        for (auto& c : input) {
            if ((c >= 'a' && c <= 'z') || c == ' ') {
                result += c;
            }
        }
        return result;
    }


    std::vector<int> Scorer::makeWordDec(const std::string& label) {
        std::string lower_label = to_lower(label);
        std::string refine_label = refineLabel(lower_label);
        std::vector<int> ids = sp_->EncodeAsIds(refine_label);
        ids.insert(ids.begin(), 2);  // SOS
        ids.push_back(3);  // EOS
        return ids;
    }

    std::vector<int> Scorer::makePhoneDec(const std::string& label) {
        std::string lower_label = to_lower(label);
        std::string refine_label = refineLabel(lower_label);
        std::vector<int> ret;
        ret.push_back(2);  // Start token

        std::istringstream iss(refine_label);
        std::string word;
        while (iss >> word) {
            // std::cout << "word: " << word << std::endl;
            auto it = phone_dict_.find(word);
            if (it == phone_dict_.end()) {
                LoggerManager::Log("ERROR", "Word not found in phone_dict: " + word);
                assert(false);
            }
            // std::cout << "it->second: " << it->second << std::endl;
            std::istringstream phones_iss(it->second);
            std::string p;
            while (phones_iss >> p) {
                auto idx_it = ctc_index_.find(p);
                if (idx_it == ctc_index_.end()) {
                    LoggerManager::Log("ERROR", "Phone not found in ctc_index: " + p);
                    assert(false);
                }
                ret.push_back(idx_it->second);
            }
            ret.push_back(4);  // Word separator
        }

        LoggerManager::Log("INFO", "ret: " + std::to_string(ret.size()));

        if (ret.size() > 1) {   // 단어가 없으면 [sos]뿐 — sos 를 지우면 안 된다
            ret.pop_back();  // Remove last word separator
        }
        LoggerManager::Log("INFO", "ret2: " + std::to_string(ret.size()));

        ret.push_back(3);  // End token
        return ret;
    }

    std::vector<WordScore> 
    Scorer::getWordScore(std::shared_ptr<Tensor<float>> w_out, 
                         const std::vector<int>& target,
                         const std::vector<int>& target_group,
                         const std::vector<std::string>& lab_words) {
        validateScoreInput(w_out, target, target_group, lab_words.size());
        std::vector<WordScore> word_score;
        for(int i=0;i<lab_words.size();++i) {
            WordScore tmp_word;
            tmp_word.word = lab_words[i];
            tmp_word.cnt = 0;
            tmp_word.score = 0.0f;
            word_score.push_back(tmp_word);
        }
        
        std::cout << "getWordScore target: " << target.size() << std::endl;
        for (size_t idx = 0; idx < target.size(); ++idx) {
            int value = target[idx];
            int group = target_group[idx];
            if(group < 0) continue;

            float prob = w_out->value(idx, value);
            float s = prob * 100.0f;
            word_score[group].score += s;
            word_score[group].cnt++;
        }
        for(int i=0;i<word_score.size();++i) {
            // 토큰이 하나도 안 붙은 단어(숫자/기호 등 발음 평가 불가) — 0/0 NaN 방지
            if (word_score[i].cnt > 0) {
                word_score[i].score /= word_score[i].cnt;
            }
        }
        return word_score;
    }

    std::vector<PhoneScore>
    Scorer::getPhoneScore(std::shared_ptr<Tensor<float>> p_out,
                          const std::vector<int>& target,
                          const std::vector<int>& target_group,
                          const std::vector<std::string>& lab_words) {
        validateScoreInput(p_out, target, target_group, lab_words.size());
        // std::vector<std::vector<WordScore>> phone_score;
        // std::vector<WordScore> p_word;
        std::vector<PhoneScore> phone_score_arr;
        for(int i=0;i<lab_words.size();++i) {
            PhoneScore tmp_phone;
            tmp_phone.word = lab_words[i];
            tmp_phone.cnt = 0;
            tmp_phone.score = 0.0f;
            tmp_phone.phone_each_score = std::vector<WordScore>();
            phone_score_arr.push_back(tmp_phone);
        }

        for (size_t idx = 0; idx < target.size(); ++idx) {
            int value = target[idx];
            int group = target_group[idx];
            if(group < 0) continue;
            if(value == 4) continue;
            float prob = p_out->value(idx, value);
            float s = prob * 100.0f;
            phone_score_arr[group].score += s;
            phone_score_arr[group].cnt++;

            WordScore tmp_word;
            tmp_word.word = ctc_dict_[value];
            tmp_word.score = s;
            phone_score_arr[group].phone_each_score.push_back(tmp_word);
        }

        for(int i=0;i<phone_score_arr.size();++i) {
            if (phone_score_arr[i].cnt > 0) {
                phone_score_arr[i].score /= phone_score_arr[i].cnt;
            }
        }
        return phone_score_arr;
    }

    /// 목록 파일을 읽어 둔다. 파일이 없거나 이름이 비어 있으면 하한을 쓰지 않는다 —
    /// 이 항목이 생기기 전에 만든 model_state 도 그대로 돌아야 한다.
    void Scorer::loadFunctionWords(const std::string& base_path, const std::string& file_name) {
        if (file_name.empty() || function_word_floor_ <= 0.0f) {
            return;
        }
        std::ifstream ifs(base_path + "/" + file_name);
        if (!ifs) {
            // 없다고 죽이지는 않는다. 하한만 못 쓸 뿐 채점 자체는 된다.
            LoggerManager::Log("WARN", "function word list not found: " + file_name);
            function_word_floor_ = 0.0f;
            return;
        }
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();   // CRLF 대비
            if (line.empty() || line[0] == '#') continue;
            std::string w = normalizeForFunctionWord(line);
            if (!w.empty()) function_words_.insert(w);
        }
        LoggerManager::Log("INFO", "function words loaded: " + std::to_string(function_words_.size())
                                   + ", floor: " + std::to_string(function_word_floor_));
    }

    /// 소문자로 바꾸고 앞뒤 문장부호를 뗀다.
    ///
    /// 채점에 넘어오는 라벨은 원문 그대로라 "The", "and,", "it." 처럼 들어온다.
    /// 아포스트로피는 남긴다 — "it's" 같은 축약형이 목록에 있기 때문이다.
    std::string Scorer::normalizeForFunctionWord(const std::string& word) {
        size_t begin = 0, end = word.size();
        auto keep = [](unsigned char c) {
            return std::isalpha(c) != 0 || c == '\'';
        };
        while (begin < end && !keep(static_cast<unsigned char>(word[begin]))) ++begin;
        while (end > begin && !keep(static_cast<unsigned char>(word[end - 1]))) --end;
        std::string out;
        out.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(word[i]))));
        }
        return out;
    }

    bool Scorer::isFunctionWord(const std::string& word) const {
        if (function_words_.empty()) return false;
        return function_words_.count(normalizeForFunctionWord(word)) > 0;
    }

    void Scorer::setProperNounFloor(const std::string& base_path, const std::string& list_file, float floor) {
        proper_nouns_.clear();
        proper_noun_floor_ = std::isfinite(floor) ? std::clamp(floor, 0.0f, 100.0f) : 0.0f;
        proper_noun_floor_min_ = proper_noun_floor_ * 0.5f;
        loadProperNouns(base_path, list_file);
    }

    /// 목록 파일을 읽어 둔다. 파일이 없거나 이름이 비어 있으면 하한을 쓰지 않는다.
    void Scorer::loadProperNouns(const std::string& base_path, const std::string& file_name) {
        if (file_name.empty() || proper_noun_floor_ <= 0.0f) {
            proper_noun_floor_ = 0.0f;
            return;
        }
        std::ifstream ifs(base_path + "/" + file_name);
        if (!ifs) {
            LoggerManager::Log("WARN", "proper noun list not found: " + file_name);
            proper_noun_floor_ = 0.0f;
            return;
        }
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();   // CRLF 대비
            if (line.empty() || line[0] == '#') continue;
            std::string w = normalizeForProperNoun(line);
            if (!w.empty()) proper_nouns_.insert(w);
        }
        LoggerManager::Log("INFO", "proper nouns loaded: " + std::to_string(proper_nouns_.size())
                                   + ", floor: " + std::to_string(proper_noun_floor_));
    }

    /// 앞뒤 문장부호만 뗀다. 대소문자는 그대로 둔다 — 고유명사 판정은 철자 그대로 비교한다.
    /// 낱말 안의 아포스트로피·하이픈·마침표는 남긴다("K's", "K-pop", "a.m.").
    std::string Scorer::normalizeForProperNoun(const std::string& word) {
        size_t begin = 0, end = word.size();
        auto keep = [](unsigned char c) {
            return std::isalnum(c) != 0;
        };
        while (begin < end && !keep(static_cast<unsigned char>(word[begin]))) ++begin;
        while (end > begin && !keep(static_cast<unsigned char>(word[end - 1]))) --end;
        return word.substr(begin, end - begin);
    }

    bool Scorer::isProperNoun(const std::string& word) const {
        if (proper_nouns_.empty()) return false;
        return proper_nouns_.count(normalizeForProperNoun(word)) > 0;
    }

    void Scorer::setUngradedWords(const std::string& base_path, const std::string& list_file) {
        ungraded_words_.clear();
        if (list_file.empty()) return;
        std::ifstream ifs(base_path + "/" + list_file);
        if (!ifs) {
            LoggerManager::Log("WARN", "ungraded word list not found: " + list_file);
            return;
        }
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();   // CRLF 대비
            if (line.empty() || line[0] == '#') continue;
            std::string w = normalizeForProperNoun(line);
            if (!w.empty()) ungraded_words_.insert(w);
        }
        LoggerManager::Log("INFO", "ungraded words loaded: " + std::to_string(ungraded_words_.size()));
    }

    bool Scorer::isUngradedWord(const std::string& word) const {
        if (ungraded_words_.empty()) return false;
        return ungraded_words_.count(normalizeForProperNoun(word)) > 0;
    }

    namespace {
        using Words = std::vector<std::string>;
        const Words small_numbers = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                     "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
        const Words tens = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

        std::string cardinal(int n) {
            if (n < 20) return small_numbers[n];
            if (n < 100) return tens[n / 10] + (n % 10 ? " " + small_numbers[n % 10] : "");
            for (const auto& unit : std::vector<std::pair<int, std::string>>{{1000000, "million"}, {1000, "thousand"}, {100, "hundred"}}) {
                if (n >= unit.first)
                    return cardinal(n / unit.first) + " " + unit.second + (n % unit.first ? " " + cardinal(n % unit.first) : "");
            }
            return "";
        }

        std::string ordinal(int n) {
            static const Words first = {"", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
                                        "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth"};
            static const Words tenth = {"", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth"};
            if (n < 20) return first[n];
            if (n < 100) return n % 10 ? tens[n / 10] + " " + first[n % 10] : tenth[n / 10];
            return "";
        }

        bool digit(char c) { return c >= '0' && c <= '9'; }

        // Finite, value-preserving readings for the integer forms in contents.
        // Unsupported numbers retain their literal token; they are never wildcards.
        std::vector<Words> numberReadings(const std::string& token) {
            if (token.empty()) return {};
            const bool currency = token[0] == '$';
            size_t i = currency ? 1 : 0;
            int n = 0, group_digits = 0;
            bool comma = false;
            for (; i < token.size() && (digit(token[i]) || token[i] == ','); ++i) {
                if (token[i] == ',') {
                    if (group_digits == 0 || (comma ? group_digits != 3 : group_digits > 3)) return {};
                    comma = true;
                    group_digits = 0;
                } else {
                    n = n * 10 + token[i] - '0';
                    if (n > 10000000) return {};
                    ++group_digits;
                }
            }
            if (group_digits == 0 || (comma && group_digits != 3)) return {};
            const std::string suffix = token.substr(i);
            if (currency && !suffix.empty()) return {};
            if (!suffix.empty() && suffix != "st" && suffix != "nd" && suffix != "rd" && suffix != "th"
                && suffix != "s" && suffix != "'s" && suffix != "%" && suffix != "d") return {};
            std::set<std::string> forms{cardinal(n)};
            if ((1000 <= n && n < 2000) || (2000 < n && n < 2100)) {
                const int low = n % 100;
                forms.insert(cardinal(n / 100) + (low == 0 ? " hundred" : low < 10 ? " oh " + cardinal(low) : " " + cardinal(low)));
            }
            const auto original_forms = forms;
            for (std::string form : original_forms) {
                size_t pos = 0;
                while ((pos = form.find("hundred ", pos)) != std::string::npos) {
                    pos += 8;
                    if (form.compare(pos, 8, "thousand") != 0 && form.compare(pos, 7, "million") != 0) {
                        form.insert(pos, "and ");
                        pos += 4;
                    }
                }
                forms.insert(form);
            }
            if (n >= 1000 && n % 1000 > 0 && n % 1000 < 100)
                forms.insert(cardinal(n - n % 1000) + " and " + cardinal(n % 1000));
            if (suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th") {
                const std::string form = ordinal(n);
                if (form.empty()) return {};
                forms = {form};
            }
            std::vector<Words> readings;
            for (std::string form : forms) {
                if (suffix == "s" || suffix == "'s")
                    form = form.back() == 'y' ? form.substr(0, form.size() - 1) + "ies" : form + "s";
                if (suffix == "%") form += " percent";
                if (suffix == "d") form += " d";
                if (currency) form += n == 1 ? " dollar" : " dollars";
                std::istringstream stream(form);
                Words words;
                for (std::string word; stream >> word;) words.push_back(word);
                readings.push_back(std::move(words));
            }
            return readings;
        }

        struct RecognitionToken {
            std::string text;
            bool has_digit;
            std::vector<Words> readings;
        };

        // Orthographic variants, not acoustic substitutions (e.g. night/knight).
        std::string comparisonSpelling(std::string token) {
            static const std::map<std::string, std::string> variants = {
                {"organisations", "organizations"}, {"tumour", "tumor"},
                {"revolutionise", "revolutionize"}, {"behaviour", "behavior"},
                {"wilfully", "willfully"}, {"realised", "realized"},
                {"tranquillity", "tranquility"}, {"ploughed", "plowed"},
                {"wracked", "racked"}
            };
            const auto found = variants.find(token);
            if (found != variants.end()) token = found->second;
            // Possessive 's and plural s have the same spoken suffix. Preserve
            // contractions: it's/its and I'd/id are not interchangeable words.
            static const std::set<std::string> contractions = {
                "it's", "he's", "she's", "that's", "there's", "here's", "what's",
                "who's", "let's", "how's", "where's", "when's", "why's"
            };
            if (token.size() > 2 && token.compare(token.size()-2, 2, "'s") == 0
                && !contractions.count(token)) token.erase(token.size()-2, 1);
            return token;
        }

        std::vector<RecognitionToken> recognitionTokens(const std::string& text) {
            std::vector<RecognitionToken> words;
            std::string current;
            bool has_digit = false;
            auto flush = [&]() {
                const size_t begin = current.find_first_not_of('\'');
                if (begin != std::string::npos) {
                    std::string token = current.substr(begin, current.find_last_not_of('\'') - begin + 1);
                    auto readings = has_digit ? numberReadings(token) : std::vector<Words>{};
                    if (!readings.empty()) token.erase(std::remove(token.begin(), token.end(), ','), token.end());
                    if (token == "i'm") {
                        words.push_back({"i", false, {}});
                        words.push_back({"am", false, {}});
                    } else {
                        if (!has_digit) token = comparisonSpelling(std::move(token));
                        words.push_back({std::move(token), has_digit, std::move(readings)});
                    }
                }
                current.clear();
                has_digit = false;
            };
            for (size_t i = 0; i < text.size(); ++i) {
                const unsigned char c = static_cast<unsigned char>(text[i]);
                const bool next_digit = i + 1 < text.size() && digit(text[i + 1]);
                if (c == 0xE2 && i + 2 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x80
                    && (static_cast<unsigned char>(text[i + 2]) == 0x99 || static_cast<unsigned char>(text[i + 2]) == 0x98)) {
                    current += '\'';
                    i += 2;
                } else if (c == 0xCA && i + 1 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0xBC) {
                    current += '\'';
                    ++i;
                } else if (c >= 'A' && c <= 'Z') {
                    current += static_cast<char>(c - 'A' + 'a');
                } else if ((c >= 'a' && c <= 'z') || c == '\'') {
                    current += static_cast<char>(c);
                } else if (digit(c)) {
                    current += static_cast<char>(c);
                    has_digit = true;
                } else if ((c == '$' && current.find_first_not_of('\'') == std::string::npos && next_digit) ||
                           ((c == ',' || c == '.' || c == ':' || c == '-') && has_digit && next_digit) ||
                           ((c == '-' || c == '.') && current.empty() && next_digit) || (c == '%' && has_digit)) {
                    current += static_cast<char>(c);
                } else {
                    flush();
                    if (c == '&') words.push_back({"and", false, {}});
                    if (c == '+') words.push_back({"plus", false, {}});
                }
            }
            flush();
            return words;
        }
    } // namespace

    float Scorer::wordSimilarity(const std::string& reference, const std::string& hypothesis) {
        const auto ref = recognitionTokens(reference);
        const auto hyp = recognitionTokens(hypothesis);
        if (ref.empty()) return 1.0f;
        if (hyp.empty()) return 0.0f;
        // Three rolling rows support 2-to-1 word boundary equivalence.
        std::vector<int> prev2(hyp.size() + 1), prev(hyp.size() + 1), cur(hyp.size() + 1);
        std::iota(prev.begin(), prev.end(), 0);
        for (size_t i = 1; i <= ref.size(); ++i) {
            cur[0] = prev[0] + 1;
            for (size_t j = 1; j <= hyp.size(); ++j) {
                const int cost = ref[i - 1].text == hyp[j - 1].text ? 0 : 1;
                cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
                for (const auto& reading : ref[i - 1].readings) {
                    if (j >= reading.size() && std::equal(reading.begin(), reading.end(), hyp.begin() + j - reading.size(),
                            [](const std::string& word, const RecognitionToken& token) { return word == token.text; }))
                        cur[j] = std::min(cur[j], prev[j - reading.size()]);
                }
                if (i >= 2 && !ref[i - 2].has_digit && !ref[i - 1].has_digit && !hyp[j - 1].has_digit
                    && ref[i - 2].text + ref[i - 1].text == hyp[j - 1].text)
                    cur[j] = std::min(cur[j], prev2[j - 1]);
                if (j >= 2 && !ref[i - 1].has_digit && !hyp[j - 2].has_digit && !hyp[j - 1].has_digit
                    && ref[i - 1].text == hyp[j - 2].text + hyp[j - 1].text)
                    cur[j] = std::min(cur[j], prev[j - 2]);
            }
            prev2.swap(prev);
            prev.swap(cur);
        }
        return std::max(0.0f, 1.0f - float(prev[hyp.size()]) / float(ref.size()));
    }

    FinalScore Scorer::applyRecognitionPenalty(FinalScore score, const std::string& label,
                                               const std::string& recognized) const {
        if (recognition_weight_ <= 0.0f) return score;
        const float similarity = wordSimilarity(label, recognized);
        score.avg_score *= 1.0f - recognition_weight_ * (1.0f - similarity);
        return score;
    }

    float Scorer::weightSum(float score1, float score2, float weight) {
        return weight * score1 + (1.0f - weight) * score2;
    }

    FinalScore Scorer::applyRecognitionBlend(FinalScore score, const std::string& label,
                                             const std::string& recognized,
                                             const SentenceScoreWeights& weights) const {
        // Validate at the API boundary too, in case a caller modified the fields.
        const SentenceScoreWeights checked(weights.pronunciation, weights.recognition);
        score.avg_score = checked.pronunciation * score.avg_score
            + checked.recognition * 100.0f * wordSimilarity(label, recognized);
        return score;
    }

    FinalScore
    Scorer::mergeScore(const std::vector<WordScore>& word_score,
                       const std::vector<PhoneScore>& phone_score,
                       const PronunciationScoreWeights* weights) {
        std::vector<PhoneScore> final_score;
        float avg_score = 0.0f;
        size_t graded_count = 0;
        const float phone_weight = weights
            ? PronunciationScoreWeights(weights->phone, weights->word).phone : phone_weight_;
        
        for (size_t i = 0; i < word_score.size(); ++i) {
            const auto& [l_word, l_score, l_cnt] = word_score[i];
            const auto& p_arr = phone_score[i].phone_each_score;
            const float p_score = phone_score[i].score;
            
            std::vector<WordScore> tmp_phone;
            for (const auto& p_word_score : p_arr) {
                const auto& [phn, score, cnt] = p_word_score;
                WordScore tmp_word;
                tmp_word.word = phn;
                tmp_word.score = weightSum(score, l_score, phone_weight);
                tmp_phone.push_back(tmp_word);
            }
            
            float finetune_score = weightSum(p_score, l_score, phone_weight);
            const float raw_score = finetune_score;
            std::string floor_kind;
            // 기능어는 약하게 발음하는 것이 정상이라 하한으로 받쳐 준다.
            // 잘 냈으면 원래 점수가 이미 하한보다 높아 손대지 않는다.
            //
            // 다만 아무 때나 받쳐 주면 안 된다. 타겟과 전혀 다른 문장을 읽어도 기능어가
            // 무조건 하한을 받으면 총점이 과도하게 올라갈 수 있다.
            // 하한의 절반보다 낮은 원점수에는 이 보정을 적용하지 않는다.
            if (function_word_floor_ > 0.0f && finetune_score >= function_word_floor_min_
                && isFunctionWord(l_word)) {
                if (finetune_score < function_word_floor_) floor_kind = "function_word";
                finetune_score = std::max(finetune_score, function_word_floor_);
            }
            // 고유명사도 같은 규칙으로 받쳐 준다. 목록은 철자 그대로(대소문자 유지) 비교하므로
            // 문장 첫 낱말이나 일반 명사는 목록에 없는 한 영향을 받지 않는다.
            // 원점수가 하한의 절반에 못 미치면 엉뚱한 발화로 보고 그대로 둔다.
            if (proper_noun_floor_ > 0.0f && finetune_score >= proper_noun_floor_min_
                && finetune_score < proper_noun_floor_ && isProperNoun(l_word)) {
                floor_kind = "proper_noun";
                finetune_score = proper_noun_floor_;
            }
            // 채점 제외 낱말은 점수를 남기되 문장 평균에는 넣지 않는다.
            const bool graded = !isUngradedWord(l_word);
            if (graded) {
                avg_score += finetune_score;
                ++graded_count;
            }

            PhoneScore tmp_phone_score;
            tmp_phone_score.word = l_word;
            tmp_phone_score.phone_each_score = tmp_phone;
            tmp_phone_score.score = finetune_score;
            tmp_phone_score.raw_score = raw_score;
            tmp_phone_score.floor_kind = floor_kind;
            tmp_phone_score.graded = graded;
            final_score.push_back(tmp_phone_score);
        }
        // 모든 낱말이 채점 제외라면 평균을 낼 근거가 없다 — 그때만 전체 평균으로 돌아간다.
        if (graded_count == 0) {
            for (const auto& item : final_score) avg_score += item.score;
            graded_count = final_score.size();
        }
        FinalScore ret = {final_score, graded_count ? avg_score / graded_count : 0.0f};

        return ret;
    }

    FinalScore
    Scorer::getScore(const std::string& label,
                    std::shared_ptr<Tensor<float>> wrd_out,
                    const std::vector<int>& wrd_idx,
                    const std::vector<int>& wrd_idx_group,
                    std::shared_ptr<Tensor<float>> phn_out,
                    const std::vector<int>& phn_idx,
                    const std::vector<int>& phn_idx_group,
                    const PronunciationScoreWeights* weights) {
        std::istringstream iss(label);
        std::vector<std::string> lab_words;
        std::string word;
        while (iss >> word) {
            lab_words.push_back(word);
        }

        auto word_score = getWordScore(wrd_out, wrd_idx, wrd_idx_group, lab_words);
        auto phone_score = getPhoneScore(phn_out, phn_idx, phn_idx_group, lab_words);
        assert(word_score.size() == phone_score.size());
        FinalScore final_score = mergeScore(word_score, phone_score, weights);
        return final_score;
    }
} // namespace MERCURY
