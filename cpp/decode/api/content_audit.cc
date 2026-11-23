// 콘텐츠 검수기.
//
// 주제 폴더의 음원을 인식하고 .eng 문장을 정답 라벨로 채점한다. TTS 로 만든 음원이므로
// 사람이 읽은 것보다 점수가 높게 나오는 것이 정상이다. 그런데도 낮게 나오면 콘텐츠 쪽에
// 문제가 있다는 신호다 — 음원과 문장이 어긋났거나, 사전에 없는 단어가 섞였거나,
// 음원 자체가 잘못 합성된 경우다.
//
//   ./content_audit <model_dir> <manifest> <out_csv>
//
// manifest 는 한 줄에 "wav경로 \t 정답문장" 이고, 뒤에 네 칸(쉼표로 이은 정수)을 더 붙이면
// .jsonl 에 저장된 채점 인덱스로도 함께 채점해 두 점수를 견준다:
//   wav \t 라벨 \t word_idx \t word_idx_group \t phn_idx \t phn_idx_group
// 선택적인 7번째 칸은 숫자를 완전한 읽기로 확장한 참고 문장이다.
// 8~9번째 칸을 함께 주면 문장별 발음/인식 가중치(합 1)를 사용한다. 생략하면 모델 기본값이다.
// 라벨에서 새로 만든 인덱스와 .jsonl 인덱스가 같은 점수를 내야 정상이다. 크게 벌어지면
// 그 주제의 .jsonl 이 지금 모델과 어긋났다는 뜻이다.
// 모델은 한 번만 올리고 전부 훑는다.
// 중간 산출물은 남기지 않는다 — 검수에 필요한 것은 점수와 인식 결과뿐이다.
//
// 출력 CSV: wav,label,recognized,label_score,jsonl_score,diff,worst_word,worst_score,note,
//           pronunciation_base,default_score,score_policy
// AUDIT_WORD_DETAILS=1: 원본 단어 순서대로 합산 점수와 디코더·음소 원점수를
// word_scores_json 열에 추가한다. 문장 인식 가중치를 적용하기 전 단어 점수다.
// raw_score 는 기능어·고유명사 하한을 적용하기 전 점수이고, score_floor 는 적용된
// 하한의 종류("" / function_word / proper_noun)다. graded=false 는 채점 제외 낱말이다
// (ungraded_word_list) — 점수는 남기되 문장 평균과 최저 낱말에서 뺐다.

#include "commands.h"
#include "json_loader/json_object.h"
#include "feature/audio_reader/audio_reader.h"
#include "feature/librosa_feature.h"
#include "decoder/asr_stream_conformer.h"
#include "score/scorer.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> loadVocabTokens(const std::string& path) {
    std::vector<std::string> tokens;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        tokens.push_back(line.substr(0, line.find('\t')));
    }
    return tokens;
}

std::set<std::string> loadDictWords(const std::string& path) {
    std::set<std::string> words;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        // Scorer 는 사전을 소문자 키로 담고 소문자로 조회한다 — 같은 기준으로 맞춘다
        for (auto& c : word) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (!word.empty()) words.insert(word);
    }
    return words;
}

bool isWordStartToken(const std::string& token) {
    return token.rfind("\xe2\x96\x81", 0) == 0;   // "▁"
}

/// Scorer::to_lower + refineLabel 과 같은 규칙.
/// 어포스트로피는 지우고, 하이픈은 사전을 보고 정한다 — 붙인 형태가 사전에 있으면
/// 한 낱말로 두고, 없으면 가른다.
std::string refine(const std::string& text, const std::set<std::string>& dict) {
    std::string cleaned;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        // U+2019 (’) = E2 80 99 — 어포스트로피와 같이 지운다
        if (c == 0xE2 && i + 2 < text.size() &&
            static_cast<unsigned char>(text[i + 1]) == 0x80 &&
            static_cast<unsigned char>(text[i + 2]) == 0x99) {
            i += 2;
            continue;
        }
        char lowered = (text[i] >= 'A' && text[i] <= 'Z')
                     ? char(text[i] - 'A' + 'a') : text[i];
        if ((lowered >= 'a' && lowered <= 'z') || lowered == ' ' || lowered == '-') {
            cleaned += lowered;
        }
    }

    std::string result;
    std::istringstream iss(cleaned);
    std::string token;
    while (iss >> token) {
        if (!result.empty()) result += ' ';
        if (token.find('-') == std::string::npos) {
            result += token;
            continue;
        }
        std::string joined;
        for (char ch : token) {
            if (ch != '-') joined += ch;
        }
        if (dict.count(joined)) {
            result += joined;
        } else {
            for (char ch : token) result += (ch == '-') ? ' ' : ch;
        }
    }
    return result;
}

std::vector<std::string> split(const std::string& text) {
    std::istringstream iss(text);
    std::vector<std::string> out;
    std::string word;
    while (iss >> word) out.push_back(word);
    return out;
}

/// "1,2,3" 을 정수 목록으로.
std::vector<int> parseInts(const std::string& text) {
    std::vector<int> out;
    std::string token;
    std::istringstream iss(text);
    while (std::getline(iss, token, ',')) {
        if (token.empty()) continue;
        try { out.push_back(std::stoi(token)); } catch (...) {}
    }
    return out;
}

/// CSV 한 칸. 쉼표와 따옴표가 섞여도 깨지지 않게 감싼다.
std::string csv(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else if (c == '\n' || c == '\r') out += ' ';
        else out += c;
    }
    return out + "\"";
}

}  // namespace

namespace {

/// 주어진 인덱스로 채점한다. 앱과 같은 규칙을 쓴다 —
/// teacher forcing 입력은 마지막 하나를 뺀 앞부분, 정답은 처음 하나를 뺀 뒷부분이다.
double scoreWith(MERCURY::AsrStreamConformer<float>& model,
                 const std::shared_ptr<MERCURY::Scorer>& scorer,
                 const std::string& refined,
                 const std::vector<int>& word_ids, const std::vector<int>& word_groups,
                 const std::vector<int>& phone_ids, const std::vector<int>& phone_groups,
                 std::string* worst_word, double* worst_score,
                 const std::string& recognized,
                 const std::string& recognition_label,
                 const MERCURY::SentenceScoreWeights* weights = nullptr,
                 double* base_score = nullptr, double* default_score = nullptr,
                 const MERCURY::PronunciationScoreWeights* pronunciation_weights = nullptr,
                 nlohmann::json* word_details = nullptr) {
    if (word_ids.size() < 2 || phone_ids.size() < 2) return -1;

    std::vector<int> word_infer(word_ids.begin(), word_ids.end() - 1);
    std::vector<int> phone_infer(phone_ids.begin(), phone_ids.end() - 1);
    auto word_posterior = model.getWordScore(word_infer);
    auto phone_posterior = model.getPhoneScore(phone_infer);
    if (word_posterior == nullptr || word_posterior->value.rows() == 0 ||
        phone_posterior == nullptr || phone_posterior->value.rows() == 0) return -1;

    std::vector<int> word_targets(word_ids.begin() + 1, word_ids.end());
    std::vector<int> phone_targets(phone_ids.begin() + 1, phone_ids.end());
    if (word_groups.size() != word_targets.size() ||
        phone_groups.size() != phone_targets.size()) return -2;

    MERCURY::FinalScore score;
    if (word_details) {
        const auto words = split(refined);
        const auto raw_words = scorer->getWordScore(word_posterior, word_targets, word_groups, words);
        const auto raw_phones = scorer->getPhoneScore(phone_posterior, phone_targets, phone_groups, words);
        score = scorer->mergeScore(raw_words, raw_phones, pronunciation_weights);
        *word_details = nlohmann::json::array();
        for (size_t i = 0; i < score.final_score.size(); ++i) {
            auto phones = nlohmann::json::array();
            for (const auto& phone : raw_phones[i].phone_each_score)
                phones.push_back({{"phone", phone.word}, {"score", phone.score}});
            word_details->push_back({{"word_position", i + 1}, {"word", words[i]},
                {"score", score.final_score[i].score}, {"raw_score", score.final_score[i].raw_score},
                {"score_floor", score.final_score[i].floor_kind}, {"graded", score.final_score[i].graded},
                {"word_decoder_score", raw_words[i].score},
                {"phone_decoder_score", raw_phones[i].score}, {"word_token_count", raw_words[i].cnt},
                {"phone_token_count", raw_phones[i].cnt}, {"phones", phones}});
        }
    } else {
        score = scorer->getScore(refined, word_posterior, word_targets, word_groups,
            phone_posterior, phone_targets, phone_groups, pronunciation_weights);
    }

    if (worst_word && worst_score) {
        *worst_score = 101;
        for (const auto& item : score.final_score) {
            if (!item.graded) continue;   // 채점 제외 낱말은 최저 낱말이 될 수 없다
            if (item.score < *worst_score) { *worst_score = item.score; *worst_word = item.word; }
        }
        if (*worst_score > 100) *worst_score = 0;
    }
    // 앱은 채점 뒤 인식 결과로 총점을 맞춘다. 검수도 같은 규칙을 써야 결과를 믿을 수 있다.
    if (base_score) *base_score = score.avg_score;
    const auto default_result = scorer->applyRecognitionPenalty(score, recognition_label, recognized);
    if (default_score) *default_score = default_result.avg_score;
    score = weights ? scorer->applyRecognitionBlend(score, recognition_label, recognized, *weights)
                    : default_result;
    return score.avg_score;
}

}  // namespace

int api_tests::audit(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <model_dir> <manifest> <out_csv>\n";
        return 1;
    }
    const std::string model_dir = argv[1];
    const std::string manifest_path = argv[2];
    const std::string out_path = argv[3];

    const int eos_id = 3;
    const int blank_id = 1;   // manifest 의 blank_id 와 같다
    const int phone_separator_id = 4;
    // 인코더에 한 번에 넣는 오디오 양. 앱의 AsrAsyncDecoder::WAKE_THRESHOLD_SAMPLES 에
    // 해당한다 — 그 값을 바꿨을 때 인식 결과가 달라지는지 재려고 밖에서 지정할 수 있게 뒀다.
    //   AUDIT_SEGMENT_SAMPLES=8000 ./content_audit ...   # 500ms
    int segment_samples = 48000;
    if (const char* env = std::getenv("AUDIT_SEGMENT_SAMPLES")) {
        int v = std::atoi(env);
        if (v > 0) segment_samples = v;
    }
    std::cerr << "segment_samples=" << segment_samples
              << " (" << (segment_samples / 16000.0) << "s)\n";
    const int silence_samples = 8000;

    MERCURY::JsonObject config(model_dir + "/model.json");
    MERCURY::JsonObject model_json = config.parseObject("model_path");
    MERCURY::LibrosaFeature feature_extractor(model_json, model_dir);
    MERCURY::AsrStreamConformer<float> asr_model(model_json.parseObject("model"), model_dir);

    auto scorer = std::make_shared<MERCURY::Scorer>(
        model_dir,
        model_json.parserString("vocab_model"),
        model_json.parserString("vocab_name"),
        model_json.parserString("phone_name"),
        model_json.parserString("dict_name"),
        int(model_json.parserValue("phone_size")),
        // manifest 의 phone_weight 를 그대로 따른다 — 앱과 같은 점수가 나와야
        // 검수 결과를 믿을 수 있다.
        model_json.parserValue("phone_weight", MERCURY::Scorer::DEFAULT_PHONE_WEIGHT),
        model_json.parserStringOr("function_word_list", ""),
        model_json.parserValue("function_word_floor",
                               MERCURY::Scorer::DEFAULT_FUNCTION_WORD_FLOOR),
        model_json.parserValue("recognition_weight",
                               MERCURY::Scorer::DEFAULT_RECOGNITION_WEIGHT));
    scorer->setProperNounFloor(model_dir, model_json.parserStringOr("proper_noun_list", ""),
                               model_json.parserValue("proper_noun_floor",
                                                      MERCURY::Scorer::DEFAULT_PROPER_NOUN_FLOOR));
    // 채점 제외 낱말 목록(마지막 수단). 없으면 모든 낱말을 채점한다.
    scorer->setUngradedWords(model_dir, model_json.parserStringOr("ungraded_word_list", ""));
    auto vocab_tokens = loadVocabTokens(model_dir + "/" + model_json.parserString("vocab_name"));
    auto dict_words = loadDictWords(model_dir + "/" + model_json.parserString("dict_name"));

    std::ofstream out(out_path);
    const bool export_words = std::getenv("AUDIT_WORD_DETAILS") &&
                             std::string(std::getenv("AUDIT_WORD_DETAILS")) == "1";
    out << "wav,label,recognized,label_score,jsonl_score,diff,worst_word,worst_score,note,pronunciation_base,default_score,score_policy,pronunciation_phone_weight,pronunciation_word_weight";
    if (export_words) out << ",word_scores_json";
    out << "\n";

    std::ifstream manifest(manifest_path);
    std::string line;
    int done = 0, skipped = 0;

    while (std::getline(manifest, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cols;
        {
            std::string field;
            std::istringstream iss(line);
            while (std::getline(iss, field, '\t')) cols.push_back(field);
        }
        if (cols.size() < 2) continue;
        const std::string wav_path = cols[0];
        const std::string label = cols[1];
        const bool has_jsonl = cols.size() >= 6;
        std::unique_ptr<MERCURY::SentenceScoreWeights> weights;
        std::unique_ptr<MERCURY::PronunciationScoreWeights> pronunciation_weights;
        if (cols.size() > 7) {
            try {
                if (cols.size() != 9 && cols.size() != 11)
                    throw std::invalid_argument("Expected pairs of score weights");
                auto parseWeight = [&](size_t index) {
                    size_t end = 0;
                    float value = std::stof(cols[index], &end);
                    if (end != cols[index].size())
                        throw std::invalid_argument("Invalid scoring weight text");
                    return value;
                };
                if (!cols[7].empty() || !cols[8].empty())
                    weights = std::make_unique<MERCURY::SentenceScoreWeights>(parseWeight(7), parseWeight(8));
                else if (cols.size() != 11)
                    throw std::invalid_argument("Empty sentence score weights");
                if (cols.size() == 11)
                    pronunciation_weights = std::make_unique<MERCURY::PronunciationScoreWeights>(parseWeight(9), parseWeight(10));
            } catch (const std::exception& error) {
                std::cerr << "Invalid scoring weights for " << wav_path << ": " << error.what() << '\n';
                return 2;
            }
        }
        const std::string score_policy = weights ? "sentence_blend" : "default";

        std::vector<int16_t> raw = audio_read(wav_path.c_str());
        std::vector<float> audio;
        audio.reserve(raw.size());
        for (int16_t sample : raw) audio.push_back(float(sample) / 32768.0f);
        if (audio.empty()) {
            out << csv(wav_path) << "," << csv(label) << ",,,,,,," << csv("오디오를 읽지 못함") << ",,," << score_policy << ",";
            if (pronunciation_weights) out << pronunciation_weights->phone;
            out << ",";
            if (pronunciation_weights) out << pronunciation_weights->word;
            if (export_words) out << ",[]";
            out << "\n";
            ++skipped;
            continue;
        }

        // 앱과 같은 경로로 흘린다 — 48000 샘플씩, 마지막에 무음을 덧붙인다
        asr_model.clear();
        for (size_t start = 0; start < audio.size(); start += segment_samples) {
            size_t end = std::min(audio.size(), start + size_t(segment_samples));
            std::vector<float> segment(audio.begin() + start, audio.begin() + end);
            if (end >= audio.size()) segment.resize(segment.size() + silence_samples, 0.0f);
            auto feature = feature_extractor.getFeature(segment.data(), int(segment.size()));
            asr_model.encoding(feature);
        }
        asr_model.encodeSecondPassIfNeeded();

        // 인식 결과(2패스 CTC). greedy id 라 blank 와 반복이 그대로 들어 있어 걷어낸다.
        std::string recognized;
        int previous = -1;
        for (int id : asr_model.getCtcAsrIds()) {
            if (id == previous) { continue; }          // 같은 프레임이 이어지면 한 글자다
            previous = id;
            if (id == blank_id || id < 0 || id >= int(vocab_tokens.size())) continue;
            const std::string& token = vocab_tokens[id];
            if (token == "<pad>" || token == "<unk>" || token == "<s>" || token == "</s>") continue;
            recognized += token;
        }
        {   // 단어 경계 표시를 공백으로
            size_t pos = 0;
            while ((pos = recognized.find("\xe2\x96\x81", pos)) != std::string::npos) {
                recognized.replace(pos, 3, " ");
                ++pos;
            }
            while (!recognized.empty() && recognized.front() == ' ') recognized.erase(0, 1);
        }

        // Optional field 7 is a validated spoken reference (numbers expanded by
        // the manifest producer). Keep the original label for recognition penalty.
        const std::string refined = refine(cols.size() > 6 ? cols[6] : label, dict_words);
        // 사전에 없는 낱말은 채점을 막지 않는다 — Scorer 가 그 낱말만 건너뛴다.
        // 다만 어느 낱말이 빠졌는지는 남긴다(검수에서 중요한 신호다).
        std::string oov_note;
        {
            std::vector<std::string> oov;
            for (const auto& word : split(refined)) {
                if (!dict_words.count(word)) oov.push_back(word);
            }
            if (!oov.empty()) {
                oov_note = "사전에 없는 낱말:";
                for (const auto& word : oov) oov_note += " " + word;
            }
        }

        // (1) 라벨에서 새로 만든 인덱스로 채점 — 이것이 기준이다
        std::vector<int> word_ids = scorer->makeWordDec(refined);
        // makePhoneDec requires every word in the dictionary (asserts otherwise).
        // JSONL scoring is independent and remains available for OOV references.
        std::vector<int> phone_ids = oov_note.empty()
            ? scorer->makePhoneDec(refined) : std::vector<int>{};

        std::vector<int> word_groups, phone_groups;
        {
            std::vector<int> word_targets(word_ids.begin() + 1, word_ids.end());
            int group = -1;
            for (int target : word_targets) {
                if (target == eos_id) { word_groups.push_back(-1); continue; }
                if (isWordStartToken(vocab_tokens[target])) ++group;
                word_groups.push_back(group);
            }
            std::vector<int> phone_targets;
            if (phone_ids.size() >= 2) {
                phone_targets.assign(phone_ids.begin() + 1, phone_ids.end());
            }
            int phone_group = 0;
            for (int target : phone_targets) {
                phone_groups.push_back(target == eos_id ? -1 : phone_group);
                if (target == phone_separator_id) ++phone_group;
            }
        }

        std::string worst_word;
        double worst_score = 0;
        double label_score = scoreWith(asr_model, scorer, refined,
                                       word_ids, word_groups, phone_ids, phone_groups,
                                       &worst_word, &worst_score, recognized, label, weights.get(),
                                       nullptr, nullptr, pronunciation_weights.get());

        // (2) .jsonl 에 저장된 인덱스로도 채점해 견준다. 앱이 쓰는 값이 바로 이것이다.
        double jsonl_score = -1;
        double pronunciation_base = -1, default_score = -1;
        auto word_details = nlohmann::json::array();
        std::string note = oov_note;
        if (label_score < 0) {
            note += (note.empty() ? "" : " / ");
            note += "참고 라벨 점수 계산 불가(사전/인덱스/인코딩)";
        }
        if (has_jsonl) {
            std::vector<int> j_word = parseInts(cols[2]);
            std::vector<int> j_word_group = parseInts(cols[3]);
            std::vector<int> j_phone = parseInts(cols[4]);
            std::vector<int> j_phone_group = parseInts(cols[5]);
            // 저장된 group 은 target 과 같은 길이여야 한다(맨 앞 -1 포함)
            if (!j_word_group.empty() && j_word_group.size() == j_word.size()) {
                j_word_group.erase(j_word_group.begin());
            }
            if (!j_phone_group.empty() && j_phone_group.size() == j_phone.size()) {
                j_phone_group.erase(j_phone_group.begin());
            }
            // 최저 낱말도 .jsonl 채점에서 가져온다. 앱이 쓰는 것이 이 채점이라
            // 라벨 기준의 최저 낱말을 함께 적으면 총점과 앞뒤가 맞지 않는다.
            std::string j_worst_word;
            double j_worst_score = 0;
            // 앱은 getScore 에 원문 문장을 넘긴다 — 그래야 점수 칸 수가 org 수와 맞는다.
            // 검수기가 정제된 문장을 넘기면 칸이 더 생겨 마지막 낱말이 0점이 된다.
            jsonl_score = scoreWith(asr_model, scorer, label,
                                    j_word, j_word_group, j_phone, j_phone_group,
                                    &j_worst_word, &j_worst_score, recognized, label, weights.get(),
                                    &pronunciation_base, &default_score, pronunciation_weights.get(),
                                    export_words ? &word_details : nullptr);
            if (jsonl_score >= 0 && !j_worst_word.empty()) {
                worst_word = j_worst_word;
                worst_score = j_worst_score;
            }
            const char* extra = nullptr;
            if (jsonl_score == -2) extra = "jsonl 인덱스와 그룹 길이가 맞지 않음";
            else if (jsonl_score < 0) extra = "jsonl 인덱스로 채점하지 못함";
            else if (j_word != word_ids) extra = "낱말 묶음이 라벨 기준과 다름(숫자·하이픈이면 정상)";
            if (extra) note = note.empty() ? extra : note + " / " + extra;
        }

        out << csv(wav_path) << "," << csv(label) << "," << csv(recognized) << ",";
        if (label_score >= 0) out << label_score;
        out << ",";
        if (jsonl_score >= 0) out << jsonl_score;
        out << ",";
        if (jsonl_score >= 0 && label_score >= 0) out << (jsonl_score - label_score);
        out << "," << csv(worst_word) << "," << worst_score << "," << csv(note) << ",";
        if (pronunciation_base >= 0) out << pronunciation_base;
        out << ",";
        if (default_score >= 0) out << default_score;
        out << "," << score_policy << ",";
        if (pronunciation_weights) out << pronunciation_weights->phone;
        out << ",";
        if (pronunciation_weights) out << pronunciation_weights->word;
        if (export_words) out << "," << csv(word_details.dump());
        out << "\n";
        ++done;

        if ((done + skipped) % 50 == 0) {
            std::cout << "[audit] " << (done + skipped) << " 처리" << std::endl;
        }
    }
    std::cout << "[audit] 완료 — 채점 " << done << ", 건너뜀 " << skipped
              << " -> " << out_path << std::endl;
    return 0;
}
