// wav.list 전체를 AsrStreamConformer 로 디코딩하며 중간 산출물을 덤프한다.
// Python 모델과의 logit 레벨 비교용 — test/api_test/python_compare_wav_list.py 가 읽는다.
//
//   ./wav_list_compare <local_wav_list> <output_dir> [label_map]
//
// wav 마다 저장하는 것 (모두 int32 rows, int32 cols 헤더 + float32 row-major):
//   <stem>.feat.bin   세그먼트별 fbank 를 이어붙인 (T, 80)
//   <stem>.enc1.bin   1패스 스트림 인코딩 누적 (T', 256)
//   <stem>.logit.bin  ctc_linear2(enc2) — softmax 전 logits (T'', 1024)
//   <stem>.ids.txt    1행: ctc2 greedy ids, 2행: second pass(beam 3) ids
//
// label_map(stem TAB 정답텍스트)을 주면 발음 평가 score 도 함께 산출한다:
//   <stem>.wout.bin   wrd_decoder teacher-forcing posterior (softmax 후)
//   <stem>.pout.bin   phn_decoder teacher-forcing posterior (softmax 후)
//   <stem>.score.txt  1행: "word <평균> phone <평균> final <종합>", 이후 단어별 줄
//
// 오디오 분할은 api_test 엔진과 동일: 48000 샘플 세그먼트, 마지막 세그먼트에
// 0.5초(8000 샘플) 무음을 덧붙인다. 인코딩은 양쪽 모두 스트리밍 경로다.
#include "json_loader/json_object.h"
#include "feature/audio_reader/audio_reader.h"
#include "feature/librosa_feature.h"
#include "decoder/asr_stream_conformer.h"
#include "decoder/asr_async_decoder.h"
#include <cstdlib>
#include <filesystem>
#include "score/scorer.h"

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static void dumpTensor(const std::shared_ptr<MERCURY::Tensor<float>>& tensor, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    int32_t rows = int32_t(tensor->value.rows());
    int32_t cols = int32_t(tensor->value.cols());
    out.write(reinterpret_cast<char*>(&rows), 4);
    out.write(reinterpret_cast<char*>(&cols), 4);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float value = tensor->value(r, c);
            out.write(reinterpret_cast<char*>(&value), 4);
        }
    }
}

static std::string wavStem(const std::string& wav_path) {
    size_t slash = wav_path.find_last_of('/');
    std::string name = (slash == std::string::npos) ? wav_path : wav_path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

// mercury.vocab 첫 컬럼(토큰 문자열) — 단어 그룹 경계("▁" 시작) 판정용
static std::vector<std::string> loadVocabTokens(const std::string& vocab_path) {
    std::vector<std::string> tokens;
    std::ifstream vocab_file(vocab_path);
    std::string line;
    while (std::getline(vocab_file, line)) {
        size_t tab = line.find('\t');
        tokens.push_back(tab == std::string::npos ? line : line.substr(0, tab));
    }
    return tokens;
}

// full.dict 의 단어 키 집합 — makePhoneDec 가 OOV 에서 죽지 않게 사전 검사용
static std::set<std::string> loadDictWords(const std::string& dict_path) {
    std::set<std::string> words;
    std::ifstream dict_file(dict_path);
    std::string line;
    while (std::getline(dict_file, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string word = line.substr(0, tab);
        for (auto& c : word) if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
        words.insert(word);
    }
    return words;
}

// 토큰이 "▁"(0xE2 0x96 0x81) 로 시작하면 새 단어의 시작이다
static bool isWordStartToken(const std::string& token) {
    return token.size() >= 3 && (unsigned char)token[0] == 0xE2
           && (unsigned char)token[1] == 0x96 && (unsigned char)token[2] == 0x81;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: wav_list_compare <wav_list> <output_dir> [label_map]" << std::endl;
        return 1;
    }
    const std::string list_path = argv[1];
    const std::string output_dir = argv[2];
    const int segment_samples = MERCURY::AsrAsyncDecoder<float>::WAKE_THRESHOLD_SAMPLES;
    const int silence_samples = 1600 * 5;
    const int eos_id = 3;
    const int phone_separator_id = 4;

    const char* assets = std::getenv("MERCURY_TEST_MODEL_DIR");
    const std::string model_dir = assets ? assets : "python/binary";
    MERCURY::JsonObject config_parser(model_dir + "/model.json");
    MERCURY::JsonObject model_json = config_parser.parseObject("model_path");
    MERCURY::LibrosaFeature feature_extractor(model_json, model_dir);
    MERCURY::AsrStreamConformer<float> asr_model(model_json.parseObject("model"), model_dir);

    // 발음 평가 준비 (label_map 이 있을 때만 사용)
    std::map<std::string, std::string> reference_texts;
    std::shared_ptr<MERCURY::Scorer> scorer;
    std::vector<std::string> vocab_tokens;
    std::set<std::string> dict_words;
    if (argc >= 4) {
        std::ifstream label_file(argv[3]);
        std::string line;
        while (std::getline(label_file, line)) {
            size_t tab = line.find('\t');
            if (tab != std::string::npos) {
                reference_texts[line.substr(0, tab)] = line.substr(tab + 1);
            }
        }
        scorer = std::make_shared<MERCURY::Scorer>(model_dir,
                                                   model_json.parserString("vocab_model"),
                                                   model_json.parserString("vocab_name"),
                                                   model_json.parserString("phone_name"),
                                                   model_json.parserString("dict_name"),
                                                   int(model_json.parserValue("phone_size")));
        vocab_tokens = loadVocabTokens(model_dir + "/" + model_json.parserString("vocab_name"));
        dict_words = loadDictWords(model_dir + "/" + model_json.parserString("dict_name"));
    }

    std::ifstream list_file(list_path);
    if (!list_file) {
        std::cerr << "Cannot open WAV list " << list_path << std::endl;
        return 1;
    }
    std::string wav_path;
    int wav_count = 0;
    while (std::getline(list_file, wav_path)) {
        if (!wav_path.empty() && wav_path.back() == '\r') wav_path.pop_back();
        if (wav_path.empty()) continue;
        wav_path = (std::filesystem::path(list_path).parent_path() / wav_path).string();
        if (!std::filesystem::is_regular_file(wav_path)) {
            std::cerr << "Missing WAV: " << wav_path << std::endl;
            return 1;
        }
        std::vector<int16_t> wave = audio_read(wav_path.c_str());
        std::vector<float> audio = loadWave(wave);
        if (audio.empty()) {
            std::cerr << "[wav-compare] skip (읽기 실패/빈 오디오): " << wav_path << std::endl;
            return 1;
        }

        asr_model.clear();
        std::shared_ptr<MERCURY::Tensor<float>> merged_features;
        for (size_t start = 0; start < audio.size(); start += segment_samples) {
            size_t end = std::min(audio.size(), start + size_t(segment_samples));
            std::vector<float> segment(audio.begin() + start, audio.begin() + end);
            if (end == audio.size()) {   // 마지막 세그먼트: 0.5초 무음 패딩
                segment.resize(segment.size() + silence_samples, 0.0f);
            }
            std::shared_ptr<MERCURY::Tensor<float>> segment_feature =
                feature_extractor.getFeature(segment.data(), int(segment.size()));
            if (merged_features == nullptr) {
                merged_features = std::make_shared<MERCURY::Tensor<float>>();
                merged_features->value = segment_feature->value;
            }
            else {
                merged_features->appendChunkRow(segment_feature);
            }
            asr_model.encoding(segment_feature);
        }

        if (merged_features == nullptr || asr_model.enc_out_val == nullptr) {
            std::cerr << "[wav-compare] skip (인코딩 프레임 없음): " << wav_path << std::endl;
            return 1;
        }
        asr_model.encodeSecondPassIfNeeded();
        std::shared_ptr<MERCURY::Tensor<float>> ctc2_logits =
            asr_model.ctc_linear2->forward(asr_model.enc2_out_val);

        const std::string stem = wavStem(wav_path);
        dumpTensor(merged_features, output_dir + "/" + stem + ".feat.bin");
        dumpTensor(asr_model.enc_out_val, output_dir + "/" + stem + ".enc1.bin");
        dumpTensor(ctc2_logits, output_dir + "/" + stem + ".logit.bin");

        std::ofstream ids_file(output_dir + "/" + stem + ".ids.txt");
        for (auto row : ctc2_logits->value.rowwise()) {
            int max_idx;
            row.maxCoeff(&max_idx);
            ids_file << max_idx << " ";
        }
        ids_file << "\n";
        std::vector<int> second_pass_ids = asr_model.secondPassDecoding(3);
        for (size_t i = 0; i < second_pass_ids.size(); ++i) {
            ids_file << second_pass_ids[i] << " ";
        }
        ids_file << "\n";

        // ---- 발음 평가 score (라벨이 있고 사전 OOV 가 없을 때) ----
        if (scorer != nullptr && reference_texts.count(stem)) {
            std::string refined_label = scorer->refineLabel(scorer->to_lower(reference_texts[stem]));
            bool has_oov_word = false;
            int scorable_word_count = 0;
            {
                std::istringstream iss(refined_label);
                std::string word;
                while (iss >> word) {
                    ++scorable_word_count;
                    if (!dict_words.count(word)) { has_oov_word = true; break; }
                }
            }
            if (scorable_word_count == 0) {   // 숫자/기호뿐인 라벨 — 0/0 nan 방지
                has_oov_word = true;
            }
            if (has_oov_word) {
                std::ofstream score_file(output_dir + "/" + stem + ".score.txt");
                score_file << "OOV\n";
            }
            else {
                std::vector<int> word_ids = scorer->makeWordDec(refined_label);
                std::vector<int> phone_ids = scorer->makePhoneDec(refined_label);

                // teacher forcing 입력: eos 를 뺀 앞부분 / target: sos 를 뺀 뒷부분
                std::vector<int> word_infer_ids(word_ids.begin(), word_ids.end() - 1);
                std::vector<int> phone_infer_ids(phone_ids.begin(), phone_ids.end() - 1);
                std::shared_ptr<MERCURY::Tensor<float>> word_posterior = asr_model.getWordScore(word_infer_ids);
                std::shared_ptr<MERCURY::Tensor<float>> phone_posterior = asr_model.getPhoneScore(phone_infer_ids);
                dumpTensor(word_posterior, output_dir + "/" + stem + ".wout.bin");
                dumpTensor(phone_posterior, output_dir + "/" + stem + ".pout.bin");

                // 단어 그룹: 토큰이 "▁" 로 시작하면 다음 단어
                std::vector<int> word_targets(word_ids.begin() + 1, word_ids.end());
                std::vector<int> word_groups;
                int word_group = -1;
                for (size_t i = 0; i < word_targets.size(); ++i) {
                    if (word_targets[i] == eos_id) { word_groups.push_back(-1); continue; }
                    if (isWordStartToken(vocab_tokens[word_targets[i]])) ++word_group;
                    word_groups.push_back(word_group);
                }
                // phone 그룹: separator(4) 마다 다음 단어
                std::vector<int> phone_targets(phone_ids.begin() + 1, phone_ids.end());
                std::vector<int> phone_groups;
                int phone_group = 0;
                for (size_t i = 0; i < phone_targets.size(); ++i) {
                    phone_groups.push_back(phone_targets[i] == eos_id ? -1 : phone_group);
                    if (phone_targets[i] == phone_separator_id) ++phone_group;
                }

                MERCURY::FinalScore final_score = scorer->getScore(
                    refined_label, word_posterior, word_targets, word_groups,
                    phone_posterior, phone_targets, phone_groups);

                std::ofstream score_file(output_dir + "/" + stem + ".score.txt");
                score_file << "final " << final_score.avg_score << "\n";
                for (size_t i = 0; i < final_score.final_score.size(); ++i) {
                    score_file << final_score.final_score[i].word << " "
                               << final_score.final_score[i].score << "\n";
                }
            }
        }

        ++wav_count;
        if (wav_count % 10 == 0) {
            std::cout << "[wav-compare] " << wav_count << " done" << std::endl;
        }
    }
    std::cout << "[wav-compare] total " << wav_count << " wavs -> " << output_dir << std::endl;
    return wav_count > 0 ? 0 : 1;
}
