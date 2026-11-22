#include "scorer.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <sstream>
#include <iomanip>

namespace {
    void expectNear(float actual, float expected, const std::string& context) {
        if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.0001f)
            throw std::runtime_error(context + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
    }

    void testSimilarity() {
        const std::vector<std::tuple<std::string, std::string, float>> cases = {
            {"Hello, WORLD!", "hello world", 1},
            {"‘Don't’ stop.", "don't stop", 1},
            {"it's good", "its good", 0.5f},
            {"I'd like it", "id like it", 2.0f / 3},
            {"Market’s role", "markets role", 1},
            {"I am ready", "i'm ready", 1},
            {"behaviour and tumour", "behavior and tumor", 1},
            {"night", "knight", 0},
            {"in", "and", 0},
            {"doctors", "doctor", 0},
            {"opened", "open", 0},
            {"k-pop", "k pop", 1},
            {"R&B and Disney+", "r and b and disney plus", 1},
            {"macro economics", "macroeconomics", 1},
            {"macroeconomics", "macro economics", 1},
            {"the cat sat", "the dog sat", 2.0f / 3},
            {"the cat sat", "the cat", 2.0f / 3},
            {"the cat", "the big cat", 0.5f},
            {"hello", "unrelated words here", 0},
            {"hello", "", 0}, {"hello", "   !!!", 0}, {"", "hello", 1},
            {"1920s", "nineteen twenties", 1},
            {"1940's", "nineteen forties", 1},
            {"2010", "twenty ten", 1},
            {"2010", "two thousand ten", 1},
            {"2007", "two thousand and seven", 1},
            {"2007", "twenty oh seven", 1},
            {"1900", "nineteen hundred", 1},
            {"$32,000", "thirty two thousand dollars", 1},
            {"'$32,000'", "thirty two thousand dollars", 1},
            {"'$32,000'", "thirty two thousand", 0},
            {"$1", "one dollar", 1},
            {"5,000", "5000", 1},
            {"600", "six hundred", 1},
            {"111", "one hundred and eleven", 1},
            {"10%", "ten percent", 1},
            {"3D", "three d", 1},
            {"1st 12th 20th 21st", "first twelfth twentieth twenty first", 1},
            {"1920s", "nineteen twenty s", 0},
            {"1920s", "nineteen thirties", 0},
            {"2007", "twenty seven", 0},
            {"5000", "fifty", 0},
            {"5000", "bananas are yellow", 0},
            {"I paid 5000 today", "i paid bananas are yellow today", 0.25f},
            {"I paid 5000 today", "i paid today", 0.75f},
            {"10%", "ten", 0}, {"$1", "one", 0},
            {"1 2", "12", 0}, {"3.5", "three five", 0},
            {"-5", "five", 0}, {"5-10", "five ten", 0},
            {"1,23", "one hundred twenty three", 0},
        };
        for (const auto& [ref, hyp, expected] : cases)
            expectNear(MERCURY::Scorer::wordSimilarity(ref, hyp), expected, ref + " -> " + hyp);
    }

    void testScores(const std::string& assets) {
        auto make = [&](float weight, float floor = 85.0f) {
            return MERCURY::Scorer(assets, "mercury.model", "mercury.vocab", "phone.txt", "full.dict",
                                   47, 0.5f, "function_words.txt", floor, weight);
        };
        auto scorer = make(0.5f);
        MERCURY::FinalScore original{{{"hello", {{"HH", 70.0f, 1}}, 80.0f, 1}}, 80.0f};
        expectNear(scorer.applyRecognitionPenalty(original, "hello", "hello").avg_score, 80, "matching sentence");
        auto changed = scorer.applyRecognitionPenalty(original, "hello", "goodbye");
        expectNear(changed.avg_score, 40, "half recognition weight");
        expectNear(changed.final_score[0].score, 80, "word score preserved");
        expectNear(changed.final_score[0].phone_each_score[0].score, 70, "phone score preserved");
        expectNear(original.avg_score, 80, "input score unchanged");
        expectNear(scorer.applyRecognitionPenalty(original, "hello", "").avg_score, 40, "empty ASR penalized");
        expectNear(scorer.applyRecognitionPenalty(original, "hello", "  ").avg_score, 40, "whitespace ASR penalized");

        const MERCURY::SentenceScoreWeights nineOne(0.9f, 0.1f);
        auto blend = scorer.applyRecognitionBlend(original, "hello", "wrong", nineOne);
        expectNear(blend.avg_score, 72, "9:1 gives zero ASR score a 10 percent share");
        expectNear(scorer.applyRecognitionBlend(original, "hello", "hello", nineOne).avg_score,
                   82, "9:1 uses ASR similarity on the 0..100 scale");
        expectNear(blend.final_score[0].score, 80, "blend preserves word score");
        expectNear(blend.final_score[0].phone_each_score[0].score, 70, "blend preserves phone score");
        expectNear(scorer.applyRecognitionPenalty(original, "hello", "wrong").avg_score,
                   40, "next default sentence does not inherit blend weights");
        expectNear(scorer.applyRecognitionBlend(original, "hello", "wrong", {1, 0}).avg_score,
                   80, "pronunciation-only endpoint");
        expectNear(scorer.applyRecognitionBlend(original, "hello", "hello", {0, 1}).avg_score,
                   100, "recognition-only endpoint");
        for (const auto& pair : std::vector<std::pair<float, float>>{
                {.9f, .2f}, {-.1f, 1.1f}, {std::numeric_limits<float>::quiet_NaN(), .1f},
                {.9f, std::numeric_limits<float>::infinity()}}) {
            bool rejected = false;
            try { MERCURY::SentenceScoreWeights invalid(pair.first, pair.second); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Invalid sentence weights accepted");
        }

        MERCURY::Scorer legacy(assets, "mercury.model", "mercury.vocab", "phone.txt", "full.dict", 47);
        expectNear(legacy.applyRecognitionPenalty(original, "hello", "wrong").avg_score, 80, "legacy default disabled");
        for (const auto& [weight, expected] : std::vector<std::pair<float, float>>{
                {0, 80}, {1, 0}, {-1, 80}, {2, 0}, {std::numeric_limits<float>::quiet_NaN(), 80}}) {
            auto weighted = make(weight);
            expectNear(weighted.applyRecognitionPenalty(original, "hello", "wrong").avg_score, expected, "weight bounds");
        }
        auto merged = [&](const std::string& word, float score) {
            return scorer.mergeScore({{word, score, 1}}, {{word, {}, score, 1}}).avg_score;
        };
        expectNear(merged("The", 40), 40, "low function word is not boosted");
        expectNear(merged("The", 42.5f), 85, "function word threshold");
        expectNear(merged("and,", 50), 85, "function word floor");
        expectNear(merged("The", 90), 90, "good function word unchanged");
        expectNear(merged("elephant", 50), 50, "content word unchanged");
        {
            // 고유명사 하한: 목록에 철자 그대로 있는 낱말만, 원점수가 하한의 절반 이상일 때만.
            auto proper = make(0.5f);
            proper.setProperNounFloor(assets, "proper_nouns.txt", 90.0f);
            auto mergedWith = [&](MERCURY::Scorer& s, const std::string& word, float score) {
                return s.mergeScore({{word, score, 1}}, {{word, {}, score, 1}}).final_score[0];
            };
            expectNear(mergedWith(proper, "Roppongi", 65.8f).score, 90, "proper noun floor");
            expectNear(mergedWith(proper, "Roppongi", 65.8f).raw_score, 65.8f, "raw score kept");
            if (mergedWith(proper, "Roppongi", 65.8f).floor_kind != "proper_noun")
                throw std::runtime_error("proper noun floor kind not recorded");
            expectNear(mergedWith(proper, "Roppongi,", 45.0f).score, 90, "punctuation stripped, threshold");
            expectNear(mergedWith(proper, "Roppongi", 44.9f).score, 44.9f, "wrong utterance stays low");
            expectNear(mergedWith(proper, "Roppongi", 95.0f).score, 95, "good proper noun unchanged");
            if (!mergedWith(proper, "Roppongi", 95.0f).floor_kind.empty())
                throw std::runtime_error("unfloored word must have empty floor kind");
            expectNear(mergedWith(proper, "roppongi", 65.8f).score, 65.8f, "case-sensitive match");
            expectNear(mergedWith(proper, "elephant", 65.8f).score, 65.8f, "common word unchanged");
            expectNear(mergedWith(proper, "The", 50.0f).score, 85, "function word floor still applies");
            if (mergedWith(proper, "The", 50.0f).floor_kind != "function_word")
                throw std::runtime_error("function word floor kind not recorded");
            expectNear(mergedWith(scorer, "Roppongi", 65.8f).score, 65.8f, "floor off without configuration");
            proper.setProperNounFloor(assets, "missing_list.txt", 90.0f);
            expectNear(mergedWith(proper, "Roppongi", 65.8f).score, 65.8f, "missing list disables floor");
            proper.setProperNounFloor(assets, "proper_nouns.txt", 0.0f);
            expectNear(mergedWith(proper, "Roppongi", 65.8f).score, 65.8f, "zero floor disables floor");
        }
        {
            // 채점 제외 낱말: 점수는 남고 문장 평균·graded 표시에서만 빠진다.
            auto ungraded = make(0.5f);
            ungraded.setUngradedWords(assets, "ungraded_words.txt");
            const std::vector<MERCURY::WordScore> words{{"an", 95.0f, 1}, {"AA", 10.0f, 1}, {"meeting.", 99.0f, 1}};
            const std::vector<MERCURY::PhoneScore> phones{{"an", {}, 95.0f, 1}, {"AA", {}, 10.0f, 1}, {"meeting.", {}, 99.0f, 1}};
            auto result = ungraded.mergeScore(words, phones);
            expectNear(result.avg_score, 97.0f, "ungraded word excluded from sentence average");
            expectNear(result.final_score[1].score, 10.0f, "ungraded word keeps its own score");
            if (result.final_score[1].graded || !result.final_score[0].graded || !result.final_score[2].graded)
                throw std::runtime_error("graded flags wrong");
            expectNear(scorer.mergeScore(words, phones).avg_score, 68.0f, "without the list every word is graded");
            expectNear(ungraded.mergeScore({{"aa", 10.0f, 1}}, {{"aa", {}, 10.0f, 1}}).avg_score, 10.0f, "case-sensitive match");
            expectNear(ungraded.mergeScore({{"AA", 10.0f, 1}}, {{"AA", {}, 10.0f, 1}}).avg_score, 10.0f, "all-ungraded sentence falls back to plain average");
            ungraded.setUngradedWords(assets, "missing_list.txt");
            expectNear(ungraded.mergeScore(words, phones).avg_score, 68.0f, "missing list grades everything");
        }
        const std::vector<MERCURY::WordScore> word{{"okay", 48.9569f, 1}};
        const std::vector<MERCURY::PhoneScore> phone{{"okay", {{"OW", 99.9867f, 1}}, 99.9911f, 1}};
        const MERCURY::PronunciationScoreWeights phoneHeavy(.9f, .1f);
        const auto defaultBefore = scorer.mergeScore(word, phone).avg_score;
        auto phoneBlend = scorer.mergeScore(word, phone, &phoneHeavy);
        expectNear(phoneBlend.avg_score, 94.88768f, "sentence phone/word shares");
        expectNear(scorer.mergeScore(word, phone).avg_score, defaultBefore,
                   "following sentence retains model phone/word defaults");
        expectNear(scorer.applyRecognitionBlend(phoneBlend, "okay", "okay", nineOne).avg_score,
                   95.39891f, "independent inner and outer weighting");
        for (const auto& pair : std::vector<std::pair<float, float>>{
                {1, 0}, {0, 1}}) {
            const MERCURY::PronunciationScoreWeights endpoint(pair.first, pair.second);
            expectNear(scorer.mergeScore(word, phone, &endpoint).avg_score,
                       pair.first * 99.9911f + pair.second * 48.9569f, "inner weight endpoints");
        }
        for (const auto& pair : std::vector<std::pair<float, float>>{
                {.9f, .2f}, {-.1f, 1.1f}, {NAN, .1f}, {.9f, INFINITY}}) {
            bool rejected = false;
            try { MERCURY::PronunciationScoreWeights invalid(pair.first, pair.second); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Invalid pronunciation weights accepted");
        }
        auto mutated = phoneHeavy;
        mutated.word = .5f;
        bool rejected = false;
        try { scorer.mergeScore(word, phone, &mutated); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Mutated pronunciation weights accepted");
    }
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--batch-similarity") {
            std::string line;
            std::cout << std::setprecision(9);
            while (std::getline(std::cin, line)) {
                const auto tab = line.find('\t');
                if (tab == std::string::npos) throw std::runtime_error("Expected reference TAB hypothesis");
                std::cout << MERCURY::Scorer::wordSimilarity(line.substr(0, tab), line.substr(tab+1)) << '\n';
            }
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--similarity") {
            std::cout << MERCURY::Scorer::wordSimilarity(argv[2], argv[3]) << '\n';
            return 0;
        }
        if (argc != 2) {
            std::cerr << "Usage: recognition_score_test <model_assets_dir>\n";
            return 2;
        }
        testSimilarity();
        testScores(argv[1]);
        std::cout << "PASS: normalization, numeric values, edit distance, score weights and function-word floor\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
