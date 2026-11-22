#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <memory>
#include <tuple>
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"
#include <logger/logger.h>

namespace MERCURY {
    struct WordScore {
        std::string word;
        float score;
        int cnt;
    };

    struct PhoneScore {
        std::string word;
        std::vector<WordScore> phone_each_score;
        float score;
        int cnt;
        /// 하한을 적용하기 전의 낱말 점수. 하한이 없으면 score 와 같다.
        float raw_score = 0.0f;
        /// 적용된 하한의 종류: "" / "function_word" / "proper_noun".
        std::string floor_kind;
        /// false 면 채점 제외 낱말이다 — 점수는 참고용으로 남기되 문장 평균·최저 낱말에서 뺀다.
        bool graded = true;
    };

    struct FinalScore {
        std::vector<PhoneScore> final_score;
        float avg_score;
    };

    // Per-sentence final-score blend; both shares must be finite and sum to 1.
    struct SentenceScoreWeights {
        float pronunciation;
        float recognition;
        SentenceScoreWeights(float pronunciation, float recognition);
    };

    // Per-sentence shares inside the pronunciation score (before CTC blending).
    struct PronunciationScoreWeights {
        float phone;
        float word;
        PronunciationScoreWeights(float phone, float word);
    };

class Scorer {
public:
    /// 낱말 점수와 음소 점수를 섞는 기본 비율(음소 쪽 몫).
    ///
    /// manifest(model.json)가 phone_weight 를 선언하지 않으면 이 값을 쓴다 —
    /// 이 항목이 생기기 전에 만든 model_state 도 그대로 동작해야 한다.
    static constexpr float DEFAULT_PHONE_WEIGHT = 0.6f;

    /// 기능어 점수의 하한. 0 이면 하한을 적용하지 않는다.
    ///
    /// 강세 박자 언어에서 관사·전치사·be동사 같은 낱말은 약해지는 것이 정상이라,
    /// 또박또박 읽지 않았다고 감점하면 실제 발음 습관과 어긋난다. 잘 발음하면 그대로
    /// 높은 점수를 받고, 뭉갰을 때만 이 값으로 받쳐 준다 — 아예 빼먹은 경우까지
    /// 봐주지는 않도록 100 이 아닌 값을 쓴다.
    static constexpr float DEFAULT_FUNCTION_WORD_FLOOR = 0.0f;

    /// 인식 결과와 타겟이 얼마나 어긋났는지를 총점에 반영하는 정도(0~1). 0 이면 반영하지 않는다.
    ///
    /// 정답 토큰을 입력하는 발음 채점에 2패스 인식 결과의 단어 일치도를 추가한다.
    /// ASR 오류도 총점에 영향을 줄 수 있다. 설정이 없는 기존 모델은 보정을 끈다.
    static constexpr float DEFAULT_RECOGNITION_WEIGHT = 0.0f;

    /// 고유명사 점수의 하한. 0 이면 하한을 적용하지 않는다.
    ///
    /// 지명·인명·외래어는 채점 모델이 학습에서 거의 보지 못해 음소·낱말 디코더가 모두
    /// 확신을 갖지 못한다. 참조 음원(TTS)조차 60~70점대에 머무는 경우가 많아 학습자의
    /// 실제 발음과 무관하게 감점된다. 다만 전혀 다른 말을 했을 때까지 받쳐 주면 안 되므로
    /// 원점수가 하한의 절반 이상일 때만 적용한다 — 기능어 하한과 같은 규칙이다.
    static constexpr float DEFAULT_PROPER_NOUN_FLOOR = 0.0f;


    // phone_size 는 manifest(model.json)가 선언한 음소 개수 — 파일에서 읽을 항목 수를
    // 코드에 고정하면 음소 집합이 바뀔 때 인덱스가 통째로 어긋난다.
    /// phone_weight 는 낱말 점수와 음소 점수를 섞는 비율이다 — mergeScore 참고.
    /// 1.0 이면 음소 점수만, 0.0 이면 낱말 점수만 쓴다.
    Scorer(std::string base_path, std::string model_path, std::string vocab_path, std::string phone_path, std::string dict_path, int phone_size, float phone_weight = DEFAULT_PHONE_WEIGHT,
           const std::string& function_word_path = "",
           float function_word_floor = DEFAULT_FUNCTION_WORD_FLOOR,
           float recognition_weight = DEFAULT_RECOGNITION_WEIGHT);
    ~Scorer() = default;

    /// 고유명사 목록과 하한을 정한다. manifest(model.json)의 proper_noun_list /
    /// proper_noun_floor 를 읽은 뒤 호출한다. 목록 파일이 없거나 하한이 0 이면 쓰지 않는다.
    void setProperNounFloor(const std::string& base_path, const std::string& list_file, float floor);

    /// 채점 제외 낱말 목록을 정한다. manifest(model.json)의 ungraded_word_list 를 읽은 뒤 호출한다.
    ///
    /// 모델이 학습으로도 배우지 못한 형태(예: 문자명 "AA")를 위한 마지막 수단이다. 목록의 낱말은
    /// 낱말 점수를 그대로 돌려주되 문장 평균과 최저 낱말 계산에서 제외하고 graded=false 로 표시한다.
    /// 목록은 아주 작게 유지해야 한다 — 넣는 만큼 학습자의 실제 오류를 놓친다.
    void setUngradedWords(const std::string& base_path, const std::string& list_file);

    // Word decoding
    std::vector<int> makeWordDec(const std::string& label);

    // Phone decoding
    std::vector<int> makePhoneDec(const std::string& label);

    // Score calculations
    std::vector<WordScore> 
    getWordScore(std::shared_ptr<Tensor<float>> w_out, 
                 const std::vector<int>& target, 
                 const std::vector<int>& target_group,
                 const std::vector<std::string>& lab_words);

    std::vector<PhoneScore>
    getPhoneScore(std::shared_ptr<Tensor<float>> p_out, 
                  const std::vector<int>& target, 
                  const std::vector<int>& target_group,
                  const std::vector<std::string>& lab_words);

    std::string to_lower(const std::string& input);
    std::string refineLabel(const std::string& input);
    float weightSum(float score1, float score2, float weight);

    FinalScore
    mergeScore(const std::vector<WordScore>& word_score,
               const std::vector<PhoneScore>& phone_score,
               const PronunciationScoreWeights* weights = nullptr);

    FinalScore
    getScore(const std::string& label,
             std::shared_ptr<Tensor<float>> wrd_out,
             const std::vector<int>& wrd_idx,
             const std::vector<int>& wrd_idx_group,
             std::shared_ptr<Tensor<float>> phn_out,
             const std::vector<int>& phn_idx,
             const std::vector<int>& phn_idx_group,
             const PronunciationScoreWeights* weights = nullptr);

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
    std::vector<std::string> vocab_;
    std::vector<std::string> phone_;
    std::unordered_map<std::string, std::string> phone_dict_;
    std::unordered_map<int, std::string> ctc_dict_;
    std::unordered_map<std::string, int> ctc_index_;

private:
    /// 낱말/음소 점수 결합 비율. 생성자에서 정해지고 이후 바뀌지 않는다.
    float phone_weight_ = DEFAULT_PHONE_WEIGHT;
    float function_word_floor_ = DEFAULT_FUNCTION_WORD_FLOOR;
    /// 하한을 적용할 최소 원점수. 낮은 원점수를 일괄적으로 올리지 않기 위한 기준이다.
    /// 하한의 절반을 쓴다 — 하한 85 면 42.5 미만은 받쳐 주지 않는다.
    float function_word_floor_min_ = 0.0f;
    float recognition_weight_ = DEFAULT_RECOGNITION_WEIGHT;
    float proper_noun_floor_ = DEFAULT_PROPER_NOUN_FLOOR;
    /// 고유명사 하한을 적용할 최소 원점수. 하한의 절반이다.
    float proper_noun_floor_min_ = 0.0f;

public:
    /// 단어 편집거리 기반 일치도(0~1). 숫자는 값이 같은 명시적 읽기만 허용한다.
    static float wordSimilarity(const std::string& reference, const std::string& hypothesis);
    /// 인식 결과를 총점에 반영해 돌려준다. recognition_weight 가 0 이면 그대로 둔다.
    FinalScore applyRecognitionPenalty(FinalScore score, const std::string& label,
                                       const std::string& recognized) const;
    /// Blend the pre-penalty pronunciation score with CTC word similarity (0..100).
    /// Does not change model defaults or per-word/per-phone scores.
    FinalScore applyRecognitionBlend(FinalScore score, const std::string& label,
                                     const std::string& recognized,
                                     const SentenceScoreWeights& weights) const;
private:
    /// 소문자·문장부호 제거를 마친 형태로 담는다 — 비교할 때마다 다듬지 않으려는 것.
    std::set<std::string> function_words_;
    /// 라벨의 낱말을 기능어 목록과 견줄 수 있는 형태로 다듬는다.
    static std::string normalizeForFunctionWord(const std::string& word);
    bool isFunctionWord(const std::string& word) const;
    void loadFunctionWords(const std::string& base_path, const std::string& file_name);
    /// 대소문자를 유지한 채 앞뒤 문장부호만 뗀 형태로 담는다 — "Apple" 과 "apple" 은 다르다.
    std::set<std::string> proper_nouns_;
    static std::string normalizeForProperNoun(const std::string& word);
    bool isProperNoun(const std::string& word) const;
    void loadProperNouns(const std::string& base_path, const std::string& file_name);
    /// 철자 그대로(대소문자 유지) 비교하는 채점 제외 낱말.
    std::set<std::string> ungraded_words_;
    bool isUngradedWord(const std::string& word) const;
};
} // namespace MERCURY
