#ifndef MECURY_ASR_MODEL
#define MECURY_ASR_MODEL
#include "tensor/tensor.h"
#include "score/scorer.h"
#include "beam/beam.h"
#include <cassert>
#include <queue>

namespace MERCURY {
    struct Score {
        int idx;
        float logit;
        float org_logit;
        
        bool operator<(const Score &org) const {
        return this->logit < org.logit;
        }
    };

    template<typename T>
    class AsrModel {
    public:
        // virtual void setDecoderCache(std::shared_ptr<DecoderCache> _decoder_cache) = 0;
        // virtual std::shared_ptr<Tensor<T>> encoding(std::shared_ptr<Tensor<T>> enc) = 0;
        // virtual int getNumDecoderLayer() = 0;
        virtual std::vector<Score> getNbest(std::shared_ptr<Tensor<T>> logits, int num_nbest) = 0;
        virtual std::vector<Beam> run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams) = 0;
        virtual std::vector<Beam> makeInitBeams() = 0;
        virtual void clear() = 0;
        virtual std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<int>> dec, std::shared_ptr<Tensor<T>> enc) = 0;
        // virtual std::shared_ptr<Tensor<float>> getCtcOut() = 0;
        // virtual std::shared_ptr<Tensor<float>> getEncOut() = 0;
        virtual std::shared_ptr<Tensor<T>> getPhoneScore(std::vector<int> phone_dec) = 0;
        virtual std::shared_ptr<Tensor<T>> getWordScore(std::vector<int> word_dec) = 0;
        virtual void warmUpInference(std::shared_ptr<Tensor<T>> feature) = 0;
        virtual std::vector<int> getCtcAsrIds() = 0;
        // 스트리밍 중간 결과용 1패스 CTC(ctc_linear1). 2-pass 구조가 아닌 모델은
        // 빈 결과를 돌려주고, 호출부는 그 경우 중간 결과를 내보내지 않는다.
        virtual std::vector<int> getFirstPassCtcAsrIds() { return std::vector<int>(); }
        virtual std::vector<int> secondPassDecoding(int num_nbest) = 0;
        /// 진단용: 지금까지 누적된 1패스 인코딩. 없으면 nullptr.
        virtual std::shared_ptr<Tensor<T>> getEncoderOutputForDiag() { return nullptr; }
    };
};


#endif

