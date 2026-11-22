#ifndef MERCURY_TRANSFORMER_HEAD
#define MERCURY_TRANSFORMER_HEAD

#include <string>
#include <memory>
#include <cmath>
#include "tensor/tensor.h"
#include "layer/softmax.h"
#include "module/decoder_cache.h"
#include "logger/logger_manager.h"

// TransformerHead: Transformer 단일 Attention Head (attention 코어 + 캐시)
// - Q/K/V projection 은 TransformerMultiHead 의 fused qkv_proj 가 담당하고,
//   이 클래스는 projection 된 per-head 슬라이스를 받아 attention 을 계산한다
// - causal 캐시(maskedKeys/Vals)는 beam search 의 DecoderCache 와 공유되고,
//   cross 캐시(crossKeys/Vals)는 인코더 출력이 불변이므로 최초 1회만 만든다

namespace MERCURY {
    template<typename T>
    class TransformerHead {
    public:
        TransformerHead(int _head_dim);
        ~TransformerHead() = default;

        void clear();
        void setHeadCache(const HeadCache& head_cache);
        void upperTriangleMasking(const std::shared_ptr<Tensor<T>>& norm_qk);
        void upperTriangleStreamMasking(const std::shared_ptr<Tensor<T>>& norm_qk);

        std::shared_ptr<Tensor<T>> forwardProjected(std::shared_ptr<Tensor<T>> query,
                                                    std::shared_ptr<Tensor<T>> key,
                                                    std::shared_ptr<Tensor<T>> value,
                                                    bool isMask);
        std::shared_ptr<Tensor<T>> causalAttentionProjected(std::shared_ptr<Tensor<T>> query,
                                                            std::shared_ptr<Tensor<T>> key,
                                                            std::shared_ptr<Tensor<T>> value);
        std::shared_ptr<Tensor<T>> crossAttentionProjected(std::shared_ptr<Tensor<T>> query,
                                                           std::shared_ptr<Tensor<T>> key,
                                                           std::shared_ptr<Tensor<T>> value);

        std::shared_ptr<Tensor<float>> maskedKeys;
        std::shared_ptr<Tensor<float>> maskedVals;
        std::shared_ptr<Tensor<float>> crossKeys;   // K^T / sqrt(d_k) 로 미리 변환해 캐시
        std::shared_ptr<Tensor<float>> crossVals;
        int head_dim;
        T inv_norm;
    };
} // namespace MERCURY

#endif
