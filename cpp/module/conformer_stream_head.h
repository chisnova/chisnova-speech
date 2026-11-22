#ifndef MERCURY_CONFORMER_STREAM_HEAD
#define MERCURY_CONFORMER_STREAM_HEAD

#include <string>
#include <memory>
#include <cmath>
#include "tensor/tensor.h"
#include "layer/softmax.h"
#include "layer/rotary_embedding.h"

// ConformerStreamHead: Streaming Conformer 단일 Attention Head
// - QKV projection 은 ConformerStreamMultiHead 의 fused qkv_proj 가 담당하고,
//   이 클래스는 projection 된 per-head q/k/v 슬라이스만 받는다
// - RoPE(cos/sin)를 새 청크의 q/k 에 적용 — 캐시된 k 는 이미 절대 위치로 회전됨
// - Key/Value 캐시를 left_context + 청크길이로 잘라 유지
// - 청크 내부는 양방향(non-causal): 마스킹 없이 캐시 전체에 attention
//   (Python ConformerStreamMultiHead.stream 과 동일)

namespace MERCURY {
    template<typename T>
    class ConformerStreamHead {
    public:
        ConformerStreamHead(int _head_dim);
        ~ConformerStreamHead() = default;

        void clear();
        // q/k/v: (chunk_len, head_dim), cos/sin: (chunk_len, head_dim)
        std::shared_ptr<Tensor<T>> forwardProjected(std::shared_ptr<Tensor<T>> query,
                                                    std::shared_ptr<Tensor<T>> key,
                                                    std::shared_ptr<Tensor<T>> value,
                                                    std::shared_ptr<Tensor<T>> cos_chunk,
                                                    std::shared_ptr<Tensor<T>> sin_chunk,
                                                    int left_context);

        std::shared_ptr<Tensor<T>> KeyCache;    // 회전 적용된 key 캐시 (좌컨텍스트로 잘림)
        std::shared_ptr<Tensor<T>> ValueCache;
        int head_dim;
        T inv_norm;                             // 1 / sqrt(head_dim)
    };
} // namespace MERCURY

#endif
