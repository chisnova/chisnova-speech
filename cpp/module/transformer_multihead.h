#ifndef MERCURY_TRANSFORMER_MULTIHEAD
#define MERCURY_TRANSFORMER_MULTIHEAD

#include <string>
#include <vector>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/linear.h"
#include "module/transformer_head.h"
#include "module/decoder_cache.h"

// TransformerMultiHead: Transformer Multi-Head Attention 모듈
// - Python MultiHead 와 동일: fused qkv_proj Linear(d_model -> 3*d_model) 로
//   한 번에 projection 하고 per-head 슬라이스를 각 head 에 전달
// - cross-attention 은 K/V 를 인코더 출력에서 최초 1회만 projection (head 캐시)
// - 공개 API(forward/causalAttention/crossAttention/setMultiheadCache/clear)는
//   기존과 동일 — TransformerDecoderLayer 는 수정 없이 사용

namespace MERCURY {
    template<typename T>
    class TransformerMultiHead {
    public:
        TransformerMultiHead(JsonObject transformer_multihead_path, const std::string& base_path);
        ~TransformerMultiHead() = default;

        void clear();
        void setMultiheadCache(const MultiHeadCache& multihead_cache);

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& q,
                                           const std::shared_ptr<Tensor<T>>& k,
                                           const std::shared_ptr<Tensor<T>>& v,
                                           bool isMask);
        std::shared_ptr<Tensor<T>> causalAttention(const std::shared_ptr<Tensor<T>>& q,
                                                   const std::shared_ptr<Tensor<T>>& k,
                                                   const std::shared_ptr<Tensor<T>>& v);
        std::shared_ptr<Tensor<T>> crossAttention(const std::shared_ptr<Tensor<T>>& q,
                                                  const std::shared_ptr<Tensor<T>>& k,
                                                  const std::shared_ptr<Tensor<T>>& v);

        std::shared_ptr<Linear<T>> qkv_proj;   // [q; k; v] 순서 fused projection
        std::shared_ptr<Linear<T>> linear;
        std::vector<std::shared_ptr<TransformerHead<T>>> heads;
        int n_head;
        int d_model;
        int head_dim;

    private:
        // fused qkv 출력에서 per-head q/k/v 슬라이스를 만든다
        void sliceHeads(const std::shared_ptr<Tensor<T>>& qkv_q,
                        const std::shared_ptr<Tensor<T>>& qkv_kv,
                        std::vector<std::shared_ptr<Tensor<T>>>& head_queries,
                        std::vector<std::shared_ptr<Tensor<T>>>& head_keys,
                        std::vector<std::shared_ptr<Tensor<T>>>& head_values);
        std::shared_ptr<Tensor<T>> concatHeadOutputs(
            std::vector<std::future<std::shared_ptr<Tensor<float>>>>& future_vector, int rows);
    };
} // namespace MERCURY

#endif
