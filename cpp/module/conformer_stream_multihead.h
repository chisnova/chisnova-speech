#ifndef MERCURY_CONFORMER_STREAM_MULTIHEAD
#define MERCURY_CONFORMER_STREAM_MULTIHEAD

#include <string>
#include <vector>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/linear.h"
#include "module/conformer_stream_head.h"

// ConformerStreamMultiHead: 스트리밍 multi-head self-attention
// - Python ConformerStreamMultiHead 와 동일: fused qkv_proj Linear(d_model -> 3*d_model)
//   GEMM 1회 후 per-head 로 슬라이스, head 별 attention -> concat -> linear projection
// - RoPE cos/sin 과 left_context 는 인코더에서 내려받아 각 head 에 전달

namespace MERCURY {
    template<typename T>
    class ConformerStreamMultiHead {
    public:
        ConformerStreamMultiHead(JsonObject conformer_multihead_path, const std::string& base_path);
        ~ConformerStreamMultiHead() = default;

        void clear();
        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input,
                                           std::shared_ptr<Tensor<T>> cos_chunk,
                                           std::shared_ptr<Tensor<T>> sin_chunk,
                                           int left_context);

        std::shared_ptr<Linear<T>> qkv_proj;   // (d_model -> 3*d_model), [q; k; v] 순서
        std::shared_ptr<Linear<T>> linear;     // output projection
        std::vector<std::shared_ptr<ConformerStreamHead<T>>> heads;
        int n_head;
        int d_model;
        int head_dim;
    };
} // namespace MERCURY

#endif
