#ifndef MERCURY_CONFORMER_MULTIHEAD
#define MERCURY_CONFORMER_MULTIHEAD

#include <string>
#include <vector>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/linear.h"
#include "module/conformer_head.h"

// ConformerMultiHead: Conformer Multi-Head Attention 모듈
// - Python ConformerMultiHead 와 동일: fused qkv_proj(d_model -> 3*d_model) 와
//   pos_proj(d_model -> d_model) GEMM 후 per-head 슬라이스로 attention 병렬 실행
// - 각 head 출력을 concat 후 linear projection 적용

namespace MERCURY {
    template<typename T>
    class ConformerMultiHead {
    public:
        ConformerMultiHead(JsonObject conformer_multihead_path, const std::string& base_path);
        ~ConformerMultiHead() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input,
                                           const std::shared_ptr<Tensor<T>>& pos_enc);

        std::shared_ptr<Linear<T>> qkv_proj;   // [q; k; v] 순서
        std::shared_ptr<Linear<T>> pos_proj;
        std::shared_ptr<Linear<T>> linear;
        std::vector<std::shared_ptr<ConformerHead<T>>> heads;
        int n_head;
        int d_model;
        int head_dim;
    };
} // namespace MERCURY

#endif
