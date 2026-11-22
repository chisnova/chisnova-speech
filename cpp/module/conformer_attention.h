#ifndef MERCURY_CONFORMER_ATTENTION
#define MERCURY_CONFORMER_ATTENTION

#include <string>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/layernorm.h"
#include "module/conformer_multihead.h"

// ConformerAttention: Conformer Attention Module
// - LayerNorm -> ConformerMultiHead (relative positional attention)
// - ConformerEncoderLayer에서 attention sub-block으로 사용

namespace MERCURY {
    template<typename T>
    class ConformerAttention {
    public:
        ConformerAttention(JsonObject conformer_attention_path, const std::string& base_path);
        ~ConformerAttention() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input,
                                           const std::shared_ptr<Tensor<T>>& pos_enc);

        std::shared_ptr<LayerNorm<T>> layernorm;
        std::shared_ptr<ConformerMultiHead<T>> multi_head;
    };
} // namespace MERCURY

#endif
