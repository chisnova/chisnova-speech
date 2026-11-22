#ifndef MERCURY_TRANSFORMER_ENCODER_LAYER
#define MERCURY_TRANSFORMER_ENCODER_LAYER

#include <string>
#include <memory>
#include <cassert>
#include "tensor/tensor.h"
#include "layer/layernorm.h"
#include "json_loader/json_object.h"
#include "module/transformer_multihead.h"
#include "module/transformer_feedforward.h"

// TransformerEncoderLayer: Transformer 인코더의 단일 레이어
// - Post-LN 구조: MultiHead -> Add & LayerNorm -> FFN -> Add & LayerNorm
// - self-attention + position-wise FFN

namespace MERCURY {
    template<typename T>
    class TransformerEncoderLayer {
    public:
        TransformerEncoderLayer(JsonObject encoder_layer_path, const std::string& base_path);
        ~TransformerEncoderLayer() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);

        std::shared_ptr<TransformerMultiHead<T>> multihead1;
        std::shared_ptr<LayerNorm<T>> layernorm1;
        std::shared_ptr<TransformerFeedForward<T>> ffn;
        std::shared_ptr<LayerNorm<T>> layernorm2;
    };
} // namespace MERCURY

#endif
