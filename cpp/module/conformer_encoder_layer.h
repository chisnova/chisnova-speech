#ifndef MERCURY_CONFORMER_ENCODER_LAYER
#define MERCURY_CONFORMER_ENCODER_LAYER

#include <string>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/layernorm.h"
#include "module/conformer_feedforward.h"
#include "module/conformer_attention.h"
#include "module/conformer_conv_module.h"

// ConformerEncoderLayer: Conformer 인코더의 단일 레이어
// - FFN1 (half-step) -> Attention -> ConvModule -> FFN2 (half-step)
// - 각 sub-block 앞에 LayerNorm 적용, 각 sub-block 후 residual connection

namespace MERCURY {
    template<typename T>
    class ConformerEncoderLayer {
    public:
        ConformerEncoderLayer(JsonObject conformer_encoder_layer_path, const std::string& base_path);
        ~ConformerEncoderLayer() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input,
                                           const std::shared_ptr<Tensor<T>>& pos_enc);

        std::shared_ptr<ConformerFeedForward<T>> conformer_feedforward1;
        std::shared_ptr<ConformerAttention<T>> conformer_attention;
        std::shared_ptr<ConformerConvModule<T>> conformer_conv_module;
        std::shared_ptr<ConformerFeedForward<T>> conformer_feedforward2;
        std::shared_ptr<LayerNorm<T>> conformer_layernorm1;
        std::shared_ptr<LayerNorm<T>> conformer_layernorm2;
        std::shared_ptr<LayerNorm<T>> conformer_layernorm3;
        std::shared_ptr<LayerNorm<T>> conformer_layernorm4;
    };
} // namespace MERCURY

#endif
