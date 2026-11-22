#ifndef MERCURY_CONFORMER_STREAM_ENCODER_LAYER
#define MERCURY_CONFORMER_STREAM_ENCODER_LAYER

#include <string>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/layernorm.h"
#include "module/conformer_stream_attention.h"
#include "module/conformer_stream_conv_module.h"
#include "module/conformer_feedforward.h"

// ConformerStreamEncoderLayer: Streaming Conformer 인코더의 단일 레이어
// - 비스트리밍 ConformerEncoderLayer의 streaming 버전
// - FFN1 -> ConvModule(streaming) -> Attention(streaming) -> FFN2
// - Conv와 Attention 모듈이 이전 프레임 캐시를 유지

namespace MERCURY {
    template<typename T>
    class ConformerStreamEncoderLayer {
    public:
        ConformerStreamEncoderLayer(JsonObject conformer_encoder_layer_path, const std::string& base_path);
        ~ConformerStreamEncoderLayer() = default;

        void clear();
        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input,
                                           std::shared_ptr<Tensor<T>> cos_chunk = nullptr,
                                           std::shared_ptr<Tensor<T>> sin_chunk = nullptr,
                                           int left_context = -1);

        std::shared_ptr<LayerNorm<T>> conformer_layernorm1;
        std::shared_ptr<ConformerFeedForward<T>> conformer_feedforward1;
        std::shared_ptr<LayerNorm<T>> conformer_layernorm2;
        std::shared_ptr<ConformerStreamConvModule<T>> conformer_stream_conv_module;
        std::shared_ptr<LayerNorm<T>> conformer_layernorm3;
        std::shared_ptr<ConformerStreamAttention<T>> conformer_stream_attention;
        std::shared_ptr<LayerNorm<T>> conformer_layernorm4;
        std::shared_ptr<ConformerFeedForward<T>> conformer_feedforward2;
    };
} // namespace MERCURY

#endif
