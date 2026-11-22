#ifndef MERCURY_TRANSFORMER_DECODER_LAYER
#define MERCURY_TRANSFORMER_DECODER_LAYER

#include <string>
#include <memory>
#include <cassert>
#include "json_loader/json_object.h"
#include "tensor/tensor.h"
#include "layer/layernorm.h"
#include "module/transformer_feedforward.h"
#include "module/decoder_cache.h"
#include "module/transformer_multihead.h"

// TransformerDecoderLayer: Transformer 디코더의 단일 레이어
// - Pre-LN 구조: LN -> Self-Attn + Add -> LN -> Cross-Attn + Add -> LN -> FFN + Add
// - forward: 전체 시퀀스 (masked self-attention)
// - forwardContext: autoregressive (causal self-attention + cross attention with cache)

namespace MERCURY {
    template<typename T>
    class TransformerDecoderLayer {
    public:
        TransformerDecoderLayer(JsonObject decoder_layer_path, const std::string& base_path);
        ~TransformerDecoderLayer() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input,
                                           const std::shared_ptr<Tensor<T>>& encoding);
        void clear();
        void setMultiHeadCache(const MultiHeadCache& multihead_cache);
        std::shared_ptr<Tensor<T>> forwardContext(const std::shared_ptr<Tensor<T>>& input,
                                                  const std::shared_ptr<Tensor<T>>& encoding);

        std::shared_ptr<TransformerMultiHead<T>> multihead1;  // self-attention
        std::shared_ptr<LayerNorm<T>> layernorm1;
        std::shared_ptr<TransformerMultiHead<T>> multihead2;  // cross-attention
        std::shared_ptr<LayerNorm<T>> layernorm2;
        std::shared_ptr<TransformerFeedForward<T>> ffn;
        std::shared_ptr<LayerNorm<T>> layernorm3;
    };
} // namespace MERCURY

#endif
