#ifndef MERCURY_TRANSFORMER_TRANSDUCER_LAYER
#define MERCURY_TRANSFORMER_TRANSDUCER_LAYER

#include <string>
#include <memory>
#include <cassert>
#include "json_loader/json_object.h"
#include "tensor/tensor.h"
#include "layer/layernorm.h"
#include "module/transformer_feedforward.h"
#include "module/decoder_cache.h"
#include "module/transformer_multihead.h"
#include "layer/linear.h"

// TransformerTransducerLayer: Transformer-Transducer의 단일 디코더 레이어
// - Pre-LN 구조: LN -> Causal Self-Attn + Add -> LN -> FFN + Add
// - TransformerDecoderLayer와 달리 cross-attention 없음 (prediction network 전용)

namespace MERCURY {
    template<typename T>
    class TransformerTransducerLayer {
    public:
        TransformerTransducerLayer(JsonObject transformer_transducer_layer_path, const std::string& base_path);
        ~TransformerTransducerLayer() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);
        void clear();
        void setMultiHeadCache(const MultiHeadCache& multihead_cache);
        std::shared_ptr<Tensor<T>> forwardContext(const std::shared_ptr<Tensor<T>>& input);

        std::shared_ptr<TransformerMultiHead<T>> multihead1;
        std::shared_ptr<LayerNorm<T>> layernorm1;
        std::shared_ptr<TransformerFeedForward<T>> ffn;
        std::shared_ptr<LayerNorm<T>> layernorm2;
    };
} // namespace MERCURY

#endif
