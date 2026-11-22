#ifndef MERCURY_CONFORMER_STREAM_ATTENTION
#define MERCURY_CONFORMER_STREAM_ATTENTION

#include <string>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/layernorm.h"
#include "module/conformer_stream_multihead.h"

// ConformerStreamAttention: Streaming Conformer Attention Module
// - LayerNorm -> ConformerStreamMultiHead
// - 이전 key/value 캐시를 활용한 streaming attention

namespace MERCURY {
    template<typename T>
    class ConformerStreamAttention {
    public:
        ConformerStreamAttention(JsonObject conformer_attention_path, const std::string& base_path);
        ~ConformerStreamAttention() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input,
                                           std::shared_ptr<Tensor<T>> cos_chunk = nullptr,
                                           std::shared_ptr<Tensor<T>> sin_chunk = nullptr,
                                           int left_context = -1);
        void clear();

        std::shared_ptr<LayerNorm<T>> layernorm;
        std::shared_ptr<ConformerStreamMultiHead<T>> multihead;
    };
} // namespace MERCURY

#endif
