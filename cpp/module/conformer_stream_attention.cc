#include "module/conformer_stream_attention.h"

namespace MERCURY {
    // 생성자: LayerNorm 및 ConformerStreamMultiHead 가중치 로드
    template<typename T>
    ConformerStreamAttention<T>::ConformerStreamAttention(JsonObject conv_attention_path, const std::string& base_path) {
        layernorm = std::make_shared<LayerNorm<T>>(conv_attention_path.parseObject("layer_norm"), base_path);
        multihead = std::make_shared<ConformerStreamMultiHead<T>>(conv_attention_path.parseObject("attention"), base_path);
    }

    // clear: streaming 상태 초기화 (multihead 내부 key/value 캐시 리셋)
    template<typename T>
    void ConformerStreamAttention<T>::clear() {
        multihead->clear();
    }

    // forward: LayerNorm 정규화 후 streaming multi-head attention 수행
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerStreamAttention<T>::forward(const std::shared_ptr<Tensor<T>>& input,
                                                                    std::shared_ptr<Tensor<T>> cos_chunk,
                                                                    std::shared_ptr<Tensor<T>> sin_chunk,
                                                                    int left_context) {
        std::shared_ptr<Tensor<float>> output = layernorm->forward(input);
        output = multihead->forward(output, cos_chunk, sin_chunk, left_context);
        return output;
    }

    template class ConformerStreamAttention<float>;
} // namespace MERCURY
