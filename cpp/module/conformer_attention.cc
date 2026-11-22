#include "module/conformer_attention.h"

namespace MERCURY {
    // 생성자: LayerNorm 및 ConformerMultiHead 가중치 로드
    template<typename T>
    ConformerAttention<T>::ConformerAttention(JsonObject conv_attention_path, const std::string& base_path) {
        layernorm = std::make_shared<LayerNorm<T>>(conv_attention_path.parseObject("layer_norm"), base_path);
        multi_head = std::make_shared<ConformerMultiHead<T>>(conv_attention_path.parseObject("attention"), base_path);
    }

    // forward: LayerNorm 정규화 후 Multi-Head Relative Positional Attention 수행
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerAttention<T>::forward(const std::shared_ptr<Tensor<T>>& input,
                                                              const std::shared_ptr<Tensor<T>>& pos_enc) {
        std::shared_ptr<Tensor<float>> output = layernorm->forward(input);
        output = multi_head->forward(output, pos_enc);
        return output;
    }

    template class ConformerAttention<float>;
} // namespace MERCURY
