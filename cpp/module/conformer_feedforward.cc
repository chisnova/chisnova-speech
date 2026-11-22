#include "module/conformer_feedforward.h"

namespace MERCURY {
    // 생성자: LayerNorm, Linear1 (확장), Linear2 (축소) 가중치 로드
    template<typename T>
    ConformerFeedForward<T>::ConformerFeedForward(JsonObject feedforward_path, const std::string& base_path) {
        layernorm = std::make_shared<LayerNorm<T>>(feedforward_path.parseObject("layer_norm"), base_path);
        linear1 = std::make_shared<Linear<T>>(feedforward_path.parseObject("linear1"), base_path);
        linear2 = std::make_shared<Linear<T>>(feedforward_path.parseObject("linear2"), base_path);
    }

    // forward: LayerNorm -> Linear1 (d_model -> d_ff) -> SiLU -> Linear2 (d_ff -> d_model)
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerFeedForward<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        std::shared_ptr<Tensor<float>> out = layernorm->forward(input);
        out = linear1->forward(out);
        out = SILU().forward(out);
        out = linear2->forward(out);
        return out;
    }

    template class ConformerFeedForward<float>;
} // namespace MERCURY
