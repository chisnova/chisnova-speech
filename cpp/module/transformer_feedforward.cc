#include "module/transformer_feedforward.h"

namespace MERCURY {
    // 생성자: Linear1 (d_model -> d_ff), Linear2 (d_ff -> d_model) 가중치 로드
    template<typename T>
    TransformerFeedForward<T>::TransformerFeedForward(JsonObject ffn_path, const std::string& base_path) {
        linear1 = std::make_shared<Linear<T>>(ffn_path.parseObject("linear1"), base_path);
        linear2 = std::make_shared<Linear<T>>(ffn_path.parseObject("linear2"), base_path);
    }

    // forward: Linear1 -> GELU activation -> Linear2
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerFeedForward<T>::forward(std::shared_ptr<Tensor<T>> input) {
        std::shared_ptr<Tensor<T>> out = linear1->forward(std::move(input));
        out = GELU<T>().forward(std::move(out));
        out = linear2->forward(std::move(out));
        return out;
    }

    template class TransformerFeedForward<float>;
} // namespace MERCURY
