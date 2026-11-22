#include "layer/silu.h"

namespace MERCURY {

// 순전파: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// 단일 표현식으로 융합하여 중간 텐서 할당 제거
std::shared_ptr<Tensor<float>> SILU::forward(std::shared_ptr<Tensor<float>> input) {
    auto out = std::make_shared<Tensor<float>>();
    out->value = input->value.array() / (1.0f + (-input->value.array()).exp());
    return out;
}

} // namespace MERCURY
