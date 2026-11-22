#include "layer/sigmoid.h"

namespace MERCURY {

// 순전파: sigmoid(x) = 1 / (1 + exp(-x))
std::shared_ptr<Tensor<float>> Sigmoid::forward(std::shared_ptr<Tensor<float>> input) {
    auto out = std::make_shared<Tensor<float>>();
    // Eigen array 연산으로 element-wise sigmoid 계산
    out->value = 1.0f / (1.0f + (-input->value.array()).exp());
    return out;
}

} // namespace MERCURY
