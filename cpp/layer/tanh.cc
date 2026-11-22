#include "layer/tanh.h"

namespace MERCURY {

// 순전파: tanh(x), Eigen의 array().tanh()로 벡터화 연산
std::shared_ptr<Tensor<float>> Tanh::forward(std::shared_ptr<Tensor<float>> input) {
    auto out = std::make_shared<Tensor<float>>();
    out->value = input->value.array().tanh();
    return out;
}

} // namespace MERCURY
