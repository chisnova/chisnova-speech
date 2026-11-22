#include "layer/relu.h"

namespace MERCURY {

// 단일 텐서 ReLU: max(0, x), Eigen의 array().max(0)으로 벡터화 연산
template<typename T>
std::shared_ptr<Tensor<T>> Relu<T>::forward(std::shared_ptr<Tensor<T>> input) {
    auto out = std::make_shared<Tensor<T>>();
    out->value = input->value.array().max(0);
    return out;
}

// 다채널 ReLU: Conv2D 출력(채널별 텐서 벡터)에 각각 ReLU 적용
template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Relu<T>::forward(std::vector<std::shared_ptr<Tensor<T>>> inputs) {
    std::vector<std::shared_ptr<Tensor<T>>> outputs;
    outputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto out = std::make_shared<Tensor<T>>();
        out->value = inputs[i]->value.array().max(0);
        outputs.push_back(std::move(out));
    }
    return outputs;
}

template class Relu<float>;
template class Relu<int>;

} // namespace MERCURY
