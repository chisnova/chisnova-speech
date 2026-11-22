// ReLU 활성화 함수: f(x) = max(0, x)
// 음수를 0으로 클리핑하여 비선형성 도입
#ifndef MERCURY_ASR_RELU_H
#define MERCURY_ASR_RELU_H

#include "tensor/tensor.h"

#include <memory>
#include <vector>

namespace MERCURY {

template<typename T>
class Relu {
public:
    // 단일 텐서에 ReLU 적용
    static std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input);
    // 다채널 텐서 벡터에 각각 ReLU 적용 (Conv2D 출력용)
    static std::vector<std::shared_ptr<Tensor<T>>> forward(std::vector<std::shared_ptr<Tensor<T>>> inputs);
};

} // namespace MERCURY

#endif
