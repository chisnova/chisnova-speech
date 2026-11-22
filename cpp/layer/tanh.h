// Tanh 활성화 함수: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// 출력 범위 (-1, 1), RNN 계열 모델에서 주로 사용
#ifndef MERCURY_ASR_TANH_H
#define MERCURY_ASR_TANH_H

#include "tensor/tensor.h"

#include <memory>

namespace MERCURY {

class Tanh {
public:
    static std::shared_ptr<Tensor<float>> forward(std::shared_ptr<Tensor<float>> input);
};

} // namespace MERCURY

#endif
