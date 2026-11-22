// Sigmoid 활성화 함수: f(x) = 1 / (1 + exp(-x))
// 출력 범위 (0, 1), 게이트 메커니즘에서 주로 사용 (LSTM, GLU 등)
#ifndef MERCURY_ASR_SIGMOID_H
#define MERCURY_ASR_SIGMOID_H

#include "tensor/tensor.h"

#include <memory>

namespace MERCURY {

class Sigmoid {
public:
    static std::shared_ptr<Tensor<float>> forward(std::shared_ptr<Tensor<float>> input);
};

} // namespace MERCURY

#endif
