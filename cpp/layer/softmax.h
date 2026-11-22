// Softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))
// 행별로 적용하여 확률 분포로 변환 (각 행의 합 = 1)
// 수치 안정성을 위해 max-subtraction 기법 사용
#ifndef MERCURY_ASR_SOFTMAX_H
#define MERCURY_ASR_SOFTMAX_H

#include "tensor/tensor.h"

#include <memory>

namespace MERCURY {

class Softmax {
public:
    // 안정적 지수 함수: exp(x - max_val)로 오버플로우 방지
    static float EXP(float x, float max_val);
    static std::shared_ptr<Tensor<float>> forward(std::shared_ptr<Tensor<float>> input);
};

} // namespace MERCURY

#endif
