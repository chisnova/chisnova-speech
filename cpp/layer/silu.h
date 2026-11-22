// SiLU (Swish) 활성화 함수: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Transformer/LLM에서 널리 사용되는 부드러운 비선형 활성화
#ifndef MERCURY_ASR_SILU_H
#define MERCURY_ASR_SILU_H

#include "tensor/tensor.h"

#include <memory>

namespace MERCURY {

class SILU {
public:
    static std::shared_ptr<Tensor<float>> forward(std::shared_ptr<Tensor<float>> input);
};

} // namespace MERCURY

#endif
