// GELU 활성화 함수: f(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// Transformer 모델에서 표준적으로 사용되는 활성화 (BERT, GPT 등)
#ifndef MERCURY_ASR_GELU_H
#define MERCURY_ASR_GELU_H

#include "tensor/tensor.h"

#include <memory>

namespace MERCURY {

template<typename T>
class GELU {
public:
    static std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input);
};

} // namespace MERCURY

#endif
