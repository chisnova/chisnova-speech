// GLU (Gated Linear Unit): GLU(x) = first_half * sigmoid(second_half)
// 입력을 절반으로 분할하여 게이트 메커니즘 적용
// Conformer, 음성 인식 모델에서 주로 사용
#ifndef MERCURY_ASR_GLU_H
#define MERCURY_ASR_GLU_H

#include "tensor/tensor.h"

#include <memory>

namespace MERCURY {

class GLU {
public:
    static std::shared_ptr<Tensor<float>> forward(std::shared_ptr<Tensor<float>> input);
    // 실제 GLU 구현 (forward에서 위임)
    static std::shared_ptr<Tensor<float>> forward2(std::shared_ptr<Tensor<float>> input);
};

} // namespace MERCURY

#endif
