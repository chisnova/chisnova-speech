#include "layer/glu.h"

namespace MERCURY {

// forward → forward2 위임
std::shared_ptr<Tensor<float>> GLU::forward(std::shared_ptr<Tensor<float>> input) {
    return forward2(input);
}

// GLU 순전파: output = first_half * sigmoid(second_half)
// PyTorch nn.GLU(dim=0)과 동일하게 행(채널) 방향으로 분할
// Conv1D 출력 [out_channels, time]에서 채널을 절반으로 나눔
std::shared_ptr<Tensor<float>> GLU::forward2(std::shared_ptr<Tensor<float>> input) {
    const int half_rows = input->shape(0) / 2;
    const int cols = input->shape(1);

    auto out = std::make_shared<Tensor<float>>();

    // 입력을 행(채널) 방향으로 절반 분할: first = x[:half, :], second = x[half:, :]
    auto first = input->value.topRows(half_rows).array();
    auto second = input->value.bottomRows(half_rows).array();
    // output = first * sigmoid(second) = first * (1 / (1 + exp(-second)))
    out->value = first * (1.0f / (1.0f + (-second).exp()));
    return out;
}

} // namespace MERCURY
