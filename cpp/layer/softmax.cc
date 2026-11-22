#include "layer/softmax.h"

namespace MERCURY {

// Legacy helper — kept for API compatibility but unused in hot path
float Softmax::EXP(float x, float max_val) {
    return exp(x - max_val);
}

// 순전파: softmax(x) = exp(x - max) / sum(exp(x - max))
// Fused pipeline: max, exp, normalize with minimal temporaries
std::shared_ptr<Tensor<float>> Softmax::forward(std::shared_ptr<Tensor<float>> input) {
    auto out = std::make_shared<Tensor<float>>();

    // Fused: compute (x - row_max), then exp, store into out->value
    // Eigen expression templates defer evaluation so this is a single pass over input
    const Eigen::VectorXf row_max = input->value.rowwise().maxCoeff();
    out->value.noalias() = (input->value.colwise() - row_max).array().exp().matrix();

    // Normalize each row by its sum (in-place)
    const Eigen::VectorXf inv_row_sum = out->value.rowwise().sum().array().inverse();
    out->value.array().colwise() *= inv_row_sum.array();
    return out;
}

} // namespace MERCURY
