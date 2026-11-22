#include "layer/gelu.h"

namespace MERCURY {

// 순전파: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// 단일 융합 표현식으로 중간 텐서 할당 최소화
template<typename T>
std::shared_ptr<Tensor<T>> GELU<T>::forward(std::shared_ptr<Tensor<T>> input) {
    auto out = std::make_shared<Tensor<T>>();
    constexpr T kInvSqrt2 = static_cast<T>(0.7071067811865476); // 1/sqrt(2) 사전 계산 상수
    // unaryExpr: Eigen 배열의 각 원소에 std::erf 적용 (element-wise)
    out->value = (static_cast<T>(0.5) * input->value.array()) *
                 (static_cast<T>(1.0) + (input->value.array() * kInvSqrt2).unaryExpr([](T x) { return std::erf(x); }));
    return out;
}

template class GELU<float>;

} // namespace MERCURY
