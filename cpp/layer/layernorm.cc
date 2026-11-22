#include "layer/layernorm.h"

namespace MERCURY {

// 생성자: gamma(weight)와 beta(bias) 파라미터 로드
template<typename T>
LayerNorm<T>::LayerNorm(JsonObject ln_path, std::string base_path) {
    weight = std::make_shared<Tensor<T>>();
    bias = std::make_shared<Tensor<T>>();

    std::string load_weight_path = base_path + "/" + ln_path.parserString("weight");
    std::string load_bias_path = base_path + "/" + ln_path.parserString("bias");
    // shape: [1, dim] - 특성 차원 크기
    std::vector<int> shape = ln_path.parserIntArray("shape");
    weight->loadValue(load_weight_path, shape);
    bias->loadValue(load_bias_path, shape);
}

template<typename T>
LayerNorm<T>::~LayerNorm() = default;

// 순전파: y = (x - mean) / sqrt(var + eps) * gamma + beta
template<typename T>
std::shared_ptr<Tensor<T>> LayerNorm<T>::forward(std::shared_ptr<Tensor<T>> input) {
    auto out = std::make_shared<Tensor<T>>();

    // 1단계: 행별 평균 계산 후 중심화 (x - mean)
    const Eigen::Matrix<T, Eigen::Dynamic, 1> mean = input->value.rowwise().mean();
    out->value.noalias() = input->value.colwise() - mean;

    // 2단계: 분산으로부터 역표준편차를 직접 계산 (중간 var 벡터 저장 생략)
    constexpr T kEps = static_cast<T>(1e-5);
    const Eigen::Matrix<T, Eigen::Dynamic, 1> inv_std =
        (out->value.array().square().matrix().rowwise().mean().array() + kEps).rsqrt();

    // 3단계: 정규화 + 스케일 + 시프트를 단일 표현식으로 융합
    // Cache weight/bias row references to avoid repeated row(0) dispatch
    const auto& gamma = weight->value.row(0).array();
    const auto& beta = bias->value.row(0).array();
    out->value = ((out->value.array().colwise() * inv_std.array()).rowwise() * gamma).rowwise() + beta;
    return out;
}

template class LayerNorm<float>;
template class LayerNorm<int>;

} // namespace MERCURY
