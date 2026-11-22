#include "layer/batchnorm1d.h"

namespace MERCURY {

// 생성자: running mean/std, gamma, beta를 로드 후 사전 융합
// 추론 시 y = (gamma/std)*x + (beta - gamma*mean/std) → 단일 affine으로 최적화
template<typename T>
BatchNorm1D<T>::BatchNorm1D(JsonObject ln_path, std::string base_path) {
    mean = std::make_shared<Tensor<T>>();
    std = std::make_shared<Tensor<T>>();
    weight = std::make_shared<Tensor<T>>();
    bias = std::make_shared<Tensor<T>>();

    // 4개 파라미터 파일 경로 파싱 및 로드
    std::string load_mean_path = base_path + "/" + ln_path.parserString("mean");
    std::string load_std_path = base_path + "/" + ln_path.parserString("std");
    std::string load_weight_path = base_path + "/" + ln_path.parserString("weight");
    std::string load_bias_path = base_path + "/" + ln_path.parserString("bias");
    std::vector<int> shape = ln_path.parserIntArray("shape");

    mean->loadValue(load_mean_path, shape);
    std->loadValue(load_std_path, shape);
    weight->loadValue(load_weight_path, shape);
    bias->loadValue(load_bias_path, shape);

    // 가중치 사전 융합: 추론 시 정규화+스케일을 단일 곱+덧셈으로 변환
    // fused_bias = beta - gamma * mean / std
    bias->value = bias->value.array() - (weight->value.array() * mean->value.array()).array() / std->value.array();
    // fused_weight = gamma / std
    weight->value = weight->value.array() / std->value.array();
}

template<typename T>
BatchNorm1D<T>::~BatchNorm1D() = default;

// 순전파: y = fused_weight * x + fused_bias (사전 융합된 단일 affine 변환)
// 입력 shape: [channels, length], 채널별로 스케일+시프트 적용
template<typename T>
std::shared_ptr<Tensor<T>> BatchNorm1D<T>::forward(std::shared_ptr<Tensor<T>> input) {
    auto out = std::make_shared<Tensor<T>>();
    // colwise 연산: 각 채널(행)에 대해 융합된 가중치 곱 + 편향 덧셈
    out->value = (input->value.array().colwise() * weight->value.col(0).array()).colwise()
                 + bias->value.col(0).array();
    return out;
}

template class BatchNorm1D<float>;
template class BatchNorm1D<int>;

} // namespace MERCURY
