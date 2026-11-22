// Conv2D: 2차원 합성곱 레이어
// out[oc, h, w] = sum_ic sum_kh sum_kw W[oc, ic, kh, kw] * x[ic, h*s+kh, w*s+kw] + b[oc]
// im2col + GEMM 최적화로 구현, grouped convolution 지원
#ifndef MERCURY_ASR_CONVOLUTION2D
#define MERCURY_ASR_CONVOLUTION2D

#include "tensor/tensor.h"
#include "json_loader/json_object.h"

#include <cassert>
#include <future>
#include <string>
#include <vector>

namespace MERCURY {

template<typename T>
class Convolution2D {
public:
    // 2D 합성곱 설정 파라미터
    struct conv2d_config {
        int in_channels;     // 입력 채널 수
        int out_channels;    // 출력 채널 수
        int kernel_size;     // 커널 크기 (정사각형)
        int stride;          // 스트라이드
        int padding;         // 제로 패딩 크기
        int dilation;        // 확장 (현재 미사용)
        int groups;          // 그룹 수
        int bias_bool;       // bias 사용 여부
        int in_group_size;   // 그룹당 입력 채널 수
        int out_group_size;  // 그룹당 출력 채널 수
        int input_height;    // 입력 높이
        int input_width;     // 입력 너비
        int output_height;   // 출력 높이
        int output_width;    // 출력 너비
    };

    Convolution2D(JsonObject cnn_path, std::string base_path);
    ~Convolution2D();

    // 각 입력 채널에 제로 패딩 적용
    std::vector<std::shared_ptr<Tensor<T>>> makePaddedInputArr(std::vector<std::shared_ptr<Tensor<T>>> inputs, int input_height, int input_width);

    // 단일 입력-커널 쌍의 2D 합성곱 (element-wise product + sum)
    static std::shared_ptr<Tensor<T>> conv_forward(std::shared_ptr<Tensor<T>> rhp_input,
                                                   std::shared_ptr<Tensor<T>> weight,
                                                   float bias_value,
                                                   conv2d_config config);

    // 단일 출력 채널: 그룹 내 모든 입력 채널 합성곱 결과 합산
    static std::shared_ptr<Tensor<T>> out_forward(std::vector<std::shared_ptr<Tensor<T>>> rhp_inputs,
                                                  std::vector<std::shared_ptr<Tensor<T>>> weight,
                                                  std::shared_ptr<Tensor<T>> bias,
                                                  int out_idx,
                                                  conv2d_config config);

    std::vector<std::shared_ptr<Tensor<T>>> forward(std::vector<std::shared_ptr<Tensor<T>>> inputs);

    // kernel[oc][ic]: [kernel_size, kernel_size] 2D 커널 가중치
    std::vector<std::vector<std::shared_ptr<Tensor<T>>>> kernel;
    std::shared_ptr<Tensor<T>> bias;    // 편향 [1, out_channels]
    conv2d_config config;
};

} // namespace MERCURY

#endif
