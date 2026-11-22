// Conv1D: 1차원 합성곱 레이어
// out[oc, t] = sum_ic sum_k W[oc, ic, k] * x[ic, t*stride + k] + b[oc]
// im2col + GEMM 최적화로 구현, grouped convolution 지원
#ifndef MERCURY_ASR_CONVOLUTION1D
#define MERCURY_ASR_CONVOLUTION1D

#include "tensor/tensor.h"
#include "json_loader/json_object.h"

#include <cassert>
#include <future>
#include <string>
#include <vector>

namespace MERCURY {

template<typename T>
class Convolution1D {
public:
    // 1D 합성곱 설정 파라미터
    struct conv1d_config {
        int in_channels;     // 입력 채널 수
        int out_channels;    // 출력 채널 수
        int kernel_size;     // 커널 크기
        int stride;          // 스트라이드
        int padding;         // 제로 패딩 크기
        int dilation;        // 확장 (현재 미사용)
        int groups;          // 그룹 수 (grouped conv 지원)
        int bias_bool;       // bias 사용 여부 (1: 사용)
        int in_group_size;   // 그룹당 입력 채널 수 = in_channels / groups
        int out_group_size;  // 그룹당 출력 채널 수 = out_channels / groups
        int input_size;      // 입력 시퀀스 길이
        int output_size;     // 출력 시퀀스 길이
    };

    Convolution1D(JsonObject cnn_path, std::string base_path);
    ~Convolution1D();

    // 입력에 제로 패딩 적용
    std::shared_ptr<Tensor<T>> makePadded(std::shared_ptr<Tensor<T>> input);
    // im2col: 입력을 행렬 곱이 가능한 형태로 변환
    std::vector<std::shared_ptr<Tensor<T>>> makeInputMatrix(std::shared_ptr<Tensor<T>> input, conv1d_config config);

    // 단일 출력 채널에 대한 합성곱 연산
    static int out_forward_1d(std::shared_ptr<Tensor<T>> input,
                              std::shared_ptr<Tensor<T>> kernel,
                              std::shared_ptr<Tensor<T>> bias,
                              std::shared_ptr<Tensor<T>> output,
                              int out_idx,
                              conv1d_config config);

    std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input);

    std::shared_ptr<Tensor<T>> bias;                       // 편향 [1, out_channels]
    std::vector<std::shared_ptr<Tensor<T>>> kernels;       // 커널 가중치 (채널별)
    conv1d_config config;
};

} // namespace MERCURY

#endif
