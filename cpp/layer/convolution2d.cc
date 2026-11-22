#include "layer/convolution2d.h"

namespace MERCURY {

// 생성자: JSON에서 합성곱 파라미터 파싱, 가중치를 [oc][ic] 형태의 2D 커널로 분리
template<typename T>
Convolution2D<T>::Convolution2D(JsonObject cnn_path, std::string base_path) {
    // JSON에서 합성곱 하이퍼파라미터 파싱
    config.in_channels = static_cast<int>(cnn_path.parserValue("in_channels"));
    config.out_channels = static_cast<int>(cnn_path.parserValue("out_channels"));
    config.kernel_size = static_cast<int>(cnn_path.parserValue("kernel_size"));
    config.stride = static_cast<int>(cnn_path.parserValue("stride"));
    config.padding = static_cast<int>(cnn_path.parserValue("padding"));
    config.dilation = static_cast<int>(cnn_path.parserValue("dilation"));
    config.groups = static_cast<int>(cnn_path.parserValue("groups"));
    config.bias_bool = static_cast<int>(cnn_path.parserValue("bias"));
    // 그룹당 채널 수 계산
    config.in_group_size = config.in_channels / config.groups;
    config.out_group_size = config.out_channels / config.groups;

    auto weight = std::make_shared<Tensor<T>>();
    bias = std::make_shared<Tensor<T>>();

    std::string weight_path = base_path + "/" + cnn_path.parserString("weight_path");
    std::string bias_path = base_path + "/" + cnn_path.parserString("bias_path");
    // shape: [out_channels, in_group_size * kernel_size * kernel_size] (flatten 형태)
    std::vector<int> shape = cnn_path.parserIntArray("shape");

    weight->loadValue(weight_path, shape);
    if (config.bias_bool == 1) {
        bias->loadValue(bias_path, {1, config.out_channels});
    }

    // 평탄화된 가중치를 [oc][ic] -> [kernel_size x kernel_size] 2D 커널로 재구성
    kernel.reserve(config.out_channels);
    for (int i = 0; i < config.out_channels; ++i) {
        std::vector<std::shared_ptr<Tensor<T>>> input_kernels;
        input_kernels.reserve(config.in_group_size);
        for (int j = 0; j < config.in_group_size; ++j) {
            auto tmp_kernel = std::make_shared<Tensor<T>>();
            tmp_kernel->value.resize(config.kernel_size, config.kernel_size);
            // 각 행을 평탄화된 가중치에서 kernel_size 만큼 추출
            for (int k = 0; k < config.kernel_size; ++k) {
                tmp_kernel->value.row(k) = weight->value.block(
                    i, j * config.kernel_size * config.kernel_size + k * config.kernel_size, 1, config.kernel_size);
            }
            input_kernels.push_back(std::move(tmp_kernel));
        }
        kernel.push_back(std::move(input_kernels));
    }
}

template<typename T>
Convolution2D<T>::~Convolution2D() = default;

// 각 입력 채널에 제로 패딩 적용: [H, W] → [H+2p, W+2p]
template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Convolution2D<T>::makePaddedInputArr(
    std::vector<std::shared_ptr<Tensor<T>>> inputs, int input_height, int input_width) {
    std::vector<std::shared_ptr<Tensor<T>>> padded_inputs;
    padded_inputs.reserve(config.in_channels);

    for (int i = 0; i < config.in_channels; ++i) {
        if (config.padding > 0) {
            int padded_height = input_height + config.padding * 2;
            int padded_width = input_width + config.padding * 2;
            auto padded_input = std::make_shared<Tensor<T>>();
            // 0으로 초기화 후 원본 데이터를 중앙에 배치
            padded_input->value.resize(padded_height, padded_width);
            padded_input->value.setZero();
            padded_input->value.block(config.padding, config.padding, inputs[i]->shape(0), inputs[i]->shape(1)) = inputs[i]->value;
            padded_inputs.push_back(std::move(padded_input));
        } else {
            padded_inputs.push_back(inputs[i]);
        }
    }
    return padded_inputs;
}

// 단일 입력-커널 쌍의 2D 합성곱: 슬라이딩 윈도우 element-wise 곱 + 합산
// output[h, w] = sum(input_patch .* kernel) + bias
template<typename T>
std::shared_ptr<Tensor<T>> Convolution2D<T>::conv_forward(std::shared_ptr<Tensor<T>> rhp_input,
                                                          std::shared_ptr<Tensor<T>> weight,
                                                          float bias_value,
                                                          conv2d_config cfg) {
    auto output = std::make_shared<Tensor<T>>();
    output->value.resize(cfg.output_height, cfg.output_width);

    // 슬라이딩 윈도우: stride 간격으로 입력을 순회
    for (int i = 0; i < rhp_input->shape(0); i += cfg.stride) {
        for (int j = 0; j < rhp_input->shape(1); j += cfg.stride) {
            // 경계 검사: 커널이 입력 범위를 벗어나면 건너뜀
            if (i + cfg.kernel_size > rhp_input->shape(0) || j + cfg.kernel_size > rhp_input->shape(1)) {
                continue;
            }
            // cwiseProduct: 입력 패치와 커널의 element-wise 곱 → sum으로 내적 계산
            float tmp = rhp_input->value.block(i, j, cfg.kernel_size, cfg.kernel_size).cwiseProduct(weight->value).sum();
            output->value(i / cfg.stride, j / cfg.stride) = tmp + bias_value;
        }
    }
    return output;
}

// 단일 출력 채널 합성곱: 그룹 내 모든 입력 채널 결과를 합산
// output = sum_ic conv(input[ic], kernel[oc][ic]) + bias[oc]
template<typename T>
std::shared_ptr<Tensor<T>> Convolution2D<T>::out_forward(std::vector<std::shared_ptr<Tensor<T>>> rhp_inputs,
                                                         std::vector<std::shared_ptr<Tensor<T>>> weight,
                                                         std::shared_ptr<Tensor<T>> bias,
                                                         int out_idx,
                                                         conv2d_config cfg) {
    // 현재 출력 채널이 속한 그룹의 입력 채널 범위 결정
    int group_idx = out_idx / cfg.out_group_size;
    int start_channel = group_idx * cfg.in_group_size;
    int num_channel = cfg.in_group_size;
    float bias_value = cfg.bias_bool == 1 ? bias->value(0, out_idx) : 0.0f;

    // 첫 번째 입력 채널에 bias 포함, 나머지는 bias=0으로 합산
    auto output = conv_forward(rhp_inputs[start_channel], weight[0], bias_value, cfg);
    for (int j = 1; j < num_channel; ++j) {
        auto tmp = conv_forward(rhp_inputs[start_channel + j], weight[j], 0.0f, cfg);
        output->value += tmp->value;
    }
    return output;
}

// 순전파: im2col + GEMM 최적화된 2D 합성곱
// output_h = (H + 2*padding - kernel_size) / stride + 1
// output_w = (W + 2*padding - kernel_size) / stride + 1
template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Convolution2D<T>::forward(std::vector<std::shared_ptr<Tensor<T>>> inputs) {
    std::vector<std::shared_ptr<Tensor<T>>> out;
    // 입출력 크기 계산
    config.input_height = inputs[0]->shape(0);
    config.input_width = inputs[0]->shape(1);
    config.output_height = ((config.input_height - config.kernel_size + 2 * config.padding) / config.stride) + 1;
    config.output_width = ((config.input_width - config.kernel_size + 2 * config.padding) / config.stride) + 1;

    // 모든 입력 채널에 제로 패딩 적용
    auto padded_inputs = makePaddedInputArr(inputs, config.input_height, config.input_width);

    // im2col + 배치 GEMM: 픽셀별 루프 대신 단일 행렬 곱으로 합성곱 수행
    int num_groups = config.groups;
    int ks = config.kernel_size;
    int ks2 = ks * ks;                // 커널 원소 수
    int out_h = config.output_height;
    int out_w = config.output_width;
    int output_size = out_h * out_w;  // 출력의 총 공간 위치 수
    out.resize(config.out_channels);

    for (int g = 0; g < num_groups; ++g) {
        int in_start = g * config.in_group_size;
        // im2col 행렬: [output_size, in_group_size * ks^2]
        // 각 행 = 하나의 출력 위치에 해당하는 모든 입력 패치 평탄화
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> im2col_mat(
            output_size, config.in_group_size * ks2);
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int row_idx = oh * out_w + ow;
                int h_start = oh * config.stride;
                int w_start = ow * config.stride;
                // 각 입력 채널의 2D 패치를 1행으로 펼침
                for (int ic = 0; ic < config.in_group_size; ++ic) {
                    int col_offset = ic * ks2;
                    for (int kh = 0; kh < ks; ++kh) {
                        im2col_mat.block(row_idx, col_offset + kh * ks, 1, ks) =
                            padded_inputs[in_start + ic]->value.block(h_start + kh, w_start, 1, ks);
                    }
                }
            }
        }
        // 가중치 행렬: [in_group_size * ks^2, out_group_size]
        // 각 열 = 하나의 출력 채널 커널을 평탄화
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weight_mat(
            config.in_group_size * ks2, config.out_group_size);
        for (int oc = 0; oc < config.out_group_size; ++oc) {
            int global_oc = g * config.out_group_size + oc;
            for (int ic = 0; ic < config.in_group_size; ++ic) {
                int col_offset = ic * ks2;
                for (int kh = 0; kh < ks; ++kh) {
                    weight_mat.block(col_offset + kh * ks, oc, ks, 1) =
                        kernel[global_oc][ic]->value.row(kh).transpose();
                }
            }
        }
        // 단일 GEMM: [output_size, out_group_size] = im2col * weight
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(output_size, config.out_group_size);
        result.noalias() = im2col_mat * weight_mat;
        // GEMM 결과를 각 출력 채널의 [out_h, out_w] 텐서로 재구성
        for (int oc = 0; oc < config.out_group_size; ++oc) {
            int global_oc = g * config.out_group_size + oc;
            auto out_tensor = std::make_shared<Tensor<T>>();
            // RowMajor 행렬의 열은 비연속이므로 열을 먼저 복사 후 Map으로 reshape
            Eigen::Matrix<T, Eigen::Dynamic, 1> col = result.col(oc);
            out_tensor->value = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                col.data(), out_h, out_w);
            if (config.bias_bool == 1) {
                out_tensor->value.array() += bias->value(0, global_oc);
            }
            out[global_oc] = std::move(out_tensor);
        }
    }

    return out;
}

template class Convolution2D<float>;
template class Convolution2D<int>;

} // namespace MERCURY
