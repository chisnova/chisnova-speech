#include "layer/convolution1d.h"
#include "utils/thread_pool.h"

namespace MERCURY {

// 생성자: JSON에서 합성곱 파라미터를 파싱하고 가중치를 채널별 커널로 분리
template<typename T>
Convolution1D<T>::Convolution1D(JsonObject cnn_path, std::string base_path) {
    // JSON에서 합성곱 하이퍼파라미터 파싱
    config.in_channels = static_cast<int>(cnn_path.parserValue("in_channels"));
    config.out_channels = static_cast<int>(cnn_path.parserValue("out_channels"));
    config.kernel_size = static_cast<int>(cnn_path.parserValue("kernel_size"));
    config.stride = static_cast<int>(cnn_path.parserValue("stride"));
    config.padding = static_cast<int>(cnn_path.parserValue("padding"));
    config.dilation = static_cast<int>(cnn_path.parserValue("dilation"));
    config.groups = static_cast<int>(cnn_path.parserValue("groups"));
    config.bias_bool = static_cast<int>(cnn_path.parserValue("bias"));

    // 그룹당 채널 수 계산 (grouped convolution 지원)
    config.in_group_size = config.in_channels / config.groups;
    config.out_group_size = config.out_channels / config.groups;

    auto weight = std::make_shared<Tensor<T>>();
    bias = std::make_shared<Tensor<T>>();

    std::string weight_path = base_path + "/" + cnn_path.parserString("weight_path");
    std::string bias_path = base_path + "/" + cnn_path.parserString("bias_path");
    // shape: [out_channels, in_group_size * kernel_size] (PyTorch flatten 형태)
    std::vector<int> shape = cnn_path.parserIntArray("shape");

    weight->loadValue(weight_path, shape);
    if (config.bias_bool == 1) {
        bias->loadValue(bias_path, {1, config.out_channels});
    }

    // 각 출력 채널의 커널을 열 벡터로 변환 (im2col GEMM에 사용)
    // kernel[oc]: [in_group_size * kernel_size, 1]
    kernels.reserve(config.out_channels);
    for (int i = 0; i < config.out_channels; ++i) {
        auto kernel = std::make_shared<Tensor<T>>();
        kernel->value.resize(config.in_group_size * config.kernel_size, 1);
        kernel->value.col(0) = weight->value.row(i);
        kernels.push_back(std::move(kernel));
    }
}

template<typename T>
Convolution1D<T>::~Convolution1D() = default;

// 입력 양쪽에 제로 패딩 적용: [C, L] → [C, padding + L + padding]
template<typename T>
std::shared_ptr<Tensor<T>> Convolution1D<T>::makePadded(std::shared_ptr<Tensor<T>> input) {
    if (config.padding == 0) {
        return input;
    }
    auto cast_input = std::make_shared<Tensor<T>>();
    int shape0 = input->shape(0);
    int shape1 = input->shape(1);
    // 새 텐서를 0으로 초기화 후 원본 데이터를 중앙에 배치
    cast_input->value.resize(shape0, shape1 + 2 * config.padding);
    cast_input->value.setZero();
    cast_input->value.block(0, config.padding, shape0, shape1) = input->value;
    return cast_input;
}

// im2col 변환: 입력을 그룹별로 [output_size, in_group_size * kernel_size] 행렬로 펼침
// 슬라이딩 윈도우를 행렬의 행으로 배치 → 합성곱을 행렬 곱으로 변환
template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Convolution1D<T>::makeInputMatrix(std::shared_ptr<Tensor<T>> input, conv1d_config cfg) {
    std::vector<std::shared_ptr<Tensor<T>>> input_matrix_vector;
    input_matrix_vector.reserve(cfg.in_channels / cfg.in_group_size);

    for (int i = 0; i < cfg.in_channels; i += cfg.in_group_size) {
        auto input_matrix = std::make_shared<Tensor<T>>();
        input_matrix->value.resize(cfg.output_size, cfg.kernel_size * cfg.in_group_size);

        // 각 출력 위치에 대해 커널 크기만큼의 윈도우를 1행으로 평탄화
        for (int j = 0; j <= input->shape(1) - cfg.kernel_size; j += cfg.stride) {
            auto block_eval = input->value.block(i, j, cfg.in_group_size, cfg.kernel_size).eval();
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> flat_block(
                block_eval.data(), 1, cfg.kernel_size * cfg.in_group_size);
            input_matrix->value.block(j / cfg.stride, 0, 1, cfg.in_group_size * cfg.kernel_size) = flat_block;
        }
        input_matrix_vector.emplace_back(std::move(input_matrix));
    }
    return input_matrix_vector;
}

// 단일 출력 채널 합성곱: im2col 행렬 * 커널 벡터 + bias
// output[out_idx, :] = im2col_matrix * kernel + bias
template<typename T>
int Convolution1D<T>::out_forward_1d(std::shared_ptr<Tensor<T>> input,
                                     std::shared_ptr<Tensor<T>> kernel,
                                     std::shared_ptr<Tensor<T>> bias,
                                     std::shared_ptr<Tensor<T>> output,
                                     int out_idx,
                                     conv1d_config cfg) {
    T bias_value = cfg.bias_bool == 1 ? bias->value(0, out_idx) : static_cast<T>(0);
    // 행렬-벡터 곱으로 해당 채널의 출력 시퀀스 계산
    output->value.row(out_idx) = (input->value * kernel->value).col(0);
    output->value.row(out_idx).array() += bias_value;
    return 0;
}

// 순전파: im2col + GEMM 최적화
// output_size = (L + 2*padding - kernel_size) / stride + 1
template<typename T>
std::shared_ptr<Tensor<T>> Convolution1D<T>::forward(std::shared_ptr<Tensor<T>> input) {
    auto out = std::make_shared<Tensor<T>>();
    // 출력 시퀀스 길이 계산
    config.output_size = ((input->shape(1) - config.kernel_size + 2 * config.padding) / config.stride) + 1;
    config.input_size = input->shape(1);
    // 출력 텐서: [out_channels, output_size]
    out->value.resize(config.out_channels, config.output_size);

    // 입력에 제로 패딩 적용
    auto padded_input = makePadded(input);

    // 그룹별 im2col + 배치 GEMM으로 합성곱 수행
    // 채널별 비동기 디스패치 대신 단일 GEMM으로 오버헤드 제거
    const int num_groups = config.groups;
    const int igs = config.in_group_size;
    const int ks = config.kernel_size;
    const int igs_ks = igs * ks;
    const int os = config.output_size;
    const int ogs = config.out_group_size;

    for (int g = 0; g < num_groups; ++g) {
        const int in_start = g * igs;
        // im2col 행렬 구축: [output_size, in_group_size * kernel_size]
        // 각 행 = 하나의 출력 위치에 해당하는 슬라이딩 윈도우 평탄화
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> im2col_mat(os, igs_ks);
        // 채널 우선 루프: padded_input 행에 대한 캐시 지역성 향상
        for (int ic = 0; ic < igs; ++ic) {
            const int col_offset = ic * ks;
            const auto& row_data = padded_input->value.row(in_start + ic);
            for (int j = 0; j < os; ++j) {
                im2col_mat.block(j, col_offset, 1, ks) =
                    row_data.segment(j * config.stride, ks);
            }
        }
        // 가중치 행렬 구축: [in_group_size * kernel_size, out_group_size]
        // 각 열 = 하나의 출력 채널 커널
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weight_mat(igs_ks, ogs);
        for (int oc = 0; oc < ogs; ++oc) {
            const int global_oc = g * ogs + oc;
            weight_mat.col(oc) = kernels[global_oc]->value.col(0);
        }
        // 단일 GEMM: result = im2col * W → [output_size, out_group_size]
        // noalias()로 Eigen 임시 할당 방지
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(os, ogs);
        result.noalias() = im2col_mat * weight_mat;
        // 결과를 전치하여 출력 텐서에 배치 ([out_group_size, output_size])
        const int out_start = g * ogs;
        out->value.block(out_start, 0, ogs, os) = result.transpose();
    }
    // 열별로 bias 더하기 (broadcasting)
    if (config.bias_bool == 1) {
        out->value.colwise() += bias->value.row(0).transpose();
    }

    return out;
}

template class Convolution1D<float>;
template class Convolution1D<int>;

} // namespace MERCURY
