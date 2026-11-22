#include "module/conformer_subsample.h"

namespace MERCURY {
    template<typename T>
    ConformerSubSample<T>::ConformerSubSample(JsonObject subsample_path, const std::string& base_path) {
        conv1 = std::make_shared<Convolution2D<T>>(subsample_path.parseObject("conv1"), base_path);
        conv2 = std::make_shared<Convolution2D<T>>(subsample_path.parseObject("conv2"), base_path);
    }

    // permute: multi-channel 2D conv 출력을 (T, C*D)로 재배열
    // 최적화: 채널 우선 루프로 연속 메모리 블록 복사 → 행별 루프 내 채널 루프 제거
    // 각 채널의 전체 행 블록을 한번에 복사하여 캐시 효율 향상
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerSubSample<T>::permute(std::vector<std::shared_ptr<Tensor<T>>> inputs) {
        const int num_channel = inputs.size();
        const int length = inputs[0]->shape(0);
        const int dim = inputs[0]->shape(1);

        auto output = std::make_shared<Tensor<T>>();
        output->value.resize(length, num_channel * dim);

        // 채널 우선: 각 채널의 모든 행을 한번에 middleCols로 복사
        // 내부 루프가 연속 열 블록이므로 RowMajor 레이아웃에서 캐시 친화적
        for(int j = 0; j < num_channel; ++j){
            output->value.middleCols(j * dim, dim).noalias() = inputs[j]->value;
        }
        return output;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerSubSample<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        std::shared_ptr<Tensor<float>> output;
        std::vector<std::shared_ptr<Tensor<float>>> output_vec;
        std::vector<std::shared_ptr<Tensor<float>>> inputs;
        inputs.push_back(std::move(input));
        output_vec = conv1->forward(std::move(inputs));
        output_vec = Relu<float>().forward(std::move(output_vec));
        output_vec = conv2->forward(std::move(output_vec));
        output_vec = Relu<float>().forward(std::move(output_vec));
        output = permute(output_vec);
        return output;
    }
    template<typename T>
    int ConformerSubSample<T>::calcConvOutputLength(int input_length) {
        if (input_length < KERNEL_SIZE) {
            return 0;
        }
        return (input_length - KERNEL_SIZE) / STRIDE + 1;
    }

    template<typename T>
    void ConformerSubSample<T>::clear() {
        input_cache = nullptr;
        conv1_output_cache.clear();
    }

    // stream: Python ConformerConvSubSample.stream 과 동일한 캐시 정책.
    // 출력 n 은 입력 [n*stride, n*stride+kernel-1] 을 보므로, out 프레임을 냈다면
    // 다음 출력이 볼 시작 위치는 out*stride — 그 앞은 소비된 것이라 버린다.
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerSubSample<T>::stream(const std::shared_ptr<Tensor<T>>& input) {
        // 1) 미소비 입력 프레임과 이어 붙인다
        std::shared_ptr<Tensor<T>> merged_input = std::make_shared<Tensor<T>>();
        if (input_cache != nullptr && input_cache->value.rows() > 0) {
            merged_input->value.resize(input_cache->value.rows() + input->value.rows(),
                                       input->value.cols());
            merged_input->value.topRows(input_cache->value.rows()) = input_cache->value;
            merged_input->value.bottomRows(input->value.rows()) = input->value;
        }
        else {
            merged_input->value = input->value;
        }

        std::shared_ptr<Tensor<T>> empty_output = std::make_shared<Tensor<T>>();
        empty_output->value.resize(0, 0);

        int conv1_output_length = calcConvOutputLength(int(merged_input->value.rows()));
        if (conv1_output_length == 0) {
            input_cache = merged_input;
            return empty_output;
        }
        input_cache = std::make_shared<Tensor<T>>();
        int consumed_rows = conv1_output_length * STRIDE;
        input_cache->value = merged_input->value.bottomRows(
            merged_input->value.rows() - consumed_rows);

        // 2) conv1 + relu
        std::vector<std::shared_ptr<Tensor<T>>> conv1_inputs;
        conv1_inputs.push_back(merged_input);
        std::vector<std::shared_ptr<Tensor<T>>> conv1_outputs = conv1->forward(std::move(conv1_inputs));
        conv1_outputs = Relu<float>().forward(std::move(conv1_outputs));

        // 3) conv2 가 아직 소비 못한 conv1 출력과 이어 붙인다 (채널별)
        if (!conv1_output_cache.empty() && conv1_output_cache[0]->value.rows() > 0) {
            for (size_t channel = 0; channel < conv1_outputs.size(); ++channel) {
                std::shared_ptr<Tensor<T>> merged_channel = std::make_shared<Tensor<T>>();
                merged_channel->value.resize(
                    conv1_output_cache[channel]->value.rows() + conv1_outputs[channel]->value.rows(),
                    conv1_outputs[channel]->value.cols());
                merged_channel->value.topRows(conv1_output_cache[channel]->value.rows())
                    = conv1_output_cache[channel]->value;
                merged_channel->value.bottomRows(conv1_outputs[channel]->value.rows())
                    = conv1_outputs[channel]->value;
                conv1_outputs[channel] = merged_channel;
            }
        }

        int conv2_output_length = calcConvOutputLength(int(conv1_outputs[0]->value.rows()));
        if (conv2_output_length == 0) {
            conv1_output_cache = conv1_outputs;
            return empty_output;
        }
        conv1_output_cache.clear();
        int conv2_consumed_rows = conv2_output_length * STRIDE;
        for (size_t channel = 0; channel < conv1_outputs.size(); ++channel) {
            std::shared_ptr<Tensor<T>> remaining_rows = std::make_shared<Tensor<T>>();
            remaining_rows->value = conv1_outputs[channel]->value.bottomRows(
                conv1_outputs[channel]->value.rows() - conv2_consumed_rows);
            conv1_output_cache.push_back(remaining_rows);
        }

        // 4) conv2 + relu -> permute
        std::vector<std::shared_ptr<Tensor<T>>> conv2_outputs = conv2->forward(std::move(conv1_outputs));
        conv2_outputs = Relu<float>().forward(std::move(conv2_outputs));
        return permute(conv2_outputs);
    }

    template class ConformerSubSample<float>;
    // template class ConformerSubSample<int>;
};
