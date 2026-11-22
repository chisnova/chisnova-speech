#ifndef MERCURY_CONFORMER_SUBSAMPLE
#define MERCURY_CONFORMER_SUBSAMPLE

#include <string>
#include <vector>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/convolution2d.h"
#include "layer/relu.h"

// ConformerSubSample: Conformer Subsampling Module
// - 2D Convolution x2 + ReLU로 입력 특징을 다운샘플링
// - permute()로 multi-channel 출력을 단일 텐서로 병합
// - 시간 축 해상도를 1/4로 줄이고 채널을 확장
//
// stream(): 청크 스트리밍용 — Python ConformerConvSubSample.stream 과 동일하게
// "아직 소비되지 않은 입력 프레임" 만 캐시에 남겨, 청크 크기와 무관하게
// full-sequence forward 와 같은 프레임을 낸다. 커널을 채울 만큼 안 모였으면
// 0행 텐서를 돌려주고 캐시에만 쌓는다 (호출자는 빈 청크를 건너뛴다).

namespace MERCURY {
    template<typename T>
    class ConformerSubSample {
    public:
        static const int KERNEL_SIZE = 3;
        static const int STRIDE = 2;

        ConformerSubSample(JsonObject subsample_path, const std::string& base_path);
        ~ConformerSubSample() = default;

        std::shared_ptr<Tensor<T>> permute(std::vector<std::shared_ptr<Tensor<T>>> inputs);
        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);
        std::shared_ptr<Tensor<T>> stream(const std::shared_ptr<Tensor<T>>& input);
        void clear();

        // kernel 3 / stride 2 conv 하나를 지난 뒤의 길이 (모자라면 0)
        static int calcConvOutputLength(int input_length);

        std::shared_ptr<Convolution2D<T>> conv1;
        std::shared_ptr<Tensor<T>> input_cache;                     // conv1 이 아직 소비 못한 입력 프레임
        std::vector<std::shared_ptr<Tensor<T>>> conv1_output_cache; // conv2 가 아직 소비 못한 conv1 출력 (채널별)
        std::shared_ptr<Convolution2D<T>> conv2;
    };
} // namespace MERCURY

#endif
