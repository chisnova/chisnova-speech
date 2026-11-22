// BatchNorm1D: y = gamma * (x - mean) / std + beta
// 추론 시 running mean/std 사용, 생성자에서 가중치를 사전 융합하여 단일 affine 변환으로 최적화
// 사전 융합: y = (gamma/std) * x + (beta - gamma*mean/std)
#ifndef MERCURY_ASR_BATCHNORM1D
#define MERCURY_ASR_BATCHNORM1D

#include "tensor/tensor.h"
#include "json_loader/json_object.h"

#include <string>

namespace MERCURY {

template<typename T>
class BatchNorm1D {
public:
    BatchNorm1D(JsonObject ln_path, std::string base_path);
    ~BatchNorm1D();

    std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input);

    std::shared_ptr<Tensor<T>> mean;    // running mean (학습 시 누적)
    std::shared_ptr<Tensor<T>> std;     // running std (학습 시 누적)
    std::shared_ptr<Tensor<T>> weight;  // gamma (생성 후 gamma/std로 융합됨)
    std::shared_ptr<Tensor<T>> bias;    // beta (생성 후 beta - gamma*mean/std로 융합됨)
};

} // namespace MERCURY

#endif
