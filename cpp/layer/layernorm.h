// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// 각 행(토큰/프레임)에 대해 독립적으로 정규화 수행
#ifndef MERCURY_ASR_LAYERNORM
#define MERCURY_ASR_LAYERNORM

#include "tensor/tensor.h"
#include "json_loader/json_object.h"

#include <string>

namespace MERCURY {

template<typename T>
class LayerNorm {
public:
    LayerNorm(JsonObject ln_path, std::string base_path);
    ~LayerNorm();

    std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input);

    std::shared_ptr<Tensor<T>> weight;  // gamma (스케일 파라미터) [1, dim]
    std::shared_ptr<Tensor<T>> bias;    // beta (시프트 파라미터) [1, dim]
};

} // namespace MERCURY

#endif
