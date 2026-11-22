// Conformer Relative Positional Encoding
// Conformer 아키텍처에서 사용하는 상대 위치 인코딩 (입력에 더하지 않고 별도 반환)
// Self-attention에서 상대 위치 정보로 사용됨
#ifndef MERCURY_ASR_CONFORMER_POSITIONAL
#define MERCURY_ASR_CONFORMER_POSITIONAL

#include "tensor/tensor.h"
#include "json_loader/json_object.h"

#include <string>

namespace MERCURY {

template<typename T>
class ConformerPositional {
public:
    ConformerPositional(JsonObject pos_encoding_path, std::string base_path);
    ~ConformerPositional();

    // 전체 시퀀스의 위치 인코딩 반환 (입력에 더하지 않음)
    std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input);
    // 특정 위치부터 시작하는 위치 인코딩 반환
    std::shared_ptr<Tensor<T>> forwardContext(std::shared_ptr<Tensor<T>> input, int position);

    std::shared_ptr<Tensor<T>> weight;  // 사전 계산된 위치 인코딩 테이블 [max_len, dim]
};

} // namespace MERCURY

#endif
