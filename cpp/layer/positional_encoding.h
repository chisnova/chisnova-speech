// Positional Encoding: 입력에 위치 정보를 더하여 순서 인식 가능하게 함
// output = input + PE[position], PE는 사전 계산된 sin/cos 테이블
#ifndef MERCURY_ASR_POSITIONAL_ENCODING
#define MERCURY_ASR_POSITIONAL_ENCODING

#include "tensor/tensor.h"
#include "json_loader/json_object.h"

#include <string>

namespace MERCURY {

template<typename T>
class PosEncoding {
public:
    PosEncoding(JsonObject pos_encoding_path, std::string base_path);
    ~PosEncoding();

    // 전체 시퀀스에 위치 인코딩 더하기 (처음부터)
    std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input);
    // 특정 위치의 단일 행 위치 인코딩 더하기 (autoregressive 디코딩용)
    std::shared_ptr<Tensor<T>> forwardContext(std::shared_ptr<Tensor<T>> input, int position);
    // 특정 위치부터 시작하는 연속 위치 인코딩 (스트리밍 추론용)
    std::shared_ptr<Tensor<T>> forwardStream(std::shared_ptr<Tensor<T>> input, int position);

    std::shared_ptr<Tensor<T>> weight;  // 사전 계산된 위치 인코딩 테이블 [max_len, dim]
};

} // namespace MERCURY

#endif
