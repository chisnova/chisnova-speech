#include "layer/positional_encoding.h"

namespace MERCURY {

// 생성자: 사전 계산된 위치 인코딩 테이블 [max_len, dim] 로드
template<typename T>
PosEncoding<T>::PosEncoding(JsonObject pos_encoding_path, std::string base_path) {
    weight = std::make_shared<Tensor<T>>();
    weight->loadValue(base_path + "/" + pos_encoding_path.parserString("encoding"),
                      pos_encoding_path.parserIntArray("shape"));
}

template<typename T>
PosEncoding<T>::~PosEncoding() = default;

// 순전파: output = input + PE[0:seq_len]
// 입력 길이만큼 위치 인코딩 테이블의 상위 행을 잘라서 더함
template<typename T>
std::shared_ptr<Tensor<T>> PosEncoding<T>::forward(std::shared_ptr<Tensor<T>> input) {
    auto out = std::make_shared<Tensor<T>>();
    // noalias: 출력이 입력과 별도 메모리이므로 aliasing 검사 생략 → 성능 향상
    out->value.noalias() = weight->value.topRows(input->shape(0)) + input->value;
    return out;
}

// 단일 위치 인코딩: output = input + PE[position]
// autoregressive 디코딩에서 한 토큰씩 생성할 때 사용
template<typename T>
std::shared_ptr<Tensor<T>> PosEncoding<T>::forwardContext(std::shared_ptr<Tensor<T>> input, int position) {
    auto out = std::make_shared<Tensor<T>>();
    out->value = weight->value.row(position).array() + input->value.array();
    return out;
}

// 스트리밍 위치 인코딩: output = input + PE[position : position+len]
// 스트리밍 추론에서 현재 청크의 시작 위치부터 인코딩 적용
template<typename T>
std::shared_ptr<Tensor<T>> PosEncoding<T>::forwardStream(std::shared_ptr<Tensor<T>> input, int position) {
    auto out = std::make_shared<Tensor<T>>();
    // noalias: 출력이 입력과 별도 메모리이므로 aliasing 검사 생략
    out->value.noalias() = weight->value.block(position, 0, input->shape(0), input->shape(1)) + input->value;
    return out;
}

template class PosEncoding<float>;

} // namespace MERCURY
