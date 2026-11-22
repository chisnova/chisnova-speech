#include "layer/conformer_positional.h"

namespace MERCURY {

// 생성자: 사전 계산된 위치 인코딩 테이블 [max_len, dim] 로드
template<typename T>
ConformerPositional<T>::ConformerPositional(JsonObject pos_encoding_path, std::string base_path) {
    weight = std::make_shared<Tensor<T>>();
    std::string weight_path = base_path + "/" + pos_encoding_path.parserString("encoding");
    std::vector<int> shape = pos_encoding_path.parserIntArray("shape");
    weight->loadValue(weight_path, shape);
}

template<typename T>
ConformerPositional<T>::~ConformerPositional() = default;

// 순전파: 위치 인코딩 테이블에서 입력 크기에 맞는 블록 반환 (더하지 않음)
// PosEncoding과 달리 입력에 더하지 않고 별도 텐서로 반환 → attention에서 사용
template<typename T>
std::shared_ptr<Tensor<T>> ConformerPositional<T>::forward(std::shared_ptr<Tensor<T>> input) {
    auto out = std::make_shared<Tensor<T>>();
    // 테이블의 [0:seq_len, 0:dim] 블록 추출
    out->value = weight->value.block(0, 0, input->shape(0), input->shape(1));
    return out;
}

// 컨텍스트 순전파: 특정 위치부터 시작하는 위치 인코딩 반환 (스트리밍용)
template<typename T>
std::shared_ptr<Tensor<T>> ConformerPositional<T>::forwardContext(std::shared_ptr<Tensor<T>> input, int position) {
    auto out = std::make_shared<Tensor<T>>();
    // 테이블의 [position:position+seq_len, 0:dim] 블록 추출
    out->value = weight->value.block(position, 0, input->shape(0), input->shape(1));
    return out;
}

template class ConformerPositional<float>;

} // namespace MERCURY
