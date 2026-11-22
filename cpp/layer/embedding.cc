#include "layer/embedding.h"
#include <cstring>

namespace MERCURY {

// 생성자: 임베딩 테이블 [vocab_size, embed_dim] 로드
Embedding::Embedding(JsonObject embedding_path, std::string base_path) {
    weight = std::make_shared<Tensor<float>>();
    std::string weight_path = base_path + "/" + embedding_path.parserString("embedding");
    weight->loadValue(weight_path, embedding_path.parserIntArray("shape"));
}

Embedding::~Embedding() = default;

// 순전파: 각 토큰 인덱스에 해당하는 임베딩 벡터를 테이블에서 조회
// input [1, seq_len] (정수) → output [seq_len, embed_dim] (실수)
std::shared_ptr<Tensor<float>> Embedding::forward(std::shared_ptr<Tensor<int>> input) {
    auto out = std::make_shared<Tensor<float>>();
    const int seq_len = input->shape(1);
    const int embed_dim = weight->shape(1);
    out->value.resize(seq_len, embed_dim);

    // RowMajor 레이아웃의 연속 메모리를 직접 memcpy → Eigen 표현식 오버헤드 회피
    const float* w_ptr = weight->value.data();
    float* o_ptr = out->value.data();
    for (int i = 0; i < seq_len; ++i) {
        const int idx = input->value(0, i);
        if (idx < 0 || idx >= weight->shape(0))
            throw std::invalid_argument("embedding token index out of range");
        // weight[idx] 행을 output[i] 행에 직접 복사
        std::memcpy(o_ptr + i * embed_dim, w_ptr + idx * embed_dim, embed_dim * sizeof(float));
    }
    return out;
}

} // namespace MERCURY
