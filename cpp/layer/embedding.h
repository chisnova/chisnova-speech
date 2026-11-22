// Embedding: 정수 토큰 인덱스 → 밀집 벡터 변환 (lookup table)
// weight[vocab_size, embed_dim] 테이블에서 행을 선택하여 출력 구성
#ifndef MERCURY_ASR_EMBEDDING
#define MERCURY_ASR_EMBEDDING

#include "tensor/tensor.h"
#include "json_loader/json_object.h"

#include <string>

namespace MERCURY {

class Embedding {
public:
    Embedding(JsonObject embedding_path, std::string base_path);
    ~Embedding();

    // 입력: 정수 토큰 [1, seq_len], 출력: 임베딩 벡터 [seq_len, embed_dim]
    std::shared_ptr<Tensor<float>> forward(std::shared_ptr<Tensor<int>> input);

    std::shared_ptr<Tensor<float>> weight;  // 임베딩 테이블 [vocab_size, embed_dim]
};

} // namespace MERCURY

#endif
