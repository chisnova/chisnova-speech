// RoPE (Rotary Position Embedding): 위치 정보를 회전 변환으로 임베딩
// x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)
// LLM(LLaMA, GPT-NeoX 등)에서 상대 위치 인코딩으로 사용
//
// Python model/layers/rotary_embedding.py 와 동일한 llama3 주파수 스케일링
// (factor=8, low_freq_factor=1, high_freq_factor=4, old_context_len=1500)으로
// cos/sin 테이블을 생성한다 — 가중치 파일 없이 수식으로만 만든다.
#ifndef MERCURY_ASR_ROTARY_EMBEDDING_H
#define MERCURY_ASR_ROTARY_EMBEDDING_H

#include "tensor/tensor.h"

#include <memory>
#include <vector>

namespace MERCURY {

template<typename T>
class RotaryEmbedding {
public:
    // head_dim 차원용 llama3 스케일 inv_freq 를 계산해 둔다 (테이블은 lazy 생성)
    RotaryEmbedding(int head_dim);
    RotaryEmbedding() = default;

    // cos/sin 테이블이 min_length 행 이상이 되도록 (재)생성한다
    void ensureTableLength(int min_length);

    // 벡터의 절반을 교환하고 부호 반전: [-x2, x1] (회전 행렬의 일부)
    static std::shared_ptr<Tensor<T>> rotate_half(std::shared_ptr<Tensor<T>> x);
    // RoPE 적용: x * cos + rotate_half(x) * sin
    static std::shared_ptr<Tensor<T>> apply_rotary_pos_emb(
        std::shared_ptr<Tensor<T>> x,
        std::shared_ptr<Tensor<T>> cos,   // 사전 계산된 cos(m*theta) 테이블
        std::shared_ptr<Tensor<T>> sin,   // 사전 계산된 sin(m*theta) 테이블
        int start,                         // 시작 위치 인덱스
        int width);                        // 처리할 시퀀스 길이

    std::shared_ptr<Tensor<T>> cos_table;  // [table_length, head_dim]
    std::shared_ptr<Tensor<T>> sin_table;  // [table_length, head_dim]
    int head_dim = 0;
    int table_length = 0;

private:
    std::vector<double> computeLlama3InverseFrequencies();
};

} // namespace MERCURY

#endif
