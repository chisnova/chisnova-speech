#include "layer/rotary_embedding.h"
#include <cmath>

namespace MERCURY {

// rotate_half: 벡터를 절반으로 나누어 교환 + 부호 반전
// [x1, x2] → [-x2, x1] (2D 회전 행렬의 구조)
template<typename T>
std::shared_ptr<Tensor<T>> RotaryEmbedding<T>::rotate_half(std::shared_ptr<Tensor<T>> x) {
    auto result = std::make_shared<Tensor<T>>();
    result->value.resize(x->shape(0), x->shape(1));

    int half_dim = x->shape(1) / 2;

    // 전반부 ← -후반부, 후반부 ← 전반부
    result->value.leftCols(half_dim) = -x->value.rightCols(half_dim);
    result->value.rightCols(half_dim) = x->value.leftCols(half_dim);

    return result;
}

// RoPE 적용: x_rot = x * cos(theta) + rotate_half(x) * sin(theta)
// cos/sin 테이블에서 [start, start+width) 범위의 위치 인코딩 사용
template<typename T>
std::shared_ptr<Tensor<T>> RotaryEmbedding<T>::apply_rotary_pos_emb(std::shared_ptr<Tensor<T>> x,
                                                                    std::shared_ptr<Tensor<T>> cos,
                                                                    std::shared_ptr<Tensor<T>> sin,
                                                                    int start,
                                                                    int width) {
    auto out = std::make_shared<Tensor<T>>();
    int rows = x->shape(0);
    int cols = x->shape(1);
    int half_dim = cols / 2;

    // cos/sin 테이블에서 현재 위치 범위의 블록 추출
    auto cos_block = cos->value.middleRows(start, width).array();
    auto sin_block = sin->value.middleRows(start, width).array();

    // rotate_half를 인라인으로 융합 → 중간 텐서 할당 제거
    // rotated = [-x2, x1] 이므로:
    //   전반부: x1 * cos - x2 * sin (2D 회전의 실수부)
    //   후반부: x2 * cos + x1 * sin (2D 회전의 허수부)
    out->value.resize(rows, cols);
    out->value.leftCols(half_dim) =
        (x->value.leftCols(half_dim).array() * cos_block.leftCols(half_dim)
         - x->value.rightCols(half_dim).array() * sin_block.leftCols(half_dim)).matrix();
    out->value.rightCols(half_dim) =
        (x->value.rightCols(half_dim).array() * cos_block.rightCols(half_dim)
         + x->value.leftCols(half_dim).array() * sin_block.rightCols(half_dim)).matrix();

    return out;
}

// llama3 주파수 스케일링 — Python RotaryEmbedding._compute_llama3_parameters 와 동일
template<typename T>
std::vector<double> RotaryEmbedding<T>::computeLlama3InverseFrequencies() {
    const double base = 10000.0;
    const double factor = 8.0;
    const double low_freq_factor = 1.0;
    const double high_freq_factor = 4.0;
    const double old_context_len = 1500.0;
    const double low_freq_wavelen = old_context_len / low_freq_factor;
    const double high_freq_wavelen = old_context_len / high_freq_factor;
    const double pi = 3.14159265358979323846;

    std::vector<double> inverse_frequencies(head_dim / 2);
    for (int i = 0; i < head_dim / 2; ++i) {
        double inv_freq = 1.0 / std::pow(base, double(2 * i) / double(head_dim));
        double wavelen = 2.0 * pi / inv_freq;
        if (wavelen > low_freq_wavelen) {
            inv_freq = inv_freq / factor;
        }
        else if (wavelen >= high_freq_wavelen) {
            double smooth_factor = (old_context_len / wavelen - low_freq_factor)
                                   / (high_freq_factor - low_freq_factor);
            inv_freq = (1.0 - smooth_factor) * inv_freq / factor
                       + smooth_factor * inv_freq;
        }
        inverse_frequencies[i] = inv_freq;
    }
    return inverse_frequencies;
}

template<typename T>
RotaryEmbedding<T>::RotaryEmbedding(int _head_dim) {
    head_dim = _head_dim;
    cos_table = std::make_shared<Tensor<T>>();
    sin_table = std::make_shared<Tensor<T>>();
}

// cos/sin 테이블 lazy 생성 — 스트리밍 절대 위치가 자라면 2배씩 늘려 다시 만든다
template<typename T>
void RotaryEmbedding<T>::ensureTableLength(int min_length) {
    if (table_length >= min_length) {
        return;
    }
    int new_length = table_length > 0 ? table_length : 1500;
    while (new_length < min_length) {
        new_length *= 2;
    }
    std::vector<double> inverse_frequencies = computeLlama3InverseFrequencies();
    const int half_dim = head_dim / 2;
    cos_table->value.resize(new_length, head_dim);
    sin_table->value.resize(new_length, head_dim);
    for (int position = 0; position < new_length; ++position) {
        for (int i = 0; i < half_dim; ++i) {
            // emb = cat(freqs, freqs): 전반부/후반부가 같은 각도를 쓴다
            double angle = double(position) * inverse_frequencies[i];
            T cos_value = static_cast<T>(std::cos(angle));
            T sin_value = static_cast<T>(std::sin(angle));
            cos_table->value(position, i) = cos_value;
            cos_table->value(position, half_dim + i) = cos_value;
            sin_table->value(position, i) = sin_value;
            sin_table->value(position, half_dim + i) = sin_value;
        }
    }
    table_length = new_length;
}

template class RotaryEmbedding<float>;

} // namespace MERCURY
