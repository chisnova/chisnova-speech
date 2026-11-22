#include "module/conformer_stream_head.h"

namespace MERCURY {
    template<typename T>
    ConformerStreamHead<T>::ConformerStreamHead(int _head_dim) {
        head_dim = _head_dim;
        inv_norm = static_cast<T>(1.0) / static_cast<T>(std::sqrt(double(head_dim)));
        KeyCache = std::make_shared<Tensor<T>>();
        ValueCache = std::make_shared<Tensor<T>>();
        KeyCache->value.resize(0, 0);
        ValueCache->value.resize(0, 0);
    }

    template<typename T>
    void ConformerStreamHead<T>::clear() {
        KeyCache = std::make_shared<Tensor<T>>();
        ValueCache = std::make_shared<Tensor<T>>();
        KeyCache->value.resize(0, 0);
        ValueCache->value.resize(0, 0);
    }

    // forwardProjected: RoPE 적용 -> 캐시 append -> left_context 로 자르기 ->
    // 마스킹 없는 attention (청크 내부 양방향 + 좌컨텍스트)
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerStreamHead<T>::forwardProjected(std::shared_ptr<Tensor<T>> query,
                                                                        std::shared_ptr<Tensor<T>> key,
                                                                        std::shared_ptr<Tensor<T>> value,
                                                                        std::shared_ptr<Tensor<T>> cos_chunk,
                                                                        std::shared_ptr<Tensor<T>> sin_chunk,
                                                                        int left_context) {
        const int chunk_length = int(query->value.rows());
        std::shared_ptr<Tensor<T>> rotated_query = query;
        std::shared_ptr<Tensor<T>> rotated_key = key;
        if (cos_chunk != nullptr && sin_chunk != nullptr) {
            rotated_query = RotaryEmbedding<T>::apply_rotary_pos_emb(query, cos_chunk, sin_chunk, 0, chunk_length);
            rotated_key = RotaryEmbedding<T>::apply_rotary_pos_emb(key, cos_chunk, sin_chunk, 0, chunk_length);
        }

        // 캐시 append (캐시된 k/v 는 이미 절대 위치로 회전되어 있다)
        const int cached_rows = int(KeyCache->value.rows());
        std::shared_ptr<Tensor<T>> merged_key = std::make_shared<Tensor<T>>();
        std::shared_ptr<Tensor<T>> merged_value = std::make_shared<Tensor<T>>();
        merged_key->value.resize(cached_rows + chunk_length, head_dim);
        merged_value->value.resize(cached_rows + chunk_length, head_dim);
        if (cached_rows > 0) {
            merged_key->value.topRows(cached_rows) = KeyCache->value;
            merged_value->value.topRows(cached_rows) = ValueCache->value;
        }
        merged_key->value.bottomRows(chunk_length) = rotated_key->value;
        merged_value->value.bottomRows(chunk_length) = value->value;

        // 좌컨텍스트 초과분을 앞에서 잘라낸다 (Python: 마지막 len_q+left_context 유지)
        if (left_context >= 0 && merged_key->value.rows() > chunk_length + left_context) {
            const int keep_rows = chunk_length + left_context;
            std::shared_ptr<Tensor<T>> trimmed_key = std::make_shared<Tensor<T>>();
            std::shared_ptr<Tensor<T>> trimmed_value = std::make_shared<Tensor<T>>();
            trimmed_key->value = merged_key->value.bottomRows(keep_rows);
            trimmed_value->value = merged_value->value.bottomRows(keep_rows);
            merged_key = trimmed_key;
            merged_value = trimmed_value;
        }
        KeyCache = merged_key;
        ValueCache = merged_value;

        // attention: 마스킹 없음 — 청크 내부 양방향 + 캐시(좌컨텍스트) 전체 참조
        auto attention_scores = std::make_shared<Tensor<T>>();
        attention_scores->value.noalias() =
            (rotated_query->value * merged_key->value.transpose()) * inv_norm;
        std::shared_ptr<Tensor<float>> attention_weights = Softmax().forward(attention_scores);

        auto out = std::make_shared<Tensor<T>>();
        out->value.noalias() = attention_weights->value * merged_value->value;
        return out;
    }

    template class ConformerStreamHead<float>;
};
