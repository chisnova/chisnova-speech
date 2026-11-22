#include "module/conformer_stream_multihead.h"
#include "utils/thread_pool.h"

#include <future>

namespace MERCURY {
    // 생성자: fused qkv projection / output linear 로드, head 는 캐시만 가진다
    template<typename T>
    ConformerStreamMultiHead<T>::ConformerStreamMultiHead(JsonObject conformer_multihead_path, const std::string& base_path) {
        n_head = int(conformer_multihead_path.parserValue("n_head"));
        d_model = int(conformer_multihead_path.parserValue("d_model"));
        head_dim = d_model / n_head;
        qkv_proj = std::make_shared<Linear<T>>(conformer_multihead_path.parseObject("qkv_proj"), base_path);
        linear = std::make_shared<Linear<T>>(conformer_multihead_path.parseObject("linear"), base_path);
        for (int i = 0; i < n_head; ++i) {
            heads.push_back(std::make_shared<ConformerStreamHead<T>>(head_dim));
        }
    }

    // clear: 모든 head의 key/value 캐시 리셋
    template<typename T>
    void ConformerStreamMultiHead<T>::clear() {
        for (int i = 0; i < n_head; ++i) {
            heads[i]->clear();
        }
    }

    // forward: fused QKV GEMM -> per-head 슬라이스 -> head 병렬 실행 -> concat -> linear
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerStreamMultiHead<T>::forward(const std::shared_ptr<Tensor<T>>& input,
                                                                    std::shared_ptr<Tensor<T>> cos_chunk,
                                                                    std::shared_ptr<Tensor<T>> sin_chunk,
                                                                    int left_context) {
        std::shared_ptr<Tensor<T>> qkv = qkv_proj->forward(input);   // (T, 3*d_model)

        std::vector<std::future<std::shared_ptr<Tensor<float>>>> future_vector;
        future_vector.reserve(n_head);
        std::vector<std::shared_ptr<Tensor<T>>> head_queries(n_head);
        std::vector<std::shared_ptr<Tensor<T>>> head_keys(n_head);
        std::vector<std::shared_ptr<Tensor<T>>> head_values(n_head);
        for (int i = 0; i < n_head; ++i) {
            head_queries[i] = std::make_shared<Tensor<T>>();
            head_keys[i] = std::make_shared<Tensor<T>>();
            head_values[i] = std::make_shared<Tensor<T>>();
            head_queries[i]->value = qkv->value.middleCols(i * head_dim, head_dim);
            head_keys[i]->value = qkv->value.middleCols(d_model + i * head_dim, head_dim);
            head_values[i]->value = qkv->value.middleCols(2 * d_model + i * head_dim, head_dim);
        }

        for (int i = 0; i < n_head; ++i) {
            #ifdef THREAD_POOL_ENABLED
            future_vector.push_back(MERCURY::ThreadPool::submit(
                &ConformerStreamHead<T>::forwardProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i], cos_chunk, sin_chunk, left_context));
            #else
            future_vector.push_back(std::async(std::launch::async,
                &ConformerStreamHead<T>::forwardProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i], cos_chunk, sin_chunk, left_context));
            #endif
        }

        auto out = std::make_shared<Tensor<T>>();
        out->value.resize(input->value.rows(), d_model);
        for (int i = 0; i < n_head; ++i) {
            std::shared_ptr<Tensor<float>> head_output = future_vector[i].get();
            out->value.middleCols(i * head_dim, head_dim).noalias() = head_output->value;
        }

        out = linear->forward(out);
        return out;
    }

    template class ConformerStreamMultiHead<float>;
} // namespace MERCURY
