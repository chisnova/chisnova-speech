#include "module/conformer_multihead.h"
#include "utils/thread_pool.h"

#include <future>

namespace MERCURY {
    // 생성자: fused qkv/pos projection 과 output linear 로드
    template<typename T>
    ConformerMultiHead<T>::ConformerMultiHead(JsonObject conformer_multihead_path, const std::string& base_path) {
        n_head = int(conformer_multihead_path.parserValue("n_head"));
        d_model = int(conformer_multihead_path.parserValue("d_model"));
        head_dim = d_model / n_head;
        qkv_proj = std::make_shared<Linear<T>>(conformer_multihead_path.parseObject("qkv_proj"), base_path);
        pos_proj = std::make_shared<Linear<T>>(conformer_multihead_path.parseObject("pos_proj"), base_path);
        linear = std::make_shared<Linear<T>>(conformer_multihead_path.parseObject("linear"), base_path);
        for (int i = 0; i < n_head; ++i) {
            heads.push_back(std::make_shared<ConformerHead<T>>(head_dim));
        }
    }

    // forward: fused QKV/Pos GEMM -> per-head 슬라이스 -> head 병렬 실행 -> concat -> linear
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerMultiHead<T>::forward(const std::shared_ptr<Tensor<T>>& input,
                                                              const std::shared_ptr<Tensor<T>>& pos_enc) {
        std::shared_ptr<Tensor<T>> qkv = qkv_proj->forward(input);      // (L, 3*d_model)
        std::shared_ptr<Tensor<T>> pos_all = pos_proj->forward(pos_enc); // (L, d_model)

        std::vector<std::future<std::shared_ptr<Tensor<float>>>> future_vector;
        future_vector.reserve(n_head);
        std::vector<std::shared_ptr<Tensor<T>>> head_queries(n_head);
        std::vector<std::shared_ptr<Tensor<T>>> head_keys(n_head);
        std::vector<std::shared_ptr<Tensor<T>>> head_values(n_head);
        std::vector<std::shared_ptr<Tensor<T>>> head_positions(n_head);
        for (int i = 0; i < n_head; ++i) {
            head_queries[i] = std::make_shared<Tensor<T>>();
            head_keys[i] = std::make_shared<Tensor<T>>();
            head_values[i] = std::make_shared<Tensor<T>>();
            head_positions[i] = std::make_shared<Tensor<T>>();
            head_queries[i]->value = qkv->value.middleCols(i * head_dim, head_dim);
            head_keys[i]->value = qkv->value.middleCols(d_model + i * head_dim, head_dim);
            head_values[i]->value = qkv->value.middleCols(2 * d_model + i * head_dim, head_dim);
            head_positions[i]->value = pos_all->value.middleCols(i * head_dim, head_dim);
        }

        for (int i = 0; i < n_head; ++i) {
            #ifdef THREAD_POOL_ENABLED
            future_vector.push_back(MERCURY::ThreadPool::submit(
                &ConformerHead<T>::forwardProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i], head_positions[i]));
            #else
            future_vector.push_back(std::async(std::launch::async,
                &ConformerHead<T>::forwardProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i], head_positions[i]));
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

    template class ConformerMultiHead<float>;
} // namespace MERCURY
