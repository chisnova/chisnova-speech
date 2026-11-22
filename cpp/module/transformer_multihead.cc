#include "module/transformer_multihead.h"
#include "utils/thread_pool.h"

#include <future>

namespace MERCURY {
    // 생성자: fused qkv projection 과 output linear 로드
    template<typename T>
    TransformerMultiHead<T>::TransformerMultiHead(JsonObject transformer_multihead_path, const std::string& base_path) {
        n_head = int(transformer_multihead_path.parserValue("n_head"));
        d_model = int(transformer_multihead_path.parserValue("d_model"));
        head_dim = d_model / n_head;
        qkv_proj = std::make_shared<Linear<T>>(transformer_multihead_path.parseObject("qkv_proj"), base_path);
        linear = std::make_shared<Linear<T>>(transformer_multihead_path.parseObject("linear"), base_path);
        for (int i = 0; i < n_head; ++i) {
            heads.push_back(std::make_shared<TransformerHead<T>>(head_dim));
        }
    }

    template<typename T>
    void TransformerMultiHead<T>::clear() {
        for (int i = 0; i < n_head; ++i) {
            heads[i]->clear();
        }
    }

    template<typename T>
    void TransformerMultiHead<T>::setMultiheadCache(const MultiHeadCache& multihead_cache) {
        assert(int(multihead_cache.heads.size()) == n_head);
        for (int i = 0; i < n_head; ++i) {
            heads[i]->setHeadCache(multihead_cache.heads[i]);
        }
    }

    // sliceHeads: qkv_q 의 앞 d_model 열에서 query, qkv_kv 의 뒤 2*d_model 열에서 key/value
    template<typename T>
    void TransformerMultiHead<T>::sliceHeads(const std::shared_ptr<Tensor<T>>& qkv_q,
                                             const std::shared_ptr<Tensor<T>>& qkv_kv,
                                             std::vector<std::shared_ptr<Tensor<T>>>& head_queries,
                                             std::vector<std::shared_ptr<Tensor<T>>>& head_keys,
                                             std::vector<std::shared_ptr<Tensor<T>>>& head_values) {
        head_queries.resize(n_head);
        head_keys.resize(n_head);
        head_values.resize(n_head);
        for (int i = 0; i < n_head; ++i) {
            head_queries[i] = std::make_shared<Tensor<T>>();
            head_keys[i] = std::make_shared<Tensor<T>>();
            head_values[i] = std::make_shared<Tensor<T>>();
            head_queries[i]->value = qkv_q->value.middleCols(i * head_dim, head_dim);
            head_keys[i]->value = qkv_kv->value.middleCols(d_model + i * head_dim, head_dim);
            head_values[i]->value = qkv_kv->value.middleCols(2 * d_model + i * head_dim, head_dim);
        }
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerMultiHead<T>::concatHeadOutputs(
            std::vector<std::future<std::shared_ptr<Tensor<float>>>>& future_vector, int rows) {
        auto out = std::make_shared<Tensor<T>>();
        out->value.resize(rows, d_model);
        for (int i = 0; i < n_head; ++i) {
            std::shared_ptr<Tensor<float>> head_output = future_vector[i].get();
            out->value.middleCols(i * head_dim, head_dim).noalias() = head_output->value;
        }
        return linear->forward(out);
    }

    // forward: 일반 multi-head attention (self 또는 cross, optional causal mask)
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerMultiHead<T>::forward(const std::shared_ptr<Tensor<T>>& q,
                                                                 const std::shared_ptr<Tensor<T>>& k,
                                                                 const std::shared_ptr<Tensor<T>>& v,
                                                                 bool isMask) {
        std::shared_ptr<Tensor<T>> qkv_q = qkv_proj->forward(q);
        std::shared_ptr<Tensor<T>> qkv_kv = (q.get() == k.get()) ? qkv_q : qkv_proj->forward(k);

        std::vector<std::shared_ptr<Tensor<T>>> head_queries, head_keys, head_values;
        sliceHeads(qkv_q, qkv_kv, head_queries, head_keys, head_values);

        std::vector<std::future<std::shared_ptr<Tensor<float>>>> future_vector;
        future_vector.reserve(n_head);
        for (int i = 0; i < n_head; ++i) {
            #ifdef THREAD_POOL_ENABLED
            future_vector.push_back(MERCURY::ThreadPool::submit(
                &TransformerHead<T>::forwardProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i], isMask));
            #else
            future_vector.push_back(std::async(std::launch::async,
                &TransformerHead<T>::forwardProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i], isMask));
            #endif
        }
        return concatHeadOutputs(future_vector, int(q->value.rows()));
    }

    // causalAttention: autoregressive 디코딩용 (head 별 key/value 누적 캐시)
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerMultiHead<T>::causalAttention(const std::shared_ptr<Tensor<T>>& q,
                                                                        const std::shared_ptr<Tensor<T>>& k,
                                                                        const std::shared_ptr<Tensor<T>>& v) {
        std::shared_ptr<Tensor<T>> qkv = qkv_proj->forward(q);   // self-attention: q == k == v

        std::vector<std::shared_ptr<Tensor<T>>> head_queries, head_keys, head_values;
        sliceHeads(qkv, qkv, head_queries, head_keys, head_values);

        std::vector<std::future<std::shared_ptr<Tensor<float>>>> future_vector;
        future_vector.reserve(n_head);
        for (int i = 0; i < n_head; ++i) {
            #ifdef THREAD_POOL_ENABLED
            future_vector.push_back(MERCURY::ThreadPool::submit(
                &TransformerHead<T>::causalAttentionProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i]));
            #else
            future_vector.push_back(std::async(std::launch::async,
                &TransformerHead<T>::causalAttentionProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i]));
            #endif
        }
        return concatHeadOutputs(future_vector, int(q->value.rows()));
    }

    // crossAttention: encoder-decoder cross attention
    // head 가 crossKeys 캐시를 이미 가지고 있으면 K/V projection 을 건너뛴다
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerMultiHead<T>::crossAttention(const std::shared_ptr<Tensor<T>>& q,
                                                                       const std::shared_ptr<Tensor<T>>& k,
                                                                       const std::shared_ptr<Tensor<T>>& v) {
        std::shared_ptr<Tensor<T>> qkv_q = qkv_proj->forward(q);

        bool needs_kv_projection = (heads[0]->crossKeys == nullptr);
        std::shared_ptr<Tensor<T>> qkv_kv = needs_kv_projection ? qkv_proj->forward(k) : qkv_q;

        std::vector<std::shared_ptr<Tensor<T>>> head_queries, head_keys, head_values;
        sliceHeads(qkv_q, qkv_kv, head_queries, head_keys, head_values);

        std::vector<std::future<std::shared_ptr<Tensor<float>>>> future_vector;
        future_vector.reserve(n_head);
        for (int i = 0; i < n_head; ++i) {
            #ifdef THREAD_POOL_ENABLED
            future_vector.push_back(MERCURY::ThreadPool::submit(
                &TransformerHead<T>::crossAttentionProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i]));
            #else
            future_vector.push_back(std::async(std::launch::async,
                &TransformerHead<T>::crossAttentionProjected, heads[i].get(),
                head_queries[i], head_keys[i], head_values[i]));
            #endif
        }
        return concatHeadOutputs(future_vector, int(q->value.rows()));
    }

    template class TransformerMultiHead<float>;
} // namespace MERCURY
