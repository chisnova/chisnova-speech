#include "module/transformer_head.h"

namespace MERCURY {
    template<typename T>
    TransformerHead<T>::TransformerHead(int _head_dim) {
        head_dim = _head_dim;
        inv_norm = static_cast<T>(1.0) / static_cast<T>(std::sqrt(double(head_dim)));
        clear();
    }

    template<typename T>
    void TransformerHead<T>::clear() {
        maskedKeys = nullptr;
        maskedVals = nullptr;
        crossKeys = nullptr;
        crossVals = nullptr;
    }

    // setHeadCache: 외부에서 미리 계산된 key/value 캐시 설정 (beam search 등)
    template<typename T>
    void TransformerHead<T>::setHeadCache(const HeadCache& head_cache) {
        maskedKeys = head_cache.maskedKeys;
        maskedVals = head_cache.maskedVals;
    }

    // upperTriangleMasking: 상삼각 영역을 -inf로 설정 (미래 토큰 마스킹)
    template<typename T>
    void TransformerHead<T>::upperTriangleMasking(const std::shared_ptr<Tensor<T>>& norm_qk) {
        const int rows = int(norm_qk->value.rows());
        const int cols = int(norm_qk->value.cols());
        for (int i = 0; i < rows; ++i) {
            if (i + 1 < cols) {
                norm_qk->value.row(i).segment(i + 1, cols - i - 1).setConstant(static_cast<T>(-99999999999.0));
            }
        }
    }

    // upperTriangleStreamMasking: streaming causal mask (캐시 offset 고려)
    template<typename T>
    void TransformerHead<T>::upperTriangleStreamMasking(const std::shared_ptr<Tensor<T>>& norm_qk) {
        const int c = int(norm_qk->value.rows());
        const int l = int(norm_qk->value.cols());
        for (int i = 0; i < c; ++i) {
            const int start = l - c + i + 1;
            if (start < l) {
                norm_qk->value.row(i).segment(start, l - start).setConstant(static_cast<T>(-99999999999.0));
            }
        }
    }

    // forwardProjected: 일반 scaled dot-product attention (optional causal mask)
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerHead<T>::forwardProjected(std::shared_ptr<Tensor<T>> query,
                                                                    std::shared_ptr<Tensor<T>> key,
                                                                    std::shared_ptr<Tensor<T>> value,
                                                                    bool isMask) {
        auto qk = std::make_shared<Tensor<T>>();
        qk->value.noalias() = (query->value * key->value.transpose()) * inv_norm;

        if (isMask) upperTriangleMasking(qk);

        std::shared_ptr<Tensor<float>> weighted = Softmax().forward(qk);
        auto out = std::make_shared<Tensor<T>>();
        out->value.noalias() = weighted->value * value->value;
        return out;
    }

    // causalAttentionProjected: autoregressive 디코딩용 attention (key/value 누적 캐시)
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerHead<T>::causalAttentionProjected(std::shared_ptr<Tensor<T>> query,
                                                                            std::shared_ptr<Tensor<T>> key,
                                                                            std::shared_ptr<Tensor<T>> value) {
        if (maskedKeys == nullptr || maskedVals == nullptr) {
            maskedKeys = std::make_shared<Tensor<float>>();
            maskedVals = std::make_shared<Tensor<float>>();
            maskedKeys->value.resize(0, 0);
            maskedVals->value.resize(0, 0);
        }

        maskedKeys->appendChunkRow(key);

        auto qk = std::make_shared<Tensor<T>>();
        qk->value.noalias() = (query->value * maskedKeys->value.transpose()) * inv_norm;

        upperTriangleStreamMasking(qk);
        std::shared_ptr<Tensor<float>> weighted = Softmax().forward(qk);

        maskedVals->appendChunkRow(value);

        auto out = std::make_shared<Tensor<T>>();
        out->value.noalias() = weighted->value * maskedVals->value;
        return out;
    }

    // crossAttentionProjected: encoder-decoder cross attention
    // key/value는 최초 1회만 캐시 (인코더 출력은 불변)
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerHead<T>::crossAttentionProjected(std::shared_ptr<Tensor<T>> query,
                                                                           std::shared_ptr<Tensor<T>> key,
                                                                           std::shared_ptr<Tensor<T>> value) {
        if (crossKeys == nullptr || crossVals == nullptr) {
            crossKeys = std::make_shared<Tensor<float>>();
            crossVals = std::make_shared<Tensor<float>>();
            // key^T / sqrt(d_k)를 미리 캐시하여 이후 호출에서 연산 절약
            crossKeys->value = key->value.transpose() * inv_norm;
            crossVals->value = value->value;
        }

        auto qk = std::make_shared<Tensor<T>>();
        qk->value.noalias() = query->value * crossKeys->value;

        std::shared_ptr<Tensor<float>> weighted = Softmax().forward(qk);

        auto out = std::make_shared<Tensor<T>>();
        out->value.noalias() = weighted->value * crossVals->value;
        return out;
    }

    template class TransformerHead<float>;
} // namespace MERCURY
