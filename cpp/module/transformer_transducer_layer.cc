#include "module/transformer_transducer_layer.h"

namespace MERCURY {
    template<typename T>
    TransformerTransducerLayer<T>::TransformerTransducerLayer(JsonObject trf_transducer_layer_path, const std::string& base_path) {
        multihead1 = std::make_shared<TransformerMultiHead<T>>(trf_transducer_layer_path.parseObject("multihead1"), base_path);
        layernorm1 = std::make_shared<LayerNorm<T>>(trf_transducer_layer_path.parseObject("layer_norm1"), base_path);
        ffn = std::make_shared<TransformerFeedForward<T>>(trf_transducer_layer_path.parseObject("ffn") , base_path);
        layernorm2 = std::make_shared<LayerNorm<T>>(trf_transducer_layer_path.parseObject("layer_norm2"), base_path);
    }

    // forward: Causal Self-Attention + FFN
    // 최적화: residual += in-place
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerTransducerLayer<T>::forward(const std::shared_ptr<Tensor<T>>& input){
        std::shared_ptr<Tensor<T>> out1 = layernorm1->forward(input);
        out1 = multihead1->causalAttention(out1, out1, out1);
        out1->value.noalias() += input->value;

        std::shared_ptr<Tensor<T>> out2 = layernorm2->forward(out1);
        out2 = ffn->forward(out2);
        out2->value.noalias() += out1->value;

        return out2;
    }

    template<typename T>
    void TransformerTransducerLayer<T>::clear() {
        multihead1->clear();
    }

    template<typename T>
    void TransformerTransducerLayer<T>::setMultiHeadCache(const MultiHeadCache& multihead_cache){
        multihead1->setMultiheadCache(multihead_cache);
    }

    // forwardContext: autoregressive 디코딩 모드
    // 최적화: residual += in-place
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerTransducerLayer<T>::forwardContext(const std::shared_ptr<Tensor<T>>& input){
        std::shared_ptr<Tensor<T>> out1 = layernorm1->forward(input);
        out1 = multihead1->causalAttention(out1, out1, out1);
        out1->value.noalias() += input->value;

        std::shared_ptr<Tensor<T>> out2 = layernorm2->forward(out1);
        out2 = ffn->forward(out2);
        out2->value.noalias() += out1->value;

        return out2;
    }
    template class TransformerTransducerLayer<float>;
    // template class TransformerDecoderLayer<int>;
};
