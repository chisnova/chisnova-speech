#include "module/transformer_decoder_layer.h"

namespace MERCURY {
    template<typename T>
    TransformerDecoderLayer<T>::TransformerDecoderLayer(JsonObject decoder_layer_path, const std::string& base_path) {
        multihead1 = std::make_shared<TransformerMultiHead<T>>(decoder_layer_path.parseObject("multihead1"), base_path);
        layernorm1 = std::make_shared<LayerNorm<T>>(decoder_layer_path.parseObject("layer_norm1"), base_path);
        multihead2 = std::make_shared<TransformerMultiHead<T>>(decoder_layer_path.parseObject("multihead2"), base_path);
        layernorm2 = std::make_shared<LayerNorm<T>>(decoder_layer_path.parseObject("layer_norm2"), base_path);
        ffn = std::make_shared<TransformerFeedForward<T>>(decoder_layer_path.parseObject("ffn") , base_path);
        layernorm3 = std::make_shared<LayerNorm<T>>(decoder_layer_path.parseObject("layer_norm3"), base_path);
    }

    // forward: Pre-LN Decoder Layer (전체 시퀀스 모드)
    // 최적화: residual += in-place로 임시 행렬 할당 제거
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerDecoderLayer<T>::forward(const std::shared_ptr<Tensor<T>>& input,
                                        const std::shared_ptr<Tensor<T>>& encoding){
        // Masked Self-Attention + residual
        std::shared_ptr<Tensor<T>> out1 = layernorm1->forward(input);
        out1 = multihead1->forward(out1, out1, out1, true);
        out1->value.noalias() += input->value;

        // Cross-Attention + residual
        std::shared_ptr<Tensor<T>> out2 = layernorm2->forward(out1);
        out2 = multihead2->forward(out2, encoding, encoding, false);
        out2->value.noalias() += out1->value;

        // FFN + residual
        std::shared_ptr<Tensor<T>> out3 = layernorm3->forward(out2);
        out3 = ffn->forward(out3);
        out3->value.noalias() += out2->value;
        return out3;
    }

    template<typename T>
    void TransformerDecoderLayer<T>::clear() {
        multihead1->clear();
        multihead2->clear();
    }

    template<typename T>
    void TransformerDecoderLayer<T>::setMultiHeadCache(const MultiHeadCache& multihead_cache){
        multihead1->setMultiheadCache(multihead_cache);
    }

    // forwardContext: autoregressive 디코딩 모드
    // 최적화: residual += in-place
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerDecoderLayer<T>::forwardContext(const std::shared_ptr<Tensor<T>>& input,
                                                const std::shared_ptr<Tensor<T>>& encoding){
        std::shared_ptr<Tensor<T>> out1 = layernorm1->forward(input);
        out1 = multihead1->causalAttention(out1, out1, out1);
        out1->value.noalias() += input->value;

        std::shared_ptr<Tensor<T>> out2 = layernorm2->forward(out1);
        out2 = multihead2->crossAttention(out2, encoding, encoding);
        out2->value.noalias() += out1->value;

        std::shared_ptr<Tensor<T>> out3 = layernorm3->forward(out2);
        out3 = ffn->forward(out3);
        out3->value.noalias() += out2->value;
        return out3;
    }
    template class TransformerDecoderLayer<float>;
    // template class TransformerDecoderLayer<int>;
};
