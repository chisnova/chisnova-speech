#include "module/transformer_encoder_layer.h"

namespace MERCURY {
    template<typename T>
    TransformerEncoderLayer<T>::TransformerEncoderLayer(JsonObject encoder_layer_path, const std::string& base_path) {
        multihead1 = std::make_shared<TransformerMultiHead<T>>(encoder_layer_path.parseObject("multihead"), base_path);
        layernorm1 = std::make_shared<LayerNorm<T>>(encoder_layer_path.parseObject("layer_norm1"), base_path);
        ffn = std::make_shared<TransformerFeedForward<T>>(encoder_layer_path.parseObject("ffn") , base_path);
        layernorm2 = std::make_shared<LayerNorm<T>>(encoder_layer_path.parseObject("layer_norm2"), base_path);
    }

    // forward: Post-LN Transformer Encoder
    // 최적화: residual += in-place로 임시 행렬 할당 제거
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerEncoderLayer<T>::forward(const std::shared_ptr<Tensor<T>>& input){
        // Self-Attention + residual → LayerNorm
        std::shared_ptr<Tensor<T>> out1 = multihead1->forward(input, input, input, false);
        out1->value.noalias() += input->value;
        out1 = layernorm1->forward(out1);

        // FFN + residual → LayerNorm
        std::shared_ptr<Tensor<T>> out2 = ffn->forward(out1);
        out2->value.noalias() += out1->value;
        out2 = layernorm2->forward(out2);
        return out2;
    }
    template class TransformerEncoderLayer<float>;
    // template class TransformerEncoderLayer<int>;
};
