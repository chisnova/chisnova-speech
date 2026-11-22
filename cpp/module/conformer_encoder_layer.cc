#include "module/conformer_encoder_layer.h"

namespace MERCURY {
    template<typename T>
    ConformerEncoderLayer<T>::ConformerEncoderLayer(JsonObject conformer_encoder_layer_path, const std::string& base_path) {
        conformer_layernorm1 = std::make_shared<LayerNorm<T>>(conformer_encoder_layer_path.parseObject("layer_norm1"), base_path);
        conformer_feedforward1 = std::make_shared<ConformerFeedForward<T>>(conformer_encoder_layer_path.parseObject("ffn1"), base_path);
        conformer_layernorm2 = std::make_shared<LayerNorm<T>>(conformer_encoder_layer_path.parseObject("layer_norm2"), base_path);
        conformer_attention = std::make_shared<ConformerAttention<T>>(conformer_encoder_layer_path.parseObject("multihead"), base_path);
        conformer_layernorm3 = std::make_shared<LayerNorm<T>>(conformer_encoder_layer_path.parseObject("layer_norm3"), base_path);
        conformer_conv_module = std::make_shared<ConformerConvModule<T>>(conformer_encoder_layer_path.parseObject("conv_module"), base_path);
        conformer_layernorm4 = std::make_shared<LayerNorm<T>>(conformer_encoder_layer_path.parseObject("layer_norm4"), base_path);
        conformer_feedforward2 = std::make_shared<ConformerFeedForward<T>>(conformer_encoder_layer_path.parseObject("ffn2"), base_path);
    }

    // forward: Conformer 블록 Macaron-style 구조
    // 최적화: residual connection을 in-place += 로 수행, *0.5를 *= 로 분리
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerEncoderLayer<T>::forward(const std::shared_ptr<Tensor<T>>& input, const std::shared_ptr<Tensor<T>>& pos_enc) {
        // 1) FFN1 (half-step residual)
        std::shared_ptr<Tensor<float>> output1;
        output1 = conformer_layernorm1->forward(input);
        output1 = conformer_feedforward1->forward(output1);
        output1->value *= static_cast<T>(0.5);
        output1->value.noalias() += input->value;

        // 2) Multi-Head Attention + residual
        std::shared_ptr<Tensor<float>> output2;
        output2 = conformer_layernorm2->forward(output1);
        output2 = conformer_attention->forward(output2, pos_enc);
        output2->value.noalias() += output1->value;

        // 3) Convolution Module + residual
        std::shared_ptr<Tensor<float>> output3;
        output3 = conformer_layernorm3->forward(output2);
        output3 = conformer_conv_module->forward(output3);
        output3->value.noalias() += output2->value;

        // 4) FFN2 (half-step residual)
        std::shared_ptr<Tensor<float>> output4;
        output4 = conformer_layernorm4->forward(output3);
        output4 = conformer_feedforward2->forward(output4);
        output4->value *= static_cast<T>(0.5);
        output4->value.noalias() += output3->value;

        return output4;
    }
    template class ConformerEncoderLayer<float>;
    // template class ConformerEncoderLayer<int>;
};
