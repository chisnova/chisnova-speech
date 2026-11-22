#include "module/conformer_stream_encoder_layer.h"

namespace MERCURY {
    template<typename T>
    ConformerStreamEncoderLayer<T>::ConformerStreamEncoderLayer(JsonObject conformer_encoder_layer_path, const std::string& base_path) {
        conformer_layernorm1 = std::make_shared<LayerNorm<T>>(conformer_encoder_layer_path.parseObject("layer_norm1"), base_path);
        conformer_feedforward1 = std::make_shared<ConformerFeedForward<T>>(conformer_encoder_layer_path.parseObject("ffn1"), base_path);
        conformer_layernorm2 = std::make_shared<LayerNorm<T>>(conformer_encoder_layer_path.parseObject("layer_norm2"), base_path);
        conformer_stream_conv_module = std::make_shared<ConformerStreamConvModule<T>>(conformer_encoder_layer_path.parseObject("conv_module"), base_path);
        conformer_layernorm3 = std::make_shared<LayerNorm<T>>(conformer_encoder_layer_path.parseObject("layer_norm3"), base_path);
        conformer_stream_attention = std::make_shared<ConformerStreamAttention<T>>(conformer_encoder_layer_path.parseObject("multihead"), base_path);
        conformer_layernorm4 = std::make_shared<LayerNorm<T>>(conformer_encoder_layer_path.parseObject("layer_norm4"), base_path);
        conformer_feedforward2 = std::make_shared<ConformerFeedForward<T>>(conformer_encoder_layer_path.parseObject("ffn2"), base_path);
    }

    template<typename T>
    void ConformerStreamEncoderLayer<T>::clear() {
        conformer_stream_conv_module->clear();
        conformer_stream_attention->clear();
    }

    // forward: Streaming Conformer 블록 (Macaron-style)
    // 최적화: residual *= 0.5 + noalias += 로 임시 행렬 할당 제거
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerStreamEncoderLayer<T>::forward(const std::shared_ptr<Tensor<T>>& input,
                                                                       std::shared_ptr<Tensor<T>> cos_chunk,
                                                                       std::shared_ptr<Tensor<T>> sin_chunk,
                                                                       int left_context) {
        // FFN1 (half-step residual)
        std::shared_ptr<Tensor<float>> output1;
        output1 = conformer_layernorm1->forward(input);
        output1 = conformer_feedforward1->forward(output1);
        output1->value *= static_cast<T>(0.5);
        output1->value.noalias() += input->value;

        // Streaming ConvModule + residual
        std::shared_ptr<Tensor<float>> output2;
        output2 = conformer_layernorm2->forward(output1);
        output2 = conformer_stream_conv_module->forward(output2);
        output2->value.noalias() += output1->value;

        // Streaming Attention + residual
        std::shared_ptr<Tensor<float>> output3;
        output3 = conformer_layernorm3->forward(output2);
        output3 = conformer_stream_attention->forward(output3, cos_chunk, sin_chunk, left_context);
        output3->value.noalias() += output2->value;

        // FFN2 (half-step residual)
        std::shared_ptr<Tensor<float>> output4;
        output4 = conformer_layernorm4->forward(output3);
        output4 = conformer_feedforward2->forward(output4);
        output4->value *= static_cast<T>(0.5);
        output4->value.noalias() += output3->value;

        return output4;
    }

    template class ConformerStreamEncoderLayer<float>;
    // template class ConformerEncoderLayer<int>;
};
