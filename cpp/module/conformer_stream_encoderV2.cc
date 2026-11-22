#include "module/conformer_stream_encoderV2.h"

namespace MERCURY {
    // 생성자: feat_norm, SubSample, Linear, LayerNorm1, PosEnc, N개의 StreamEncoderLayer 가중치 로드
    template<typename T>
    ConformerStreamEncoderV2<T>::ConformerStreamEncoderV2(JsonObject conformer_encoder_path, const std::string& base_path) {
        num_head = int(conformer_encoder_path.parserValue("num_head"));
        num_layer = int(conformer_encoder_path.parserValue("num_layer"));
        dim_model = int(conformer_encoder_path.parserValue("dim_model"));
        pos_enc = std::make_shared<PosEncoding<T>>(conformer_encoder_path.parseObject("pos_enc"), base_path);
        convsubsample = std::make_shared<ConformerSubSample<T>>(conformer_encoder_path.parseObject("feat_convsubsample"), base_path);
        feat_linear = std::make_shared<Linear<T>>(conformer_encoder_path.parseObject("feat_linear"), base_path);
        auto layers_json = conformer_encoder_path.parseObject("layers");
        for (int i = 0; i < num_layer; ++i) {
            std::string object_name = std::to_string(i);
            auto layer = std::make_shared<ConformerStreamEncoderLayer<T>>(layers_json.parseObject(object_name.c_str()), base_path);
            conformer_stream_encoder_layers.push_back(std::move(layer));
        }
        feat_norm = std::make_shared<LayerNorm<T>>(conformer_encoder_path.parseObject("feat_norm"), base_path);
        layer_norm1 = std::make_shared<LayerNorm<T>>(conformer_encoder_path.parseObject("layer_norm1"), base_path);
        position = 0;
    }

    // clear: 모든 streaming 레이어의 캐시 초기화 및 position 리셋
    template<typename T>
    void ConformerStreamEncoderV2<T>::clear() {
        for (int i = 0; i < num_layer; ++i) {
            conformer_stream_encoder_layers[i]->clear();
        }
        position = 0;
    }

    // forward: feat_norm -> SubSample -> Linear -> LayerNorm1 -> Streaming PosEnc -> N개 StreamEncoderLayer
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerStreamEncoderV2<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        // 1) 입력 feature 정규화
        std::shared_ptr<Tensor<float>> output = feat_norm->forward(input);
        // 2) Conv2D subsampling (시간 축 1/4)
        output = convsubsample->forward(output);
        // 3) Linear projection -> LayerNorm
        output = feat_linear->forward(output);
        output = layer_norm1->forward(output);
        // 4) Streaming positional encoding (현재 position 오프셋 적용)
        output = pos_enc->forwardStream(output, position);
        position += output->shape(0);
        // 5) N개의 StreamEncoderLayer 순차 적용
        for (int i = 0; i < num_layer; ++i) {
            output = conformer_stream_encoder_layers[i]->forward(output);
        }
        return output;
    }

    template class ConformerStreamEncoderV2<float>;
} // namespace MERCURY
