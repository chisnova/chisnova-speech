#include "module/conformer_ctc.h"

namespace MERCURY {
    // 생성자: LayerNorm x2, PosEnc, N개의 ConformerEncoderLayer 가중치 로드
    template<typename T>
    ConformerCTC<T>::ConformerCTC(JsonObject conformer_ctc_path, const std::string& base_path) {
        num_head = int(conformer_ctc_path.parserValue("num_head"));
        num_layer = int(conformer_ctc_path.parserValue("num_layer"));
        dim_model = int(conformer_ctc_path.parserValue("dim_model"));
        pos_enc = std::make_shared<ConformerPositional<T>>(conformer_ctc_path.parseObject("pos_enc"), base_path);
        auto layers_json = conformer_ctc_path.parseObject("layers");
        for (int i = 0; i < num_layer; ++i) {
            std::string object_name = std::to_string(i);
            auto layer = std::make_shared<ConformerEncoderLayer<T>>(layers_json.parseObject(object_name.c_str()), base_path);
            conformer_encoder_layers.push_back(std::move(layer));
        }
        layer_norm1 = std::make_shared<LayerNorm<T>>(conformer_ctc_path.parseObject("layer_norm1"), base_path);
        layer_norm2 = std::make_shared<LayerNorm<T>>(conformer_ctc_path.parseObject("layer_norm2"), base_path);
    }

    // forward: LayerNorm1 -> PosEnc 생성 -> N개 EncoderLayer -> LayerNorm2
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerCTC<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        // 1) 입력 정규화
        std::shared_ptr<Tensor<float>> output = layer_norm1->forward(input);
        // 2) relative positional encoding 생성
        std::shared_ptr<Tensor<float>> pos_encoding = pos_enc->forward(output);
        // 3) N개의 ConformerEncoderLayer 순차 적용
        for (int i = 0; i < num_layer; ++i) {
            output = conformer_encoder_layers[i]->forward(output, pos_encoding);
        }
        // 4) 최종 LayerNorm 적용
        output = layer_norm2->forward(output);
        return output;
    }

    template class ConformerCTC<float>;
} // namespace MERCURY
