#include "module/conformer_encoder.h"

namespace MERCURY {
    // 생성자: SubSample, Linear, PositionalEncoding, N개의 EncoderLayer 가중치 로드
    template<typename T>
    ConformerEncoder<T>::ConformerEncoder(JsonObject conformer_encoder_path, const std::string& base_path) {
        num_head = int(conformer_encoder_path.parserValue("num_head"));
        num_layer = int(conformer_encoder_path.parserValue("num_layer"));
        dim_model = int(conformer_encoder_path.parserValue("dim_model"));
        pos_enc = std::make_shared<ConformerPositional<T>>(conformer_encoder_path.parseObject("pos_enc"), base_path);
        convsubsample = std::make_shared<ConformerSubSample<T>>(conformer_encoder_path.parseObject("feat_convsubsample"), base_path);
        feat_linear = std::make_shared<Linear<T>>(conformer_encoder_path.parseObject("feat_linear"), base_path);
        auto layers_json = conformer_encoder_path.parseObject("layers");
        for (int i = 0; i < num_layer; ++i) {
            std::string object_name = std::to_string(i);
            auto layer = std::make_shared<ConformerEncoderLayer<T>>(layers_json.parseObject(object_name.c_str()), base_path);
            conformer_encoder_layers.push_back(std::move(layer));
        }
    }

    // forward: SubSample -> Linear -> PosEnc 생성 -> N개 EncoderLayer 순차 적용
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerEncoder<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        // 1) Conv2D subsampling: 시간 축 1/4 다운샘플링
        std::shared_ptr<Tensor<float>> output = convsubsample->forward(input);
        // 2) Linear projection: subsampled feature -> d_model 차원으로 변환
        output = feat_linear->forward(output);
        // 3) Relative positional encoding 생성
        std::shared_ptr<Tensor<float>> pos_encoding = pos_enc->forward(output);
        // 4) N개의 ConformerEncoderLayer 순차 적용
        for (int i = 0; i < num_layer; ++i) {
            output = conformer_encoder_layers[i]->forward(output, pos_encoding);
        }
        enc_out_val = output;
        return enc_out_val;
    }

    template class ConformerEncoder<float>;
} // namespace MERCURY
