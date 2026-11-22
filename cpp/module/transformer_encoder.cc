#include "module/transformer_encoder.h"

namespace MERCURY {
    // 생성자: PosEncoding 및 N개의 TransformerEncoderLayer 가중치 로드
    template<typename T>
    TransformerEncoder<T>::TransformerEncoder(JsonObject encoder_path, const std::string& base_path) {
        pos_encoding = std::make_shared<PosEncoding<T>>(encoder_path.parseObject("pos_enc"), base_path);
        num_layer = int(encoder_path.parserValue("num_layer"));
        auto layers_json = encoder_path.parseObject("layers");
        for (int i = 0; i < num_layer; ++i) {
            std::string object_name = std::to_string(i);
            auto layer = std::make_shared<TransformerEncoderLayer<T>>(layers_json.parseObject(object_name.c_str()), base_path);
            encoder_layers.push_back(std::move(layer));
        }
    }

    // forward: Positional Encoding -> N개 EncoderLayer 순차 적용
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerEncoder<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        // 1) positional encoding 적용
        std::shared_ptr<Tensor<T>> out = pos_encoding->forward(input);
        // 2) N개의 encoder layer 순차 통과
        for (int i = 0; i < num_layer; ++i) {
            out = encoder_layers[i]->forward(out);
        }
        return out;
    }

    template class TransformerEncoder<float>;
} // namespace MERCURY
