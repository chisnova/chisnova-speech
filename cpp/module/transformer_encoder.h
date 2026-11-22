#ifndef MERCURY_TRANSFORMER_ENCODER
#define MERCURY_TRANSFORMER_ENCODER

#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/positional_encoding.h"
#include "module/transformer_encoder_layer.h"

// TransformerEncoder: Transformer 인코더 전체 모듈
// - Positional Encoding -> N개의 TransformerEncoderLayer 순차 적용
// - 입력 시퀀스를 고수준 표현으로 인코딩

namespace MERCURY {
    template<typename T>
    class TransformerEncoder {
    public:
        TransformerEncoder(JsonObject encoder_path, const std::string& base_path);
        ~TransformerEncoder() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);

        int num_layer;
        std::shared_ptr<PosEncoding<T>> pos_encoding;
        std::vector<std::shared_ptr<TransformerEncoderLayer<T>>> encoder_layers;
    };
} // namespace MERCURY

#endif
