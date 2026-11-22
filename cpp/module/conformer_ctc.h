#ifndef MERCURY_CONFORMER_CTC
#define MERCURY_CONFORMER_CTC

#include <string>
#include <vector>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/conformer_positional.h"
#include "module/conformer_encoder_layer.h"
#include "layer/linear.h"
#include "layer/layernorm.h"

// ConformerCTC: Conformer CTC 인코더 모듈
// - LayerNorm1 -> Positional Encoding -> N개의 ConformerEncoderLayer -> LayerNorm2
// - CTC 디코딩을 위한 인코더 출력 생성

namespace MERCURY {
    template<typename T>
    class ConformerCTC {
    public:
        ConformerCTC(JsonObject conformer_ctc_path, const std::string& base_path);
        ~ConformerCTC() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);

        int num_head;
        int num_layer;
        int dim_model;
        std::shared_ptr<ConformerPositional<T>> pos_enc;
        std::vector<std::shared_ptr<ConformerEncoderLayer<T>>> conformer_encoder_layers;
        std::shared_ptr<LayerNorm<T>> layer_norm1;
        std::shared_ptr<LayerNorm<T>> layer_norm2;
    };
} // namespace MERCURY

#endif
