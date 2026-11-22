#ifndef MERCURY_CONFORMER_ENCODER
#define MERCURY_CONFORMER_ENCODER

#include <string>
#include <vector>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/conformer_positional.h"
#include "module/conformer_subsample.h"
#include "module/conformer_encoder_layer.h"
#include "layer/linear.h"
#include "layer/layernorm.h"

// ConformerEncoder: Conformer 인코더 전체 모듈
// - ConvSubSample (4x downsampling) -> Linear -> Positional Encoding -> N개의 ConformerEncoderLayer 순차 적용
// - 음성 특징 입력을 고수준 인코딩 표현으로 변환

namespace MERCURY {
    template<typename T>
    class ConformerEncoder {
    public:
        ConformerEncoder(JsonObject conformer_encoder_path, const std::string& base_path);
        ~ConformerEncoder() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);

        int num_head;
        int num_layer;
        int dim_model;
        std::shared_ptr<ConformerPositional<T>> pos_enc;
        std::shared_ptr<ConformerSubSample<T>> convsubsample;
        std::shared_ptr<Linear<T>> feat_linear;
        std::vector<std::shared_ptr<ConformerEncoderLayer<T>>> conformer_encoder_layers;
        std::shared_ptr<Tensor<T>> enc_out_val;
    };
} // namespace MERCURY

#endif
