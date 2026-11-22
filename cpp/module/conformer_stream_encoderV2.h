#ifndef MERCURY_CONFORMER_STREAM_ENCODER_V2
#define MERCURY_CONFORMER_STREAM_ENCODER_V2

#include <string>
#include <vector>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "module/conformer_subsample.h"
#include "module/conformer_stream_encoder_layer.h"
#include "layer/linear.h"
#include "layer/layernorm.h"
#include "layer/positional_encoding.h"

// ConformerStreamEncoderV2: Streaming Conformer 인코더 V2
// - V1과 달리 feat_norm, layer_norm1 추가 (입력 정규화 강화)
// - feat_norm -> SubSample -> Linear -> LayerNorm1 -> Streaming PosEnc -> N개 StreamEncoderLayer

namespace MERCURY {
    template<typename T>
    class ConformerStreamEncoderV2 {
    public:
        ConformerStreamEncoderV2(JsonObject conformer_encoder_path, const std::string& base_path);
        ~ConformerStreamEncoderV2() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);
        void clear();

        int num_head;
        int num_layer;
        int dim_model;
        std::shared_ptr<PosEncoding<T>> pos_enc;
        std::shared_ptr<ConformerSubSample<T>> convsubsample;
        std::shared_ptr<Linear<T>> feat_linear;
        std::shared_ptr<LayerNorm<T>> layer_norm1;
        std::shared_ptr<LayerNorm<T>> feat_norm;
        std::vector<std::shared_ptr<ConformerStreamEncoderLayer<T>>> conformer_stream_encoder_layers;
        int position;  // streaming positional encoding 오프셋
    };
} // namespace MERCURY

#endif
