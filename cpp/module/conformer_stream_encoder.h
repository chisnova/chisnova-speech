#ifndef MERCURY_CONFORMER_STREAM_ENCODER
#define MERCURY_CONFORMER_STREAM_ENCODER

#include <string>
#include <vector>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/rotary_embedding.h"
#include "module/conformer_subsample.h"
#include "module/conformer_stream_encoder_layer.h"
#include "layer/linear.h"
#include "layer/layernorm.h"

// ConformerStreamEncoder: Streaming Conformer 인코더 (1패스)
// Python module/conformer_stream_encoder.py 의 stream() 경로와 동일:
//   feat_norm(LayerNorm 80) -> ConvSubSample.stream -> feat_linear -> layer_norm1
//   -> RoPE cos/sin (발화 시작부터의 절대 위치) -> N개의 StreamEncoderLayer
// 위치 인코딩은 additive 가 아니라 attention 내부의 RoPE 다.
// 청크가 subsample 커널을 못 채우면 0행 텐서를 돌려준다 (호출자가 건너뛴다).

namespace MERCURY {
    template<typename T>
    class ConformerStreamEncoder {
    public:
        ConformerStreamEncoder(JsonObject conformer_encoder_path, const std::string& base_path);
        ~ConformerStreamEncoder() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);
        void clear();
        void setStreamLeftContext(int _stream_left_context);

        int num_head;
        int num_layer;
        int dim_model;
        std::shared_ptr<LayerNorm<T>> feat_norm;      // 입력 log-mel(80) 정규화
        std::shared_ptr<ConformerSubSample<T>> convsubsample;
        std::shared_ptr<Linear<T>> feat_linear;
        std::shared_ptr<LayerNorm<T>> layer_norm1;
        std::vector<std::shared_ptr<ConformerStreamEncoderLayer<T>>> conformer_stream_encoder_layers;
        RotaryEmbedding<T> rotary;
        int stream_left_context;  // attention KV 캐시 좌컨텍스트 (프레임)
        int position;             // 발화 시작부터 누적된 subsample 후 프레임 수
    };
} // namespace MERCURY

#endif
