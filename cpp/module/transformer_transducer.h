#ifndef MERCURY_TRANSFORMER_TRANSDUCER
#define MERCURY_TRANSFORMER_TRANSDUCER

#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/positional_encoding.h"
#include "module/transformer_transducer_layer.h"
#include "module/decoder_cache.h"
#include "layer/embedding.h"
#include "layer/tanh.h"
#include "module/transformer_encoder.h"
#include "module/transformer_decoder.h"

// TransformerTransducer: Transformer-Transducer (RNN-T) 전체 모듈
// - Prediction Network: Embedding -> PosEnc -> N개 TransducerLayer
// - Joint Network: enc_proj + prd_proj -> concat -> fc1 -> tanh -> fc2
// - 인코더와 예측 네트워크의 출력을 결합하여 토큰 확률 생성

namespace MERCURY {
    template<typename T>
    class TransformerTransducer {
    public:
        TransformerTransducer(JsonObject transformer_transducer_path, const std::string& base_path);
        ~TransformerTransducer() = default;

        void clear();
        void setDecoderCache(const std::shared_ptr<DecoderCache>& _decoder_cache);

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<int>>& input,
                                           const std::shared_ptr<Tensor<T>>& encoding);
        std::shared_ptr<Tensor<T>> forwardContext(const std::shared_ptr<Tensor<int>>& input,
                                                  const std::shared_ptr<Tensor<T>>& encoding,
                                                  int position);

        int getNumLayer() const;

        int num_layer;
        int d_model;
        std::shared_ptr<Embedding> t_in_emb;
        std::shared_ptr<PosEncoding<T>> pos_encoding;
        std::vector<std::shared_ptr<TransformerTransducerLayer<T>>> decoder_layers;
        std::shared_ptr<Linear<T>> enc_proj;   // encoder output projection
        std::shared_ptr<Linear<T>> prd_proj;   // prediction output projection
        std::shared_ptr<Linear<T>> fc1;        // joint network FC1
        std::shared_ptr<Linear<T>> fc2;        // joint network FC2
        std::shared_ptr<TransformerEncoder<T>> encoder;
        std::shared_ptr<TransformerDecoder<T>> decoder;
        std::shared_ptr<TransformerTransducerLayer<T>> transducer;
    };
} // namespace MERCURY

#endif
