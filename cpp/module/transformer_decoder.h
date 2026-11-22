#ifndef MERCURY_TRANSFORMER_DECODER
#define MERCURY_TRANSFORMER_DECODER

#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/positional_encoding.h"
#include "module/transformer_decoder_layer.h"
#include "module/decoder_cache.h"

// TransformerDecoder: Transformer 디코더 전체 모듈
// - PosEncoding -> N개의 TransformerDecoderLayer (self-attn + cross-attn + FFN)
// - 인코더 출력을 cross-attention으로 참조하여 시퀀스 디코딩

namespace MERCURY {
    template<typename T>
    class TransformerDecoder {
    public:
        TransformerDecoder(JsonObject decoder_path, const std::string& base_path);
        ~TransformerDecoder() = default;

        void clear();
        void setDecoderCache(const std::shared_ptr<DecoderCache>& _decoder_cache);

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input,
                                           const std::shared_ptr<Tensor<T>>& encoding);
        std::shared_ptr<Tensor<T>> forwardContext(const std::shared_ptr<Tensor<T>>& input,
                                                  const std::shared_ptr<Tensor<T>>& encoding,
                                                  int position);

        int num_layer;
        int d_model;
        std::shared_ptr<PosEncoding<T>> pos_encoding;
        std::vector<std::shared_ptr<TransformerDecoderLayer<T>>> decoder_layers;
    };
} // namespace MERCURY

#endif
