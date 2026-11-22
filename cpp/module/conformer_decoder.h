#ifndef MERCURY_CONFORMER_DECODER
#define MERCURY_CONFORMER_DECODER

#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/embedding.h"
#include "layer/linear.h"
#include "layer/positional_encoding.h"
#include "module/transformer_decoder_layer.h"
#include "module/decoder_cache.h"

// ConformerDecoder: Conformer 디코더 모듈
// - Embedding -> PosEncoding -> N개의 TransformerDecoderLayer -> LayerNorm -> Linear(output)
// - 인코더 출력을 cross-attention으로 참조하여 토큰 시퀀스 디코딩

namespace MERCURY {
    template<typename T>
    class ConformerDecoder {
    public:
        ConformerDecoder(JsonObject conformer_decoder_path, const std::string& base_path);
        ~ConformerDecoder() = default;

        void clear();
        void setDecoderCache(const std::shared_ptr<DecoderCache>& _decoder_cache);

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<int>>& input,
                                           const std::shared_ptr<Tensor<T>>& encoding);
        std::shared_ptr<Tensor<T>> forwardContext(const std::shared_ptr<Tensor<int>>& input,
                                                  const std::shared_ptr<Tensor<T>>& encoding,
                                                  int position);

        int getNumLayer() const;

        int num_layer;
        int n_head;
        int d_model;
        int ffn_hidden;
        std::shared_ptr<Embedding> t_in_emb;
        std::shared_ptr<PosEncoding<T>> pos_encoding;
        std::vector<std::shared_ptr<TransformerDecoderLayer<T>>> decoder_layers;
        std::shared_ptr<LayerNorm<T>> layer_norm;
        std::shared_ptr<Linear<T>> t_out_emb;
    };
} // namespace MERCURY

#endif
