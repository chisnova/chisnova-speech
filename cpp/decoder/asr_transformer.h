#ifndef MECURY_ASR_TRANSFORMER
#define MECURY_ASR_TRANSFORMER

#include <string>
#include <cassert>
#include "time.h"
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/linear.h"
#include "layer/embedding.h"
#include "layer/softmax.h"
#include "module/transformer_encoder.h"
#include "module/transformer_decoder.h"
#include "module/decoder_cache.h"
#include "decoder/asr_model.h"


namespace MERCURY {
    template<typename T>
    class AsrTransformer : public AsrModel<T> {
    public:
        AsrTransformer(JsonObject &asr_transformer_path, std::string base_path);
        ~AsrTransformer();
        void setDecoderCache(std::shared_ptr<DecoderCache> _decoder_cache);
        std::shared_ptr<Tensor<T>> encoding(std::shared_ptr<Tensor<T>> enc);
        std::shared_ptr<Tensor<T>> decoding(std::shared_ptr<Tensor<int>> dec, int position);
        std::vector<int> getCtcAsrIds();
        int getNumDecoderLayer();
        std::shared_ptr<Linear<T>> f_in_emb1;
        std::shared_ptr<Linear<T>> f_in_emb2;
        std::shared_ptr<Linear<T>> f_in_emb3;
        std::shared_ptr<Embedding> t_in_emb;
        std::shared_ptr<Linear<T>> t_out_emb;
        std::shared_ptr<TransformerEncoder<T>> encoder;
        std::shared_ptr<TransformerDecoder<T>> decoder;
        std::shared_ptr<Linear<T>> enc_ctc_out;
        std::shared_ptr<Tensor<T>> enc_out_val;
        
    };
};

#endif

