#ifndef MECURY_ASR_CONFORMER
#define MECURY_ASR_CONFORMER
#include "time.h"
#include <string>
#include "tensor/tensor.h"
#include "layer/softmax.h"
#include "json_loader/json_object.h"
#include "module/conformer_encoder.h"
#include "module/conformer_decoder.h"
#include "module/decoder_cache.h"
#include "beam/beam.h"
#include "decoder/asr_model.h"

namespace MERCURY {
    template<typename T>
    class AsrConformer : public AsrModel<T> {
    public:
    
        AsrConformer(JsonObject asr_conformer_path, std::string base_path);
        ~AsrConformer();
        int getNumDecoderLayer();
        void setDecoderCache(std::shared_ptr<DecoderCache> _decoder_cache);
        void clear();
        std::shared_ptr<Tensor<float>> getCtcOut();
        std::vector<Beam> makeInitBeams();
        std::shared_ptr<Tensor<T>> encoding(std::shared_ptr<Tensor<T>> enc);
        std::shared_ptr<Tensor<T>> decoding(std::shared_ptr<Tensor<int>> dec, int position);
        std::vector<Score> getNbest(std::shared_ptr<Tensor<float>> logits, int num_nbest);
        std::vector<int> getCtcAsrIds();
        std::vector<Beam> run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams);
        std::shared_ptr<ConformerEncoder<T>> conformer_encoder;
        std::shared_ptr<ConformerDecoder<T>> conformer_decoder;
        std::shared_ptr<Tensor<T>> enc_out_val;
        std::shared_ptr<Tensor<T>> ctc_out_val;
        float score;
        int sos;
        int eos;
    };
};

#endif

