#ifndef MECURY_ASR_CONFORMER_TRANSFORMER_TRANSDUCER_V2
#define MECURY_ASR_CONFORMER_TRANSFORMER_TRANSDUCER_V2
#include "time.h"
#include <string>

#include "tensor/tensor.h"
#include "layer/softmax.h"
#include "json_loader/json_object.h"
#include "module/conformer_stream_encoderV2.h"
#include "module/transformer_transducer.h"
#include "module/decoder_cache.h"
#include "module/conformer_ctc.h"
#include "beam/beam.h"
#include "decoder/asr_model.h"
#include <memory>

namespace MERCURY {
    template<typename T>
    class AsrConformerTrfTransducerV2 : public AsrModel<T> {
    public:
        AsrConformerTrfTransducerV2(JsonObject asr_conformer_path, std::string base_path);
        ~AsrConformerTrfTransducerV2();
        int getNumDecoderLayer();
        void setDecoderCache(std::shared_ptr<DecoderCache> _decoder_cache);
        void clear();
        std::shared_ptr<Tensor<float>> getCtcOut();
        std::vector<Beam> makeInitBeams();
        
        std::shared_ptr<Tensor<T>> encoding(std::shared_ptr<Tensor<T>> enc);
        std::shared_ptr<Tensor<T>> decoding(std::shared_ptr<Tensor<int>> dec, int position);
        std::vector<Score> getNbest(std::shared_ptr<Tensor<float>> logits, int num_nbest);
        bool isBreak(std::priority_queue<Beam> stop, std::priority_queue<Beam> running, int num_nbest);
        std::vector<Beam> run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams);
        std::vector<int> getCtcAsrIds();
        std::shared_ptr<ConformerStreamEncoderV2<T>> conformer_stream_encoder;
        std::shared_ptr<TransformerTransducer<T>> transformer_transducer;
        std::shared_ptr<ConformerCTC<T>> conformer_ctc;

        std::vector<std::shared_ptr<Tensor<T>>> enc_arr;

        int num_layer;
        int position;
        int sos;
        int eos;
    };
};

#endif

