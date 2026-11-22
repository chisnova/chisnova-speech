#ifndef MECURY_ASR_STREAM_CONFORMER_TRANSFORMER_TRANSDUCER
#define MECURY_ASR_STREAM_CONFORMER_TRANSFORMER_TRANSDUCER
#include "time.h"
#include <string>

#include "tensor/tensor.h"
#include "layer/softmax.h"
#include "json_loader/json_object.h"
#include "module/conformer_stream_encoderV2.h"
#include "module/conformer_ctc.h"

#include "module/transformer_transducer.h"
#include "module/conformer_decoder.h"
#include "module/decoder_cache.h"
#include "beam/beam.h"
#include "decoder/asr_model.h"
#include "utils/thread_pool.h"
#include <memory>
#include <future>
#include <mutex>
#include <queue>

namespace MERCURY {
    template<typename T>
    class AsrStreamConformerTrfTransducer : public AsrModel<T> {
    public:
        AsrStreamConformerTrfTransducer(JsonObject asr_conformer_path, std::string base_path);
        ~AsrStreamConformerTrfTransducer();
        void clear();
        std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<int>> dec, std::shared_ptr<Tensor<T>> enc);
        std::vector<Beam> makeInitBeams();
        std::shared_ptr<Tensor<T>> encoding(std::shared_ptr<Tensor<T>> feature);

        std::vector<Score> getNbest(std::shared_ptr<Tensor<T>> logits, int num_nbest);
        // bool isBreak(std::vector<Beam>& stop, std::vector<Beam>& running, int num_nbest);
        bool isBreak(std::vector<Beam> stop, std::vector<Beam> running, int num_nbest);
        std::vector<Beam> run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams);
        void warmUpInference(std::shared_ptr<Tensor<T>> feature);
        
        std::pair<Beam, std::vector<Beam>> eachBeamSearch(Beam& beam, std::shared_ptr<Tensor<T>> enc_in, int beam_size);
        // Thread-safe beam 처리를 위한 함수들
        std::pair<Beam, std::vector<Beam>> processBeam(const Beam& cur_beam, 
                                                       std::shared_ptr<Tensor<T>> each_enc, 
                                                       int num_nbest);
        
        // 병렬 beam 처리 함수
        std::pair<std::vector<Beam>, std::vector<Beam>> runEachBeams(const std::vector<Beam>& worker, 
                                                                   std::shared_ptr<Tensor<T>> each_enc, 
                                                                   int num_nbest);
        
        std::vector<int> getCtcAsrIds();
        std::vector<int> secondPassDecoding(int num_nbest);
        std::shared_ptr<Tensor<T>> getWordScore(std::vector<int> word_dec);
        std::shared_ptr<Tensor<T>> getPhoneScore(std::vector<int> phone_dec);
        
        
        // TransformerTransducer 인스턴스 풀링
        std::shared_ptr<TransformerTransducer<T>> getTransformerInstance();
        void initializeTransformerPool(int num_instance);
        void returnTransformerInstance(std::shared_ptr<TransformerTransducer<T>> instance);
        std::shared_ptr<ConformerStreamEncoderV2<T>> conformer_stream_encoder;
        std::shared_ptr<ConformerCTC<T>> conformer_ctc;
        std::shared_ptr<TransformerTransducer<T>> transformer_transducer;  // 메인 인스턴스 (단일 스레드용)
        std::shared_ptr<ConformerDecoder<T>> phone_decoder;
        std::shared_ptr<ConformerDecoder<T>> word_decoder;
        std::shared_ptr<Linear<T>> ctc_linear1;
        std::shared_ptr<Linear<T>> ctc_linear2;
        
        std::shared_ptr<Tensor<T>> enc_out_val;
        std::shared_ptr<Tensor<T>> ctc_out_val;

        int num_layer;
        int position;
        int sos;
        int eos;
        
    private:
        // TransformerTransducer 인스턴스 풀 (간단한 접근법: 미리 생성된 인스턴스만 사용)
        std::vector<std::shared_ptr<TransformerTransducer<T>>> transformer_pool;
        std::mutex pool_mutex;
        JsonObject transformer_config;
        std::string config_base_path;
        bool pool_initialized;

    };
};

#endif

