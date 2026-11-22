#ifndef MERCURY_ASR_DECODER
#define MERCURY_ASR_DECODER

#include <queue>
#include <numeric> 
#include "tensor/tensor.h"
#include "beam/beam.h"
#include "feature/librosa_feature.h"
#include "decoder/asr_model.h"
#include "decoder/asr_conformer.h"
#include "decoder/asr_conformer_trf_transducer.h"
#include "decoder/asr_conformer_trf_transducerV2.h"
#include "decoder/asr_stream_conformer_trf_transducer.h"
#include <thread>
#include <functional>
#include <condition_variable>
#include <mutex>

namespace MERCURY {
    template<typename T>
    class AsrDecoder {
    public:
        AsrDecoder(JsonObject &asr_model_path, std::string base_path);
        ~AsrDecoder();
        void clear();
        void sendAudio(float audio[], int size);
        void runEngine(int num_nbest);
        std::vector<int> getAsrResult();
        std::shared_ptr<Tensor<T>> getPhoneScore(std::vector<int> phone_dec);
        std::shared_ptr<Tensor<T>> getWordScore(std::vector<int> word_dec);

        std::shared_ptr<LibrosaFeature> feature_extractor;
        std::shared_ptr<AsrModel<T>> asr_model;
        std::vector<float> chunk_audio;
        std::vector<Beam> beams;
    };
};
#endif