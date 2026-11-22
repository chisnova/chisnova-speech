#include "decoder/asr_decoder.h"
#include "logger/logger.h"
#include "time.h"

namespace MERCURY {
    template<typename T>
    AsrDecoder<T>::AsrDecoder(JsonObject &asr_model_path, std::string base_path) {
        std::string type = asr_model_path.parserString("model_type");
        if (type.compare("stream-conformer-transformer-transducer") == 0) {
            asr_model = std::make_shared<AsrStreamConformerTrfTransducer<T>>(asr_model_path.parseObject("model"), base_path);
            std::cout << "conformer stream transducer init done" << std::endl;
        }       
        feature_extractor = std::make_shared<LibrosaFeature>(asr_model_path, base_path);   
        beams.clear();
        beams = asr_model->makeInitBeams();
        chunk_audio.clear();
    }

    template<typename T>
    AsrDecoder<T>::~AsrDecoder() {
        asr_model = nullptr;
        feature_extractor = nullptr;
    }

    template<typename T>
    void AsrDecoder<T>::clear() {
        chunk_audio.clear();
        asr_model->clear();
        beams.clear();
    }

    template<typename T>
    void AsrDecoder<T>::sendAudio(float audio[], int size) {
        for(int i=0;i<size;++i) {
            chunk_audio.push_back(audio[i]);
        }
    }

    template<typename T>
    void AsrDecoder<T>::runEngine(int num_nbest) {
        std::shared_ptr<Tensor<T>> feature = feature_extractor->getFeature(chunk_audio.data(), chunk_audio.size());
        beams = asr_model->run(feature, num_nbest);
    }

    template<typename T>
    std::vector<int> AsrDecoder<T>::getAsrResult() {
        return beams[0].word_ids;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrDecoder<T>::getPhoneScore(std::vector<int> phone_dec){
        return asr_model->getPhoneScore(phone_dec);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrDecoder<T>::getWordScore(std::vector<int> word_dec){
        return asr_model->getWordScore(word_dec);
    }




    template class AsrDecoder<float>;
    // template class AsrDecoder<int>;
};