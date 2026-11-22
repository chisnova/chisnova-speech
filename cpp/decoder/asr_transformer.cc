#include "decoder/asr_transformer.h"

namespace MERCURY {
    template<typename T>
    AsrTransformer<T>::AsrTransformer(JsonObject &asr_transformer_path, std::string base_path) {
        f_in_emb1 = std::make_shared<Linear<T>>(asr_transformer_path.parseObject("feat_emb1"), base_path);
        f_in_emb2 = std::make_shared<Linear<T>>(asr_transformer_path.parseObject("feat_emb2"), base_path);
        f_in_emb3 = std::make_shared<Linear<T>>(asr_transformer_path.parseObject("feat_emb3"), base_path);
        t_in_emb = std::make_shared<Embedding>(asr_transformer_path.parseObject("txt_in_emb"), base_path);
        t_out_emb = std::make_shared<Linear<T>>(asr_transformer_path.parseObject("txt_out_emb"), base_path);
        encoder = std::make_shared<TransformerEncoder<T>>(asr_transformer_path.parseObject("encoder"), base_path);
        decoder = std::make_shared<TransformerDecoder<T>>(asr_transformer_path.parseObject("decoder"), base_path);
        enc_ctc_out = std::make_shared<Linear<T>>(asr_transformer_path.parseObject("enc_ctc_out"), base_path);
    }

    template<typename T>
    AsrTransformer<T>::~AsrTransformer() {
        f_in_emb1 = nullptr;
        f_in_emb2 = nullptr;
        f_in_emb3 = nullptr;
        t_in_emb = nullptr;
        t_out_emb = nullptr;
        encoder = nullptr;
        decoder = nullptr;
        enc_ctc_out = nullptr;
        enc_out_val = nullptr;
    }

    template<typename T>
    void AsrTransformer<T>::setDecoderCache(std::shared_ptr<DecoderCache> _decoder_cache) {
        decoder->setDecoderCache(_decoder_cache);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrTransformer<T>::encoding(std::shared_ptr<Tensor<T>> enc) {
        std::shared_ptr<Tensor<T>> out1;
        out1 = f_in_emb1->forward(enc);
        out1 = f_in_emb2->forward(out1);
        out1 = f_in_emb3->forward(out1);
        enc_out_val = encoder->forward(out1);
        decoder->clear();
        return enc_out_val;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> AsrTransformer<T>::decoding(std::shared_ptr<Tensor<int>> dec, int position) {
        std::shared_ptr<Tensor<T>> ret;
        if (enc_out_val == nullptr){
            std::cout << "enc_output is nullptr" << std::endl;
            assert(false);
        }
        ret = t_in_emb->forward(dec);
        ret = decoder->forwardContext(ret, enc_out_val, position);
        ret = t_out_emb->forward(ret);
        ret = Softmax().forward(ret);
        return ret;
    }

    template<typename T>
    int AsrTransformer<T>::getNumDecoderLayer() {
        return -1;
    }

    template<typename T>
    std::vector<int> AsrTransformer<T>::getCtcAsrIds() {
        std::vector<int> ctc_asr_ids;
        return ctc_asr_ids;
    }

    template class AsrTransformer<float>;
    // template class AsrTransformer<int>;
};

