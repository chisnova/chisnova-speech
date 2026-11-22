#include "module/conformer_decoder.h"

namespace MERCURY {
    // 생성자: Embedding, PosEncoding, N개 DecoderLayer, LayerNorm, 출력 Linear 가중치 로드
    template<typename T>
    ConformerDecoder<T>::ConformerDecoder(JsonObject conformer_decoder_path, const std::string& base_path) {
        t_in_emb = std::make_shared<Embedding>(conformer_decoder_path.parseObject("tin_emb"), base_path);
        pos_encoding = std::make_shared<PosEncoding<T>>(conformer_decoder_path.parseObject("pos_enc"), base_path);
        num_layer = int(conformer_decoder_path.parserValue("num_layer"));
        n_head = int(conformer_decoder_path.parserValue("num_head"));
        d_model = int(conformer_decoder_path.parserValue("dim_model"));
        ffn_hidden = int(conformer_decoder_path.parserValue("ffn_hidden"));
        auto layers_json = conformer_decoder_path.parseObject("layers");
        for (int i = 0; i < num_layer; ++i) {
            std::string object_name = std::to_string(i);
            auto layer = std::make_shared<TransformerDecoderLayer<T>>(layers_json.parseObject(object_name.c_str()), base_path);
            decoder_layers.push_back(std::move(layer));
        }
        layer_norm = std::make_shared<LayerNorm<T>>(conformer_decoder_path.parseObject("layer_norm"), base_path);
        t_out_emb = std::make_shared<Linear<T>>(conformer_decoder_path.parseObject("tout_emb"), base_path);
    }

    // clear: 모든 decoder layer의 attention 캐시 초기화
    template<typename T>
    void ConformerDecoder<T>::clear() {
        for (int i = 0; i < num_layer; ++i) {
            decoder_layers[i]->clear();
        }
    }

    // setDecoderCache: 외부에서 미리 계산된 key/value 캐시를 각 layer에 설정
    template<typename T>
    void ConformerDecoder<T>::setDecoderCache(const std::shared_ptr<DecoderCache>& _decoder_cache) {
        assert(_decoder_cache->multihead_caches.size() == decoder_layers.size());
        for (int i = 0; i < num_layer; ++i) {
            decoder_layers[i]->setMultiHeadCache(_decoder_cache->multihead_caches[i]);
        }
    }

    // forward: 전체 시퀀스 디코딩 (학습/평가 모드)
    // Embedding -> PosEnc -> N개 DecoderLayer (self-attn + cross-attn) -> LayerNorm -> Linear
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerDecoder<T>::forward(const std::shared_ptr<Tensor<int>>& input,
                                                             const std::shared_ptr<Tensor<T>>& encoding) {
        std::shared_ptr<Tensor<T>> out = t_in_emb->forward(input);
        out = pos_encoding->forward(out);
        for (int i = 0; i < num_layer; ++i) {
            out = decoder_layers[i]->forward(out, encoding);
        }
        out = layer_norm->forward(out);
        out = t_out_emb->forward(out);
        return out;
    }

    // forwardContext: autoregressive 디코딩 (추론 모드)
    // 캐시를 활용하여 한 토큰씩 디코딩, softmax로 확률 분포 출력
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerDecoder<T>::forwardContext(const std::shared_ptr<Tensor<int>>& input,
                                                                    const std::shared_ptr<Tensor<T>>& encoding,
                                                                    int position) {
        std::shared_ptr<Tensor<T>> out = t_in_emb->forward(input);
        out = pos_encoding->forwardContext(out, position);
        for (int i = 0; i < num_layer; ++i) {
            out = decoder_layers[i]->forwardContext(out, encoding);
        }
        out = layer_norm->forward(out);
        out = t_out_emb->forward(out);
        out = Softmax().forward(out);
        return out;
    }

    template<typename T>
    int ConformerDecoder<T>::getNumLayer() const {
        return num_layer;
    }

    template class ConformerDecoder<float>;
} // namespace MERCURY
