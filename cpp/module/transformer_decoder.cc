#include "module/transformer_decoder.h"

namespace MERCURY {
    // 생성자: PosEncoding 및 N개의 TransformerDecoderLayer 가중치 로드
    template<typename T>
    TransformerDecoder<T>::TransformerDecoder(JsonObject decoder_path, const std::string& base_path) {
        pos_encoding = std::make_shared<PosEncoding<T>>(decoder_path.parseObject("pos_enc"), base_path);
        num_layer = int(decoder_path.parserValue("num_layer"));
        auto layers_json = decoder_path.parseObject("layers");
        for (int i = 0; i < num_layer; ++i) {
            std::string object_name = std::to_string(i);
            auto layer = std::make_shared<TransformerDecoderLayer<T>>(layers_json.parseObject(object_name.c_str()), base_path);
            decoder_layers.push_back(std::move(layer));
        }
    }

    // clear: 모든 decoder layer의 attention 캐시 초기화
    template<typename T>
    void TransformerDecoder<T>::clear() {
        for (int i = 0; i < num_layer; ++i) {
            decoder_layers[i]->clear();
        }
    }

    // setDecoderCache: 외부 캐시를 각 layer에 설정 (beam search 등)
    template<typename T>
    void TransformerDecoder<T>::setDecoderCache(const std::shared_ptr<DecoderCache>& _decoder_cache) {
        assert(_decoder_cache->multihead_caches.size() == decoder_layers.size());
        for (int i = 0; i < num_layer; ++i) {
            decoder_layers[i]->setMultiHeadCache(_decoder_cache->multihead_caches[i]);
        }
    }

    // forward: 전체 시퀀스 디코딩 (학습/평가 모드)
    // PosEnc -> N개 DecoderLayer (self-attn + cross-attn + FFN)
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerDecoder<T>::forward(const std::shared_ptr<Tensor<T>>& input,
                                                               const std::shared_ptr<Tensor<T>>& encoding) {
        std::shared_ptr<Tensor<T>> out = pos_encoding->forward(input);
        for (int i = 0; i < num_layer; ++i) {
            out = decoder_layers[i]->forward(out, encoding);
        }
        return out;
    }

    // forwardContext: autoregressive 디코딩 (추론 모드)
    // position offset으로 positional encoding 적용 후 캐시 활용 디코딩
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerDecoder<T>::forwardContext(const std::shared_ptr<Tensor<T>>& input,
                                                                     const std::shared_ptr<Tensor<T>>& encoding,
                                                                     int position) {
        // forwardContext 가 내부에서 input + PE[position] 을 이미 계산한다 —
        // 여기서 input 을 또 더하면 임베딩이 2배가 되어 Python 과 어긋난다
        std::shared_ptr<Tensor<T>> out = pos_encoding->forwardContext(input, position);
        for (int i = 0; i < num_layer; ++i) {
            out = decoder_layers[i]->forwardContext(out, encoding);
        }
        return out;
    }

    template class TransformerDecoder<float>;
} // namespace MERCURY
