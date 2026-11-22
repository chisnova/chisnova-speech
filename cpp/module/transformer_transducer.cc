#include "module/transformer_transducer.h"

namespace MERCURY {
    template<typename T>
    TransformerTransducer<T>::TransformerTransducer(JsonObject decoder_path, const std::string& base_path) {
        t_in_emb = std::make_shared<Embedding>(decoder_path.parseObject("emb"), base_path);
        pos_encoding = std::make_shared<PosEncoding<T>>(decoder_path.parseObject("pos_enc"), base_path);
        num_layer = int(decoder_path.parserValue("num_layer"));
        auto layers_json = decoder_path.parseObject("layers");
        for (int i = 0; i < num_layer; ++i) {
            std::string object_name = std::to_string(i);
            auto layer = std::make_shared<TransformerTransducerLayer<T>>(layers_json.parseObject(object_name.c_str()), base_path);
            decoder_layers.push_back(std::move(layer));
        }
        enc_proj = std::make_shared<Linear<T>>(decoder_path.parseObject("enc_proj"), base_path);
        prd_proj = std::make_shared<Linear<T>>(decoder_path.parseObject("prd_proj"), base_path);
        fc1 = std::make_shared<Linear<T>>(decoder_path.parseObject("fc1"), base_path);
        fc2 = std::make_shared<Linear<T>>(decoder_path.parseObject("fc2"), base_path);
    }

    template<typename T>
    int TransformerTransducer<T>::getNumLayer() const {
        return num_layer;
    }

    template<typename T>
    void TransformerTransducer<T>::clear() {
        for (int i = 0; i < num_layer; ++i) {
            decoder_layers[i]->clear();
        }
    }

    template<typename T>
    void TransformerTransducer<T>::setDecoderCache(const std::shared_ptr<DecoderCache>& _decoder_cache) {
        assert(_decoder_cache->multihead_caches.size() == decoder_layers.size());
        for (int i = 0; i < num_layer; ++i) {
            decoder_layers[i]->setMultiHeadCache(_decoder_cache->multihead_caches[i]);
        }
    }

    // forward: 전체 시퀀스 Transducer 연산
    // 최적화: concat 루프를 Eigen rowwise replicate로 대체 → O(T) 스칼라 루프 제거
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerTransducer<T>::forward(const std::shared_ptr<Tensor<int>>& input,
                                                                  const std::shared_ptr<Tensor<T>>& encoding) {
        // 1) Prediction Network: Embedding -> PosEnc -> N개 TransducerLayer
        std::shared_ptr<Tensor<T>> out = t_in_emb->forward(input);
        out = pos_encoding->forward(out);
        for (int i = 0; i < num_layer; ++i) {
            out = decoder_layers[i]->forward(out);
        }

        // 2) Joint Network: encoder/prediction 출력을 각각 projection
        std::shared_ptr<Tensor<T>> enc_out = enc_proj->forward(encoding);
        out = prd_proj->forward(out);

        // 3) concat: enc_out 각 행에 prediction 출력(단일 행)을 broadcast
        // 최적화: 행별 루프 대신 rowwise replicate로 벡터화된 broadcast
        const int enc_rows = enc_out->shape(0);
        const int enc_cols = enc_out->shape(1);
        const int prd_cols = out->shape(1);
        auto concat_out = std::make_shared<Tensor<T>>();
        concat_out->value.resize(enc_rows, enc_cols + prd_cols);
        // 왼쪽: encoder projection 결과 복사
        concat_out->value.leftCols(enc_cols).noalias() = enc_out->value;
        // 오른쪽: prediction 결과를 모든 행에 broadcast (replicate)
        // out->value는 [1, prd_cols] 이므로 rowwise broadcast
        concat_out->value.rightCols(prd_cols) = out->value.replicate(enc_rows, 1);

        // 4) FC1 -> Tanh -> FC2
        out = fc1->forward(concat_out);
        out = Tanh().forward(out);
        out = fc2->forward(out);
        return out;
    }

    // forwardContext: autoregressive 디코딩 모드 (position offset 적용)
    // 최적화: 동일한 concat broadcast 패턴 적용
    template<typename T>
    std::shared_ptr<Tensor<T>> TransformerTransducer<T>::forwardContext(const std::shared_ptr<Tensor<int>>& input,
                                                                        const std::shared_ptr<Tensor<T>>& encoding,
                                                                        int position) {
        std::shared_ptr<Tensor<T>> out = t_in_emb->forward(input);
        out = pos_encoding->forwardContext(out, position);
        for (int i = 0; i < num_layer; ++i) {
            out = decoder_layers[i]->forwardContext(out);
        }

        std::shared_ptr<Tensor<T>> enc_out = enc_proj->forward(encoding);
        out = prd_proj->forward(out);

        const int enc_rows = enc_out->shape(0);
        const int enc_cols = enc_out->shape(1);
        const int prd_cols = out->shape(1);
        auto concat_out = std::make_shared<Tensor<T>>();
        concat_out->value.resize(enc_rows, enc_cols + prd_cols);
        concat_out->value.leftCols(enc_cols).noalias() = enc_out->value;
        // broadcast: prediction [1, prd_cols] → [enc_rows, prd_cols]
        concat_out->value.rightCols(prd_cols) = out->value.replicate(enc_rows, 1);

        out = fc1->forward(concat_out);
        out = Tanh().forward(out);
        out = fc2->forward(out);
        return out;
    }

    template class TransformerTransducer<float>;
} // namespace MERCURY
