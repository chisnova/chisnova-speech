#include "module/conformer_stream_encoder.h"

namespace MERCURY {
    // 생성자: feat_norm, SubSample, Linear, layer_norm1, N개의 StreamEncoderLayer 로드
    template<typename T>
    ConformerStreamEncoder<T>::ConformerStreamEncoder(JsonObject conformer_encoder_path, const std::string& base_path) {
        num_head = int(conformer_encoder_path.parserValue("num_head"));
        num_layer = int(conformer_encoder_path.parserValue("num_layer"));
        dim_model = int(conformer_encoder_path.parserValue("dim_model"));
        feat_norm = std::make_shared<LayerNorm<T>>(conformer_encoder_path.parseObject("feat_norm"), base_path);
        convsubsample = std::make_shared<ConformerSubSample<T>>(conformer_encoder_path.parseObject("feat_convsubsample"), base_path);
        feat_linear = std::make_shared<Linear<T>>(conformer_encoder_path.parseObject("feat_linear"), base_path);
        layer_norm1 = std::make_shared<LayerNorm<T>>(conformer_encoder_path.parseObject("layer_norm1"), base_path);
        auto layers_json = conformer_encoder_path.parseObject("layers");
        for (int i = 0; i < num_layer; ++i) {
            std::string object_name = std::to_string(i);
            auto layer = std::make_shared<ConformerStreamEncoderLayer<T>>(layers_json.parseObject(object_name.c_str()), base_path);
            conformer_stream_encoder_layers.push_back(std::move(layer));
        }
        rotary = RotaryEmbedding<T>(dim_model / num_head);
        stream_left_context = -1;   // 음수 = 무제한 (AsrStreamConformer 가 설정)
        position = 0;
    }

    template<typename T>
    void ConformerStreamEncoder<T>::setStreamLeftContext(int _stream_left_context) {
        stream_left_context = _stream_left_context;
    }

    // clear: 모든 streaming 캐시 초기화 및 position 리셋
    template<typename T>
    void ConformerStreamEncoder<T>::clear() {
        for (int i = 0; i < num_layer; ++i) {
            conformer_stream_encoder_layers[i]->clear();
        }
        convsubsample->clear();
        position = 0;
    }

    // forward: 청크 스트리밍 인코딩 (Python stream() 과 동일)
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerStreamEncoder<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        std::shared_ptr<Tensor<float>> output = feat_norm->forward(input);
        output = convsubsample->stream(output);
        if (output->value.rows() == 0) {
            return output;   // subsample 커널을 채울 만큼 아직 안 모였다
        }
        output = feat_linear->forward(output);
        output = layer_norm1->forward(output);

        // RoPE: 발화 시작부터 누적된 절대 위치의 cos/sin 슬라이스
        const int chunk_length = int(output->value.rows());
        rotary.ensureTableLength(position + chunk_length);
        std::shared_ptr<Tensor<T>> cos_chunk = std::make_shared<Tensor<T>>();
        std::shared_ptr<Tensor<T>> sin_chunk = std::make_shared<Tensor<T>>();
        cos_chunk->value = rotary.cos_table->value.middleRows(position, chunk_length);
        sin_chunk->value = rotary.sin_table->value.middleRows(position, chunk_length);
        position += chunk_length;

        for (int i = 0; i < num_layer; ++i) {
            output = conformer_stream_encoder_layers[i]->forward(output, cos_chunk, sin_chunk,
                                                                 stream_left_context);
        }
        return output;
    }

    template class ConformerStreamEncoder<float>;
} // namespace MERCURY
