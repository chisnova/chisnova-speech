#ifndef MECURY_ASR_STREAM_CONFORMER
#define MECURY_ASR_STREAM_CONFORMER
#include "time.h"
#include <string>
#include <memory>
#include <vector>
#include <queue>

#include "tensor/tensor.h"
#include "layer/softmax.h"
#include "layer/linear.h"
#include "layer/layernorm.h"
#include "layer/embedding.h"
#include "json_loader/json_object.h"
#include "module/conformer_stream_encoder.h"
#include "module/conformer_ctc.h"
#include "module/transformer_decoder.h"
#include "module/decoder_cache.h"
#include "beam/beam.h"
#include "decoder/asr_model.h"

// AsrStreamConformer: model/app/asr_stream_conformer.py (AsrStreamConformer) 의 추론용 C++ 구현.
//
// 2-pass cascade 구조:
//   1패스  stream_encoder(청크 스트리밍) -> ctc_linear1   : 청크마다 바로 나오는 partial 결과
//   2패스  second_encoder(full-context)  -> ctc_linear2   : 발화 끝에서 한 번 (avg_pool 시간 축소 포함)
//         wrd_decoder(attention beam search)              : enc2 를 보고 최종 결과
//
// second_encoder 는 Python ConformerEncoder(ln1 -> pos -> layers -> ln2)와 구조가 같은
// ConformerCTC 모듈을 재사용한다. ctc_time_reduction > 1 이면 second_encoder 입력을
// avg_pool(ceil) 로 먼저 줄인다 — Python forward 와 같은 식.

namespace MERCURY {
    template<typename T>
    class AsrStreamConformer : public AsrModel<T> {
    public:
        AsrStreamConformer(JsonObject asr_stream_conformer_config, std::string base_path);
        ~AsrStreamConformer();
        void clear();

        // AsrModel 인터페이스
        std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<int>> dec, std::shared_ptr<Tensor<T>> enc);
        std::vector<Beam> makeInitBeams();
        std::vector<Score> getNbest(std::shared_ptr<Tensor<T>> logits, int num_nbest);
        std::vector<Beam> run(std::shared_ptr<Tensor<T>> feature, int num_nbest, std::vector<Beam> beams);
        void warmUpInference(std::shared_ptr<Tensor<T>> feature);
        std::vector<int> getCtcAsrIds();                     // 2패스: ctc_linear2(enc2) greedy
        std::vector<int> secondPassDecoding(int num_nbest);  // wrd_decoder attention beam search
        std::shared_ptr<Tensor<T>> getWordScore(std::vector<int> word_dec);
        std::shared_ptr<Tensor<T>> getPhoneScore(std::vector<int> phone_dec);

        // 스트리밍용: 지금까지 누적된 enc1 에 대한 ctc_linear1 greedy (partial 결과)
        std::vector<int> getFirstPassCtcAsrIds();
        std::shared_ptr<Tensor<T>> getEncoderOutputForDiag() { return enc_out_val; }

        // 청크 하나를 stream_encoder 로 인코딩해 enc1 누적 버퍼에 붙인다
        std::shared_ptr<Tensor<T>> encoding(std::shared_ptr<Tensor<T>> feature);

        // ctc_time_reduction 만큼 시간축 avg_pool (ceil_mode) — second_encoder 입력용
        std::shared_ptr<Tensor<T>> reduceEncodingTimeResolution(std::shared_ptr<Tensor<T>> encoded_frames);

        // enc2 가 아직 없으면 second_encoder 를 한 번 돌려 캐시한다
        void encodeSecondPassIfNeeded();

        // 모듈 (Python 모델의 같은 이름 속성과 1:1 대응)
        std::shared_ptr<ConformerStreamEncoder<T>> stream_encoder;
        std::shared_ptr<ConformerCTC<T>> second_encoder;
        std::shared_ptr<Linear<T>> ctc_linear1;
        std::shared_ptr<Linear<T>> ctc_linear2;

        // wrd_decoder: tin_emb -> pos/layers(TransformerDecoder) -> layer_norm -> tout_emb
        std::shared_ptr<Embedding> word_token_embedding;
        std::shared_ptr<TransformerDecoder<T>> word_decoder;
        std::shared_ptr<LayerNorm<T>> word_decoder_norm;
        std::shared_ptr<Linear<T>> word_output_linear;

        std::shared_ptr<Embedding> phone_token_embedding;
        std::shared_ptr<TransformerDecoder<T>> phone_decoder;
        std::shared_ptr<LayerNorm<T>> phone_decoder_norm;
        std::shared_ptr<Linear<T>> phone_output_linear;

        std::shared_ptr<Tensor<T>> enc_out_val;   // 누적된 1패스 인코딩 (enc1)
        std::shared_ptr<Tensor<T>> enc2_out_val;  // 2패스 인코딩 캐시 (enc2)

        int ctc_time_reduction;
        int num_layer;   // wrd_decoder 레이어 수 — Beam cache 초기화용
        int sos;
        int eos;
    };
};

#endif
