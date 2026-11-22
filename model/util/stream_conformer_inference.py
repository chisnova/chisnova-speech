"""AsrStreamConformer 추론 진입점.

탐색 로직은 전부 InferenceWrapper 가 갖고, 이 클래스는 그 앞뒤(특징 추출 ->
청크 스트리밍 인코딩 -> 토큰 id 를 텍스트로)만 담당한다.

RNN-T 가 없는 모델이라 디코딩 경로가 transducer 계열과 다르다:
  ctc1  스트림 인코더 직후의 CTC — 청크마다 바로 나오는 partial 결과
  ctc2  cascade 인코더(second_encoder) 뒤의 CTC — 발화 끝에서 prefix beam search
  attn  wrd_decoder 로 enc2 를 보고 하는 label-synchronous beam search (최종 결과)

주의: attention 탐색은 반드시 enc2(second_encoder 출력)에 걸어야 한다 —
학습에서 wrd_decoder 가 cross-attention 으로 본 게 enc2 이기 때문이다.

실행:
    PYTHONPATH=$PWD python3 util/stream_conformer_inference.py \
        --model_path stream_conformer --wav_path sample.wav
"""
import argparse
import json
import os

import torch
from torch.nn import functional as F

from app.asr_stream_conformer import AsrStreamConformer
from util.fbank_feature_extractor import FbankFeatureExtractor
from util.inference_wrapper import InferenceWrapper


class StreamConformerInference:
    SILENCE_PAD_MS = 300
    WORD_PIECE_PREFIX = "▁"                     # sentencepiece 의 단어 시작 표시
    SPECIAL_TOKENS = ("<pad>", "<unk>", "<s>", "</s>")
    DEFAULT_VOCAB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "data", "mercury.vocab")

    def __init__(self, model, vocab_tokens, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.vocab_tokens = vocab_tokens
        self.wrapper = InferenceWrapper(self.model)
        self.feature_extractor = FbankFeatureExtractor()

    @classmethod
    def from_model_path(cls, model_path, device="cpu", vocab_path=None):
        """모델 디렉토리(config.json + asr_model.pt)로 추론기를 만든다."""
        with open(os.path.join(model_path, "config.json"), 'r') as config_file:
            model_config = json.load(config_file)["model"]

        model = AsrStreamConformer.from_config(model_config)
        checkpoint_path = os.path.join(model_path, "asr_model.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError("체크포인트가 없습니다: {}".format(checkpoint_path))
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print("ASR model loaded: {}".format(checkpoint_path))

        resolved_vocab_path = vocab_path or model_config.get("vocab_path") or cls.DEFAULT_VOCAB_PATH
        return cls(model, cls.load_vocab_tokens(resolved_vocab_path), device=device)

    @staticmethod
    def load_vocab_tokens(vocab_path):
        """sentencepiece .vocab (토큰\\t점수) 을 id -> 토큰 리스트로 읽는다."""
        with open(vocab_path, 'r') as vocab_file:
            return [line.split('\t')[0].strip() for line in vocab_file if line.strip()]

    def decode_token_ids_to_text(self, token_ids):
        """토큰 id 열 -> 텍스트. 특수 토큰은 버리고 ▁ 는 공백으로 되돌린다."""
        pieces = [self.vocab_tokens[token_id] for token_id in token_ids
                  if 0 <= token_id < len(self.vocab_tokens)]
        pieces = [piece for piece in pieces if piece not in self.SPECIAL_TOKENS]
        return "".join(pieces).replace(self.WORD_PIECE_PREFIX, " ").strip()

    def make_batched_feature(self, feature):
        """(T, 80) -> (1, T, 80), 모델 디바이스로."""
        return feature.unsqueeze(0).to(self.device)

    def build_utterance_feature(self, wav_path):
        """발화 전체의 특징을 한 번에 뽑고 뒤에 무음을 붙인다.

        특징은 발화 단위로 계산한다 — 청크마다 따로 뽑으면 melspectrogram 의
        center 패딩이 청크 경계마다 끼어들어 학습 때 본 특징과 달라진다.
        무음 패딩이 없으면 마지막 프레임들이 conv subsample 의 오른쪽 컨텍스트를
        못 채워 끝 단어가 잘린다.
        """
        feature = self.feature_extractor.extract_from_wav_file(wav_path)
        silence_feature = self.feature_extractor.extract_silence(self.SILENCE_PAD_MS)
        return torch.cat((feature, silence_feature), dim=0)

    @torch.no_grad()
    def encode_feature_in_chunks(self, feature, chunk_duration_ms):
        """특징을 청크로 잘라 순서대로 스트림 인코딩한다.

        chunk_duration_ms 가 None 이면 통짜로 한 번에 인코딩한다(비스트리밍).

        Returns: (enc1, partial_texts)
          enc1          : (1, T', D) 발화 전체의 스트림 인코더 출력
          partial_texts : 청크마다 누적된 ctc1 greedy 결과 (스트리밍 partial)
        """
        self.model.clear()
        if chunk_duration_ms is None:
            chunk_frame_count = feature.size(0)
        else:
            chunk_frame_count = self.feature_extractor.frames_per_chunk(chunk_duration_ms)

        encoded_chunks = []
        partial_texts = []
        partial_token_ids = []
        for start in range(0, feature.size(0), chunk_frame_count):
            chunk_feature = feature[start:start + chunk_frame_count]
            chunk_enc = self.model.enc_stream(self.make_batched_feature(chunk_feature))
            if chunk_enc.size(1) == 0:          # 남은 프레임이 subsample 커널보다 짧은 경우
                continue
            encoded_chunks.append(chunk_enc)

            partial_token_ids += self.wrapper.ctc_greedy_search(chunk_enc, head="ctc1")
            partial_texts.append(self.decode_token_ids_to_text(partial_token_ids))

        return torch.cat(encoded_chunks, dim=1), partial_texts

    @torch.no_grad()
    def transcribe_wav_file(self, wav_path, chunk_duration_ms=100, beam_size=5):
        """wav 하나를 스트리밍으로 인식한다.

        Returns: {"ctc1_partials", "ctc1_text", "ctc2_text", "attention_text"}
        """
        feature = self.build_utterance_feature(wav_path)
        enc1, partial_texts = self.encode_feature_in_chunks(feature, chunk_duration_ms)

        # cascade 인코더는 발화 끝에서 한 번만 돌린다 — ctc2 와 attention 이 같이 쓴다.
        enc2 = self.model.second_encoder(enc1)
        ctc2_log_probs = F.log_softmax(self.model.ctc_linear2(enc2), dim=-1)
        ctc2_beams = self.wrapper.ctc_prefix_beam_search(ctc2_log_probs[0], beam_size=beam_size)

        attention_beams = self.wrapper.attention_beam_search(enc2, [self.wrapper.initBeam()],
                                                            beam_size=beam_size)
        return {
            "ctc1_partials": partial_texts,
            "ctc1_text": partial_texts[-1] if partial_texts else "",
            "ctc2_text": self.decode_token_ids_to_text(ctc2_beams[0]["context"]) if ctc2_beams else "",
            "attention_text": self.decode_token_ids_to_text(attention_beams[0]["context"]),
        }

    def transcribe_wav_files(self, wav_paths, chunk_duration_ms=100, beam_size=5):
        """목록 인식. 한 파일이 실패해도 나머지는 계속 돌린다(빈 결과로 자리를 채운다)."""
        results = []
        for wav_index, wav_path in enumerate(wav_paths):
            try:
                results.append(self.transcribe_wav_file(wav_path,
                                                        chunk_duration_ms=chunk_duration_ms,
                                                        beam_size=beam_size))
            except Exception as transcribe_error:
                print("[ERROR] {} 인식 실패: {}".format(wav_path, transcribe_error))
                results.append({"ctc1_partials": [], "ctc1_text": "",
                                "ctc2_text": "", "attention_text": ""})
            print("[{}/{}] {}".format(wav_index + 1, len(wav_paths), wav_path))
        return results


LIST_FILE_EXTENSIONS = (".txt", ".list")


def readListFile(list_path):
    with open(list_path, 'r') as list_file:
        return [line.strip() for line in list_file if line.strip()]


def resolveWavPaths(args):
    """인식할 오디오 경로 목록과 "목록 입력인가" 여부.

    --wav_path 는 오디오 파일 하나, --wav_list 는 목록 파일이다. 목록을 --wav_path
    로 주면 librosa 가 .txt 를 오디오로 열려다 NoBackendError 로 죽으므로,
    그 전에 무엇을 써야 하는지 알려 주고 멈춘다.
    """
    if args.wav_list:
        return readListFile(args.wav_list), True
    if os.path.splitext(args.wav_path)[1].lower() in LIST_FILE_EXTENSIONS:
        raise SystemExit("[ERROR] --wav_path 는 오디오 파일 하나입니다. "
                         "목록 파일은 --wav_list 로 주세요: --wav_list {}".format(args.wav_path))
    return [args.wav_path], False


def printWordErrorRate(reference_texts, hypothesis_texts):
    """get_wer.py 와 같은 지표(1 - 평균 WER)를 그 자리에서 찍는다.

    비교 전에 소문자로 맞춘다 — vocab 에 대문자 토큰이 아예 없어서(모델은 항상
    소문자를 뱉는다) 대문자 정답과 그대로 비교하면 전 단어가 substitution 이 되어
    WER 이 무조건 1.0 이 나온다.
    """
    try:
        import jiwer
    except ImportError:
        print("[WARN] jiwer 가 없어 WER 을 건너뛴다 — get_wer.py 로 따로 계산하세요")
        return
    total_word_error_rate = 0.0
    for reference_text, hypothesis_text in zip(reference_texts, hypothesis_texts):
        reference_text = reference_text.lower().strip()
        hypothesis_text = hypothesis_text.lower().strip()
        word_error_rate = jiwer.wer(reference_text, hypothesis_text or " ")
        total_word_error_rate += word_error_rate
        print("  ref: {}".format(reference_text))
        print("  hyp: {}".format(hypothesis_text))
        print("  wer: {:.3f}".format(word_error_rate))
    print("Total Score: {:.2f}".format(
        (1 - total_word_error_rate / max(1, len(reference_texts))) * 100))


def parseArguments():
    parser = argparse.ArgumentParser(description="AsrStreamConformer 스트리밍 추론")
    parser.add_argument('--model_path', type=str, required=True,
                        help="config.json 과 asr_model.pt 가 있는 모델 디렉토리")
    parser.add_argument('--wav_path', type=str, default="", help="오디오 파일 하나")
    parser.add_argument('--wav_list', type=str, default="",
                        help="오디오 경로가 한 줄에 하나씩 든 목록 파일")
    parser.add_argument('--lab_list', type=str, default="",
                        help="정답 텍스트 목록 파일 (wav_list 와 같은 순서). 주면 WER 을 찍는다")
    parser.add_argument('--hyp_out', type=str, default="",
                        help="인식 결과를 저장할 파일 (get_wer.py 입력용)")
    parser.add_argument('--decode', type=str, default="attention",
                        choices=("attention", "ctc2", "ctc1"),
                        help="목록 모드에서 가설로 쓸 디코딩 결과")
    parser.add_argument('--vocab_path', type=str, default=None)
    parser.add_argument('--chunk_ms', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--device', type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--show_partial', action='store_true',
                        help="청크마다 나오는 ctc1 partial 결과를 전부 출력")
    args = parser.parse_args()
    if not args.wav_path and not args.wav_list:
        parser.error("--wav_path 또는 --wav_list 중 하나는 있어야 합니다")
    return args


if __name__ == "__main__":
    args = parseArguments()
    inference = StreamConformerInference.from_model_path(args.model_path, device=args.device,
                                                         vocab_path=args.vocab_path)

    wav_paths, is_list_input = resolveWavPaths(args)
    if is_list_input:
        results = inference.transcribe_wav_files(wav_paths, chunk_duration_ms=args.chunk_ms,
                                                 beam_size=args.beam_size)
        hypothesis_texts = [result["{}_text".format(args.decode)] for result in results]
        if args.hyp_out:
            with open(args.hyp_out, 'w') as hypothesis_file:
                hypothesis_file.write("\n".join(hypothesis_texts) + "\n")
            print("saved: {}".format(args.hyp_out))
        if args.lab_list:
            printWordErrorRate(readListFile(args.lab_list), hypothesis_texts)
    else:
        result = inference.transcribe_wav_file(wav_paths[0], chunk_duration_ms=args.chunk_ms,
                                               beam_size=args.beam_size)
        if args.show_partial:
            for partial_index, partial_text in enumerate(result["ctc1_partials"]):
                print("partial[{}]: {}".format(partial_index, partial_text))
        print("ctc1 result     : {}".format(result["ctc1_text"]))
        print("ctc2 result     : {}".format(result["ctc2_text"]))
        print("attention result: {}".format(result["attention_text"]))
