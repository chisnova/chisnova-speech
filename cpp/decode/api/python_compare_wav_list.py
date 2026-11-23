"""wav_list_compare(C++) 덤프를 Python 모델과 단계별로 대조한다.

    cd model && PYTHONPATH=$PWD python3 \
        ../cpp/decode/api/python_compare_wav_list.py <wav_local.list> <cpp_dump_dir> \
        [--print_transcripts] [--label_map <stem TAB 정답텍스트 파일>]

인식 결과 전문은 <cpp_dump_dir>/transcripts.txt 에 wav 별로 저장되며,
--label_map 을 주면 wav 마다 정답(ref) 을 첫 줄에 함께 보여준다.
--print_transcripts 를 주면 stdout 으로도 그대로 출력한다.

단계별 비교 (오차의 진입 지점을 분리하기 위해):
  feature  : C++ librosa vs Python librosa 의 fbank 최대 오차
  enc1     : C++ 특징을 그대로 Python 인코더에 넣었을 때의 1패스 인코딩 오차
             (특징 차이와 무관한 순수 모델 비교)
  logit    : ctc_linear2(enc2) softmax 전 logits 오차 — 요청한 logit 레벨 비교
  ctc text : C++ greedy ids vs Python greedy ids 의 전사 일치 여부
  attn text: C++ second pass(beam 3) vs Python attention beam(3) 전사 일치 여부
"""
import os
import json
import struct
import sys

import numpy as np
import torch

from util.asr_evaluator import AsrEvaluator
from util.fbank_feature_extractor import FbankFeatureExtractor
from util.inference_wrapper import InferenceWrapper
from util.stream_conformer_inference import StreamConformerInference

HOP_LENGTH = 160


def load_dumped_tensor(path):
    with open(path, "rb") as dump_file:
        rows, cols = struct.unpack("ii", dump_file.read(8))
        data = np.frombuffer(dump_file.read(rows * cols * 4), dtype=np.float32)
    return data.reshape(rows, cols)


def decode_ids_to_text(token_ids, vocab_tokens):
    text = "".join(vocab_tokens[i] for i in token_ids)
    text = text.replace("▁", " ")
    for special in ("-", "<unk>", "<s>", "</s>"):
        text = text.replace(special, "")
    return " ".join(text.split())


def collapse_repeats(token_ids):
    collapsed, previous_id = [], -1
    for token_id in token_ids:
        if token_id != previous_id:
            collapsed.append(token_id)
        previous_id = token_id
    return collapsed


def load_pronounce_scorer_assets(scorer_dir):
    """C++ Scorer 와 같은 자산 로드: spm / phone 인덱스 / 발음사전."""
    import sentencepiece
    with open(os.path.join(scorer_dir, "model.json")) as manifest:
        assets = json.load(manifest)["model_path"]
    spm_processor = sentencepiece.SentencePieceProcessor(
        model_file=os.path.join(scorer_dir, assets["vocab_model"]))
    phone_tokens = open(os.path.join(scorer_dir, assets["phone_name"]),
                        errors="replace").read().split()[:assets["phone_size"]]
    phone_index = {p: i for i, p in enumerate(phone_tokens)}
    pronounce_dict = {}
    for line in open(os.path.join(scorer_dir, assets["dict_name"]), errors="replace"):
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            pronounce_dict[parts[0].lower()] = parts[1]
    return spm_processor, phone_index, pronounce_dict


def refine_label_for_scoring(text):
    """C++ Scorer.refineLabel 과 동일: 소문자화 후 [a-z 공백] 만 남긴다."""
    return "".join(c for c in text.lower() if c.islower() and c.isascii() or c == " ")


def compute_pronounce_scores(model, enc2, refined_label, spm_processor, phone_index,
                             pronounce_dict, vocab_tokens):
    """C++ AsrAsyncDecoder.getPronounceScore + Scorer.getScore 재현.

    Returns (word_posterior, phone_posterior, word_avg, phone_avg, final_avg)
    또는 사전 OOV 단어가 있으면 None.
    """
    words = refined_label.split()
    if not words or any(w not in pronounce_dict for w in words):
        return None
    word_ids = [2] + spm_processor.encode(refined_label) + [3]
    phone_ids = [2]
    for word in words:
        phone_ids += [phone_index[p] for p in pronounce_dict[word].split()]
        phone_ids.append(4)                    # 단어 구분자
    phone_ids[-1] = 3                          # 마지막 구분자 -> eos

    def teacher_forcing_posterior(decoder, ids):
        dec_input = torch.tensor(ids[:-1], dtype=torch.long).view(1, -1)
        logits = decoder(dec_input, enc2)
        return torch.softmax(logits, dim=-1)[0]

    word_posterior = teacher_forcing_posterior(model.wrd_decoder, word_ids)
    phone_posterior = teacher_forcing_posterior(model.phn_decoder, phone_ids)

    word_targets = word_ids[1:]
    word_groups, group = [], -1
    for t in word_targets:
        if t == 3:
            word_groups.append(-1); continue
        if vocab_tokens[t].startswith("\u2581"):
            group += 1
        word_groups.append(group)
    phone_targets = phone_ids[1:]
    phone_groups, group = [], 0
    for t in phone_targets:
        phone_groups.append(-1 if t == 3 else group)
        if t == 4:
            group += 1

    word_scores = [[0.0, 0] for _ in words]
    for row, (t, g) in enumerate(zip(word_targets, word_groups)):
        if g < 0: continue
        word_scores[g][0] += float(word_posterior[row, t]) * 100.0
        word_scores[g][1] += 1
    phone_scores = [[0.0, 0] for _ in words]
    for row, (t, g) in enumerate(zip(phone_targets, phone_groups)):
        if g < 0 or t == 4: continue
        phone_scores[g][0] += float(phone_posterior[row, t]) * 100.0
        phone_scores[g][1] += 1

    word_avgs = [s / c for s, c in word_scores]
    phone_avgs = [s / c for s, c in phone_scores]
    phone_weight = 0.6
    final_per_word = [phone_weight * p + (1 - phone_weight) * w
                      for w, p in zip(word_avgs, phone_avgs)]
    return (word_posterior.numpy(), phone_posterior.numpy(),
            sum(word_avgs) / len(word_avgs), sum(phone_avgs) / len(phone_avgs),
            sum(final_per_word) / len(final_per_word))


def main():
    wav_list_path, cpp_dump_dir = sys.argv[1], sys.argv[2]
    print_transcripts = "--print_transcripts" in sys.argv[3:]
    scorer_assets = None
    if "--scorer_dir" in sys.argv:
        scorer_assets = load_pronounce_scorer_assets(sys.argv[sys.argv.index("--scorer_dir") + 1])
    reference_texts = {}
    reference_texts_raw = {}
    if "--label_map" in sys.argv:
        label_map_path = sys.argv[sys.argv.index("--label_map") + 1]
        for line in open(label_map_path, errors="replace"):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                reference_texts_raw[parts[0]] = parts[1]
                reference_texts[parts[0]] = AsrEvaluator.normalize_text_for_wer(parts[1])
    extractor = FbankFeatureExtractor()
    evaluator = AsrEvaluator.from_model_path(os.environ.get("MODEL_PATH", "stream_conformer"))
    model = evaluator.model
    model.eval()
    wrapper = InferenceWrapper(model)
    vocab_tokens = evaluator.vocab_tokens

    stats = {"feat": [], "enc1": [], "logit_max": [], "logit_mean": [],
             "wout": [], "pout": [], "final_score": []}
    scored_count = oov_count = 0
    transcript_lines = []
    ctc_text_matches = attn_text_matches = ctc_id_matches = 0
    mismatch_examples = []
    wav_paths = [line.strip() for line in open(wav_list_path) if line.strip()]

    with torch.no_grad():
        for wav_index, wav_path in enumerate(wav_paths):
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            cpp_features = load_dumped_tensor(os.path.join(cpp_dump_dir, stem + ".feat.bin"))
            cpp_enc1 = load_dumped_tensor(os.path.join(cpp_dump_dir, stem + ".enc1.bin"))
            cpp_logits = load_dumped_tensor(os.path.join(cpp_dump_dir, stem + ".logit.bin"))
            with open(os.path.join(cpp_dump_dir, stem + ".ids.txt")) as ids_file:
                cpp_ctc_ids = [int(t) for t in ids_file.readline().split()]
                cpp_attention_ids = [int(t) for t in ids_file.readline().split()]

            waveform = extractor.load_waveform(
                os.path.join(os.path.dirname(os.path.abspath(wav_list_path)), wav_path)
                if not os.path.isabs(wav_path) else wav_path)
            segments = list(StreamConformerInference.runtime_segments(waveform))

            # 1) feature 비교 (각자 자기 fbank)
            python_features = np.concatenate(
                [extractor.extract_from_waveform(seg).numpy() for seg in segments], axis=0)
            if python_features.shape == cpp_features.shape:
                stats["feat"].append(float(np.abs(python_features - cpp_features).max()))
            else:
                # 스테레오/비16kHz 등으로 프레임 수가 다르면 이후 비교가 전부 무의미 —
                # 이 wav 만 건너뛰고 배치는 계속한다
                print("[py-compare] skip {} — feature shape 불일치 (C++ {}, PY {})".format(
                    stem, cpp_features.shape, python_features.shape), flush=True)
                continue

            # 2) C++ 특징을 그대로 Python 모델에 넣어 순수 모델 오차 측정
            segment_frame_counts = [1 + len(seg) // HOP_LENGTH for seg in segments]
            model.clear()
            encoded_chunks = []
            offset = 0
            for frame_count in segment_frame_counts:
                segment_feature = torch.tensor(cpp_features[offset:offset + frame_count])
                offset += frame_count
                chunk_enc = model.enc_stream(segment_feature.unsqueeze(0))
                if chunk_enc.size(1) > 0:
                    encoded_chunks.append(chunk_enc)
            enc1 = torch.cat(encoded_chunks, dim=1)
            stats["enc1"].append(float((enc1[0].numpy() - cpp_enc1).__abs__().max()))

            enc2 = model.second_encoder(enc1)
            python_logits = model.ctc_linear2(enc2)[0].numpy()
            logit_diff = np.abs(python_logits - cpp_logits)
            stats["logit_max"].append(float(logit_diff.max()))
            stats["logit_mean"].append(float(logit_diff.mean()))

            # 3) 전사 비교
            python_ctc_ids = python_logits.argmax(axis=-1).tolist()
            ctc_id_matches += int(python_ctc_ids == cpp_ctc_ids)
            python_ctc_text = decode_ids_to_text(collapse_repeats(python_ctc_ids), vocab_tokens)
            cpp_ctc_text = decode_ids_to_text(collapse_repeats(cpp_ctc_ids), vocab_tokens)
            ctc_text_matches += int(python_ctc_text == cpp_ctc_text)

            attention_beams = wrapper.attention_beam_search(
                enc2, [wrapper.initBeam()], beam_size=3)
            python_attention_text = decode_ids_to_text(attention_beams[0]["context"], vocab_tokens)
            cpp_attention_text = decode_ids_to_text(cpp_attention_ids, vocab_tokens)
            attn_text_matches += int(python_attention_text == cpp_attention_text)

            if python_ctc_text != cpp_ctc_text or python_attention_text != cpp_attention_text:
                mismatch_examples.append((stem, cpp_ctc_text, python_ctc_text,
                                          cpp_attention_text, python_attention_text))

            # 발음 평가 score 재현·비교 (C++ .score.txt 가 있을 때)
            score_line = None
            cpp_score_path = os.path.join(cpp_dump_dir, stem + ".score.txt")
            if scorer_assets is not None and stem in reference_texts_raw \
                    and os.path.isfile(cpp_score_path):
                cpp_first_line = open(cpp_score_path).readline().split()
                refined_label = refine_label_for_scoring(reference_texts_raw[stem])
                python_scores = compute_pronounce_scores(
                    model, enc2, refined_label, *scorer_assets, vocab_tokens)
                if cpp_first_line[0] == "OOV" or python_scores is None:
                    oov_count += 1
                    score_line = "  score: 사전 OOV 단어 포함 — 평가 제외"
                else:
                    cpp_final = float(cpp_first_line[1])
                    (py_wout, py_pout, py_word_avg, py_phone_avg, py_final) = python_scores
                    cpp_wout = load_dumped_tensor(os.path.join(cpp_dump_dir, stem + ".wout.bin"))
                    cpp_pout = load_dumped_tensor(os.path.join(cpp_dump_dir, stem + ".pout.bin"))
                    stats["wout"].append(float(np.abs(py_wout - cpp_wout).max()))
                    stats["pout"].append(float(np.abs(py_pout - cpp_pout).max()))
                    stats["final_score"].append(abs(py_final - cpp_final))
                    scored_count += 1
                    if abs(py_final - cpp_final) < 0.05:
                        score_line = "  score= word {:.1f} / phone {:.1f} / final {:.1f}".format(
                            py_word_avg, py_phone_avg, py_final)
                    else:
                        score_line = ("  score C++: final {:.2f}\n"
                                      "  score PY : word {:.1f} / phone {:.1f} / final {:.2f}"
                                      .format(cpp_final, py_word_avg, py_phone_avg, py_final))

            # wav 별 인식 결과 기록 — 일치하면 한 줄(=), 다르면 양쪽 다 보여준다
            transcript_block_start = len(transcript_lines)
            transcript_lines.append("[{}]".format(stem))
            if stem in reference_texts:
                transcript_lines.append("  ref  : {}".format(reference_texts[stem]))
            if python_ctc_text == cpp_ctc_text:
                transcript_lines.append("  ctc  = {}".format(cpp_ctc_text))
            else:
                transcript_lines.append("  ctc  C++: {}".format(cpp_ctc_text))
                transcript_lines.append("  ctc  PY : {}".format(python_ctc_text))
            if python_attention_text == cpp_attention_text:
                transcript_lines.append("  attn = {}".format(cpp_attention_text))
            else:
                transcript_lines.append("  attn C++: {}".format(cpp_attention_text))
                transcript_lines.append("  attn PY : {}".format(python_attention_text))
            if score_line is not None:
                transcript_lines.extend(score_line.split("\n"))
            if print_transcripts:
                for line in transcript_lines[transcript_block_start:]:
                    print(line, flush=True)
            if (wav_index + 1) % 20 == 0:
                print("[py-compare] {}/{}".format(wav_index + 1, len(wav_paths)), flush=True)

    transcripts_path = os.path.join(cpp_dump_dir, "transcripts.txt")
    with open(transcripts_path, "w") as transcripts_file:
        transcripts_file.write("\n".join(transcript_lines) + "\n")

    total = len(wav_paths)
    if total == 0 or not stats["feat"]:
        print("\n비교할 wav 가 없습니다 — 목록({})과 리소스 폴더를 확인하세요.".format(wav_list_path))
        raise SystemExit(1)
    print("\n===== wav {}개 비교 결과 =====".format(total))
    print("인식 결과 전문: {}".format(transcripts_path))
    print("feature  max|diff|   : median {:.3e} / max {:.3e}".format(
        float(np.nanmedian(stats["feat"])), float(np.nanmax(stats["feat"]))))
    print("enc1     max|diff|   : median {:.3e} / max {:.3e}  (C++ 특징 공유 — 순수 모델)".format(
        float(np.median(stats["enc1"])), float(np.max(stats["enc1"]))))
    print("logit    max|diff|   : median {:.3e} / max {:.3e}".format(
        float(np.median(stats["logit_max"])), float(np.max(stats["logit_max"]))))
    print("logit    mean|diff|  : median {:.3e} / max {:.3e}".format(
        float(np.median(stats["logit_mean"])), float(np.max(stats["logit_mean"]))))
    if scored_count:
        print("발음 posterior wout max|diff|: median {:.3e} / max {:.3e}".format(
            float(np.median(stats["wout"])), float(np.max(stats["wout"]))))
        print("발음 posterior pout max|diff|: median {:.3e} / max {:.3e}".format(
            float(np.median(stats["pout"])), float(np.max(stats["pout"]))))
        print("종합 score |C++-PY| : median {:.4f} / max {:.4f}  (평가 {}건, OOV 제외 {}건)".format(
            float(np.median(stats["final_score"])), float(np.max(stats["final_score"])),
            scored_count, oov_count))
    print("ctc greedy id 완전일치: {}/{}".format(ctc_id_matches, total))
    print("ctc 전사 일치         : {}/{}".format(ctc_text_matches, total))
    print("attention 전사 일치   : {}/{}".format(attn_text_matches, total))
    if mismatch_examples:
        print("\n----- 전사 불일치 사례 ({}개) -----".format(len(mismatch_examples)))
        for stem, cpp_ctc, py_ctc, cpp_att, py_att in mismatch_examples[:10]:
            print("[{}]".format(stem))
            if cpp_ctc != py_ctc:
                print("  ctc  C++:", cpp_ctc)
                print("  ctc  PY :", py_ctc)
            if cpp_att != py_att:
                print("  attn C++:", cpp_att)
                print("  attn PY :", py_att)

    if mismatch_examples or len(stats["feat"]) != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
