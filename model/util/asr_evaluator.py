"""테스트셋 WER/WRR 평가기.

    PYTHONPATH=$PWD python3 util/asr_evaluator.py --model_path stream_conformer --sample 500 --device cpu
    PYTHONPATH=$PWD python3 util/asr_evaluator.py --model_path stream_conformer --device cuda

특징은 테스트 LMDB 레코드에서 읽는다(학습과 같은 파이프라인으로 이미 뽑혀 있다).
wav 에서 매번 다시 뽑으면 발화당 252ms 가 들어 34,203 개면 2.4 시간인데, 레코드에서
읽으면 그 비용이 0 이다. 정답 텍스트만 test_tot.lab 에서 가져온다.

디코딩 헤드별 비용이 크게 달라 취급이 다르다 (GPU 실측, 발화당).

    CTC1 (stream)    ~0 ms   인코더 출력에 linear 하나 — 항상 같이 잰다
    CTC2 (cascade)   5.5 ms  배치 argmax — 기본 헤드
    attention beam5   523 ms  발화마다 디코더를 수십 번 — CTC2 의 47배

전체 34,185 발화 기준 CTC 는 3분, attention 은 5시간이다. 그래서 attention 은
--attention_sample 로 소표본에만 건다(500 발화면 4.4분). 주 지표(top-level wrr)는
--primary 로 고르고 기본은 ctc2 다.
"""
import argparse
import json
import os
import random
import time
from datetime import datetime

import torch
from torch.nn import functional as F

from app.asr_stream_conformer import AsrStreamConformer
from data.loader import AsrDataLoader
from util.inference_wrapper import InferenceWrapper
from util.stream_conformer_inference import StreamConformerInference


class AsrEvaluator:
    """테스트 레코드를 배치로 흘려 WER / WRR 을 낸다.

    WRR = (1 - WER) * 100 — get_wer.py 가 "Total Score" 로 찍던 값과 같은 정의다.
    다만 WER 은 코퍼스 단위(전체 편집거리 / 전체 정답 단어 수)로 잡는다. 발화별 WER 을
    평균 내면 짧은 발화 하나가 전체를 흔들기 때문이다(단어 2개짜리 발화에서 1개 틀리면
    WER 0.5). 비교용으로 발화별 평균도 같이 기록한다.
    """

    DEFAULT_RECORD = "/home/chisnova/data/records/test"
    DEFAULT_LABEL_LIST = "/home/chisnova/archive1/ASR_DATA/asr_list/test_tot.lab"
    DEFAULT_WAV_LIST = "/home/chisnova/archive1/ASR_DATA/asr_list/test_tot.wlist"

    # 코퍼스별로 난이도가 완전히 다르다(읽기 낭독 vs 크라우드소싱 vs 유튜브).
    # 합쳐 놓은 WRR 하나로는 "무엇이 좋아졌는지" 를 알 수 없다.
    CORPUS_RULES = (("LibriSpeech/test-clean", "libri-clean"),
                    ("LibriSpeech/test-other", "libri-other"),
                    ("common_voice", "common_voice"),
                    ("giga", "giga_speech"))

    def __init__(self, model, vocab_tokens, device="cpu", chunk_size=64, left_chunks=16):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.vocab_tokens = vocab_tokens
        self.wrapper = InferenceWrapper(self.model)
        # 학습의 dynamic chunk 와 같은 receptive field 로 평가한다 — 스트리밍 루프 대신
        # 청크 마스크를 씌운 full-sequence forward 라 배치 matmul 한 번이면 끝난다.
        self.chunk_size = chunk_size
        self.left_chunks = left_chunks

    @classmethod
    def from_model_path(cls, model_path, device="cpu", vocab_path=None, **kwargs):
        with open(os.path.join(model_path, "config.json")) as config_file:
            model_config = json.load(config_file)["model"]
        model = AsrStreamConformer.from_config(model_config)
        checkpoint_path = os.path.join(model_path, "asr_model.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        vocab_file = (vocab_path or model_config.get("vocab_path")
                      or StreamConformerInference.DEFAULT_VOCAB_PATH)
        return cls(model, StreamConformerInference.load_vocab_tokens(vocab_file),
                   device=device, **kwargs)

    @staticmethod
    def normalize_text_for_wer(text):
        """WER 비교 전 정규화 — 학습 데이터가 배운 형태와 같게 만든다.

        data_gen.refineText 가 하이픈을 공백으로 바꿔 저장했으므로 모델은
        "twenty one" 이라 답한다. 정답의 "twenty-one" 을 그대로 두면 맞은 답이
        치환+삽입 2건으로 계산된다 (테스트셋 하이픈 단어 0.3% -> WER ~0.6%p 인플레).
        아포스트로피는 vocab 에 있어 모델이 낼 수 있으므로 건드리지 않는다.
        """
        return " ".join(text.lower().replace("-", " ").split())

    @staticmethod
    def load_reference_texts(label_list_path):
        """test_tot.lab (utt_id<TAB>text) -> {utt_id: text}."""
        references = {}
        with open(label_list_path, errors="replace") as label_file:
            for line in label_file:
                tokens = line.rstrip("\n").split("\t")
                if len(tokens) >= 2:
                    references[tokens[0].strip()] = tokens[1].strip()
        return references

    @classmethod
    def load_corpus_groups(cls, wav_list_path):
        """wlist 의 경로로 발화를 코퍼스에 묶는다 -> {utt_id: 코퍼스명}."""
        groups = {}
        if not wav_list_path or not os.path.isfile(wav_list_path):
            return groups
        with open(wav_list_path, errors="replace") as list_file:
            for line in list_file:
                tokens = line.rstrip("\n").split("\t")
                if len(tokens) < 2:
                    continue
                path = tokens[1]
                for needle, corpus_name in cls.CORPUS_RULES:
                    if needle in path:
                        groups[tokens[0].strip()] = corpus_name
                        break
        return groups

    def decode_token_ids_to_text(self, token_ids):
        pieces = [self.vocab_tokens[token_id] for token_id in token_ids
                  if 0 <= token_id < len(self.vocab_tokens)]
        pieces = [piece for piece in pieces
                  if piece not in StreamConformerInference.SPECIAL_TOKENS]
        return "".join(pieces).replace(StreamConformerInference.WORD_PIECE_PREFIX, " ").strip()

    @torch.no_grad()
    def decode_batch(self, feat, feat_mask, heads=("ctc2",), chunk_size=-1):
        """(B,T,80) -> {헤드: [텍스트...]}

        기본은 ctc2(cascade) 하나다 — 경향만 보면 되는 주기 평가에서 헤드를 늘릴
        이유가 없다. ctc1 을 넣어도 같은 인코더 forward 를 공유하므로 linear 하나
        만큼만 더 든다.
        """
        # chunk_size=None 이면 전체 문맥(비스트리밍) — 스트리밍이 얼마를 잃는지 재는 상한
        effective_chunk = self.chunk_size if chunk_size == -1 else chunk_size
        enc1 = self.model.stream_encoder(feat, chunk_size=effective_chunk,
                                         left_chunks=self.left_chunks)
        decoded = {}
        if "ctc1" in heads:
            enc1_lengths = self.compute_encoder_lengths(feat_mask, enc1.size(1))
            decoded["ctc1"] = self.greedy_decode_rows(self.model.ctc_linear1(enc1), enc1_lengths)
        if "ctc2" in heads:
            enc2 = self.model.second_encoder(enc1)
            enc2_lengths = self.model.make_enc2_mask(feat_mask, enc2.size(1)).sum(-1).long()
            decoded["ctc2"] = self.greedy_decode_rows(self.model.ctc_linear2(enc2), enc2_lengths)
        return decoded

    def compute_encoder_lengths(self, feat_mask, encoded_len, time_reduction=1):
        """conv subsample 2회(+time_reduction) 뒤의 발화별 유효 길이."""
        lengths = feat_mask.sum(-1).long()
        for _ in range(2):
            lengths = torch.div(lengths - 3, 2, rounding_mode='floor') + 1
        if time_reduction > 1:
            lengths = torch.div(lengths + time_reduction - 1, time_reduction,
                                rounding_mode='floor')
        return lengths.clamp(min=1, max=encoded_len)

    def greedy_decode_rows(self, logits, encoded_lengths):
        """(B,T,V) 로짓 -> 발화별 텍스트. 패딩 프레임이 argmax 에 끼면 뒤에 헛토큰이
        붙으므로 발화별 유효 길이까지만 본다."""
        token_ids = torch.argmax(logits, dim=-1).cpu()
        texts = []
        for row_index in range(token_ids.size(0)):
            valid_ids = token_ids[row_index, :int(encoded_lengths[row_index])].tolist()
            collapsed, previous = [], None
            for token_id in valid_ids:
                if token_id != previous and token_id != self.wrapper.blank_id:
                    collapsed.append(token_id)
                previous = token_id
            texts.append(self.decode_token_ids_to_text(collapsed))
        return texts

    @torch.no_grad()
    def decode_attention(self, feat, feat_mask, beam_size=5):
        """attention 디코더 beam search — 발화 하나씩 (배치가 안 된다).

        wrd_decoder 가 학습에서 본 것은 enc2 이므로 반드시 enc2 에 건다.
        """
        texts = []
        for row_index in range(feat.size(0)):
            valid_frames = int(feat_mask[row_index].sum())
            enc1 = self.model.stream_encoder(feat[row_index:row_index + 1, :valid_frames],
                                             chunk_size=self.chunk_size,
                                             left_chunks=self.left_chunks)
            enc2 = self.model.second_encoder(enc1)
            beams = self.wrapper.attention_beam_search(enc2, [self.wrapper.initBeam()],
                                                      beam_size=beam_size)
            texts.append(self.decode_token_ids_to_text(beams[0]["context"]))
        return texts

    def evaluate_record(self, record_path=None, label_list_path=None, sample_size=None,
                        batch_size=8, seed=None, max_feat_len=3000, progress_every=20,
                        attention_sample=0, beam_size=5, primary_head="ctc2",
                        heads=("ctc2",), wav_list_path=None, compare_full_context=False,
                        per_corpus_size=0):
        """레코드를 흘려 헤드별 WER/WRR 을 낸다.

        sample_size 를 주면 무작위 표본만 쓰고, attention_sample 을 주면 그 앞쪽
        N 발화에 한해 attention beam search 도 함께 잰다(발화당 500ms 이므로 소표본만).
        """
        record_path = record_path or self.DEFAULT_RECORD
        reference_texts = self.load_reference_texts(label_list_path or self.DEFAULT_LABEL_LIST)
        seed = random.randrange(10 ** 6) if seed is None else int(seed)

        corpus_groups = self.load_corpus_groups(wav_list_path or self.DEFAULT_WAV_LIST)
        # 층화 표본이면 어떤 발화를 볼지 먼저 정하고 로더에 그대로 넘긴다 —
        # 모든 헤드(ctc/attention)가 같은 발화를 보므로 헤드끼리도 그대로 비교된다.
        subset_utt_ids = (self.sample_utt_ids_per_corpus(record_path, corpus_groups,
                                                         per_corpus_size, seed)
                          if per_corpus_size else None)
        loader = self.build_loader(record_path, sample_size, batch_size, seed, max_feat_len,
                                   subset_utt_ids=subset_utt_ids)
        planned_utts = int(loader.data_size)   # 길이 필터를 통과해 실제로 돌 발화 수
        started_at = time.time()
        references, hypotheses = [], {head: [] for head in heads}
        corpora = []                                   # 발화별 코퍼스 (참조와 같은 순서)
        full_context_hypotheses = []                   # compare_full_context 일 때만
        attention_references, attention_hypotheses = [], []
        for batch_index, batch in enumerate(loader.iter_prefetched_batches(num_workers=2), 1):
            feat, _, _, feat_mask, _, _, utt_ids = batch
            decoded = self.decode_batch(feat, feat_mask, heads=heads)
            keep_rows = [row for row, utt_id in enumerate(utt_ids)
                         if str(utt_id) in reference_texts]   # 목록에 없는 발화는 버린다
            full_context = (self.decode_batch(feat, feat_mask, heads=(primary_head,),
                                              chunk_size=None)[primary_head]
                            if compare_full_context else None)
            for row in keep_rows:
                utt_id = str(utt_ids[row])
                references.append(self.normalize_text_for_wer(reference_texts[utt_id]))
                corpora.append(corpus_groups.get(utt_id, "기타"))
                for head in hypotheses:
                    hypotheses[head].append(
                        self.normalize_text_for_wer(decoded[head][row]) or " ")
                if full_context is not None:
                    full_context_hypotheses.append(
                        self.normalize_text_for_wer(full_context[row]) or " ")

            if attention_sample and len(attention_references) < attention_sample and keep_rows:
                # 배치는 길이순으로 묶여 있어 앞에서 그냥 자르면 특정 코퍼스/길이에 몰린다.
                # 배치마다 고르게 조금씩 걷어 전체에 흩어지게 한다.
                per_batch_quota = max(1, attention_sample // max(1, len(loader)))
                needed = min(attention_sample - len(attention_references), per_batch_quota)
                rows = keep_rows[:needed]
                attention_texts = self.decode_attention(feat[rows], feat_mask[rows],
                                                        beam_size=beam_size)
                attention_references.extend(
                    self.normalize_text_for_wer(reference_texts[str(utt_ids[row])])
                    for row in rows)
                attention_hypotheses.extend(self.normalize_text_for_wer(text) or " "
                                            for text in attention_texts)

            if progress_every and batch_index % progress_every == 0:
                print("  [eval] 배치 {} / 발화 {}".format(batch_index, len(references)))

        # 헤드마다 코퍼스별 breakdown 도 함께 남긴다 — primary 가 아닌 헤드(예:
        # full train 구간의 ctc1)도 코퍼스별 추세를 끊기지 않고 추적하기 위해서다.
        head_metrics = {head: dict(self.compute_metrics(references, texts),
                                   num_utts=len(references),
                                   corpora=self.compute_corpus_metrics(references, texts,
                                                                       corpora))
                        for head, texts in hypotheses.items()}
        if full_context_hypotheses:
            head_metrics["full_context"] = dict(
                self.compute_metrics(references, full_context_hypotheses),
                num_utts=len(references))
        if attention_references:
            head_metrics["attention"] = dict(
                self.compute_metrics(attention_references, attention_hypotheses),
                num_utts=len(attention_references))

        metrics = dict(head_metrics[primary_head])
        metrics.update({"primary_head": primary_head, "heads": head_metrics,
                        "error_patterns": self.collect_error_patterns(
                            references, hypotheses[primary_head]),
                        "corpora": self.compute_corpus_metrics(
                            references, hypotheses[primary_head], corpora),
                        "num_utts": len(references),
                        # 길이 상한/목록 불일치로 빠진 발화 — 조용히 줄어들면 안 되므로 남긴다
                        "excluded_utts": max(0, planned_utts - len(references)),
                        "max_feat_len": max_feat_len,
                        "elapsed_seconds": round(time.time() - started_at, 1),
                        "device": str(self.device), "seed": seed,
                        "sample_size": sample_size, "per_corpus_size": per_corpus_size,
                        "batch_size": batch_size})
        return metrics, references, hypotheses

    def sample_utt_ids_per_corpus(self, record_path, corpus_groups, per_corpus_size, seed):
        """코퍼스마다 같은 수를 뽑는다 (층화 표본).

        비율 표본으로 뽑으면 코퍼스 구성비가 그대로 반영돼(giga 58% / libri-clean 8%)
        작은 코퍼스는 표본이 수십 개밖에 안 잡힌다 — 그 숫자는 표본오차가 커서
        추세를 못 읽는다. 코퍼스마다 같은 수를 뽑으면 전부 같은 정밀도로 비교된다.

        대신 전체 WRR 은 실제 구성비가 아니라 균등 가중이 되므로, 코퍼스별 값과
        같이 봐야 한다(기록에 corpora 가 함께 남는 이유다).
        """
        grouped = {}
        with open(os.path.join(record_path, "manifest.jsonl"), errors="replace") as manifest_file:
            for line in manifest_file:
                line = line.strip()
                if not line:
                    continue
                utt_id = json.loads(line)["utt_id"]
                grouped.setdefault(corpus_groups.get(utt_id, "기타"), []).append(utt_id)

        rng = random.Random(seed)
        selected = []
        for corpus_name in sorted(grouped):
            candidates = grouped[corpus_name]
            selected.extend(rng.sample(candidates, min(per_corpus_size, len(candidates))))
        return selected

    def build_loader(self, record_path, sample_size, batch_size, seed, max_feat_len,
                     subset_utt_ids=None):
        """평가용 로더 — 증강 없음, 표본이면 shuffle 로 매번 다른 발화가 뽑힌다."""
        usage_ratio = 1.0
        if sample_size and subset_utt_ids is None:
            with open(os.path.join(record_path, "meta.json")) as meta_file:
                total_utts = int(json.load(meta_file).get("count", 0))
            if total_utts:
                usage_ratio = min(1.0, sample_size / float(total_utts))
        return AsrDataLoader(data_path=record_path, batch_size=batch_size,
                             shuffle=bool(sample_size) and subset_utt_ids is None,
                             isTrain=False, usage_ratio=usage_ratio,
                             subset_utt_ids=subset_utt_ids, max_feat_len=max_feat_len,
                             seed=seed, num_workers=2,
                             device=None if self.device.type == "cpu" else str(self.device))

    @classmethod
    def compute_corpus_metrics(cls, references, hypotheses, corpora):
        """코퍼스별 WRR — 어느 도메인이 좋아졌는지는 합산값으로 알 수 없다."""
        grouped = {}
        for reference, hypothesis, corpus_name in zip(references, hypotheses, corpora):
            bucket = grouped.setdefault(corpus_name, ([], []))
            bucket[0].append(reference)
            bucket[1].append(hypothesis)
        return {corpus_name: dict(cls.compute_metrics(group_refs, group_hyps),
                                  num_utts=len(group_refs))
                for corpus_name, (group_refs, group_hyps) in sorted(grouped.items())}

    @staticmethod
    def compute_metrics(references, hypotheses):
        """코퍼스 단위 WER 과 WRR. jiwer 가 없으면 자체 편집거리로 계산한다."""
        if not references:
            return {"wer": None, "wrr": None, "wer_utt_mean": None}
        try:
            import jiwer
            measures = jiwer.process_words(references, hypotheses)
            corpus_wer = float(measures.wer)
            utt_mean = sum(jiwer.wer(reference, hypothesis)
                           for reference, hypothesis in zip(references, hypotheses)) / len(references)
            detail = {"substitutions": measures.substitutions, "deletions": measures.deletions,
                      "insertions": measures.insertions, "hits": measures.hits}
        except ImportError:
            corpus_wer = AsrEvaluator.compute_corpus_wer(references, hypotheses)
            utt_mean, detail = None, {}
        # 가설/정답 단어수 비 — 1 에서 멀어지면 붕괴(짧아짐)나 반복(길어짐)이다
        reference_words = sum(len(text.split()) for text in references)
        hypothesis_words = sum(len(text.split()) for text in hypotheses)
        return dict({"wer": round(corpus_wer, 5), "wrr": round((1.0 - corpus_wer) * 100.0, 3),
                     "wer_utt_mean": None if utt_mean is None else round(utt_mean, 5),
                     "reference_words": reference_words, "hypothesis_words": hypothesis_words,
                     "length_ratio": round(hypothesis_words / max(1, reference_words), 4)},
                    **detail)

    @staticmethod
    def compute_corpus_wer(references, hypotheses):
        """jiwer 없을 때의 대체 경로 — 전체 편집거리 / 전체 정답 단어 수."""
        total_distance, total_words = 0, 0
        for reference, hypothesis in zip(references, hypotheses):
            reference_words, hypothesis_words = reference.split(), hypothesis.split()
            previous_row = list(range(len(hypothesis_words) + 1))
            for reference_index, reference_word in enumerate(reference_words, 1):
                current_row = [reference_index]
                for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, 1):
                    current_row.append(min(previous_row[hypothesis_index] + 1,
                                           current_row[hypothesis_index - 1] + 1,
                                           previous_row[hypothesis_index - 1]
                                           + (reference_word != hypothesis_word)))
                previous_row = current_row
            total_distance += previous_row[-1]
            total_words += len(reference_words)
        return total_distance / max(1, total_words)

    @staticmethod
    def collect_error_patterns(references, hypotheses, top_k=12, worst_k=5):
        """자주 틀리는 패턴 — 치환 쌍 / 삭제 단어 / 삽입 단어 상위와 최악 발화.

        WER 숫자는 "얼마나" 틀리는지만 말한다. "무엇을" 틀리는지(고유명사 치환인지,
        기능어 삭제인지)가 처방을 정하므로 평가 기록에 같이 남긴다.
        """
        try:
            import jiwer
        except ImportError:
            return {}
        from collections import Counter
        substitution_pairs, deleted_words, inserted_words = Counter(), Counter(), Counter()
        per_utterance = []
        measures = jiwer.process_words(references, hypotheses)
        for reference, hypothesis, chunks in zip(references, hypotheses,
                                                 measures.alignments):
            reference_words = reference.split()
            hypothesis_words = hypothesis.split()
            error_count = 0
            for chunk in chunks:
                if chunk.type == "substitute":
                    for ref_word, hyp_word in zip(
                            reference_words[chunk.ref_start_idx:chunk.ref_end_idx],
                            hypothesis_words[chunk.hyp_start_idx:chunk.hyp_end_idx]):
                        substitution_pairs[(ref_word, hyp_word)] += 1
                        error_count += 1
                elif chunk.type == "delete":
                    for ref_word in reference_words[chunk.ref_start_idx:chunk.ref_end_idx]:
                        deleted_words[ref_word] += 1
                        error_count += 1
                elif chunk.type == "insert":
                    for hyp_word in hypothesis_words[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                        inserted_words[hyp_word] += 1
                        error_count += 1
            per_utterance.append((error_count / max(1, len(reference_words)),
                                  reference, hypothesis))
        worst = sorted(per_utterance, key=lambda item: -item[0])[:worst_k]
        return {
            "substitutions": [[ref_word, hyp_word, count] for (ref_word, hyp_word), count
                              in substitution_pairs.most_common(top_k)],
            "deletions": [[word, count] for word, count in deleted_words.most_common(top_k)],
            "insertions": [[word, count] for word, count in inserted_words.most_common(top_k)],
            "worst_utts": [{"wer": round(utt_wer, 3), "ref": reference[:180],
                            "hyp": hypothesis[:180]}
                           for utt_wer, reference, hypothesis in worst],
        }

    @staticmethod
    def append_history(history_path, record):
        """평가 결과 한 줄을 wer_history.jsonl 에 덧붙인다."""
        with open(history_path, "a") as history_file:
            history_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return history_path


def parseArguments():
    parser = argparse.ArgumentParser(description="테스트셋 WER/WRR 평가")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--record', type=str, default=AsrEvaluator.DEFAULT_RECORD)
    parser.add_argument('--lab_list', type=str, default=AsrEvaluator.DEFAULT_LABEL_LIST)
    parser.add_argument('--sample', type=int, default=0, help="0 이면 전체")
    parser.add_argument('--per_corpus', type=int, default=0,
                        help="코퍼스마다 이 수만큼 뽑는다(층화 표본). 주면 --sample 은 무시")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_feat_len', type=int, default=3000,
                        help="이보다 긴 발화는 제외 (프레임, 100fps)")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--history', type=str, default=None,
                        help="결과를 덧붙일 jsonl (기본: <model_path>/wer_history.jsonl)")
    parser.add_argument('--show', type=int, default=5, help="예시로 출력할 발화 수")
    parser.add_argument('--attention_sample', type=int, default=0,
                        help="attention beam search 로도 잴 발화 수 (발화당 ~0.5s)")
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--primary', type=str, default="ctc2", choices=("ctc1", "ctc2"),
                        help="대표 지표로 쓸 헤드")
    parser.add_argument('--heads', type=str, default="ctc2",
                        help="CTC 헤드 목록 (쉼표 구분: ctc2 / ctc1,ctc2)")
    parser.add_argument('--wav_list', type=str, default=None, help="코퍼스 분류용 wlist")
    parser.add_argument('--full_context', action='store_true',
                        help="전체 문맥(비스트리밍)으로도 재서 스트리밍 손해를 본다")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArguments()
    evaluator = AsrEvaluator.from_model_path(args.model_path, device=args.device)
    metrics, references, hypotheses = evaluator.evaluate_record(
        record_path=args.record, label_list_path=args.lab_list,
        sample_size=args.sample or None, batch_size=args.batch_size, seed=args.seed,
        max_feat_len=args.max_feat_len, per_corpus_size=args.per_corpus,
        attention_sample=args.attention_sample, beam_size=args.beam_size,
        primary_head=args.primary, heads=tuple(args.heads.split(",")),
        wav_list_path=args.wav_list, compare_full_context=args.full_context)

    for index in range(min(args.show, len(references))):
        print("  ref : {}".format(references[index]))
        for head, texts in hypotheses.items():
            print("  {:<4}: {}".format(head, texts[index]))
    for head, head_metrics in metrics["heads"].items():
        print("[eval] {:<9} 발화 {:>6,} | WRR {:6.2f}% | WER {:.4f}".format(
            head, head_metrics.get("num_utts", metrics["num_utts"]),
            head_metrics["wrr"], head_metrics["wer"]))
    for corpus_name, corpus_metrics in metrics["corpora"].items():
        print("[eval] {:<14} 발화 {:>6,} | WRR {:6.2f}%".format(
            corpus_name, corpus_metrics["num_utts"], corpus_metrics["wrr"]))
    print("[eval] {:.1f}s on {} (대표 헤드 {}, 제외 {} 발화)".format(
        metrics["elapsed_seconds"], metrics["device"], metrics["primary_head"],
        metrics["excluded_utts"]))

    history_path = args.history or os.path.join(args.model_path, "wer_history.jsonl")
    AsrEvaluator.append_history(history_path, dict(
        metrics, at=datetime.now().replace(microsecond=0).isoformat(),
        scope="sample" if args.sample else "full"))
    print("[eval] 기록: {}".format(history_path))
