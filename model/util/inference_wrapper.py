"""Inference wrapper: 모델을 입력으로 받아 디코딩을 수행한다.

app/ 의 모델 클래스는 순수 모델(encoder/decoder/forward)만 갖고,
탐색(beam search) 로직은 전부 이 래퍼에 모은다.

지원 디코딩:
  - Transducer beam search   : transducer_stream_search (스트리밍, frame-synchronous)
  - Transducer greedy search : transducer_greedy_search (CTC blank-skipping 지원)
  - CTC prefix beam search   : ctc_beam_search / ctc_prefix_beam_search
  - CTC greedy search        : ctc_greedy_search
  - Attention decoder beam search : attention_beam_search / attention_partial_search
    (wrd_decoder를 이용한 label-synchronous 탐색, 2-pass rescoring 용)

모델 요구 인터페이스(duck typing):
  - enc_stream(feat) -> enc          : 스트리밍 인코더
  - clear()                          : 스트리밍 상태 초기화
  - decoder(dec, enc) -> (B,T,U,V), decoder.prediction / decoder.joint : transducer
  - ctc_linear1(enc) -> logits       : CTC head (blank-skip, ctc1 beam search)
  - second_encoder(enc), ctc_linear2 : cascade CTC head (ctc2 beam search)
  - wrd_decoder(dec, enc) -> logits  : attention decoder (attention beam search)
"""
import copy

import numpy as np
import torch
from torch.nn import functional as F


class InferenceWrapper:
    def __init__(self, model, blank_id=1, bos_id=2, eos_id=3):
        self.model = model
        self.model.eval()
        self.blank_id = blank_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.vocab_size = getattr(model, "vocab_size", None)

    def clear(self):
        self.model.clear()

    # ------------------------------------------------------------------
    # beam 상태 헬퍼
    # ------------------------------------------------------------------
    def initBeam(self):
        ret = {}
        ret["context"] = [self.bos_id]
        ret["score"] = 0.0
        ret["normscore"] = 0.0
        ret["isFinal"] = False
        return ret

    def initContextBeam(self, context):
        ret = {}
        ret["context"] = copy.deepcopy(context)
        ret["score"] = 0.0
        ret["normscore"] = 0.0
        ret["isFinal"] = False
        return ret

    def makeNewBeam(self, context, score, isFinal):
        ret = {}
        ret["context"] = context[:]
        ret["score"] = score
        ret["normscore"] = score / len(context)
        ret["isFinal"] = isFinal
        return ret

    def makeDecInput(self, token, device=None):
        token = torch.tensor(token, dtype=torch.long)
        token = token.view(1, -1)
        if device is not None:
            token = token.to(device)
        return token

    def removeDuplicated(self, beams):
        ret = []
        history = {}
        for beam in beams:
            key = " ".join(map(str, beam["context"]))
            if key in history:
                continue
            ret.append(beam)
            history[key] = True
        return ret

    def normScore(self, beams):
        ret = []
        for beam in beams:
            beam["score"] = beam["score"] / len(beam["context"])
            ret.append(beam)
        return ret

    # ------------------------------------------------------------------
    # Transducer beam search (frame-synchronous)
    # ------------------------------------------------------------------
    def isBreak(self, stop, worker, beam_size):
        count = 0
        best_score = worker[0]["normscore"]
        for i in range(len(stop)):
            if stop[i]["normscore"] > best_score:
                count += 1
        if count < beam_size:
            return False
        return True

    @torch.no_grad()
    def eachBeamSearch(self, beam, enc_in, beam_size):
        stop = []
        worker = []

        context = beam["context"]
        score = beam["score"]
        dec_in = self.makeDecInput(context, device=enc_in.device)

        prd = self.model.decoder(dec_in, enc_in)
        prd = prd[:, :, -1, :]
        prd = torch.flatten(prd)
        log_out = F.log_softmax(prd, dim=-1)

        stop.append(self.makeNewBeam(context, score + log_out[self.blank_id].item(), None))

        top_k_value, top_k_indexes = torch.topk(log_out[2:], beam_size)
        top_k_value = top_k_value.tolist()
        top_k_indexes = top_k_indexes.tolist()
        for i in range(len(top_k_indexes)):
            value = top_k_value[i]
            index = top_k_indexes[i] + 2
            worker.append(self.makeNewBeam(context + [index], score + value, None))
        return stop, worker

    def frameBeamSearch(self, worker, enc_in, beam_size, max_expand=10):
        """한 인코더 프레임에 대한 transducer beam 확장.
        max_expand: 프레임당 확장 횟수 상한 — isBreak 조건이 성립하지 않는
        비정상 상황에서 context가 무한 성장하는 것을 방지한다."""
        stop = []
        count = 0
        while True:
            count += 1
            beam_worker = []
            for beam in worker:
                each_stop, each_worker = self.eachBeamSearch(beam, enc_in, beam_size)
                stop = stop + each_stop
                beam_worker = beam_worker + each_worker
            worker = beam_worker[:]

            stop = sorted(stop, key=lambda x: x["normscore"], reverse=True)
            worker = sorted(worker, key=lambda x: x["normscore"], reverse=True)

            worker = worker[:beam_size]
            if self.isBreak(stop, worker, beam_size) or count >= max_expand:
                break

        stop = stop[:beam_size]
        return stop[:]

    @torch.no_grad()
    def ctc_keep_frames(self, enc, blank_threshold=0.95):
        """CTC posterior 기준으로 transducer 디코딩이 필요한 프레임만 선별.
        blank 확률이 threshold 이상인 프레임은 skip 대상(False)."""
        post = F.softmax(self.model.ctc_linear1(enc), dim=-1)
        keep = post[..., self.blank_id] < blank_threshold
        return keep

    @torch.no_grad()
    def transducer_stream_search(self, feat, beams, beam_size, blank_threshold=None):
        """스트리밍 transducer beam search. feat 청크를 인코딩하고 프레임 단위로
        beam을 확장한다. blank_threshold를 주면 CTC blank-skipping이 적용된다
        (blank 프레임의 beam 확장을 건너뛰어 디코딩 스텝 대폭 감소).
        Returns: (beams, enc)"""
        enc = self.model.enc_stream(feat)

        keep = None
        if blank_threshold is not None:
            keep = self.ctc_keep_frames(enc, blank_threshold)

        (bs, T, dim) = enc.size()
        worker = beams[:]
        for i in range(T):
            if keep is not None and not bool(keep[0, i]):
                continue
            enc_in = enc[:, i, :]
            enc_in = enc_in.view(bs, 1, dim)
            worker = self.frameBeamSearch(worker, enc_in, beam_size)

        return worker, enc

    @torch.no_grad()
    def transducer_greedy_search(self, enc, blank_threshold=0.95, max_sym_per_frame=3):
        """CTC blank-skipping greedy transducer 디코딩.
        enc: (1, T, D) 인코더 출력. 가장 빠른 1-pass 디코딩 경로."""
        (bs, T, dim) = enc.size()
        keep = None
        if blank_threshold is not None:
            keep = self.ctc_keep_frames(enc, blank_threshold)

        context = [self.bos_id]
        for t in range(T):
            if keep is not None and not bool(keep[0, t]):
                continue
            enc_t = enc[:, t:t + 1, :]
            emitted = 0
            while emitted < max_sym_per_frame:
                dec_in = self.makeDecInput(context, device=enc.device)
                prd = self.model.decoder.prediction(dec_in)[:, -1:, :]
                logits = self.model.decoder.joint(enc_t, prd)
                token = int(torch.argmax(logits.view(-1)))
                if token == self.blank_id:
                    break
                context.append(token)
                emitted += 1
        return context

    # ------------------------------------------------------------------
    # CTC beam search (prefix beam search)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ctc_log_probs(self, enc, head="ctc1"):
        """인코더 출력에서 CTC log posterior를 얻는다.
        head='ctc1': stream encoder 출력의 CTC head
        head='ctc2': CTC cascade encoder를 거친 두 번째 head"""
        if head == "ctc2":
            enc2 = self.model.second_encoder(enc)
            logits = self.model.ctc_linear2(enc2)
        else:
            logits = self.model.ctc_linear1(enc)
        return F.log_softmax(logits, dim=-1)

    def ctc_prefix_beam_search(self, log_probs, beam_size=5, prune_size=16):
        """표준 CTC prefix beam search (log domain).
        log_probs: (T, V) log posterior (torch tensor 또는 numpy)
        Returns: [{"context": [ids], "score": float}] score 내림차순."""
        if torch.is_tensor(log_probs):
            if log_probs.dim() == 3:
                log_probs = log_probs[0]
            lp = log_probs.detach().cpu().numpy()
        else:
            lp = np.asarray(log_probs)
        T, V = lp.shape
        NEG_INF = -float("inf")

        # prefix(tuple) -> [log P(prefix, 마지막이 blank), log P(prefix, 마지막이 non-blank)]
        beams = {(): [0.0, NEG_INF]}
        prune_size = min(max(prune_size, beam_size), V)

        for t in range(T):
            cand = np.argpartition(lp[t], -prune_size)[-prune_size:]
            new_beams = {}

            def entry(prefix):
                if prefix not in new_beams:
                    new_beams[prefix] = [NEG_INF, NEG_INF]
                return new_beams[prefix]

            for prefix, (pb, pnb) in beams.items():
                p_tot = np.logaddexp(pb, pnb)
                for v in cand:
                    p = lp[t, v]
                    if v == self.blank_id:
                        e = entry(prefix)
                        e[0] = np.logaddexp(e[0], p_tot + p)
                    elif len(prefix) > 0 and v == prefix[-1]:
                        # 같은 토큰 반복: blank 없이 이어지면 같은 prefix 유지,
                        # 직전이 blank였으면 새 토큰으로 확장
                        e = entry(prefix)
                        e[1] = np.logaddexp(e[1], pnb + p)
                        e2 = entry(prefix + (int(v),))
                        e2[1] = np.logaddexp(e2[1], pb + p)
                    else:
                        e2 = entry(prefix + (int(v),))
                        e2[1] = np.logaddexp(e2[1], p_tot + p)

            ranked = sorted(new_beams.items(),
                            key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
                            reverse=True)
            beams = dict(ranked[:beam_size])

        results = []
        for prefix, (pb, pnb) in beams.items():
            results.append({"context": list(prefix), "score": float(np.logaddexp(pb, pnb))})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    @torch.no_grad()
    def ctc_beam_search(self, enc, beam_size=5, head="ctc1", prune_size=16):
        """인코더 출력에 대한 CTC prefix beam search."""
        log_probs = self.ctc_log_probs(enc, head=head)
        return self.ctc_prefix_beam_search(log_probs[0], beam_size=beam_size, prune_size=prune_size)

    def ctc_greedy_from_log_probs(self, log_probs):
        """CTC greedy 디코딩 (argmax → 반복 제거 → blank 제거).
        log_probs: (T, V) 또는 (1, T, V). 이미 계산해 둔 posterior 를 재사용할 때 쓴다."""
        if log_probs.dim() == 3:
            log_probs = log_probs[0]
        ids = torch.argmax(log_probs, dim=-1).tolist()
        out = []
        prev = None
        for i in ids:
            if i != prev and i != self.blank_id:
                out.append(i)
            prev = i
        return out

    @torch.no_grad()
    def ctc_greedy_search(self, enc, head="ctc1"):
        """인코더 출력에 대한 CTC greedy 디코딩."""
        return self.ctc_greedy_from_log_probs(self.ctc_log_probs(enc, head=head))

    # ------------------------------------------------------------------
    # Attention decoder beam search (label-synchronous, 2-pass rescoring 용)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def attention_beam_search(self, enc, beams, beam_size, end_idx=None, max_len=None):
        """wrd_decoder를 이용한 label-synchronous beam search (구 model.infer)."""
        if end_idx is None:
            end_idx = self.eos_id
        (bs, Len, dim) = enc.size()
        if max_len is None:
            max_len = Len

        count = 0
        while count < max_len:
            count += 1
            run_beams = []
            beam_finish = True

            for beam in beams:
                context = beam["context"]
                score = beam["score"]
                isFinal = beam["isFinal"]

                if isFinal:
                    run_beams.append(beam)
                    continue
                beam_finish = False

                dec_context = self.makeDecInput(context, device=enc.device)
                logits = self.model.wrd_decoder(dec_context, enc)
                logits = logits.view(-1, logits.size(-1))[-1]
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs[0] = float("-inf")             # pad 는 후보에서 제외
                log_probs[self.blank_id] = float("-inf")  # blank 도 제외 (id 는 설정값)
                values, indexes = torch.topk(log_probs, beam_size)
                values = values.tolist()
                indexes = indexes.tolist()

                for value, idx in zip(values, indexes):
                    new_final = (idx == end_idx)
                    run_beams.append(self.makeNewBeam(context + [idx], score + value, new_final))

            # raw 누적 log prob(= 문장 확률의 log) 정렬. 2026-08 스코어링 3-way 실측
            # (34k 테스트셋 표본 1,600발화, beam5)에서 전 corpus 일관되게
            #   raw Σlog p > 확률 산술평균(normscore) > log p 기하평균
            # 순이었다 — 길이 정규화는 과잉 생성을 부추겨 오히려 WRR 을 깎는다.
            # C++ 엔진(beam.h)도 같은 기준으로 맞춰져 있다.
            beams = sorted(run_beams, key=lambda x: x["score"], reverse=True)[:beam_size]
            if beam_finish:
                break
        return beams

    @torch.no_grad()
    def attention_partial_search(self, enc, context, end_idx, beam_size):
        """주어진 context에서 시작해 end_idx까지 이어 붙이는 부분 탐색 (구 model.partial_infer)."""
        beams = [self.initContextBeam(context)]
        return self.attention_beam_search(enc, beams, beam_size, end_idx=end_idx)

    # ------------------------------------------------------------------
    # 구 모델 인터페이스와의 호환 alias
    # ------------------------------------------------------------------
    stream_infer = transducer_stream_search
    greedy_infer = transducer_greedy_search
    infer = attention_beam_search
    partial_infer = attention_partial_search
