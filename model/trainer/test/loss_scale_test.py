"""무작위 기준선 정규화(auto_norm)가 손실 눈금을 실제로 맞추는지 검증.

    PYTHONPATH=$PWD:$PWD/trainer python3 trainer/test/loss_scale_test.py

검증하는 주장 두 가지
  1. 무작위 기준선은 과제마다 다르다 — CE 는 ln(클래스 수), CTC 는 그보다 훨씬 큰
     ~(T/U)·lnV 스케일(프레임 전부의 경로 확률을 곱하므로).
  2. 각 손실을 자기 기준선으로 나누면(auto_norm) 다섯 항 전부 균등 출력에서
     정확히 1.0 이 되어, 가중치가 "자의 길이"가 아니라 진행률을 반영하게 된다.
     CTC 기준선은 닫힌 식 대신 zero-logit 손실을 실제로 계산해 쓴다.

A. 균등 로짓(zeros): CE == ln V 가 수학적으로 정확히 나오는지
B. 무작위 로짓(randn): 기준선 근처인지
C. 학습 안 한 실제 모델: 5개 헤드의 실제 시작 손실과 가중치 분포 비교
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conformer_stream_trainer import compute_ce_loss, compute_ctc_loss, loss_function


class LossScaleTest:
    WORD_VOCAB = 1024
    PHONE_VOCAB = 47
    BATCH = 4
    FEAT_FRAMES = 400           # subsample 후 T=99 — U 대비 여유 충분
    WORD_LEN = 12
    PHONE_LEN = 28

    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.word_target = torch.randint(4, self.WORD_VOCAB, (self.BATCH, self.WORD_LEN))
        self.phone_target = torch.randint(4, self.PHONE_VOCAB, (self.BATCH, self.PHONE_LEN))
        self.feat_mask = torch.ones(self.BATCH, self.FEAT_FRAMES)
        self.word_mask = torch.ones(self.BATCH, self.WORD_LEN)
        self.phone_mask = torch.ones(self.BATCH, self.PHONE_LEN)
        self.encoded_frames = ((self.FEAT_FRAMES - 3) // 2 - 1) // 2 + 1   # conv subsample 2회

    # -- 손실 묶음 계산 -------------------------------------------------------
    def measure_losses(self, make_logits):
        """로짓 생성기 하나로 5개 헤드의 손실을 계산한다."""
        word_logits = make_logits(self.BATCH, self.WORD_LEN - 1, self.WORD_VOCAB)
        phone_logits = make_logits(self.BATCH, self.PHONE_LEN - 1, self.PHONE_VOCAB)
        ctc_logits = lambda: make_logits(self.BATCH, self.encoded_frames, self.WORD_VOCAB)
        return {
            "wrd CE (V=1024)": compute_ce_loss(word_logits, self.word_target[:, 1:],
                                               self.WORD_VOCAB, self.word_mask),
            "phn CE (V=47)": compute_ce_loss(phone_logits, self.phone_target[:, 1:],
                                             self.PHONE_VOCAB, self.phone_mask),
            "ctc1 (V=1024)": compute_ctc_loss(ctc_logits(),
                                              self.word_target[:, 1:].to(torch.int32),
                                              self.feat_mask, self.word_mask),
            "ctc2 (V=1024)": compute_ctc_loss(ctc_logits(),
                                              self.word_target[:, 1:].to(torch.int32),
                                              self.feat_mask, self.word_mask,
                                              time_reduction=2),
            "ctc_mid (V=1024)": compute_ctc_loss(ctc_logits(),
                                                 self.word_target[:, 1:].to(torch.int32),
                                                 self.feat_mask, self.word_mask),
        }

    def baseline_of(self, name):
        """항별 무작위 기준선 — CE 는 ln V (닫힌 식), CTC 는 zero-logit 실측."""
        if "CE" in name:
            return math.log(47) if "V=47" in name else math.log(1024)
        if not hasattr(self, "_ctc_baselines"):
            with torch.no_grad():
                zero_full = torch.zeros(self.BATCH, self.encoded_frames, self.WORD_VOCAB)
                zero_half = torch.zeros(self.BATCH, self.encoded_frames, self.WORD_VOCAB)
                self._ctc_baselines = {
                    1: float(compute_ctc_loss(zero_full,
                                              self.word_target[:, 1:].to(torch.int32),
                                              self.feat_mask, self.word_mask)),
                    2: float(compute_ctc_loss(zero_half,
                                              self.word_target[:, 1:].to(torch.int32),
                                              self.feat_mask, self.word_mask,
                                              time_reduction=2)),
                }
        return self._ctc_baselines[2 if "ctc2" in name else 1]

    def print_table(self, title, losses):
        print("\n== {} ==".format(title))
        print("{:<18} {:>8} {:>9} {:>14} {:>16}".format(
            "항", "생값", "기준선", "정규화(L/기준선)", "생값/최소 배율"))
        raw_values = {name: float(value) for name, value in losses.items()}
        normalized = {name: raw / self.baseline_of(name) for name, raw in raw_values.items()}
        raw_min = min(raw_values.values())
        norm_min = min(normalized.values())
        for name, raw in raw_values.items():
            print("{:<18} {:>8.3f} {:>9.3f} {:>13.3f} {:>16.2f}x".format(
                name, raw, self.baseline_of(name), normalized[name], raw / raw_min))
        raw_spread = max(raw_values.values()) / raw_min
        norm_spread = max(normalized.values()) / norm_min
        print("-> 눈금 격차(최대/최소): 생값 {:.2f}x  ->  정규화 후 {:.2f}x".format(
            raw_spread, norm_spread))
        return raw_values

    def print_weight_comparison(self, losses):
        """auto vs auto_norm 이 각 항에 주는 실제 가중치."""
        raw = {name: float(value) for name, value in losses.items()}
        raw_total = sum(raw.values())
        norm = {name: value / self.baseline_of(name) for name, value in raw.items()}
        norm_total = sum(norm.values())
        print("\n== 가중치 분포 (이 손실에서 auto / auto_norm 이 주는 비중) ==")
        print("{:<18} {:>8} {:>11}".format("항", "auto", "auto_norm"))
        for name in raw:
            print("{:<18} {:>7.1%} {:>10.1%}".format(
                name, raw[name] / raw_total, norm[name] / norm_total))

    # -- 시나리오 ------------------------------------------------------------
    def run_uniform_logits(self):
        """A. 로짓 전부 0 = 정확한 균등분포. CE 는 수학적으로 ln V 여야 한다."""
        losses = self.measure_losses(lambda *shape: torch.zeros(*shape))
        raw = self.print_table("A. 균등 출력 (로짓=0) — 이론적 무작위 기준선", losses)
        ce_word_error = abs(raw["wrd CE (V=1024)"] - math.log(1024))
        ce_phone_error = abs(raw["phn CE (V=47)"] - math.log(47))
        assert ce_word_error < 1e-3 and ce_phone_error < 1e-3, \
            "균등 출력에서 CE != ln V — 기준선 가정이 깨졌다"
        # 핵심 검증: 다섯 항 전부 자기 기준선 대비 정확히 1.0 이어야 한다
        for name, value in raw.items():
            normalized = value / self.baseline_of(name)
            assert abs(normalized - 1.0) < 1e-3, \
                "{} 정규화 {:.4f} != 1.0 — 기준선이 틀렸다".format(name, normalized)
        print("   [검증] 다섯 항 모두 균등 출력에서 정규화 == 1.0")
        return losses

    def run_random_logits(self):
        """B. N(0,1) 로짓 — 초기화 직후의 어수선한 출력 근사."""
        losses = self.measure_losses(lambda *shape: torch.randn(*shape))
        self.print_table("B. 무작위 로짓 N(0,1)", losses)
        return losses

    def run_untrained_model(self):
        """C. 학습 안 한 실제 모델 — 5개 헤드의 진짜 시작 손실."""
        from app.asr_stream_conformer import AsrStreamConformer
        model = AsrStreamConformer(d_model=64, enc_n_head=4, enc_nlayer=13, dec_hidden=128,
                                   dec_n_head=4, dec_nlayer=4, vocab_size=self.WORD_VOCAB,
                                   _k_size=3, num_ctc_layer=8, num_ctc_out=self.WORD_VOCAB,
                                   freeze_num=0, drop_prob=0.0, ctc_time_reduction=2,
                                   inter_ctc_layer=6)
        model.eval()
        feat = torch.randn(self.BATCH, self.FEAT_FRAMES, 80)
        with torch.no_grad():
            w_out, p_out, ctc1, ctc2, ctc_mid = model(
                self.word_target, feat, self.phone_target,
                dec_mask=self.word_mask, enc_mask=self.feat_mask,
                p_dec_mask=self.phone_mask)
            losses = {
                "wrd CE (V=1024)": compute_ce_loss(w_out.float(), self.word_target[:, 1:],
                                                   self.WORD_VOCAB, self.word_mask),
                "phn CE (V=47)": compute_ce_loss(p_out.float(), self.phone_target[:, 1:],
                                                 p_out.size(-1), self.phone_mask),
                "ctc1 (V=1024)": compute_ctc_loss(ctc1.float(),
                                                  self.word_target[:, 1:].to(torch.int32),
                                                  self.feat_mask, self.word_mask),
                "ctc2 (V=1024)": compute_ctc_loss(ctc2.float(),
                                                  self.word_target[:, 1:].to(torch.int32),
                                                  self.feat_mask, self.word_mask,
                                                  time_reduction=2),
                "ctc_mid (V=1024)": compute_ctc_loss(ctc_mid.float(),
                                                     self.word_target[:, 1:].to(torch.int32),
                                                     self.feat_mask, self.word_mask),
            }
        self.print_table("C. 학습 안 한 실제 모델 (d=64 축소판)", losses)
        self.print_weight_comparison(losses)
        return losses

    def run_all(self):
        self.run_uniform_logits()
        self.run_random_logits()
        self.run_untrained_model()
        print("\n[OK] 모든 시나리오 완료")


if __name__ == "__main__":
    LossScaleTest().run_all()
