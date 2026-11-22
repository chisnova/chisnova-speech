"""SpecAugment — 주파수/시간 마스킹 (Park et al., 2019)."""
import torch

from .feature_transform import (FeatureTransform, get_padding_row_mask,
                                make_span_mask, sample_random_span)


class SpecAugment(FeatureTransform):
    """발화마다 서로 다른 주파수/시간 구간을 마스킹한다.

      num_freq / freq_param : 주파수 마스크 개수 / 폭 상한 F (멜 빈, 폭은 0..F)
      num_time / time_param : 시간   마스크 개수 / 폭 상한 T (프레임, 폭은 0..T)
      time_ratio            : 시간 마스크 폭의 발화 길이 대비 상한 p — 짧은 발화가
                              통째로 지워지는 것을 막는다 (논문의 adaptive time masking)
      mask_value            : "mean" | "zero" | 실수
                              "mean" = 발화 평균값으로 덮는다 (정규화 안 된 dB 특징의
                              기본값. 0 은 dB 스케일에서 최댓값보다 큰 값이라
                              '지우기'가 아니라 '에너지 주입'이 된다)
                              "zero" = 0 으로 덮는다. Cmvn 을 먼저 태웠다면 이쪽.
      prob                  : 적용 확률

    이전 구현과 달라진 점
      - 마스크가 배치 전체 공유가 아니라 발화별이다. 공유하면 실효 증강 다양성이
        1/B 로 줄어 배치가 클수록 증강이 사실상 사라진다.
      - 시간 마스크가 패딩이 아니라 실제 발화 길이 안에서만 잡힌다.
      - 가우시안 노이즈는 GaussianNoise 로 분리했다.

    프리셋: SpecAugment.from_policy("LB"/"LD"/"SM"/"SS") — 논문 정책은 정규화된
    특징 기준이라 Cmvn 을 먼저 넣어야 한다.
    """

    _PARAMS = ("num_freq", "freq_param", "num_time", "time_param",
               "time_ratio", "mask_value", "prob")

    # 논문 Table 1 (LibriSpeech / Switchboard) — (F, mF, T, mT, p)
    POLICIES = {
        "LB": dict(freq_param=27, num_freq=1, time_param=100, num_time=1, time_ratio=1.0),
        "LD": dict(freq_param=27, num_freq=2, time_param=100, num_time=2, time_ratio=1.0),
        "SM": dict(freq_param=15, num_freq=2, time_param=70, num_time=2, time_ratio=0.2),
        "SS": dict(freq_param=27, num_freq=2, time_param=70, num_time=2, time_ratio=0.2),
    }

    def __init__(self, num_freq=2, num_time=2, freq_param=5, time_param=5,
                 time_ratio=0.2, mask_value="mean", prob=1.0):
        self.num_freq = int(num_freq)
        self.num_time = int(num_time)
        self.freq_param = int(freq_param)
        self.time_param = int(time_param)
        self.time_ratio = float(time_ratio)
        if isinstance(mask_value, str) and mask_value not in ("mean", "zero"):
            raise ValueError("mask_value 는 'mean' | 'zero' | 실수: {!r}".format(mask_value))
        self.mask_value = mask_value
        self.prob = float(prob)

    @classmethod
    def from_policy(cls, policy_name, **kw):
        """논문 정책 프리셋 — SpecAugment.from_policy('LD')."""
        policy_key = str(policy_name).upper()
        if policy_key not in cls.POLICIES:
            raise ValueError("알 수 없는 정책: {!r} ({})".format(
                policy_name, "/".join(cls.POLICIES)))
        return cls(**dict(cls.POLICIES[policy_key], **kw))

    def __call__(self, feat):
        if feat.dim() != 3:
            raise ValueError("feat 은 (B,T,D) 여야 합니다: {}".format(tuple(feat.shape)))
        if self.prob < 1.0 and float(torch.rand(())) >= self.prob:
            return feat

        B, T, D = feat.shape
        padding_rows = get_padding_row_mask(feat)          # (B,T)
        utt_lengths = (~padding_rows).sum(1)               # (B,)
        fill_value = self._compute_fill_value(feat, padding_rows)

        # 주파수 마스크 — 전 시간대에 걸리므로 (B,1,D) 로 broadcast
        freq_extent = torch.full((B,), D, dtype=torch.long, device=feat.device)
        for _ in range(self.num_freq):
            start, width = sample_random_span(freq_extent, self.freq_param)
            self._apply_mask_fill_(feat, make_span_mask(start, width, D).unsqueeze(1),
                                   fill_value)

        # 시간 마스크 — 발화 길이 안에서만, (B,T,1) 로 broadcast
        for _ in range(self.num_time):
            start, width = sample_random_span(utt_lengths, self.time_param,
                                              self.time_ratio)
            self._apply_mask_fill_(feat, make_span_mask(start, width, T).unsqueeze(-1),
                                   fill_value)

        if fill_value is not None:  # 주파수 마스크가 패딩에도 fill 을 썼으면 되돌린다
            feat.masked_fill_(padding_rows.unsqueeze(-1), 0.0)
        return feat

    # -- 내부 ---------------------------------------------------------------
    def _compute_fill_value(self, feat, padding_rows):
        """마스크로 덮을 값. "zero" 면 None(=0), "mean" 이면 발화 평균 (B,1,1)."""
        if self.mask_value == "zero":
            return None
        if self.mask_value != "mean":
            return float(self.mask_value)
        valid = (~padding_rows).unsqueeze(-1).to(feat.dtype)
        num_valid_values = (valid.sum(dim=(1, 2), keepdim=True)
                            * feat.shape[2]).clamp(min=1.0)
        return (feat * valid).sum(dim=(1, 2), keepdim=True) / num_valid_values

    @staticmethod
    def _apply_mask_fill_(feat, mask, fill_value):
        """mask(broadcast 가능) 위치를 fill_value 로 채운다 — in-place, 전체 재할당 없음."""
        feat.masked_fill_(mask, 0.0)
        if fill_value is not None:
            feat.add_(mask.to(feat.dtype) * fill_value)
        return feat
