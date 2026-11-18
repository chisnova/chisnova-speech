"""SpecSubstitute — 시간축 구간 치환 (WeNet 의 spec_sub)."""
import torch

from .feature_transform import (FeatureTransform, get_padding_row_mask,
                                make_span_mask, sample_random_span)


class SpecSubstitute(FeatureTransform):
    """구간 [t0, t0+w) 를 같은 발화의 앞쪽 [t0-s, t0-s+w) 로 덮어쓴다.

      num_sub   : 치환 구간 개수
      max_width : 구간 폭 상한 (프레임)
      max_shift : 얼마나 앞에서 가져올지 상한 (None 이면 max_width 와 동일)
      max_ratio : 폭의 발화 길이 대비 상한 (짧은 발화 보호)
      prob      : 적용 확률

    마스킹과의 차이 — 마스킹은 구간을 상수로 덮어 정보를 지우는 반면 이건 실제
    스펙트럼 값으로 덮는다. 그래서 입력 분포가 유지되고, 대신 시간 정렬이
    국소적으로 어긋난다. 모델이 프레임 단위 정렬에 과하게 의존하는 것을 막아 주며
    특히 스트리밍/청크 기반 인코더에서 효과가 있다. SpecAugment 와 성격이 겹치지
    않아 같이 써도 된다 (prob 0.3 정도부터).

    구현은 시간축 gather 한 번 — 구간 개수와 무관하게 전체 복사 1회다.
    """

    _PARAMS = ("num_sub", "max_width", "max_shift", "max_ratio", "prob")

    def __init__(self, num_sub=3, max_width=30, max_shift=None, max_ratio=0.2, prob=1.0):
        self.num_sub = int(num_sub)
        self.max_width = int(max_width)
        self.max_shift = int(max_width if max_shift is None else max_shift)
        self.max_ratio = float(max_ratio)
        self.prob = float(prob)

    def __call__(self, feat):
        if feat.dim() != 3:
            raise ValueError("feat 은 (B,T,D) 여야 합니다: {}".format(tuple(feat.shape)))
        if self.prob < 1.0 and float(torch.rand(())) >= self.prob:
            return feat

        B, T, D = feat.shape
        utt_lengths = (~get_padding_row_mask(feat)).sum(1)

        # 시간축 인덱스를 만들어 두고, 치환 구간만 shift 만큼 당긴다
        time_index = torch.arange(T, device=feat.device).unsqueeze(0).expand(B, T).clone()
        for _ in range(self.num_sub):
            start, width = sample_random_span(utt_lengths, self.max_width, self.max_ratio)
            max_shift_per_utt = start.clamp(max=self.max_shift)  # 앞에서 가져오므로 start 이하
            shift = (torch.rand_like(max_shift_per_utt, dtype=torch.float)
                     * (max_shift_per_utt + 1)).long().clamp(max=max_shift_per_utt)
            substitute_mask = make_span_mask(start, width, T)     # (B,T)
            time_index = torch.where(substitute_mask,
                                     (time_index - shift.unsqueeze(1)).clamp(min=0),
                                     time_index)

        return feat.gather(1, time_index.unsqueeze(-1).expand(B, T, D))
