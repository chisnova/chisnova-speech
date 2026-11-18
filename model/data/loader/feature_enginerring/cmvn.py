"""Cmvn — 평균/분산 정규화 (증강이 아니라 학습을 성립시키는 전처리)."""
import torch

from .feature_transform import FeatureTransform, get_padding_row_mask


class Cmvn(FeatureTransform):
    """발화별(또는 전역) 평균/분산 정규화. 패딩 프레임은 통계에서 빼고 0 으로 남긴다.

    data_gen.py 는 power_to_db 결과를 그대로 저장하므로 feat 은 대략 -100 ~ +5 dB
    범위의 정규화 안 된 값이다. 이 상태에서는

      - 스케일이 커서 학습 초기 gradient 가 불안정하고,
      - SpecAugment 의 마스크 값 0 이 '무음'이 아니라 '최댓값보다 큰 값'이 된다.

    Cmvn 을 먼저 태우면 평균이 0 이 되어 마스크 0 이 곧 중립값이 되고 두 문제가
    같이 사라진다. 파이프라인의 맨 앞에 두는 것을 권장한다.

      norm_means / norm_vars : 평균만 뺄지(CMN), 분산까지 나눌지(CMVN)
      mean / std : 전역 통계 (D,) 를 주면 발화별 통계 대신 이걸 쓴다.
                   학습/추론 통계를 고정해야 하는 스트리밍에서는 이쪽이 맞다.

    주의: 증강이 아니라 전처리다. train/dev/test 모두에 똑같이 적용해야 한다.
    """

    _PARAMS = ("norm_means", "norm_vars", "eps")

    def __init__(self, norm_means=True, norm_vars=True, mean=None, std=None, eps=1e-5):
        self.norm_means = bool(norm_means)
        self.norm_vars = bool(norm_vars)
        self.eps = float(eps)
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        self.std = None if std is None else torch.as_tensor(std, dtype=torch.float32)

    def __call__(self, feat):
        valid = (~get_padding_row_mask(feat)).unsqueeze(-1).to(feat.dtype)  # (B,T,1)

        if self.mean is not None or self.std is not None:           # 전역 통계
            def as_broadcastable(stat):
                if stat is None:
                    return None
                return stat.to(device=feat.device, dtype=feat.dtype).view(1, 1, -1)
            mean, std = as_broadcastable(self.mean), as_broadcastable(self.std)
        else:                                                       # 발화별 통계
            num_valid_frames = valid.sum(1, keepdim=True).clamp(min=1.0)
            mean = (feat * valid).sum(1, keepdim=True) / num_valid_frames
            mean_square = (feat * feat * valid).sum(1, keepdim=True) / num_valid_frames
            std = ((mean_square - mean * mean).clamp(min=0.0) + self.eps).sqrt()

        if self.norm_means and mean is not None:
            feat = feat - mean
        if self.norm_vars and std is not None:
            feat = feat / std
        return feat * valid                                          # 패딩은 다시 0
