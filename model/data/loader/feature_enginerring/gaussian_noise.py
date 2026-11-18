"""GaussianNoise — 가우시안 노이즈 주입 (마스킹과 분리된 독립 변환)."""
import torch

from .feature_transform import FeatureTransform, get_padding_row_mask


class GaussianNoise(FeatureTransform):
    """feat 에 N(0, std^2) 노이즈를 더한다. 패딩 프레임에는 더하지 않는다.

      std      : 노이즈 표준편차
      relative : True 면 std 를 발화별 표준편차 대비 비율로 해석한다.
                 정규화 전(dB, 편차 ~15)과 Cmvn 후(편차 1)에서 같은 세기를
                 유지하려면 이쪽이 안전하다.
      prob     : 적용 확률

    이전 SpecAugment 에 섞여 있던 노이즈를 떼어낸 것이다. 마스킹은 "정보를 지워
    문맥으로 복원하게" 하는 정칙화이고 노이즈는 "입력 섭동에 둔감하게" 만드는
    정칙화라 목적이 다르고, 세기와 확률도 따로 조절하는 편이 낫다.

    파이프라인에서는 마스킹 뒤에 두는 것을 권장한다 — 노이즈를 먼저 주면
    마스킹이 그 노이즈를 다시 지워서 효과가 줄어든다.
    """

    _PARAMS = ("std", "relative", "prob")

    def __init__(self, std=0.05, relative=False, prob=1.0):
        self.std = float(std)
        self.relative = bool(relative)
        self.prob = float(prob)

    def __call__(self, feat):
        if self.std <= 0.0:
            return feat
        if self.prob < 1.0 and float(torch.rand(())) >= self.prob:
            return feat

        valid = (~get_padding_row_mask(feat)).unsqueeze(-1).to(feat.dtype)  # (B,T,1)
        noise = torch.randn_like(feat)
        if self.relative:
            num_valid_values = (valid.sum(dim=(1, 2), keepdim=True)
                                * feat.shape[2]).clamp(min=1.0)
            mean = (feat * valid).sum(dim=(1, 2), keepdim=True) / num_valid_values
            mean_square = (feat * feat * valid).sum(dim=(1, 2),
                                                    keepdim=True) / num_valid_values
            noise.mul_((mean_square - mean * mean).clamp(min=0.0).sqrt())
        noise.mul_(valid)                                            # 패딩은 0 유지
        return feat.add_(noise, alpha=self.std)
