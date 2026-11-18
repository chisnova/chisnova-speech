"""Compose — 여러 FeatureTransform 을 순서대로 적용하는 파이프라인."""
from .feature_transform import FeatureTransform


class Compose(FeatureTransform):
    """변환들을 등록 순서대로 적용한다. 자신도 FeatureTransform 이라 중첩 가능.

        tf = Compose([Cmvn(), SpecSubstitute(prob=0.3), SpecAugment(prob=0.7)])
        feat = tf(feat)

    순서 권장: Cmvn -> SpecSubstitute -> SpecAugment -> GaussianNoise
      정규화를 먼저 해야 SpecAugment 의 마스크 값 0 이 '평균'이라는 의미를 갖고,
      노이즈를 마지막에 둬야 마스킹이 노이즈를 다시 지우지 않는다.
    """

    def __init__(self, transforms):
        self.transforms = list(transforms)
        for transform in self.transforms:
            if not isinstance(transform, FeatureTransform):
                raise TypeError(
                    "Compose 에는 FeatureTransform 만 넣을 수 있습니다: {!r}".format(transform))

    def __call__(self, feat):
        for transform in self.transforms:
            feat = transform(feat)
        return feat

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, index):
        return self.transforms[index]

    def __iter__(self):
        return iter(self.transforms)

    def __repr__(self):
        return "Compose([\n{}\n])".format(
            "\n".join("  " + repr(transform) for transform in self.transforms))
