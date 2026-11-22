"""특징 변환 패키지 — collate 된 배치 (B,T,D) 에 적용하는 전처리/증강.

모든 변환은 FeatureTransform 을 상속하고 인터페이스는 하나뿐이다:

    feat = transform(feat)

모양 (B,T,D) 은 바뀌지 않고, 패딩 프레임은 0 으로 유지된다. 난수는 torch 전역
RNG 를 쓰므로 재현이 필요하면 torch.manual_seed() 를 부르면 된다.

    Cmvn            평균/분산 정규화 (전처리 — train/dev/test 모두에 적용)
    SpecSubstitute  시간축 구간 치환 (WeNet spec_sub)
    SpecAugment     주파수/시간 마스킹 (Park et al., 2019) — 발화별 마스크
    SnrNoiseMix     실제 잡음을 목표 SNR 로 합성 (log-add) — 잡음 강건성
    GaussianNoise   가우시안 노이즈
    Compose         위를 순서대로 엮는 파이프라인

권장 조합 (순서가 중요하다 — 정규화가 먼저여야 마스크 값 0 이 중립값이 된다):

    from data.loader.feature_enginerring import (Cmvn, Compose, GaussianNoise,
                                                 SpecAugment, SpecSubstitute)
    train_tf = Compose([
        SnrNoiseMix.from_record(NOISE_RECORD, prob=0.5),   # dB 인 동안에만 가능
        Cmvn(),                                   # 항상
        SpecSubstitute(prob=0.3),                 # 증강
        SpecAugment.from_policy("SM", mask_value="zero", prob=0.7),
        GaussianNoise(std=0.05, prob=0.5),
    ])
    eval_tf = Cmvn()                              # 정규화만

순서가 중요한 두 번째 이유: SnrNoiseMix 는 feat 이 아직 dB 여야 파워로 되돌릴 수
있으므로 Cmvn 보다 앞에 와야 한다. 정규화 후에는 그 값이 더 이상 dB 가 아니다.

길이(T)를 바꾸는 변환(speed perturb, time warp)은 여기 두지 않는다 — 배치 예산과
feat_mask 가 이미 확정된 뒤라 모양을 바꿀 수 없고, 애초에 파형 단계(data_gen.py)
에서 3배 데이터로 만드는 쪽이 표준이고 효과도 크다.
"""
from .cmvn import Cmvn
from .compose import Compose
from .feature_transform import FeatureTransform
from .gaussian_noise import GaussianNoise
from .snr_noise_mix import SnrNoiseMix
from .spec_augment import SpecAugment
from .spec_substitute import SpecSubstitute

__all__ = [
    "FeatureTransform",
    "Compose",
    "Cmvn",
    "SpecAugment",
    "SpecSubstitute",
    "SnrNoiseMix",
    "GaussianNoise",
]
