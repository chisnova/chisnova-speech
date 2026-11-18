"""레코드 디렉토리 기반 ASR 데이터 로더 패키지.

    from data.loader import AsrDataLoader

data_gen.py 가 만든 레코드 디렉토리를 받아 배치 단위 텐서를 내보낸다.
배치 구성은 manifest(길이 정보)만으로 하고, 실제 feat 은 배치가 정해진 뒤
저장소에서 꺼낸다 — "배치는 학습 때 자유롭게" 라는 data_gen 설계와 동일.

파일 하나에 클래스 하나이며, 파일 이름은 그 클래스 이름이다.

    asr_data_loader.py     AsrDataLoader — 전체 조립 + 배치 이터레이터
    database/              저장소: Conn(인터페이스) + LmdbConn/LmdbManifest(구현)
    batching/              배치 구성: Sorter, BatchPlanner, Collator, BatchDataset
    feature_enginerring/   정규화/증강: FeatureTransform 과 그 구현들

의존 방향은 한 방향뿐이다:  AsrDataLoader -> {database, batching, feature_enginerring}
batching 은 database 를 임포트하지 않는다 — 저장소를 바꿔도 배치 쪽은 그대로다.
"""
from .asr_data_loader import AsrDataLoader
from .batching import BatchBudget, BatchDataset, BatchPlanner, Collator, Sorter
from .database import Conn, LmdbConn, LmdbManifest
from .feature_enginerring import (Cmvn, Compose, FeatureTransform, GaussianNoise,
                                  SpecAugment, SpecSubstitute)

__all__ = [
    "AsrDataLoader",
    "Conn",
    "LmdbConn",
    "LmdbManifest",
    "BatchBudget",
    "Sorter",
    "BatchPlanner",
    "Collator",
    "BatchDataset",
    "FeatureTransform",
    "Compose",
    "Cmvn",
    "SpecAugment",
    "SpecSubstitute",
    "GaussianNoise",
]
