"""배치 구성 — 카탈로그 길이 정보만으로 순서와 배치 경계를 만들고, 배치를 텐서로 묶는다.

저장소에 무관하다. 이 패키지는 database 를 임포트하지 않는다 — 주입받은
manifest(Conn 카탈로그 계약)와 conn(Conn 계약)의 인터페이스만 쓴다.

    manifest 에서 쓰는 것 : utt_ids / locators / feat_len / word_len / phone_len
                            (전부 배치/전체 단위 팬시 인덱싱으로만 쓴다)
    conn 에서 쓰는 것     : read_record(utt_id, locator)

배치가 만들어지는 순서
    Sorter        발화를 셔플하고 chunk 안에서 길이순으로 정렬
    BatchBudget   배치 하나의 학습 피크 메모리 상한 (있으면 배치 크기가 가변이 된다)
    BatchPlanner  그 순서를 예산 또는 batch_size 로 끊어 배치 경계를 만들고 순서를 섞는다
    BatchDataset  1 item = 1 배치. conn 으로 읽고 Collator 로 묶는다
    Collator      가변 길이 레코드들을 패딩된 배치 텐서로 묶는다
"""
from .batch_budget import BatchBudget
from .batch_dataset import BatchDataset
from .batch_planner import BatchPlanner
from .collator import Collator
from .sorter import Sorter

__all__ = ["BatchBudget", "Sorter", "BatchPlanner", "Collator", "BatchDataset"]
