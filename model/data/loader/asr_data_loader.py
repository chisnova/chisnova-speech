"""AsrDataLoader — 레코드 디렉토리(train/dev/test) -> 배치 이터레이터.

배치를 만드는 순서
    manifest 에서 발화 목록을 받아 -> 셔플 -> chunk 로 분할 -> chunk 안에서 길이
    정렬 -> batch_size 개씩 끊기 -> 배치 순서 셔플.
    앞의 세 단계는 Sorter, 뒤의 두 단계는 BatchPlanner 가 맡는다.

로더는 아래 객체들을 조립해 배치를 만든다:
    Conn             저장소 커넥션 — 카탈로그 + 레코드 읽기 (기본 구현 LmdbConn)
    manifest         conn.load_manifest() 가 낸 발화별 길이 카탈로그
    Sorter           셔플 + chunk 내 길이 정렬
    BatchPlanner     manifest + Sorter -> 배치 경계(batch_offsets) 계산
    BatchDataset     1 item = 1 배치 (워커에서 실행, Conn/Collator 주입)
    Collator         패딩 + 마스크
    FeatureTransform 정규화/증강 (feature_enginerring 패키지, 디바이스 전송 후 적용)

사용 예
    train = AsrDataLoader("/home/chisnova/data/records/train",
                          batch_size=32, shuffle=True, isTrain=True)
    for feat, word, phone, feat_mask, word_mask, phone_mask, utt_ids in train:
        ...
    train.reshuffle_for_next_epoch()  # 에폭마다 재셔플(배치 재구성)

    dev = AsrDataLoader("/home/chisnova/data/records/dev",
                        batch_size=16, shuffle=False, isTrain=False)
"""
import os

import numpy as np
import torch

from .batching import BatchDataset, BatchPlanner, Collator, Sorter
from .database import LmdbConn
from .feature_enginerring import Compose, GaussianNoise, SpecAugment


class AsrDataLoader:
    """train/dev/test 레코드 디렉토리 -> 배치 이터레이터.

    배칭
      budget      : BatchBudget. 주면 배치 크기가 가변이 되어 예상 학습 피크
                    메모리가 상한을 넘지 않는 선까지 발화를 담는다. 짧은 발화
                    구간에서는 배치가 커지고 긴 구간에서는 작아져 GPU 메모리를
                    고르게 쓴다. None 이면 batch_size 로 고정한다.
      batch_size  : 배치 하나의 발화 수 (기본 16). budget 이 있으면 chunk 크기를
                    정하는 데만 쓰인다
      chunk       : 길이 정렬 단위. "auto"(기본)면 batch_size 에 맞춰 정한다
                    (Sorter.resolve_auto_chunk — 배치가 크면 chunk 도 커야 패딩이 준다)
      shuffle     : 발화 순서와 배치 순서를 섞는다. False 면 순서가 고정된다
      drop_last   : 마지막 자투리 배치를 버린다

    그 외
      usage_ratio : 데이터 일부만 사용 (0~1)
      subset_utt_ids : 이 발화들만 쓴다 (utt_id 목록). 평가에서 코퍼스별로 같은
                    수를 뽑는 것처럼, 무작위 비율이 아니라 "정확히 이 발화들" 을
                    지정해야 할 때 쓴다. usage_ratio 보다 먼저 적용된다
      min/max_feat_len : 길이 필터 (프레임, 100fps)
      conn        : database.Conn 구현. None 이면 LmdbConn(data_path) 을 쓴다.
                    저장소를 바꾸려면 여기만 갈아끼우면 된다 — 배치 쪽은 그대로다.
      transform   : FeatureTransform (보통 Compose). 주면 모든 배치에 이것만 적용하고
                    spec_augment 인자는 무시한다 — 정규화(Cmvn)처럼 train/dev 공통으로
                    타야 하는 변환이 있으므로 isTrain 과 묶지 않는다.
      spec_augment: transform 을 안 줬을 때만 의미가 있다. None 이면 isTrain 값을 따라
                    SpecAugment + GaussianNoise 를 조립한다
      device      : 배치를 올릴 디바이스. None/"cpu" 면 CPU 유지
    """

    def __init__(self, data_path, batch_size=16, budget=None, chunk="auto",
                 shuffle=True, isTrain=True, usage_ratio=1.0,
                 seed=0, drop_last=False, conn=None, transform=None,
                 subset_utt_ids=None, spec_augment=None,
                 spec_augment_prob=0.7, min_feat_len=1, max_feat_len=None,
                 num_workers=4, pin_memory=True, prefetch_factor=2, device="cuda"):
        self.data_path = data_path
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size 는 양수여야 합니다: {!r}".format(batch_size))
        self.budget = budget
        self.shuffle = shuffle
        self.isTrain = isTrain
        self.usage_ratio = float(usage_ratio)
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.epoch = 0

        # -- 협력 객체 조립 -------------------------------------------------
        # conn 은 database.Conn 구현이면 무엇이든 된다 — 저장소를 바꾸는 지점.
        self.conn = LmdbConn(data_path) if conn is None else conn
        self.feat_dim = self.conn.feat_dim
        self.manifest = self.conn.load_manifest()   # 카탈로그도 저장소가 낸다
        self.collator = Collator()
        self.transform = self._resolve_transform(transform, spec_augment,
                                                 spec_augment_prob, isTrain)

        self.sorter = Sorter(chunk=(Sorter.resolve_auto_chunk(self.batch_size)
                                    if chunk == "auto" else chunk),
                             shuffle=shuffle)
        self.planner = BatchPlanner(self.manifest, self.sorter, drop_last=drop_last)

        self.device = None if device in (None, "cpu") else torch.device(device)
        if self.device is not None and not torch.cuda.is_available():
            print("[loader] CUDA 없음 — CPU 텐서로 내보냅니다")
            self.device = None

        # -- 길이 필터 (한 번만) -------------------------------------------
        feat_lens = self.manifest.feat_len
        keep_mask = feat_lens >= max(1, int(min_feat_len))
        if max_feat_len:
            keep_mask &= feat_lens <= int(max_feat_len)
        keep_mask &= self.manifest.word_len > 0
        keep_mask &= self.manifest.phone_len > 0
        if subset_utt_ids is not None:
            keep_mask &= self._make_subset_mask(subset_utt_ids)
        self.base_index = np.nonzero(keep_mask)[0].astype(np.int64)
        num_filtered_out = len(self.manifest) - self.base_index.shape[0]
        if num_filtered_out:
            print("[loader] 길이/부분집합 필터로 {:,} utts 제외".format(num_filtered_out))
        self.full_data_size = int(self.base_index.shape[0])

        self._make_batches()

    def _make_subset_mask(self, subset_utt_ids):
        """지정한 utt_id 만 True 인 마스크. 카탈로그 순회 한 번으로 만든다."""
        wanted = {utt_id if isinstance(utt_id, bytes) else str(utt_id).encode("utf-8")
                  for utt_id in subset_utt_ids}
        catalog_ids = self.manifest.utt_ids
        return np.fromiter((bytes(utt_id) in wanted for utt_id in catalog_ids),
                           dtype=bool, count=len(catalog_ids))

    # -- 특징 변환 ----------------------------------------------------------
    def _resolve_transform(self, transform, spec_augment, spec_augment_prob, isTrain):
        """transform 이 있으면 그대로, 없으면 isTrain 에 맞춰 기본 증강을 조립한다.

        정규화(Cmvn)는 기본에 넣지 않는다 — 넣으면 dev/test 로더에도 같이 넣어야
        분포가 맞는데, 그건 호출자가 명시적으로 정할 문제다. feature_enginerring
        패키지 독스트링의 권장 조합을 참고할 것.
        """
        if transform is not None:
            return transform
        if not (isTrain if spec_augment is None else spec_augment):
            return None
        return Compose([SpecAugment(prob=spec_augment_prob),
                        GaussianNoise(std=0.05, prob=spec_augment_prob)])

    # -- 배치 구성 ----------------------------------------------------------
    def _make_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        utt_indices = self.base_index
        if self.usage_ratio < 1.0:
            num_sampled_utts = max(1, int(self.full_data_size * self.usage_ratio))
            utt_indices = (rng.permutation(utt_indices)[:num_sampled_utts]
                           if self.shuffle else utt_indices[:num_sampled_utts])

        self.utt_order, self.batch_offsets, self.batch_order = self.planner.plan_batches(
            utt_indices, self.batch_size, rng, shuffle_batches=self.shuffle,
            budget=self.budget)
        self.data_size = int(self.utt_order.shape[0])

        batch_sizes = np.diff(self.batch_offsets)
        padding_eff = self.planner.compute_padding_efficiency(self.utt_order,
                                                              self.batch_offsets)
        if self.budget is None:
            sizing = "B {}~{}".format(batch_sizes.min(), batch_sizes.max())
        else:
            batch_costs = self.planner.compute_batch_costs(self.utt_order,
                                                           self.batch_offsets, self.budget)
            sizing = "B {}~{} (평균 {:.1f})  예산 {} 사용률 {:.0f}~{:.0f}%".format(
                batch_sizes.min(), batch_sizes.max(), batch_sizes.mean(),
                self.budget.describe(),
                100.0 * batch_costs.min() / self.budget.limit_bytes,
                100.0 * batch_costs.max() / self.budget.limit_bytes)
        print("[loader] {}  utts {:,}  batches {:,}  {}  "
              "패딩효율 {:.1f}%  chunk {:,}  shuffle={}".format(
                  os.path.basename(os.path.normpath(self.data_path)), self.data_size,
                  self.num_batches, sizing,
                  100.0 * padding_eff, self.sorter.chunk, self.shuffle))

    @property
    def num_batches(self):
        return int(self.batch_offsets.shape[0] - 1)

    def __len__(self):
        return self.num_batches

    def reshuffle_for_next_epoch(self):
        """에폭 경계에서 호출 — 셔플/버킷을 다시 굴려 배치를 재구성한다."""
        self.epoch += 1
        self._make_batches()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._make_batches()

    # -- 이터레이션 ---------------------------------------------------------
    def _make_batch_dataset(self):
        return BatchDataset(self.conn, self.manifest, self.collator,
                                self.utt_order, self.batch_offsets)

    def _move_and_transform_batch(self, batch):
        """디바이스 전송 + 특징 변환(정규화/증강)."""
        feat, word, phone, feat_mask, word_mask, phone_mask, utt_ids = batch
        if self.device is not None:
            non_blocking = self.pin_memory
            feat = feat.to(self.device, non_blocking=non_blocking)
            word = word.to(self.device, non_blocking=non_blocking)
            phone = phone.to(self.device, non_blocking=non_blocking)
            feat_mask = feat_mask.to(self.device, non_blocking=non_blocking)
            word_mask = word_mask.to(self.device, non_blocking=non_blocking)
            phone_mask = phone_mask.to(self.device, non_blocking=non_blocking)
        if self.transform is not None:
            feat = self.transform(feat)
        return feat, word, phone, feat_mask, word_mask, phone_mask, utt_ids

    def iter_prefetched_batches(self, num_workers=None):
        """워커 프로세스가 LMDB 읽기/패딩을 미리 수행하는 경로 (학습용 기본).

        yield: feat(B,T,D), word(B,Lw), phone(B,Lp),
               feat_mask(B,T), word_mask(B,Lw), phone_mask(B,Lp), utt_ids(list[str])
        """
        resolved_num_workers = self.num_workers if num_workers is None else num_workers
        prefetch_loader = torch.utils.data.DataLoader(
            self._make_batch_dataset(),
            batch_size=None,                 # Dataset 이 이미 배치 단위
            sampler=self.batch_order.tolist(),
            num_workers=resolved_num_workers,
            pin_memory=self.pin_memory and self.device is not None,
            prefetch_factor=(self.prefetch_factor
                             if resolved_num_workers > 0 else None),
            persistent_workers=False,
        )
        for batch in prefetch_loader:
            if batch is None:                # 배치 전체가 손상된 경우
                continue
            yield self._move_and_transform_batch(batch)

    def iter_batches_in_main_process(self):
        """워커 없이 현재 프로세스에서 읽는 경로 (디버깅/추론용)."""
        batch_dataset = self._make_batch_dataset()
        for batch_index in self.batch_order:
            batch = batch_dataset[int(batch_index)]
            if batch is None:
                continue
            yield self._move_and_transform_batch(batch)

    def __iter__(self):
        return self.iter_prefetched_batches()

    def close(self):
        self.conn.close()
