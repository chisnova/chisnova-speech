"""BatchPlanner — manifest 길이 정보만으로 발화 순서와 배치 경계를 만든다.

실제 feat 은 배치가 정해진 뒤 저장소에서 꺼낸다. 여기서는 길이 배열만 다루므로
927만 발화도 에폭마다 배치를 다시 짤 수 있다.

협력 객체: Sorter(순서).
"""
import numpy as np


class BatchPlanner:
    """manifest + Sorter 를 들고 배치 계획을 만든다.

    계획 = (utt_order, batch_offsets, batch_order)
      utt_order     : 발화 순서 (Sorter 가 셔플 + chunk 내 길이정렬한 인덱스 배열)
      batch_offsets : 배치 경계. 배치 i = utt_order[batch_offsets[i]:batch_offsets[i+1]]
      batch_order   : 배치를 내보내는 순서 (길이순으로 묶인 배치를 다시 섞는다 —
                      안 섞으면 짧은 배치부터 순서대로 나가 학습이 편향된다)
    """

    def __init__(self, manifest, sorter, drop_last=False):
        self.manifest = manifest
        self.sorter = sorter
        self.drop_last = drop_last

    def cut_into_batches(self, num_utts, batch_size):
        """발화 num_utts 개를 batch_size 개씩 끊은 경계 배열을 만든다.

        마지막 자투리 배치는 drop_last 면 버린다. 단 배치가 하나뿐이면 버리지
        않는다 — 전부 사라지면 이터레이션이 빈 채로 끝난다.
        """
        if num_utts == 0:
            return np.zeros(1, dtype=np.int64)
        batch_offsets = np.arange(0, num_utts + 1, batch_size, dtype=np.int64)
        if (batch_offsets[-1] != num_utts
                and not (self.drop_last and batch_offsets.shape[0] > 1)):
            batch_offsets = np.append(batch_offsets, np.int64(num_utts))
        return batch_offsets

    def pack_into_batches(self, utt_order, budget):
        """utt_order 를 메모리 예산에 맞춰 배치 경계로 나눈다 (greedy bin packing).

        앞에서부터 담다가 예산을 넘기면 거기서 끊는다. chunk 정렬로 길이가 비슷한
        발화가 모여 있으므로 배치마다 예산을 고르게 채운다 — 짧은 발화 구간에서는
        배치가 커지고 긴 구간에서는 작아져, 고정 batch_size 와 달리 피크가 평탄해진다.

        발화 하나가 이미 예산을 넘으면 단독 배치로 둔다 — 더 줄일 수 없다
        (그런 발화는 AsrDataLoader 의 max_feat_len 으로 걸러내는 게 맞다).
        """
        num_utts = int(utt_order.shape[0])
        if num_utts == 0:
            return np.zeros(1, dtype=np.int64)

        feat_lens = self.manifest.feat_len[utt_order]
        word_lens = self.manifest.word_len[utt_order]

        batch_offsets = [0]
        cur_size = cur_max_feat = cur_max_word = 0
        num_oversize = 0
        for i in range(num_utts):
            feat_len = int(feat_lens[i])
            word_len = int(word_lens[i])
            next_max_feat = feat_len if feat_len > cur_max_feat else cur_max_feat
            next_max_word = word_len if word_len > cur_max_word else cur_max_word
            if cur_size and not budget.fits(cur_size + 1, next_max_feat, next_max_word):
                batch_offsets.append(i)
                cur_size, cur_max_feat, cur_max_word = 1, feat_len, word_len
                if not budget.fits(1, feat_len, word_len):
                    num_oversize += 1
            else:
                cur_size += 1
                cur_max_feat, cur_max_word = next_max_feat, next_max_word
        if cur_size and not (self.drop_last and len(batch_offsets) > 1):
            batch_offsets.append(num_utts)

        if num_oversize:
            print("[loader] 발화 {:,}개가 단독으로 예산({})을 초과 — 단독 배치로 처리".format(
                num_oversize, budget.describe()))
        return np.asarray(batch_offsets, dtype=np.int64)

    def compute_batch_costs(self, utt_order, batch_offsets, budget):
        """배치별 예상 피크 바이트 (예산 사용률 확인용)."""
        feat_lens = self.manifest.feat_len[utt_order].astype(np.int64)
        word_lens = self.manifest.word_len[utt_order].astype(np.int64)
        batch_starts = batch_offsets[:-1]
        batch_sizes = np.diff(batch_offsets)
        max_feat_len = np.maximum.reduceat(feat_lens, batch_starts)
        max_word_len = np.maximum.reduceat(word_lens, batch_starts)
        return budget.compute_cost(batch_sizes, max_feat_len, max_word_len)

    def compute_padding_efficiency(self, utt_order, batch_offsets):
        """실제 프레임 / 패딩 포함 프레임 (0~1). chunk 정렬이 먹었는지 보는 지표."""
        feat_lens = self.manifest.feat_len[utt_order].astype(np.int64)
        batch_sizes = np.diff(batch_offsets)
        max_feat_len = np.maximum.reduceat(feat_lens, batch_offsets[:-1])
        return float(feat_lens.sum()) / float((batch_sizes * max_feat_len).sum())

    def plan_batches(self, utt_indices, batch_size, rng, shuffle_batches, budget=None):
        """utt_indices -> (utt_order, batch_offsets, batch_order).

        budget 이 있으면 메모리 예산으로 배치를 짜고(가변 크기), 없으면
        batch_size 개씩 고정으로 자른다.
        """
        utt_order = self.sorter.make_order(self.manifest.feat_len, utt_indices, rng)
        batch_offsets = (self.pack_into_batches(utt_order, budget) if budget is not None
                         else self.cut_into_batches(int(utt_order.shape[0]), batch_size))
        batch_order = np.arange(batch_offsets.shape[0] - 1, dtype=np.int64)
        if shuffle_batches:
            # 지우지 말 것. chunk 안이 길이순이라 이 셔플이 없으면 배치의 max_len 이
            # chunk 주기로 톱니처럼 오르내린다. HF Trainer 가 이 단계를 빼먹어
            # chunk 주기마다 loss spike 가 나는 버그가 보고돼 있다(transformers#20810,
            # 미해결). fastai·torchtext·OpenNMT 는 모두 이 단계를 넣는다.
            rng.shuffle(batch_order)
        return utt_order, batch_offsets, batch_order
