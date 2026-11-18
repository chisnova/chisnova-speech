"""Collator — 가변 길이 레코드들을 하나의 패딩된 배치 텐서로 묶는다."""
import numpy as np
import torch


class Collator:
    """레코드 리스트 -> 패딩 배치 + 마스크.

    채널 수(D)는 레코드에서 직접 읽는다 — 메타데이터를 믿지 않으므로 여기서
    feat_dim 이 틀릴 일이 없다.
    """

    def __call__(self, records, utt_ids):
        """레코드 리스트 -> 패딩 배치 + 마스크 (1=실제, 0=패딩).

        반환: feat(B,T,D) f32, word(B,Lw) i64, phone(B,Lp) i64,
              feat_mask(B,T) f32, word_mask(B,Lw) f32, phone_mask(B,Lp) f32, utt_ids
        """
        batch_size = len(records)
        feat_dim = records[0]["feat"].shape[1]
        max_feat_len = max(r["feat"].shape[0] for r in records)
        max_word_len = max(r["word_ids"].shape[0] for r in records)
        max_phone_len = max(r["phone_ids"].shape[0] for r in records)

        feat = torch.zeros(batch_size, max_feat_len, feat_dim, dtype=torch.float32)
        word = torch.zeros(batch_size, max_word_len, dtype=torch.long)
        phone = torch.zeros(batch_size, max_phone_len, dtype=torch.long)
        feat_mask = torch.zeros(batch_size, max_feat_len, dtype=torch.float32)
        word_mask = torch.zeros(batch_size, max_word_len, dtype=torch.float32)
        phone_mask = torch.zeros(batch_size, max_phone_len, dtype=torch.float32)

        for row_index, record in enumerate(records):
            feat_arr = np.asarray(record["feat"], dtype=np.float32)   # 저장은 float16
            word_arr = np.asarray(record["word_ids"], dtype=np.int64)
            phone_arr = np.asarray(record["phone_ids"], dtype=np.int64)
            feat_len = feat_arr.shape[0]
            word_len = word_arr.shape[0]
            phone_len = phone_arr.shape[0]
            feat[row_index, :feat_len] = torch.from_numpy(feat_arr)
            word[row_index, :word_len] = torch.from_numpy(word_arr)
            phone[row_index, :phone_len] = torch.from_numpy(phone_arr)
            feat_mask[row_index, :feat_len] = 1.0
            word_mask[row_index, :word_len] = 1.0
            phone_mask[row_index, :phone_len] = 1.0
        return feat, word, phone, feat_mask, word_mask, phone_mask, utt_ids
