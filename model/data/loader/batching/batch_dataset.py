"""BatchDataset — 1 item = 1 배치 (레코드 읽기 + 패딩까지 워커에서 끝낸다)."""
import sys

import torch


class BatchDataset(torch.utils.data.Dataset):
    """배치 하나를 통째로 읽어 패딩까지 끝낸 텐서로 돌려주는 Dataset.

    DataLoader(batch_size=None) 로 쓰면 collate 없이 그대로 흘러간다.

    저장소에 무관하다 — conn 이 database.Conn 계약(read_record(utt_id, locator))만
    지키면 LMDB 든 무엇이든 상관없고, 이 파일은 database 패키지를 임포트하지 않는다.
    커넥션을 언제 여는지도 conn 의 사정이다(LmdbConn 은 워커에서 처음 접근할 때 연다).

    협력 객체(주입):
      conn     : Conn      레코드 읽기
      manifest : Manifest  utt_id / locator 조회
      collator : Collator  패딩 + 마스크

    배치 구성(주입):
      utt_order     : 정렬/셔플이 끝난 발화 인덱스 순열
      batch_offsets : utt_order 를 배치로 자르는 경계 오프셋 (len = num_batches + 1)
    """

    def __init__(self, conn, manifest, collator, utt_order, batch_offsets):
        self.conn = conn
        self.manifest = manifest
        self.collator = collator
        self.utt_order = utt_order
        self.batch_offsets = batch_offsets

    def __len__(self):
        return int(self.batch_offsets.shape[0] - 1)

    def __getitem__(self, batch_index):
        utt_indices = self.utt_order[
            self.batch_offsets[batch_index]:self.batch_offsets[batch_index + 1]]
        # 배치 전체를 한 번에 팬시 인덱싱한다 — 카탈로그가 numpy 배열이어야 하는
        # 이유를 여기서 쓴다. tolist() 는 순수 bytes/int 를 만들어 Conn 계약
        # (utt_id: bytes|str, locator: int) 에 정확히 맞는 타입으로 넘긴다.
        batch_utt_id_bytes = self.manifest.utt_ids[utt_indices].tolist()
        batch_locators = self.manifest.locators[utt_indices].tolist()

        records, utt_ids = [], []
        for utt_id_bytes, locator in zip(batch_utt_id_bytes, batch_locators):
            try:
                records.append(self.conn.read_record(utt_id_bytes, locator))
            except Exception as error:  # 레코드 손상/누락 — 배치에서 빼고 계속
                print("[loader] skip {}: {}".format(utt_id_bytes, error), file=sys.stderr)
                continue
            utt_ids.append(utt_id_bytes.decode("utf-8", "replace"))
        if not records:
            return None
        return self.collator(records, utt_ids)
