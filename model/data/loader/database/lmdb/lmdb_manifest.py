"""LmdbManifest — LMDB 저장소의 카탈로그(manifest.jsonl + npz 캐시)."""
import json
import os
import sys

import numpy as np


class LmdbManifest:
    """LmdbConn 이 내보내는 Conn 카탈로그 구현 — manifest.jsonl 을 numpy 배열로.

    manifest.jsonl 은 data_gen.py 가 LMDB 를 만들면서 같이 쓰는 사이드카이고,
    'shard' 컬럼처럼 LMDB 레이아웃에 매인 정보를 담는다. 그래서 카탈로그를 내는
    책임은 저장소(LmdbConn)에 있고, 배치 쪽은 Conn 카탈로그 계약
    (utt_ids / locators / feat_len / word_len / phone_len / __len__) 만 본다.

    LMDB 를 열지 않고 텍스트만 읽는다 — 배치는 이 길이 배열만으로 짜고 실제
    레코드는 배치가 정해진 뒤에 꺼내므로, 927만 발화도 에폭마다 다시 배치할 수 있다.

    9M 줄 규모라 파이썬 dict/list 로 들고 있으면 메모리도 크고 fork 워커에서
    refcount 때문에 COW 가 깨진다. 고정폭 바이트 배열 + int32 배열로 유지하면
    워커가 늘어도 실제 메모리는 공유된다.  최초 파싱 결과는 npz 로 캐시한다.

    필드 (모두 같은 길이의 배열)
      utt_ids   : bytes  (고정폭 'S' dtype) — LMDB 키
      shard     : int32  (샤드 인덱스, 단일 레이아웃이면 0) = 계약상의 locators
      feat_len  : int32  (프레임 수)
      word_len  : int32  (wordpiece id 개수, SOS/EOS 포함)
      phone_len : int32  (phone id 개수, SOS/EOS 포함)
    """

    @property
    def locators(self):
        """Conn 계약의 위치 힌트 배열 = 샤드 인덱스.

        배치 쪽은 이 값의 의미를 해석하지 않고 read_record 로 되돌려 주기만 한다.
        """
        return self.shard

    def __init__(self, db_dir, use_cache=True):
        self.db_dir = db_dir
        manifest_path = os.path.join(db_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(manifest_path)
        manifest_stat = os.stat(manifest_path)
        cache_path = os.path.join(db_dir, "manifest_cache.npz")
        self.utt_ids = self.shard = self.feat_len = self.word_len = self.phone_len = None

        if use_cache and self._load_cache(cache_path, manifest_stat):
            print("[loader] manifest cache 사용: {} ({:,} utts)".format(cache_path, len(self)))
            return

        print("[loader] manifest 파싱 중: {}".format(manifest_path))
        utt_id_list, shard_index_list = [], []
        feat_len_list, word_len_list, phone_len_list = [], [], []
        with open(manifest_path) as manifest_file:
            for line in manifest_file:
                if not line.strip():
                    continue
                utt_entry = json.loads(line)
                utt_id_list.append(utt_entry["utt_id"])
                shard_index_list.append(utt_entry.get("shard", 0))
                feat_len_list.append(utt_entry["feat_len"])
                word_len_list.append(utt_entry["word_len"])
                phone_len_list.append(utt_entry["phone_len"])

        self.utt_ids = np.array([utt_id.encode("utf-8") for utt_id in utt_id_list], dtype="S")
        self.shard = np.asarray(shard_index_list, dtype=np.int32)
        self.feat_len = np.asarray(feat_len_list, dtype=np.int32)
        self.word_len = np.asarray(word_len_list, dtype=np.int32)
        self.phone_len = np.asarray(phone_len_list, dtype=np.int32)
        print("[loader] manifest {:,} utts 로드 완료".format(len(self)))
        if use_cache:
            self._save_cache(cache_path, manifest_stat)

    def __len__(self):
        return int(self.feat_len.shape[0])

    def get_utt_id(self, utt_index):
        """인덱스 -> utt_id 문자열. LmdbManifest 고유 — Conn 카탈로그 계약이 아니다.

        저장소 구현을 아는 자리(테스트/진단)에서만 쓴다. batching 은 구현을 모르고
        발화 단위로 부를 자리도 없다 — 배치 단위로 utt_ids[utt_indices] 를 쓴다.
        """
        return self.utt_ids[utt_index].decode("utf-8", "replace")

    def _load_cache(self, cache_path, manifest_stat):
        if not os.path.exists(cache_path):
            return False
        try:
            cached_npz = np.load(cache_path)
            # 원본이 바뀌면 캐시 무효화 (크기/mtime)
            if (int(cached_npz["src_size"]) != manifest_stat.st_size
                    or int(cached_npz["src_mtime"]) != int(manifest_stat.st_mtime)):
                return False
            self.utt_ids = cached_npz["utt_ids"]
            self.shard = cached_npz["shard"]
            self.feat_len = cached_npz["feat_len"]
            self.word_len = cached_npz["word_len"]
            self.phone_len = cached_npz["phone_len"]
            return True
        except Exception as error:
            print("[loader] manifest cache 무시 ({})".format(error), file=sys.stderr)
            return False

    def _save_cache(self, cache_path, manifest_stat):
        try:
            tmp_cache_path = cache_path + ".tmp.npz"   # np.savez 는 .npz 가 없으면 붙인다
            np.savez(tmp_cache_path, utt_ids=self.utt_ids, shard=self.shard,
                     feat_len=self.feat_len, word_len=self.word_len,
                     phone_len=self.phone_len,
                     src_size=np.int64(manifest_stat.st_size),
                     src_mtime=np.int64(manifest_stat.st_mtime))
            os.replace(tmp_cache_path, cache_path)
            print("[loader] manifest cache 저장: {}".format(cache_path))
        except Exception as error:  # 읽기 전용 디렉토리 등 — 캐시는 있으면 좋은 정도
            print("[loader] manifest cache 저장 실패 ({})".format(error), file=sys.stderr)
