"""NoiseLmdbConn — 잡음 특징 LMDB 커넥션. 임의 클립의 임의 구간만 꺼낸다."""
import json
import os
import weakref

import numpy as np

try:
    import lmdb
except ImportError:
    raise SystemExit("lmdb 가 필요합니다:  python3 -m pip install lmdb")


class NoiseLmdbConn:
    """noise_record_gen.py 가 만든 잡음 레코드 디렉토리로의 커넥션.

    카탈로그는 발화 레코드와 같은 규약이다 — manifest.jsonl 이 클립당 한 줄로
    (clip_id, feat_len) 을 갖고, LMDB 는 clip_id 로 값을 낸다. 학습 중에는 이
    카탈로그만 보고 "아무 클립의 아무 구간" 을 고르므로 LMDB 를 열 필요조차 없다.

    읽기가 싼 이유
      begin(buffers=True) 로 열면 txn.get 이 mmap 안을 가리키는 memoryview 를
      돌려준다. 클립 전체(길게는 7천 프레임)를 복사하지 않고 필요한 구간만
      잘라 복사하므로, 한 번 읽어 페이지 캐시에 올라간 뒤에는 memcpy 비용만 든다.
      그래서 342MB 를 GPU 에 상주시킬 이유가 없다.

    fork 안전
      Environment(mmap/fd)는 fork 를 넘겨 쓰면 안 된다. lazy open + pid 검사로
      자기 프로세스 몫을 그때 연다 — LmdbConn 과 같은 방식이다.
    """

    DB_NAME = "noise.lmdb"
    META_NAME = "meta.json"
    MANIFEST_NAME = "manifest.jsonl"

    def __init__(self, db_dir):
        self.db_dir = db_dir
        with open(os.path.join(db_dir, self.META_NAME)) as meta_file:
            self.meta = json.load(meta_file)
        self.feat_dim = int(self.meta["feat_dim"])
        self.dtype = np.dtype(self.meta.get("feat_dtype", "float16"))
        self.frame_rate = float(self.meta.get("frame_rate", 100.0))
        self.clip_ids, self.frame_counts = self.load_manifest()
        self.num_clips = int(self.frame_counts.shape[0])
        self._env = None
        self._opened_pid = None
        NoiseLmdbConn._live_conns.add(self)

    def load_manifest(self):
        """manifest.jsonl(클립당 한 줄) -> (clip_ids bytes 배열, frame_counts).

        고정폭 numpy 배열로 든다 — 파이썬 list 로 들면 fork 워커에서 refcount 때문에
        COW 가 깨진다(LmdbManifest 와 같은 이유).
        """
        clip_ids, frame_counts = [], []
        with open(os.path.join(self.db_dir, self.MANIFEST_NAME)) as manifest_file:
            for line in manifest_file:
                if not line.strip():
                    continue
                entry = json.loads(line)
                clip_ids.append(entry["clip_id"].encode("utf-8"))
                frame_counts.append(int(entry["feat_len"]))
        if not clip_ids:
            raise ValueError("빈 잡음 manifest: {}".format(self.db_dir))
        return np.asarray(clip_ids), np.asarray(frame_counts, dtype=np.int64)

    @property
    def total_frames(self):
        return int(self.frame_counts.sum())

    def make_clip_key(self, clip_index):
        return bytes(self.clip_ids[clip_index])

    # -- 프로세스 단위 Environment 공유 -------------------------------------
    # py-lmdb 는 한 프로세스에서 같은 DB 를 두 번 열지 못한다("already open in this
    # process"). 같은 잡음 레코드를 보는 transform 이 둘 이상이면(train 로더 + 실험용
    # 로더 등) 두 번째가 터지므로 (pid, path) 로 공유하고 참조 수로 닫는다.
    # LmdbConn 과 같은 방식이다.
    _shared_envs = {}                   # (pid, path) -> [Environment, 참조 수]
    _live_conns = weakref.WeakSet()     # fork 청소 대상

    @classmethod
    def _acquire_shared_env(cls, pid, path):
        entry = cls._shared_envs.get((pid, path))
        if entry is None:
            entry = [lmdb.open(path, subdir=True, readonly=True,
                               lock=False, readahead=False, meminit=False), 0]
            cls._shared_envs[(pid, path)] = entry
        entry[1] += 1
        return entry[0]

    @classmethod
    def _release_shared_env(cls, pid, path):
        entry = cls._shared_envs.get((pid, path))
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            entry[0].close()
            del cls._shared_envs[(pid, path)]

    @classmethod
    def _drop_envs_from_other_pids(cls, current_pid):
        """fork 로 상속된 Environment 는 닫지 말고 참조만 버린다 (LmdbConn 과 같은 이유)."""
        cls._shared_envs = {key: entry for key, entry in cls._shared_envs.items()
                            if key[0] == current_pid}
        for conn in list(cls._live_conns):
            if conn._opened_pid is not None and conn._opened_pid != current_pid:
                conn._env = None
                conn._opened_pid = None

    @property
    def db_path(self):
        return os.path.join(self.db_dir, self.DB_NAME)

    def _get_or_open_env(self):
        current_pid = os.getpid()
        if current_pid != self._opened_pid:
            self._env = None
            self._drop_envs_from_other_pids(current_pid)
            self._opened_pid = current_pid
        if self._env is None:
            self._env = self._acquire_shared_env(current_pid, self.db_path)
        return self._env

    def read_segment(self, clip_index, start_frame, num_frames):
        """클립 하나에서 num_frames 프레임을 꺼낸다 — 끝에 닿으면 앞으로 감는다.

        Returns: (num_frames, feat_dim) 배열 (저장 dtype 그대로)
        """
        clip_frame_count = int(self.frame_counts[clip_index])
        start_frame = int(start_frame) % clip_frame_count
        with self._get_or_open_env().begin(buffers=True) as read_txn:
            clip_buffer = read_txn.get(self.make_clip_key(clip_index))
            if clip_buffer is None:
                raise KeyError("잡음 클립 {!r} 이 없습니다: {}".format(
                    self.make_clip_key(clip_index), self.db_dir))
            clip = np.frombuffer(clip_buffer, dtype=self.dtype).reshape(clip_frame_count,
                                                                        self.feat_dim)
            end_frame = start_frame + num_frames
            if end_frame <= clip_frame_count:
                return np.array(clip[start_frame:end_frame])        # 필요한 구간만 복사
            # 클립보다 긴 발화: 뒤에서 이어 앞으로 감는다
            head = np.array(clip[start_frame:])
            tail_indexes = np.arange(end_frame - clip_frame_count) % clip_frame_count
            return np.concatenate([head, np.array(clip[tail_indexes])], axis=0)

    def close(self):
        if self._env is not None and os.getpid() == self._opened_pid:
            self._release_shared_env(self._opened_pid, self.db_path)
        self._env = None
        self._opened_pid = None

    def __getstate__(self):
        """워커로 pickle 될 때 Environment 는 빼고 보낸다 (자기 프로세스에서 새로 연다)."""
        state = self.__dict__.copy()
        state["_env"] = None
        state["_opened_pid"] = None
        return state
