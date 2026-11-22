"""LmdbConn — Conn 의 LMDB 구현. 샤드별 lazy open (fork 워커 안전)."""
import json
import os
import pickle
import weakref

from ..conn import Conn
from .lmdb_manifest import LmdbManifest

try:
    import lmdb
except ImportError:
    raise SystemExit("lmdb 가 필요합니다:  python3 -m pip install lmdb")


class LmdbConn(Conn):
    """data_gen.py 가 만든 레코드 디렉토리로의 커넥션 — 발화 하나를 꺼낸다.

    lmdb.open() 이 돌려주는 lmdb.Environment 가 DB 커넥션에 해당하고, 이 클래스는
    그 커넥션들을 열어두고 재사용하는 관리자다. 샤드 파일마다 커넥션이 하나씩
    필요하므로(하나의 Environment 가 여러 파일을 못 덮는다) 실제로는 커넥션 풀이다.

      LMDB 계층 정리 (혼동 주의)
        Environment(=conn)  DB 파일 전체 + mmap. 비싸다. 재사용한다.
        Transaction         읽기 스냅샷. 싸다. read_record 마다 열고 닫는다.
        Cursor              순차 순회용 위치 포인터. 여기서는 쓰지 않는다 —
                            읽을 순서를 BatchPlanner 가 정하므로 키 조회만 한다.

    Conn 의 locator = 샤드 번호(int). manifest 의 shard 컬럼이 그대로 들어온다.

    레이아웃은 __init__ 에서 shard_paths 목록 하나로 흡수하고, 그 뒤로는 구분하지
    않는다 — 단일 data.lmdb 도 "샤드가 1개인 DB" 로 취급한다.

      - 샤딩 : meta.json + shard_00000.lmdb ...  -> shard_paths 가 num_shards 개
      - 단일 : data.lmdb                          -> shard_paths 가 1 개
    """

    _SINGLE_DB_NAME = "data.lmdb"          # meta.json 이 없는 구형 레이아웃
    _DB_META_KEY = b"__meta__"             # data_gen 이 샤드마다 넣는 스키마 레코드

    def __init__(self, db_dir):
        self.db_dir = db_dir
        meta_path = os.path.join(db_dir, "meta.json")
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as meta_file:
                self.meta = json.load(meta_file)

        self.shard_paths = self._resolve_shard_paths()
        self._conn_by_shard = {}
        self._opened_pid = None
        self._manifest = None                  # load_manifest() 가 채워 들고 있는다
        LmdbConn._live_conns.add(self)         # fork 청소 대상 (아래 _drop... 참고)
        self.feat_dim = self._resolve_feat_dim()

    @property
    def num_shards(self):
        return len(self.shard_paths)

    def _resolve_shard_paths(self):
        """레이아웃 차이를 여기서 한 번만 흡수해 샤드 파일 경로 목록으로 만든다."""
        if not self.meta:
            return [os.path.join(self.db_dir, self._SINGLE_DB_NAME)]
        pattern = self.meta.get("shard_pattern", "shard_%05d.lmdb")
        num_shards = int(self.meta.get("num_shards", 1))
        return [os.path.join(self.db_dir, pattern % i) for i in range(num_shards)]

    def _resolve_feat_dim(self):
        """feat 채널 수 — 기본값을 두지 않는다.

        틀린 feat_dim 은 배치 텐서 모양을 조용히 어긋나게 만들므로
        추측하지 않고 DB 에서 읽는다. meta.json 이 있으면 그걸 쓰고(커넥션 불필요),
        없는 구형 DB 는 샤드 안의 __meta__ 레코드를 읽는다 —
        data_gen.saveLMDB 가 샤드마다 넣어두므로 두 레이아웃 모두 값이 있다.
        """
        if "feat_dim" in self.meta:
            return int(self.meta["feat_dim"])
        db_meta = self.read_db_meta()
        if "feat_dim" not in db_meta:
            raise ValueError(
                "feat_dim 을 결정할 수 없습니다 — {} 에 meta.json 도, "
                "샤드 안 {!r} 레코드도 없습니다".format(self.db_dir, self._DB_META_KEY))
        return int(db_meta["feat_dim"])

    def load_manifest(self, reload=False):
        """Conn 카탈로그 — 처음 부를 때 한 번 읽고 커넥션이 들고 있는다.

        LMDB 를 열지 않는다(manifest.jsonl + npz 캐시만 읽는다). locators = 샤드 인덱스.
        927만 발화면 배열이 370MB 라 매번 새로 만들면 그만큼 그냥 늘어난다.
        고정폭 numpy 배열이라 fork 워커에는 COW 로 공유된다.
        """
        if self._manifest is None or reload:
            self._manifest = LmdbManifest(self.db_dir)
        return self._manifest

    def read_db_meta(self, shard_index=0):
        """샤드 안의 __meta__ 레코드 (feat_dim/feat_dtype/sos/eos/pad/count)."""
        conn = self._get_or_open_conn(shard_index)
        with conn.begin(buffers=True) as read_txn:
            pickled_meta = read_txn.get(self._DB_META_KEY)
            return {} if pickled_meta is None else pickle.loads(pickled_meta)

    def _get_or_open_conn(self, shard_index=0):
        """샤드 커넥션을 돌려준다 — 처음 쓰는 샤드면 그때 연다(lazy).

        커넥션(mmap/fd)은 fork 를 넘어 공유하면 안 되는데 DataLoader 워커는 fork 로
        만들어진다. 아래 pid 검사가 그 방어이고, lazy open 이라 부모가 안 건드린
        샤드는 애초에 열리지도 않는다.
        """
        current_pid = os.getpid()
        if current_pid != self._opened_pid:  # fork 직후: 상속된 커넥션은 버리고 새로 연다
            self._conn_by_shard = {}
            self._drop_conns_from_other_pids(current_pid)
            self._opened_pid = current_pid
        conn = self._conn_by_shard.get(shard_index)
        if conn is None:
            if not 0 <= shard_index < self.num_shards:
                raise IndexError("shard {} 가 없습니다 — {} 의 샤드는 {}개".format(
                    shard_index, self.db_dir, self.num_shards))
            conn = self._acquire_shared_conn(current_pid, self.shard_paths[shard_index])
            self._conn_by_shard[shard_index] = conn
        return conn

    # -- 프로세스 단위 커넥션 공유 ------------------------------------------
    # py-lmdb 는 한 프로세스에서 같은 DB 를 두 번 열지 못한다("already open in this
    # process"). 같은 DB 를 보는 로더가 둘 이상이면(예: train 로더 + 검증용 로더)
    # 두 번째가 터지므로, (pid, path) 로 Environment 를 공유하고 참조 수로 닫는다.
    _shared_conns = {}          # (pid, path) -> [Environment, 참조 수]
    _live_conns = weakref.WeakSet()   # 이 프로세스에 살아있는 LmdbConn 들 (fork 청소용)

    @classmethod
    def _acquire_shared_conn(cls, pid, path):
        entry = cls._shared_conns.get((pid, path))
        if entry is None:
            entry = [lmdb.open(path, subdir=True, readonly=True,
                               lock=False, readahead=False, meminit=False), 0]
            cls._shared_conns[(pid, path)] = entry
        entry[1] += 1
        return entry[0]

    @classmethod
    def _release_shared_conn(cls, pid, path):
        entry = cls._shared_conns.get((pid, path))
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            entry[0].close()
            del cls._shared_conns[(pid, path)]

    @classmethod
    def _drop_conns_from_other_pids(cls, current_pid):
        """fork 로 상속된 남의 프로세스 커넥션은 닫지 말고 참조만 버린다.

        같은 DB 를 보는 LmdbConn 이 둘 이상이면(train/dev 를 같은 경로로 둔 경우 등)
        내 _conn_by_shard 만 비워서는 부족하다 — 다른 인스턴스가 상속된 Environment 를
        계속 붙잡고 있으면 py-lmdb 가 같은 파일을 다시 열지 못해
        "already open in this process" 로 모든 읽기가 실패한다. 참조가 전부 끊겨야
        Environment 가 GC 되고 py-lmdb 의 열린 경로 목록에서도 빠진다.
        """
        cls._shared_conns = {key: entry for key, entry in cls._shared_conns.items()
                             if key[0] == current_pid}
        for conn in list(cls._live_conns):
            if conn._opened_pid != current_pid:
                conn._conn_by_shard = {}
                conn._opened_pid = None     # 다음 사용 때 자기 몫으로 새로 연다

    def read_record(self, utt_id, locator=None):
        """utt_id(bytes|str) 레코드 반환 — dict(utt_id, feat, word_ids, phone_ids).

        locator 는 샤드 번호. 없으면 KeyError.
        """
        utt_id_bytes = utt_id if isinstance(utt_id, bytes) else utt_id.encode("utf-8")
        conn = self._get_or_open_conn(0 if locator is None else int(locator))
        with conn.begin(buffers=True) as read_txn:
            pickled_record = read_txn.get(utt_id_bytes)
            if pickled_record is None:
                raise KeyError(utt_id_bytes.decode("utf-8", "replace"))
            return pickle.loads(pickled_record)

    def close(self):
        for shard_index in self._conn_by_shard:
            self._release_shared_conn(self._opened_pid, self.shard_paths[shard_index])
        self._conn_by_shard = {}
