"""레코드 저장소 접근 — 커넥션과 그 카탈로그. 배치 코드는 batching 패키지에 있다.

    Conn          커넥션 인터페이스 (배치 쪽이 아는 유일한 계약)
    lmdb/         LMDB 구현 — LmdbConn, LmdbManifest, NoiseLmdbConn(잡음 증강용)

    conn = LmdbConn(db_dir)
    manifest = conn.load_manifest()            # 길이 카탈로그 (배치 계산의 입력)
    record = conn.read_record(utt_id, locator) # 실제 레코드

저장소를 바꾸려면 Conn 을 구현한 클래스를 하나 더 만들어 AsrDataLoader(conn=...)
에 넘기면 된다. batching 은 이 패키지를 임포트하지 않으므로 함께 고칠 곳이 없다.

동작 확인:  python3 database/test_database.py
"""
from .conn import Conn
from .lmdb import LmdbConn, LmdbManifest, NoiseLmdbConn

__all__ = ["Conn", "LmdbConn", "LmdbManifest", "NoiseLmdbConn"]
