#!/usr/bin/env python3
"""database 패키지 동작 확인 — dev/test/train 에서 각각 100개씩 꺼내 값을 검증한다.

배치·증강은 건드리지 않는다. 여기서 보는 것은 저장소 계약뿐이다:
커넥션이 열리는가, 카탈로그가 맞는가, 레코드가 카탈로그와 일치하는가.

    python3 test_database.py
"""
import os
import sys
import time

import numpy as np

# 이 파일의 디렉토리와 그 부모는 sys.path 에서 빼야 한다 — 하위 패키지 database/lmdb
# 가 pip 패키지 lmdb(C 바인딩)를 가려 `import lmdb` 가 깨진다.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SHADOWING = {_HERE, os.path.dirname(_HERE)}
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in _SHADOWING]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))  # -> model/

from data.loader.database import LmdbConn

RECORDS_ROOT = "/home/chisnova/data/records"
DB_NAMES = ("dev", "test", "train")
NUM_RECORDS = 100          # DB 하나에서 꺼내 볼 레코드 수

PASS, FAIL = "[ OK ]", "[FAIL]"
_check_results = []


def check(cond, msg):
    print("  {} {}".format(PASS if cond else FAIL, msg))
    _check_results.append(bool(cond))
    return bool(cond)


def read_and_verify(db_name):
    """DB 하나를 열어 NUM_RECORDS 개를 꺼내고 카탈로그와 대조한다."""
    db_dir = os.path.join(RECORDS_ROOT, db_name)
    print("\n" + "=" * 78)
    print("### {}  ({})".format(db_name, db_dir))
    print("=" * 78)
    if not os.path.isdir(db_dir):
        return check(False, "레코드 디렉토리가 없습니다")

    with LmdbConn(db_dir) as conn:
        manifest = conn.load_manifest()
        print("  샤드 {}개   feat_dim {}   총 발화 {:,}".format(
            conn.num_shards, conn.feat_dim, len(manifest)))

        # 카탈로그 전체에 고루 퍼진 100개 — train 이면 여러 샤드가 걸린다
        picked = np.unique(np.linspace(0, len(manifest) - 1, NUM_RECORDS).astype(np.int64))
        started_at = time.time()
        records = [conn.read_record(manifest.utt_ids[i], manifest.locators[i])
                   for i in picked]
        elapsed = time.time() - started_at

        total_frames = sum(np.asarray(r["feat"]).shape[0] for r in records)
        print("  {}개 읽음  {:.3f}s  ({:,.0f} 레코드/s, {:,} 프레임)  샤드 {}종".format(
            len(records), elapsed, len(records) / max(elapsed, 1e-9), total_frames,
            len(np.unique(manifest.locators[picked]))))

        for row, i in enumerate(picked[:2]):        # 눈으로 볼 샘플 2개
            record = records[row]
            feat = np.asarray(record["feat"], dtype=np.float32)
            print("    utt={} locator={}  feat{} {}  min/mean/max {:.1f}/{:.1f}/{:.1f} dB"
                  .format(record["utt_id"], manifest.locators[i], feat.shape,
                          np.asarray(record["feat"]).dtype,
                          feat.min(), feat.mean(), feat.max()))
            print("      word_ids  {}{}".format(
                [int(x) for x in record["word_ids"][:12]],
                " ..." if len(record["word_ids"]) > 12 else ""))
            print("      phone_ids {}{}".format(
                [int(x) for x in record["phone_ids"][:12]],
                " ..." if len(record["phone_ids"]) > 12 else ""))

        # -- 검증 : 꺼낸 레코드가 카탈로그와 맞는가 --------------------------
        shapes_match = lengths_match = tokens_wrapped = values_finite = True
        for row, i in enumerate(picked):
            record = records[row]
            feat = np.asarray(record["feat"])
            shapes_match &= feat.shape == (manifest.feat_len[i], conn.feat_dim)
            lengths_match &= (len(record["word_ids"]) == manifest.word_len[i]
                              and len(record["phone_ids"]) == manifest.phone_len[i])
            tokens_wrapped &= (int(record["word_ids"][0]) == 2
                               and int(record["word_ids"][-1]) == 3
                               and int(record["phone_ids"][0]) == 2
                               and int(record["phone_ids"][-1]) == 3)
            values_finite &= bool(np.isfinite(feat.astype(np.float32)).all())

        check(len(records) == len(picked), "{}개 전부 읽힘".format(len(picked)))
        check(shapes_match, "feat 모양 == (카탈로그 feat_len, conn.feat_dim)")
        check(lengths_match, "word/phone 개수 == 카탈로그 word_len/phone_len")
        check(tokens_wrapped, "word_ids/phone_ids 가 SOS(2)/EOS(3) 로 감싸짐")
        check(values_finite, "feat 에 NaN/Inf 없음")
        check(int(manifest.locators.max()) < conn.num_shards,
              "카탈로그의 locator 가 전부 실재하는 샤드 범위 안")
    return True


def main():
    print("레코드 루트: {}   DB 하나당 {}개씩".format(RECORDS_ROOT, NUM_RECORDS))
    for db_name in DB_NAMES:
        read_and_verify(db_name)

    print("\n" + "=" * 78)
    num_passed = sum(_check_results)
    print("  검증 {}/{} 통과".format(num_passed, len(_check_results)))
    if num_passed != len(_check_results):
        print("  실패한 항목이 있습니다 — 위 {} 표시 확인".format(FAIL))
    return 0 if num_passed == len(_check_results) else 1


if __name__ == "__main__":
    sys.exit(main())
