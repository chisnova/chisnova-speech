#!/usr/bin/env bash
# sample 음성 100개로 LMDB 레코드를 만든다.
#
#   ./make_lmdb.sh
#
# 입력  sample/sample.wlist   발화 id<TAB>음성 경로 (model/ 기준 상대경로)
#       sample/sample.lab     발화 id<TAB>전사
#       mercury.model         토큰 사전
#       full.dict             발음사전 (data_gen.py 가 항상 읽는다)
# 출력  sample/database/      shard_00000.lmdb, manifest.jsonl, meta.json, skipped.txt
#
# sample.wlist 의 경로가 model/ 기준이라 어디서 실행하든 model/ 로 옮겨서 돈다.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."      # -> model/

SAMPLE=data/sample
OUT=$SAMPLE/database

# 다시 만들 때 예전 레코드가 섞이지 않게 비우고 시작한다.
rm -rf "$OUT"

python3 data/generation/record/data_gen.py \
    --wavpath  $SAMPLE/sample.wlist \
    --txtpath  $SAMPLE/sample.lab \
    --savepath $OUT \
    --vocab    data/mercury.model \
    --map_size_gb 1
