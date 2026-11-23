#!/bin/bash
set -e
cd "$(dirname "$0")"

SRC="${MERCURY_EXPORT_DIR:-../../../../../cpp/build/dist}/model_state"
if [[ -z "${MERCURY_EXPORT_DIR:-}" && ! -d "$SRC" ]]; then
    SRC="../../../../../cpp/export/model_state"
fi
if [ ! -f "$SRC/model.json" ]; then
    echo "산출물이 없다: $SRC" >&2
    echo "먼저 cpp 빌드에서 export_model 타깃을 실행하세요." >&2
    exit 1
fi

if [[ "$SRC" -ef model_state ]]; then exit 0; fi
mkdir -p model_state
rm -rf model_state/*
cp "$SRC"/* model_state/
echo "복사: model_state ($(ls model_state | wc -l | tr -d ' ') 개 파일, $(du -sh model_state | awk '{print $1}'))"
