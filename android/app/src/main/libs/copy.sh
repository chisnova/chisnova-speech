#!/bin/bash
set -e
cd "$(dirname "$0")"

SRC="${MERCURY_EXPORT_DIR:-../../../../../cpp/build/dist}/lib/android"
if [[ ! -f "$SRC/arm64-v8a/libasr_api.so" && ! -f "$SRC/armeabi-v7a/libasr_api.so" ]]; then
    echo "산출물이 없다: $SRC" >&2
    echo "먼저 cpp 빌드에서 export_android 타깃을 실행하세요." >&2
    exit 1
fi

for abi in arm64-v8a armeabi-v7a; do
    if [ -f "$SRC/$abi/libasr_api.so" ]; then
        mkdir -p "$abi"
        cp "$SRC/$abi/libasr_api.so" "$abi/"
        echo "복사: $abi/libasr_api.so ($(ls -l "$abi/libasr_api.so" | awk '{print $5}') bytes)"
    else
        echo "건너뜀: $abi (빌드되지 않음)"
    fi
done
