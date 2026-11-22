#!/bin/bash
# iOS 용 libasr_api.a 빌드.
#
#   ./ios.sh                # 시뮬레이터 + 실기기
#   ./ios.sh simulator      # 시뮬레이터만
#   ./ios.sh device         # 실기기만
#
# 산출물은 이 디렉터리 아래 lib/ios/<sdk>/ 에 모인다. 앱 프로젝트는 여기서 복사해 간다
# (ios_mercury/Engine/copy.sh). 안드로이드와 같은 규칙이다 — 라이브러리는 무조건
# cpp/export 에서만 만들고, 프로젝트는 결과물만 가져간다.
set -e
cd "$(dirname "$0")"

EXPORT_DIR="$(pwd)"
DEPLOYMENT_TARGET=16.0
MODE=${1:-all}
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

build_one() {
    local sdk=$1           # iphonesimulator | iphoneos
    local build_dir="$EXPORT_DIR/ios_${sdk}_build"
    local out_dir="$EXPORT_DIR/lib/ios/$sdk"

    echo "== iOS $sdk 빌드 =="
    rm -rf "$build_dir"
    mkdir -p "$build_dir" "$out_dir"
    (
        cd "$build_dir"
        cmake "$EXPORT_DIR" \
            -DCMAKE_SYSTEM_NAME=iOS \
            -DCMAKE_OSX_SYSROOT="$sdk" \
            -DCMAKE_OSX_ARCHITECTURES=arm64 \
            -DCMAKE_OSX_DEPLOYMENT_TARGET=$DEPLOYMENT_TARGET \
            -DCMAKE_BUILD_TYPE=Release > /dev/null
        make asr_api -j"$JOBS"
    )

    cp "$build_dir/libasr_api.a" "$out_dir/"
    cp "$build_dir/sentencepiece_build/src/libsentencepiece.a" "$out_dir/"
    echo "   -> $out_dir/libasr_api.a"
}

case "$MODE" in
    simulator) build_one iphonesimulator ;;
    device)    build_one iphoneos ;;
    all)       build_one iphonesimulator; build_one iphoneos ;;
    *) echo "usage: $0 [all|simulator|device]" >&2; exit 2 ;;
esac

# Swift 에서 쓸 헤더도 같이 모아 둔다 — 앱은 lib 과 include 만 가져가면 된다
mkdir -p "$EXPORT_DIR/lib/ios/include"
cp "$EXPORT_DIR/../api/obj_c/MercuryAsr.h" "$EXPORT_DIR/lib/ios/include/"
echo "헤더: lib/ios/include/MercuryAsr.h"

echo "완료."
