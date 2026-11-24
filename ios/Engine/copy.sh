#!/bin/bash
set -e
cd "$(dirname "$0")"

SRC="${MERCURY_EXPORT_DIR:-../../cpp/build/dist}/lib/ios"
if [[ ! -f "$SRC/include/MercuryAsr.h" ]]; then
    echo "산출물이 없다: $SRC" >&2
    echo "먼저 cpp 빌드에서 export_ios 타깃을 실행하세요." >&2
    exit 1
fi

SRC=$(cd "$SRC" && pwd)
frameworks=()
for sdk in iphonesimulator iphoneos; do
    framework="$SRC/$sdk/MercuryAsr.framework"
    if [[ -f "$framework/MercuryAsr" ]]; then
        frameworks+=(-framework "$framework")
    fi
done
if [[ ${#frameworks[@]} -eq 0 ]]; then
    echo "MercuryAsr.framework가 없습니다. export_ios 타깃을 빌드하세요." >&2
    exit 1
fi

mkdir -p lib
stage=$(mktemp -d "$PWD/lib/.xcframework.XXXXXX")
trap 'rm -rf "$stage"' EXIT
xcodebuild -create-xcframework "${frameworks[@]}" -output "$stage/MercuryAsr.xcframework"
rsync -a --delete "$stage/MercuryAsr.xcframework/" lib/MercuryAsr.xcframework/
for sdk in iphonesimulator iphoneos; do
    if [ -f "$SRC/$sdk/libasr_api.a" ]; then
        mkdir -p "lib/$sdk"
        cp "$SRC/$sdk/"*.a "lib/$sdk/"
        echo "복사: lib/$sdk/ ($(ls "lib/$sdk" | tr '\n' ' '))"
    else
        echo "건너뜀: $sdk (빌드되지 않음)"
    fi
done

mkdir -p include
cp "$SRC/include/MercuryAsr.h" include/
echo "복사: include/MercuryAsr.h"
