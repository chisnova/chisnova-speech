#!/bin/bash
# Android 용 libasr_api.so 빌드.
#
#   ./android.sh                # 두 ABI 모두
#   ./android.sh arm64-v8a      # 하나만
#   ./android.sh --install      # 빌드 후 android_mercury/app/src/main/libs 로 복사
#
# NDK 위치는 아래 순서로 찾는다. 호스트(리눅스/macOS)에 따라 프리빌트 디렉터리가
# 다르므로 경로를 코드에 박지 않는다 — 리눅스용 NDK 를 macOS 로 복사해 오면 동작하지 않는다.
#   1) 환경변수 ANDROID_NDK_HOME
#   2) ~/Android/android-ndk-*  (내림차순으로 최신)
#   3) ~/Library/Android/sdk/ndk/*  (Android Studio 설치 위치)
set -e
cd "$(dirname "$0")"

find_ndk() {
    if [ -n "$ANDROID_NDK_HOME" ] && [ -d "$ANDROID_NDK_HOME" ]; then
        echo "$ANDROID_NDK_HOME"
        return
    fi
    local candidate
    for pattern in "$HOME/Android/android-ndk-"* "$HOME/Library/Android/sdk/ndk/"*; do
        # 툴체인이 실제로 들어 있는 것만 고른다(호스트용 프리빌트가 있어야 한다)
        for candidate in $(ls -d $pattern 2>/dev/null | sort -r); do
            if ls "$candidate/toolchains/llvm/prebuilt/" >/dev/null 2>&1; then
                echo "$candidate"
                return
            fi
        done
    done
}

NDK=$(find_ndk)
if [ -z "$NDK" ]; then
    echo "NDK 를 찾지 못했다. ANDROID_NDK_HOME 을 지정하거나 ~/Android 아래에 설치하라." >&2
    exit 1
fi

HOST_TAG=$(ls "$NDK/toolchains/llvm/prebuilt/" | head -1)
echo "NDK: $NDK (host: $HOST_TAG)"

INSTALL=0
ABIS=()
for arg in "$@"; do
    case "$arg" in
        --install) INSTALL=1 ;;
        arm64-v8a|armeabi-v7a) ABIS+=("$arg") ;;
        *) echo "usage: $0 [arm64-v8a|armeabi-v7a] [--install]" >&2; exit 2 ;;
    esac
done
[ ${#ABIS[@]} -eq 0 ] && ABIS=(arm64-v8a armeabi-v7a)

JOBS=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu )

build_abi() {
    local abi=$1
    local build_dir="${abi//-/_}_build"

    echo "== $abi 빌드 =="
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    (
        cd "$build_dir"
        export TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/"
        cmake -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
              -DANDROID_NDK="$NDK" \
              -DCMAKE_ANDROID_NDK="$NDK" \
              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
              -DCMAKE_BUILD_TYPE=Release \
              -DBUILD_SHARED_LIBS=ON \
              -DCMAKE_CROSSCOMPILING=True \
              -DNEON_ENABLE=ON \
              -DANDROID_ABI="$abi" \
              -DANDROID_PLATFORM=android-21 \
              ../.
        make -j"$JOBS"
    )
    # 산출물은 export/lib 아래로 모은다 — 프로젝트는 여기서 복사해 간다
    local out_dir="$(pwd)/lib/android/$abi"
    mkdir -p "$out_dir"
    cp "$build_dir/libasr_api.so" "$out_dir/"
    echo "   -> lib/android/$abi/libasr_api.so"
}

for abi in "${ABIS[@]}"; do
    build_abi "$abi"
done

if [ "$INSTALL" = "1" ]; then
    # 앱 쪽 복사는 프로젝트의 copy.sh 가 담당한다(양쪽 규칙을 하나로 유지)
    ../../android_mercury/app/src/main/libs/copy.sh
fi

echo "완료."
