#!/bin/bash
# 배포 산출물(model_state + shared lib)을 한 번에 생성한다.
#
# 사용법:
#   ./package_release.sh                 # export 확인 -> model_state 조립 -> android 양 ABI 빌드
#   ./package_release.sh --with-linux    # linux libasr_api.so 도 함께 빌드
#   ./package_release.sh --force-export  # 체크포인트가 바뀌었을 때 export 부터 강제 재생성
#
# 산출물:
#   model_state/                  앱 assets 에 넣을 모델 폴더 (텐서 + manifest + vocab/사전)
#   arm64_v8a_build/libasr_api.so
#   armeabi_v7a_build/libasr_api.so
#   linux_build/libasr_api.so     (--with-linux 일 때)
#
# 앱 반영 (android.sh 의 기존 관례):
#   cp -r model_state              ../../android/app/src/main/assets/
#   cp arm64_v8a_build/libasr_api.so   .../libs/arm64-v8a/
#   cp armeabi_v7a_build/libasr_api.so .../libs/armeabi-v7a/
set -e
cd "$(dirname "$0")"

MODEL_ROOT=../../model
EXPORT_DIR=$MODEL_ROOT/stream_conformer_cpp_export

WITH_LINUX=false
FORCE_EXPORT=false
for arg in "$@"; do
    case "$arg" in
        --with-linux)   WITH_LINUX=true ;;
        --force-export) FORCE_EXPORT=true ;;
        *) echo "unknown option: $arg"; exit 1 ;;
    esac
done

echo "== 1) 모델 export =="
if [ "$FORCE_EXPORT" = true ] || [ ! -f "$EXPORT_DIR/model.json" ]; then
    (cd $MODEL_ROOT && PYTHONPATH=$PWD python3 exporter/stream_conformer_cpp_exporter.py stream_conformer)
else
    echo "  기존 export 재사용: $EXPORT_DIR (재생성하려면 --force-export)"
fi

echo "== 2) model_state 조립 =="
# vocab/사전/phone 은 항상 git 추적 원본에서 동기화한다 (run_api_test.sh 와 동일 규약)
rm -rf model_state
mkdir model_state
cp -r "$EXPORT_DIR"/. model_state/
cp -f $MODEL_ROOT/data/mercury.model $MODEL_ROOT/data/mercury.vocab $MODEL_ROOT/data/full.dict model_state/
cp -f ../score/phone.txt model_state/
echo "  model_state: $(ls model_state | wc -l)개 파일, $(du -sh model_state | cut -f1)"

echo "== 3) android shared lib 빌드 (arm64-v8a, armeabi-v7a) =="
./android.sh

if [ "$WITH_LINUX" = true ]; then
    echo "== 4) linux shared lib 빌드 =="
    ./linux.sh
fi

echo ""
echo "== 산출물 =="
ls -la arm64_v8a_build/libasr_api.so armeabi_v7a_build/libasr_api.so 2>/dev/null
[ "$WITH_LINUX" = true ] && ls -la linux_build/libasr_api.so 2>/dev/null
echo "model_state/ 준비 완료 — 앱 assets 에 복사해서 사용"
