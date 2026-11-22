#!/bin/bash
# 모델 export 부터 Android / iOS 라이브러리까지 한 곳에서 만든다.
#
#   ./excute.sh                     # 이 환경에서 가능한 단계 전부
#   ./excute.sh model               # model_state 만
#   ./excute.sh android ios         # 라이브러리만
#   ./excute.sh score --install     # 채점 자산(phone.txt, function_words.txt)만 갱신
#   ./excute.sh --install           # 만든 뒤 각 프로젝트로 복사까지
#   MODEL_DIR=../../model/other ./excute.sh model    # 다른 체크포인트로
#
# 단계
#   model    학습 체크포인트 -> model_state/   (torch + 체크포인트 필요 = 보통 학습 서버)
#   score    cpp/score 의 채점 자산을 model_state 로 복사 (export 없이)
#   android  lib/android/<abi>/libasr_api.so   (NDK 필요)
#   ios      lib/ios/<sdk>/libasr_api.a + include/  (macOS + Xcode 필요)
#   install  각 프로젝트의 copy.sh 를 불러 결과물을 가져가게 한다
#
# 원칙: 산출물은 무조건 이 디렉터리(cpp/export) 아래에서만 만든다.
# 프로젝트는 자기 copy.sh 로 복사만 해 간다 — 빌드 위치가 갈리면 어느 소스로 만든
# 라이브러리인지 추적할 수 없게 된다.
set -e
# cd 뒤에는 $0 이 더 이상 유효하지 않으므로 미리 절대 경로로 잡아 둔다(--help 용)
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

EXPORT_DIR="$(pwd)"
REPO_ROOT="$(cd ../.. && pwd)"
MODEL_ROOT="$REPO_ROOT/model"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/stream_conformer}"

DO_MODEL=0
DO_ANDROID=0
DO_IOS=0
DO_INSTALL=0
DO_SCORE=0
EXPLICIT=0

for arg in "$@"; do
    case "$arg" in
        model)     DO_MODEL=1;   EXPLICIT=1 ;;
        android)   DO_ANDROID=1; EXPLICIT=1 ;;
        ios)       DO_IOS=1;     EXPLICIT=1 ;;
        score)     DO_SCORE=1;   EXPLICIT=1 ;;
        --install) DO_INSTALL=1 ;;
        -h|--help) sed -n '2,20p' "$SCRIPT_PATH"; exit 0 ;;
        *) echo "모르는 인자: $arg" >&2
           echo "쓸 수 있는 것: model android ios score --install" >&2; exit 2 ;;
    esac
done

# 단계를 지정하지 않았으면 이 환경에서 가능한 것을 모두 한다
if [ "$EXPLICIT" = "0" ]; then
    DO_MODEL=1
    DO_ANDROID=1
    [ "$(uname)" = "Darwin" ] && DO_IOS=1
fi

log() { echo ""; echo "=============== $* ==============="; }

# ---------------------------------------------------------------- score assets

# 채점에만 쓰는 자산을 model_state 로 모은다.
#
# 학습에서 나온 것(vocab, full.dict)과 달리 이것들은 모델과 무관하게 사람이 정하는
# 규칙이라 원본을 cpp/score 에 둔다. 목록 한 줄 고치자고 export 를 통째로 다시
# 돌릴 이유가 없으므로 단독 단계(score)로도 부를 수 있게 했다.
copy_score_assets() {
    local dest=$1
    cp -f "$EXPORT_DIR/../score/phone.txt" "$dest/"
    cp -f "$EXPORT_DIR/../score/function_words.txt" "$dest/"
}

sync_score_assets() {
    log "채점 자산 갱신"
    if [ ! -d "$EXPORT_DIR/model_state" ]; then
        echo "   model_state 가 없다. 먼저 ./excute.sh model 로 만들어야 한다." >&2
        return 1
    fi
    copy_score_assets "$EXPORT_DIR/model_state"
    echo "   phone.txt, function_words.txt -> model_state"
}

# ---------------------------------------------------------------- model_state

can_build_model() {
    if ! python3 -c "import torch" >/dev/null 2>&1; then
        echo "   torch 가 없다 — 모델 export 는 학습 환경에서 한다."
        return 1
    fi
    if [ ! -f "$MODEL_DIR/asr_model.pt" ]; then
        echo "   체크포인트가 없다: $MODEL_DIR/asr_model.pt"
        return 1
    fi
    return 0
}

build_model_state() {
    log "model_state 생성"
    if ! can_build_model; then
        if [ -d "$EXPORT_DIR/model_state" ]; then
            echo "   기존 model_state 를 그대로 쓴다."
            return 0
        fi
        echo "   model_state 가 없다. 학습 환경에서 만들어 가져와야 한다." >&2
        return 1
    fi

    local model_name
    model_name="$(basename "$MODEL_DIR")"
    (cd "$MODEL_ROOT" && PYTHONPATH="$PWD" \
        python3 exporter/stream_conformer_cpp_exporter.py "$model_name")

    # exporter 는 manifest 에 파일 이름만 적는다. vocab/사전/음소 원본은 여기서 채운다 —
    # 이것을 빠뜨리면 C++ 이 model.json 이 가리키는 파일을 찾지 못한다.
    local staged="${MODEL_DIR}_cpp_export"
    for data_file in data/mercury.model data/mercury.vocab data/full.dict; do
        cp -f "$MODEL_ROOT/$data_file" "$staged/"
    done
    copy_score_assets "$staged"

    rm -rf "$EXPORT_DIR/model_state"
    cp -r "$staged" "$EXPORT_DIR/model_state"
    echo "   -> model_state ($(ls "$EXPORT_DIR/model_state" | wc -l | tr -d ' ') 개 파일)"
}

# ---------------------------------------------------------------- libraries

build_android() {
    log "Android 라이브러리"
    ./android.sh
}

build_ios() {
    log "iOS 라이브러리"
    if [ "$(uname)" != "Darwin" ]; then
        echo "   macOS 가 아니라 건너뛴다."
        return 0
    fi
    ./ios.sh
}

# ---------------------------------------------------------------- install

install_all() {
    log "프로젝트로 복사"
    local android_libs="$REPO_ROOT/android_mercury/app/src/main/libs/copy.sh"
    local android_assets="$REPO_ROOT/android_mercury/app/src/main/assets/copy.sh"
    local ios_engine="$REPO_ROOT/ios_mercury/Engine/copy.sh"

    [ -x "$android_libs" ] && "$android_libs"
    [ -x "$android_assets" ] && [ -d "$EXPORT_DIR/model_state" ] && "$android_assets"
    [ -x "$ios_engine" ] && [ -d "$EXPORT_DIR/lib/ios" ] && "$ios_engine"
}

# ---------------------------------------------------------------- run

[ "$DO_SCORE" = "1" ]   && sync_score_assets
[ "$DO_MODEL" = "1" ]   && build_model_state
[ "$DO_ANDROID" = "1" ] && build_android
[ "$DO_IOS" = "1" ]     && build_ios
[ "$DO_INSTALL" = "1" ] && install_all

log "요약"
[ -d "$EXPORT_DIR/model_state" ] && \
    echo "model_state : $(ls "$EXPORT_DIR/model_state" | wc -l | tr -d ' ') 개 파일"
if [ -d "$EXPORT_DIR/lib" ]; then
    find "$EXPORT_DIR/lib" -type f \( -name "*.so" -o -name "*.a" -o -name "*.h" \) | sort | while read -r f; do
        printf "%-46s %s\n" "${f#$EXPORT_DIR/lib/}" "$(ls -lh "$f" | awk '{print $5}')"
    done
fi
echo ""
echo "완료."
