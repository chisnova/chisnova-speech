#!/bin/bash
# 모델을 학습한다. 결과는 모델 폴더(기본 sample_model/)에 쌓인다.
#
#   ./train.sh                           sample_model/config.json 으로 학습
#   MODEL_PATH=$PWD/my_model ./train.sh  다른 모델 폴더로 학습 (config.json 이 있어야 한다)
#   RUN_EVAL=1 ./train.sh                인식률을 주기적으로 재는 평가 스케줄러도 띄운다
#
# 학습은 백그라운드로 돈다. 진행은 <모델 폴더>/train.log 로 본다.
# 모델 폴더에 asr_model.pt 가 있으면 그 가중치에서 이어서 학습한다.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
set -u

export PYTHONPATH=$PWD
model_path="$(readlink -f "${MODEL_PATH:-$PWD/sample_model}")"

if [ ! -f "$model_path/config.json" ]; then
    echo "[train.sh] 설정이 없습니다: $model_path/config.json" >&2
    exit 1
fi

start_background() {          # 표시이름  중복확인패턴  로그파일  명령...
    local name=$1 pattern=$2 log_file=$3
    shift 3
    if pgrep -f "$pattern" > /dev/null; then
        echo "[train.sh] $name — 이미 실행 중이라 건너뜁니다"
        return
    fi
    nohup "$@" >> "$log_file" 2>&1 &
    echo "[train.sh] $name 시작 (pid $!)  로그: $log_file"
}

# 학습 — 어느 모델이든 하나만 돈다. 두 개를 띄우면 GPU 를 나눠 쓰다 둘 다 OOM 이 난다.
# 로그는 이어 쓴다(덮어쓰면 재시작할 때마다 손실 이력이 날아간다).
start_background "학습" "trainer/train_main.py" "$model_path/train.log" \
    python3 -u trainer/train_main.py "$model_path" \
        --config_path "$model_path/config.json"
echo "[train.sh]   진행 보기: tail -f $model_path/train.log"

# 평가 스케줄러는 모델 폴더마다 하나씩 뜬다.
if [ "${RUN_EVAL:-0}" = "1" ]; then
    # 대표 지표는 ctc2 다. EVAL_PRIMARY / EVAL_CHUNK 환경변수로 바꿀 수 있다
    # (예: EVAL_PRIMARY=ctc1 EVAL_CHUNK=12 ./train.sh).
    eval_head_args="--heads ctc1,ctc2 --primary ${EVAL_PRIMARY:-ctc2}"
    if [ -n "${EVAL_CHUNK:-}" ]; then
        eval_head_args="$eval_head_args --eval_chunk ${EVAL_CHUNK}"
    fi
    start_background "평가 스케줄러" "eval_scheduler.py --model_path $model_path " \
        "$model_path/eval_scheduler.log" \
        python3 -u util/eval_scheduler.py --model_path "$model_path" --per_corpus 200 \
            $eval_head_args
fi
