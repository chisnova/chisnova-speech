#!/bin/bash
# 사용법: ./stream_conformer_infer.sh <wav_path> [chunk_ms]
export PYTHONPATH=$PWD
model_path="${MODEL_PATH:-$PWD/sample_model}"
python3 util/stream_conformer_inference.py \
    --model_path "$model_path" \
    --wav_path "$1" \
    --chunk_ms "${2:-100}"
