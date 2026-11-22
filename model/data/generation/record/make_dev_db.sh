#!/usr/bin/env bash
# 어디서 실행해도 되도록 스크립트 위치로 이동한다.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
set -euo pipefail

python3 data_gen.py \
    --wavpath /home/chisnova/archive1/ASR_DATA/asr_list/dev_tot.wlist \
    --txtpath /home/chisnova/archive1/ASR_DATA/asr_list/dev_tot.lab \
    --savepath /home/chisnova/data/records/dev \
    --vocab /home/chisnova/Desktop/mercury/model/data/mercury.model \
    --map_size_gb 1024 \
    --commit_interval 2000
