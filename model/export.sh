#!/bin/bash
# 학습한 모델을 C++ 엔진이 읽는 폴더로 내보낸다.
#
#   ./export.sh                           sample_model -> sample_model_cpp_export
#   MODEL_PATH=$PWD/my_model ./export.sh  다른 모델 폴더를 내보낸다
#   EXPORT_PATH=/tmp/out ./export.sh      결과를 다른 곳에 쓴다
#
# 입력  <모델 폴더>/config.json, asr_model.pt
# 출력  <모델 폴더>_cpp_export/
#         *.pt, model.json                  텐서와 그 목록 (exporter 가 만든다)
#         mercury.model, mercury.vocab      토큰 사전        (data/ 에서 복사)
#         full.dict                         발음사전         (data/ 에서 복사)
#         phone.txt, *_words.txt            채점용 목록      (cpp/score/ 에서 복사)
#
# exporter 는 model.json 에 사전·목록 파일의 이름만 적는다. 그 파일들을 같은 폴더에
# 넣어 줘야 C++ 엔진이 이 폴더 하나만으로 모델을 읽는다.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

export PYTHONPATH=$PWD
model_path="$(readlink -f "${MODEL_PATH:-$PWD/sample_model}")"
export_path="${EXPORT_PATH:-${model_path}_cpp_export}"
score_dir="$PWD/../cpp/score"

for required in "$model_path/config.json" "$model_path/asr_model.pt"; do
    if [ ! -f "$required" ]; then
        echo "[export.sh] 파일이 없습니다: $required" >&2
        exit 1
    fi
done

# 예전 텐서가 섞이지 않게 비우고 시작한다.
rm -rf "$export_path"

python3 exporter/stream_conformer_cpp_exporter.py "$model_path" --export_path "$export_path"

cp -f data/mercury.model data/mercury.vocab data/full.dict "$export_path/"
for score_file in phone.txt function_words.txt proper_nouns.txt ungraded_words.txt; do
    cp -f "$score_dir/$score_file" "$export_path/"
done

echo "[export.sh] 완료: $export_path ($(ls "$export_path" | wc -l) 개 파일, $(du -sh "$export_path" | cut -f1))"
