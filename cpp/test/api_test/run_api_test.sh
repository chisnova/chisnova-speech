#!/usr/bin/env bash
# Run the API smoke test, compare C++ and Python on a WAV list, or run both.
# MODEL_PATH selects the Python checkpoint. MERCURY_TEST_MODEL_DIR can select
# an existing matching export. Resources in a supplied export are read-only.
set -euo pipefail
# Resolve caller-supplied paths before switching to the test resource directory.
for path_var in MODEL_PATH MERCURY_TEST_MODEL_DIR MERCURY_COMPARE_OUT MERCURY_TEST_BUILD_ROOT; do
    if [[ -n "${!path_var:-}" ]]; then
        printf -v "$path_var" '%s' "$(realpath -m "${!path_var}")"
    fi
done
cd "$(dirname "$0")"
mode=${1:-api}
case "$mode" in api|compare|all) ;; *) echo "Usage: $0 [api|compare|all] [wav_list]" >&2; exit 2 ;; esac
root=$(realpath ../../..)
default_model="$root/model/stream_conformer"
if [[ ! -f "$default_model/asr_model.pt" ]]; then
    default_model="$root/arifact/stream_conformer_mercury3_adapt_1p5s"
fi
MODEL_PATH=${MODEL_PATH:-$default_model}
export MODEL_PATH
export_dir=${MERCURY_TEST_MODEL_DIR:-${MODEL_PATH}_cpp_export}
if [[ ! -f "$export_dir/model.json" ]]; then
    if [[ -n "${MERCURY_TEST_MODEL_DIR:-}" ]]; then
        echo "Missing manifest in $export_dir" >&2
        exit 1
    fi
    PYTHONPATH="$root/model" python3 "$root/model/exporter/stream_conformer_cpp_exporter.py" "$MODEL_PATH"
    cp "$root/model/data/"{mercury.model,mercury.vocab,full.dict} "$root/cpp/score/phone.txt" "$export_dir/"
fi
MERCURY_TEST_MODEL_DIR=$(realpath "$export_dir")
export MERCURY_TEST_MODEL_DIR
build_dir="${MERCURY_TEST_BUILD_ROOT:-$root/cpp/test/build}/api"
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
targets=()
if [[ "$mode" != compare ]]; then targets+=(api_tester); fi
if [[ "$mode" != api ]]; then targets+=(wav_list_compare); fi
cmake --build "$build_dir" --target "${targets[@]}" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

if [[ "$mode" != compare ]]; then "$build_dir/api_tester"; fi
if [[ "$mode" != api ]]; then
    list=${2:-wav_local.list}
    if [[ "$list" == wav_local.list && ! -f "$list" ]]; then
        python3 - <<'PY'
from pathlib import Path
folder = Path("../resource/StevJobs_Standford/processing")
paths = [folder / Path(line).name for line in Path("wav.list").read_text().splitlines() if line]
Path("wav_local.list").write_text("".join(f"{p}\n" for p in paths if p.is_file()))
PY
    fi
    output=${MERCURY_COMPARE_OUT:-compare_out}
    labels=../resource/StevJobs_Standford/steve_order.txt
    mkdir -p "$output"
    "$build_dir/wav_list_compare" "$list" "$output" "$labels"
    PYTHONPATH="$root/model" python3 python_compare_wav_list.py "$list" "$output" \
        --print_transcripts --label_map "$labels" --scorer_dir "$MERCURY_TEST_MODEL_DIR"
fi
