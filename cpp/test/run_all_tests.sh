#!/bin/bash
#
# run_all_tests.sh - Comprehensive test runner for the Mercury ASR C++ engine.
#
# Runs all 4 test suites sequentially: layer -> module -> api -> java_api
# Each suite failure stops the pipeline unless --continue is specified.
#
# Dependency chain (model weights → test data → C++ tests):
#   layer_test:     NONE — uses random weights, self-contained
#   module_test:    model/ CODE only (class definitions via PYTHONPATH), random weights
#   api_test:       model/ CODE + WEIGHTS — loads checkpoint via saveModelAndInfer.py
#   java_api_test:  model/ CODE + WEIGHTS — loads checkpoint via saveModelAndInfer.py
#
# When model checkpoint files in model/stream_conformer_transducer/ change,
# api_test and java_api_test binary data must be regenerated. Use --regen-data
# to force this, or rely on automatic staleness detection (compares model
# checkpoint mtime against generated binary data).
#
# Usage:
#   ./run_all_tests.sh                  # Run all test suites
#   ./run_all_tests.sh --layer          # Run only layer tests
#   ./run_all_tests.sh --module --api   # Run module and api tests
#   ./run_all_tests.sh --skip-data      # Skip Python test data regeneration
#   ./run_all_tests.sh --regen-data     # Force regeneration of all test data
#   ./run_all_tests.sh --clean          # Clean build dirs before building
#   ./run_all_tests.sh --continue       # Don't stop on first suite failure
#   ./run_all_tests.sh --help           # Show this help
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#   2 - Build failed

set -uo pipefail

# ==============================================================================
# Color codes
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ==============================================================================
# Paths
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_DIR="${SCRIPT_DIR}/layer_test"
MODULE_DIR="${SCRIPT_DIR}/module_test"
API_DIR="${SCRIPT_DIR}/../decode/api"
JAVA_DIR="${SCRIPT_DIR}/../decode/java_api"
MODEL_DIR="${SCRIPT_DIR}/../../model"
MODEL_CHECKPOINT_DIR="${MODEL_DIR}/stream_conformer_transducer"

# Keep caches copied from another checkout out of the build.
BUILD_ROOT="${MERCURY_TEST_BUILD_ROOT:-${SCRIPT_DIR}/build}"
BUILD_ROOT=$(realpath -m "$BUILD_ROOT")
RUNTIME_BUILD_DIR="${SCRIPT_DIR}/../build"
if [[ -n "${MERCURY_TEST_BUILD_ROOT:-}" ]]; then RUNTIME_BUILD_DIR="$BUILD_ROOT/runtime"; fi
declare -A cleaned_build_dirs
if [[ -n "${MERCURY_TEST_MODEL_DIR:-}" ]]; then
    MERCURY_TEST_MODEL_DIR=$(realpath -m "$MERCURY_TEST_MODEL_DIR")
    export MERCURY_TEST_MODEL_DIR
elif [[ -f "$RUNTIME_BUILD_DIR/dist/model_state/model.json" ]]; then
    export MERCURY_TEST_MODEL_DIR="$RUNTIME_BUILD_DIR/dist/model_state"
elif [[ -f "${SCRIPT_DIR}/../../android/app/src/main/assets/model_state/model.json" && ! -d "$MODEL_CHECKPOINT_DIR" ]]; then
    export MERCURY_TEST_MODEL_DIR="${SCRIPT_DIR}/../../android/app/src/main/assets/model_state"
fi

# ==============================================================================
# Global state
# ==============================================================================
SUITES_PASSED=0
SUITES_FAILED=0
SUITES_SKIPPED=0
TOTAL_TESTS_PASSED=0
TOTAL_TESTS_FAILED=0
SUITE_RESULTS=()   # array of "name:status:time:passed:failed"
OVERALL_START=0
OPT_SKIP_DATA=false
OPT_REGEN_DATA=false
OPT_CLEAN=false
OPT_CONTINUE=false
OPT_CLEANUP_DATA=false   # [2026-04-08] 테스트 후 생성된 .pt 데이터 파일 정리
RUN_LAYER=false
RUN_MODULE=false
RUN_API=false
RUN_JAVA=false
RUN_ALL=true

# ==============================================================================
# Helper functions
# ==============================================================================

print_banner() {
    echo ""
    echo -e "${BOLD}${CYAN}================================================================${NC}"
    echo -e "${BOLD}${CYAN}       Mercury ASR Engine - Test Runner${NC}"
    echo -e "${BOLD}${CYAN}================================================================${NC}"
    echo -e "${DIM}  Started: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
}

print_usage() {
    cat <<'USAGE'
Usage: run_all_tests.sh [OPTIONS]

Options:
  --layer       Run only layer tests
  --module      Run only module tests
  --api         Run only api tests
  --java        Run only java_api tests
  --skip-data   Skip Python test data regeneration
  --regen-data    Force regeneration of test data (use when model weights change)
  --cleanup-data  Remove generated .pt data files after tests complete
  --clean         Clean build directories before building
  --continue    Continue running suites even if one fails
  --help        Show this help

When no suite options are given, all 4 suites run sequentially:
  layer -> module -> api -> java_api

Exit codes:
  0  All tests passed
  1  One or more tests failed
  2  Build failed
USAGE
}

# Returns current time in seconds (with nanoseconds if available)
now_seconds() {
    date +%s
}

# Format elapsed seconds as "Xm Ys"
format_duration() {
    local secs=$1
    if (( secs >= 60 )); then
        printf "%dm %ds" $((secs / 60)) $((secs % 60))
    else
        printf "%ds" "$secs"
    fi
}

print_suite_header() {
    local name="$1"
    local index="$2"
    local total="$3"
    echo ""
    echo -e "${BOLD}${CYAN}----------------------------------------------------------------${NC}"
    echo -e "${BOLD}${CYAN}  [${index}/${total}] ${name}${NC}"
    echo -e "${BOLD}${CYAN}----------------------------------------------------------------${NC}"
}

print_suite_result() {
    local name="$1"
    local status="$2"
    local duration="$3"
    if [[ "$status" == "PASS" ]]; then
        echo -e "${GREEN}${BOLD}  [PASS]${NC} ${name}  ${DIM}(${duration})${NC}"
    elif [[ "$status" == "FAIL" ]]; then
        echo -e "${RED}${BOLD}  [FAIL]${NC} ${name}  ${DIM}(${duration})${NC}"
    else
        echo -e "${YELLOW}${BOLD}  [SKIP]${NC} ${name}"
    fi
}

# ==============================================================================
# Model staleness detection
# ==============================================================================
# Returns 0 (true) if model checkpoint is newer than the generated binary data,
# meaning the test data is stale and should be regenerated.
is_model_data_stale() {
    local binary_dir="$1"

    # If binary dir doesn't exist, data is "stale" (needs generation)
    if [[ ! -d "$binary_dir" ]]; then
        return 0
    fi

    # If no checkpoint dir exists, can't compare — not stale
    if [[ ! -d "$MODEL_CHECKPOINT_DIR" ]]; then
        return 1
    fi

    # Find the newest .pt file in the model checkpoint directory
    local newest_model
    newest_model=$(find "$MODEL_CHECKPOINT_DIR" -maxdepth 1 -name "*.pt" -printf '%T@ %p\n' 2>/dev/null \
                   | sort -rn | head -1 | cut -d' ' -f2-)

    if [[ -z "$newest_model" ]]; then
        return 1  # No checkpoint files found
    fi

    # Find the newest file in the binary output directory
    local newest_binary
    newest_binary=$(find "$binary_dir" -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null \
                    | sort -rn | head -1 | cut -d' ' -f2-)

    if [[ -z "$newest_binary" ]]; then
        return 0  # No binary files — stale
    fi

    # Compare: if model is newer than binary, data is stale
    if [[ "$newest_model" -nt "$newest_binary" ]]; then
        echo -e "  ${YELLOW}>> Model checkpoint newer than test data — regeneration needed${NC}"
        echo -e "  ${DIM}>>   model: $(basename "$newest_model") ($(date -r "$newest_model" '+%Y-%m-%d %H:%M'))${NC}"
        echo -e "  ${DIM}>>   data:  $(basename "$newest_binary") ($(date -r "$newest_binary" '+%Y-%m-%d %H:%M'))${NC}"
        return 0
    fi

    return 1  # Data is up to date
}

# ==============================================================================
# Generate Python test data if needed
# ==============================================================================
# For layer_test and module_test: runs python3 main.py in the python/ subdir.
# These tests use random weights; the marker file check is sufficient.
generate_data_if_needed() {
    local suite_dir="$1"
    local json_marker="$2"
    local pythonpath_extra="${3:-}"

    if [[ "$OPT_SKIP_DATA" == "true" ]]; then
        echo -e "  ${DIM}>> Skipping data generation (--skip-data)${NC}"
        return 0
    fi

    local python_dir="${suite_dir}/python"
    local binary_dir="${python_dir}/binary"
    local marker_file="${binary_dir}/${json_marker}"

    if [[ "$OPT_REGEN_DATA" == "true" ]]; then
        echo -e "  ${YELLOW}>> Forcing data regeneration (--regen-data)${NC}"
        rm -rf "$binary_dir"
    elif [[ -f "$marker_file" && "$marker_file" -nt "$python_dir/main.py" ]]; then
        # [2026-04-08] json 마커 존재 + .pt 파일도 존재하는지 확인
        # 기존: json만 있으면 스킵 → .pt 파일 없이 테스트 실행되어 FAIL
        # 변경: .pt 파일 최소 1개 존재 여부도 확인
        local pt_count
        pt_count=$(find "$binary_dir" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
        if [[ "$pt_count" -gt 0 ]]; then
            echo -e "  ${DIM}>> Test data exists (${json_marker} + ${pt_count} .pt files), skipping generation${NC}"
            return 0
        else
            echo -e "  ${YELLOW}>> JSON marker exists but no .pt data files — regenerating${NC}"
            rm -rf "$binary_dir"
        fi
    fi

    echo -e "  ${YELLOW}>> Generating test data via Python...${NC}"
    mkdir -p "$binary_dir"

    local old_pythonpath="${PYTHONPATH:-}"
    if [[ -n "$pythonpath_extra" ]]; then
        export PYTHONPATH="${pythonpath_extra}:${old_pythonpath}"
    fi

    if ! (cd "$python_dir" && python3 main.py 2>&1); then
        echo -e "  ${RED}>> FAILED: Test data generation failed${NC}"
        if [[ -n "$pythonpath_extra" ]]; then
            export PYTHONPATH="$old_pythonpath"
        fi
        return 1
    fi

    if [[ -n "$pythonpath_extra" ]]; then
        export PYTHONPATH="$old_pythonpath"
    fi

    echo -e "  ${GREEN}>> Test data generated${NC}"
    return 0
}

# ==============================================================================
# Generate model-dependent test data (api_test, java_api_test)
# ==============================================================================
# These suites load actual model checkpoints via saveModelAndInfer.py.
# Data flow: model/stream_conformer_transducer/*.pt → saveModelAndInfer.py → binary/
generate_model_data_if_needed() {
    local suite_dir="$1"
    local suite_name="$2"

    if [[ -n "${MERCURY_TEST_MODEL_DIR:-}" ]]; then
        if [[ ! -r "$MERCURY_TEST_MODEL_DIR/model.json" ]]; then
            echo "Model manifest not found in $MERCURY_TEST_MODEL_DIR" >&2
            return 1
        fi
        echo "  Using exported model: $MERCURY_TEST_MODEL_DIR"
        return 0
    fi

    local python_dir="${suite_dir}/python"
    local binary_dir="${python_dir}/binary"

    # [2026-08-20] python/binary 가 심링크면 신형 export(stream_conformer_cpp_export)를
    # 가리키는 것이다 — 구형(mercury5) 재생성 플로우가 심링크 너머의 export 원본을
    # 덮어쓰지 않도록 여기서 끝낸다. 갱신은 run_api_test.sh / exporter 가 담당한다.
    if [[ -L "$binary_dir" ]]; then
        echo -e "  ${DIM}>> python/binary is a symlink to the cpp export — legacy regeneration skipped${NC}"
        return 0
    fi

    # [2026-04-08] scorer 리소스 파일은 데이터 재생성 여부와 무관하게 항상 확인/복사
    # C++ AsrAPI 초기화 시 phone.txt, full.dict, mercury5.model, mercury5.vocab 필요
    mkdir -p "$binary_dir"
    # score_dir: cpp/score/ (SCRIPT_DIR = cpp/test/)
    local score_dir="${SCRIPT_DIR}/../score"
    local resource_dir="${SCRIPT_DIR}/resource"
    for f in phone.txt full.dict mercury5.model; do
        if [[ -f "${score_dir}/${f}" ]] && [[ ! -f "${binary_dir}/${f}" ]]; then
            cp "${score_dir}/${f}" "${binary_dir}/${f}"
        fi
    done
    if [[ -f "${resource_dir}/mercury5.vocab" ]] && [[ ! -f "${binary_dir}/mercury5.vocab" ]]; then
        cp "${resource_dir}/mercury5.vocab" "${binary_dir}/mercury5.vocab"
    fi

    if [[ "$OPT_SKIP_DATA" == "true" ]]; then
        echo -e "  ${DIM}>> Skipping data generation (--skip-data)${NC}"
        return 0
    fi

    local needs_regen=false

    # [2026-04-08] .pt weight 파일 존재 여부로 데이터 유효성 판단
    # 기존: 디렉토리가 비어있지 않으면 "up to date" → 리소스 파일만 있고 .pt 없어도 스킵
    # 변경: .pt 파일 최소 1개 존재 확인
    local pt_count
    pt_count=$(find "$binary_dir" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)

    if [[ "$OPT_REGEN_DATA" == "true" ]]; then
        echo -e "  ${YELLOW}>> Forcing data regeneration (--regen-data)${NC}"
        needs_regen=true
    elif [[ "$pt_count" -eq 0 ]]; then
        echo -e "  ${YELLOW}>> No .pt weight files found, generation required${NC}"
        needs_regen=true
    elif is_model_data_stale "$binary_dir"; then
        needs_regen=true
    else
        echo -e "  ${DIM}>> Test data up to date (${pt_count} .pt files)${NC}"
        return 0
    fi

    if [[ "$needs_regen" != "true" ]]; then
        return 0
    fi

    # Verify model checkpoint directory exists
    if [[ ! -d "$MODEL_CHECKPOINT_DIR" ]]; then
        echo -e "  ${RED}>> Model checkpoint dir not found: ${MODEL_CHECKPOINT_DIR}${NC}"
        echo -e "  ${RED}>> Cannot generate ${suite_name} test data without model weights${NC}"
        return 1
    fi

    echo -e "  ${YELLOW}>> Generating model-dependent test data via saveModelAndInfer.py...${NC}"

    mkdir -p "$binary_dir"

    local old_pythonpath="${PYTHONPATH:-}"
    export PYTHONPATH="${MODEL_DIR}:${old_pythonpath}"

    local wav_path="${SCRIPT_DIR}/resource/StevJobs_Standford/processing/SteveJobsStanford00001.wav"

    if ! (cd "$python_dir" && python3 saveModelAndInfer.py \
            --model_path "$MODEL_CHECKPOINT_DIR" \
            --output_path binary \
            --wav_path "$wav_path" 2>&1); then
        echo -e "  ${RED}>> FAILED: Model data generation failed for ${suite_name}${NC}"
        export PYTHONPATH="$old_pythonpath"
        return 1
    fi

    export PYTHONPATH="$old_pythonpath"
    echo -e "  ${GREEN}>> Model test data generated for ${suite_name}${NC}"
    return 0
}

# ==============================================================================
# Build a cmake project
# ==============================================================================
build_cmake() {
    local build_dir="$1"
    local source_dir="$2"

    if [[ "$OPT_CLEAN" == "true" && -z "${cleaned_build_dirs[$build_dir]:-}" ]]; then
        cleaned_build_dirs[$build_dir]=1
        echo -e "  ${YELLOW}>> Cleaning build directory...${NC}"
        rm -rf "$build_dir"
    fi

    local options=()
    if [[ "$build_dir" == "$RUNTIME_BUILD_DIR" ]]; then options+=(-DMERCURY_BUILD_TESTS=ON); fi
    cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release "${options[@]}" || return 2
    cmake --build "$build_dir" --target "$3" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}" || return 2
    echo -e "  ${GREEN}>> Build succeeded${NC}"
    return 0
}

# ==============================================================================
# Suite: layer_test
# ==============================================================================
run_layer_test() {
    echo -e "  ${YELLOW}>> Phase: data generation${NC}"
    generate_data_if_needed "$LAYER_DIR" "layer_test.json" "" || return 1

    echo -e "  ${YELLOW}>> Phase: build${NC}"
    build_cmake "${BUILD_ROOT}/layer" "${LAYER_DIR}" layer_tester || return 2

    local test_binary="${BUILD_ROOT}/layer/layer_tester"
    if [[ ! -x "$test_binary" ]]; then
        echo -e "  ${RED}>> Test binary not found: ${test_binary}${NC}"
        return 2
    fi

    echo -e "  ${YELLOW}>> Phase: test${NC}"
    local output
    local exit_code=0
    output=$(cd "$LAYER_DIR" && "$test_binary" 2>&1) || exit_code=$?

    local passed=0
    local failed=0
    local last_latency=""

    while IFS= read -r line; do
        if [[ "$line" == RESULT:*:PASS ]]; then
            local lname="${line#RESULT:}"
            lname="${lname%:PASS}"
            echo -e "    ${GREEN}[PASS]${NC} ${lname}  ${DIM}${last_latency}${NC}"
            ((passed++)) || true
            last_latency=""
        elif [[ "$line" == RESULT:*:FAIL ]]; then
            local lname="${line#RESULT:}"
            lname="${lname%:FAIL}"
            echo -e "    ${RED}[FAIL]${NC} ${lname}  ${DIM}${last_latency}${NC}"
            ((failed++)) || true
            last_latency=""
        elif [[ "$line" == *"infernece time:"* ]]; then
            # [2026-04-08] latency 표시: inference time 라인을 파싱하여 PASS/FAIL 옆에 표시
            last_latency="(${line##*infernece time: }s)"
        elif [[ "$line" == *"RMS ERROR"* ]]; then
            : # suppress RMS lines (shown via PASS/FAIL)
        elif [[ "$line" == *"EXCEPTION"* ]] || [[ "$line" == *"ERROR"* ]]; then
            echo -e "    ${RED}${line}${NC}"
        fi
    done <<< "$output"

    TOTAL_TESTS_PASSED=$((TOTAL_TESTS_PASSED + passed))
    TOTAL_TESTS_FAILED=$((TOTAL_TESTS_FAILED + failed))

    echo -e "  ${DIM}>> Results: ${passed} passed, ${failed} failed${NC}"

    if [[ $failed -gt 0 || $exit_code -ne 0 ]]; then
        return 1
    fi
    return 0
}

# ==============================================================================
# Suite: module_test
# ==============================================================================
run_module_test() {
    # module_test imports model architecture classes (ConformerEncoder, etc.) from model/
    # via PYTHONPATH, but uses RANDOM weights — no checkpoint loading.
    # Only needs regeneration when Python class signatures change, NOT when model weights change.
    # Data flow: model/ (code only) → python main.py (random weights) → binary/*.pt → C++ module_tester
    local pythonpath_extra="${MODEL_DIR}"

    echo -e "  ${YELLOW}>> Phase: data generation${NC}"
    generate_data_if_needed "$MODULE_DIR" "module_test.json" "$pythonpath_extra" || return 1

    echo -e "  ${YELLOW}>> Phase: build${NC}"
    build_cmake "${BUILD_ROOT}/module" "${MODULE_DIR}" module_tester || return 2

    local test_binary="${BUILD_ROOT}/module/module_tester"
    if [[ ! -x "$test_binary" ]]; then
        echo -e "  ${RED}>> Test binary not found: ${test_binary}${NC}"
        return 2
    fi

    echo -e "  ${YELLOW}>> Phase: test${NC}"
    local output
    local exit_code=0
    output=$(cd "$MODULE_DIR" && "$test_binary" 2>&1) || exit_code=$?

    local passed=0
    local failed=0
    local last_latency=""

    while IFS= read -r line; do
        if [[ "$line" == RESULT:*:PASS ]]; then
            local lname="${line#RESULT:}"
            lname="${lname%:PASS}"
            echo -e "    ${GREEN}[PASS]${NC} ${lname}  ${DIM}${last_latency}${NC}"
            ((passed++)) || true
            last_latency=""
        elif [[ "$line" == RESULT:*:FAIL ]]; then
            local lname="${line#RESULT:}"
            lname="${lname%:FAIL}"
            echo -e "    ${RED}[FAIL]${NC} ${lname}  ${DIM}${last_latency}${NC}"
            ((failed++)) || true
            last_latency=""
        elif [[ "$line" == *"infernece time:"* ]]; then
            last_latency="(${line##*infernece time: }s)"
        elif [[ "$line" == *"RMS ERROR"* ]]; then
            :
        elif [[ "$line" == *"EXCEPTION"* ]] || [[ "$line" == *"ERROR"* ]]; then
            echo -e "    ${RED}${line}${NC}"
        fi
    done <<< "$output"

    TOTAL_TESTS_PASSED=$((TOTAL_TESTS_PASSED + passed))
    TOTAL_TESTS_FAILED=$((TOTAL_TESTS_FAILED + failed))

    echo -e "  ${DIM}>> Results: ${passed} passed, ${failed} failed${NC}"

    if [[ $failed -gt 0 || $exit_code -ne 0 ]]; then
        return 1
    fi
    return 0
}

# ==============================================================================
# Suite: api_test
# ==============================================================================
run_api_test() {
    # api_test loads actual model weights (not random) via saveModelAndInfer.py
    # Data flow: model/stream_conformer_transducer/*.pt → saveModelAndInfer.py → binary/*.pt → C++ api_tester
    echo -e "  ${YELLOW}>> Phase: data generation${NC}"
    generate_model_data_if_needed "$API_DIR" "api_test" || return 1

    echo -e "  ${YELLOW}>> Phase: build${NC}"
    build_cmake "$RUNTIME_BUILD_DIR" "$SCRIPT_DIR/.." api_tester || return 2

    local test_binary="$RUNTIME_BUILD_DIR/api_tester"
    if [[ ! -x "$test_binary" ]]; then
        echo -e "  ${RED}>> Test binary not found: ${test_binary}${NC}"
        return 2
    fi

    echo -e "  ${YELLOW}>> Phase: test${NC}"
    local output
    local exit_code=0
    output=$(cd "$API_DIR" && "$test_binary" smoke 2>&1) || exit_code=$?

    local passed=0
    local failed=0
    local last_latency=""

    while IFS= read -r line; do
        if [[ "$line" == RESULT:*:PASS ]]; then
            local lname="${line#RESULT:}"
            lname="${lname%:PASS}"
            echo -e "    ${GREEN}[PASS]${NC} ${lname}  ${DIM}${last_latency}${NC}"
            ((passed++)) || true
            last_latency=""
        elif [[ "$line" == RESULT:*:FAIL ]]; then
            local lname="${line#RESULT:}"
            lname="${lname%:FAIL}"
            echo -e "    ${RED}[FAIL]${NC} ${lname}  ${DIM}${last_latency}${NC}"
            ((failed++)) || true
            last_latency=""
        elif [[ "$line" == *"initialization time:"* ]]; then
            last_latency="(init: ${line##*initialization time: })"
        elif [[ "$line" == *"ctc asr result:"* ]] || [[ "$line" == *"second pass"* ]]; then
            echo -e "    ${DIM}${line}${NC}"
        elif [[ "$line" == *"ASR RESULT:"* ]]; then
            echo -e "    ${DIM}${line}${NC}"
        elif [[ "$line" == *"Segmentation"* ]] || [[ "$line" == *"terminate called"* ]] || [[ "$line" == *"EXCEPTION"* ]]; then
            # [2026-04-08] 치명적 에러만 FAIL로 카운트 — [ERROR] 로그는 정상 흐름일 수 있음
            echo -e "    ${RED}${line}${NC}"
            ((failed++)) || true
        elif [[ "$line" == *"[ERROR]"* ]]; then
            echo -e "    ${DIM}${line}${NC}"
        fi
    done <<< "$output"

    TOTAL_TESTS_PASSED=$((TOTAL_TESTS_PASSED + passed))
    TOTAL_TESTS_FAILED=$((TOTAL_TESTS_FAILED + failed))

    # If no RESULT: lines were parsed, count by exit code
    if [[ $passed -eq 0 && $failed -eq 0 ]]; then
        if [[ $exit_code -eq 0 ]]; then
            passed=1
            TOTAL_TESTS_PASSED=$((TOTAL_TESTS_PASSED + 1))
        else
            failed=1
            TOTAL_TESTS_FAILED=$((TOTAL_TESTS_FAILED + 1))
        fi
    fi

    echo -e "  ${DIM}>> Results: ${passed} passed, ${failed} failed${NC}"

    if [[ $failed -gt 0 || $exit_code -ne 0 ]]; then
        return 1
    fi
    return 0
}

# ==============================================================================
# Suite: java_api_test
# ==============================================================================
run_java_api_test() {
    # java_api_test loads actual model weights (not random) via saveModelAndInfer.py
    # Data flow: model/stream_conformer_transducer/*.pt → saveModelAndInfer.py → binary/*.pt → libasr_api.so → Java
    echo -e "  ${YELLOW}>> Phase: data generation${NC}"
    generate_model_data_if_needed "$JAVA_DIR" "java_api_test" || return 1

    echo -e "  ${YELLOW}>> Phase: build${NC}"
    local build_dir="$RUNTIME_BUILD_DIR"

    build_cmake "$build_dir" "$SCRIPT_DIR/.." java_api_test || return 2

    echo -e "  ${YELLOW}>> Phase: test${NC}"
    local tmpfile
    tmpfile=$(mktemp /tmp/mercury_java_test.XXXXXX)

    local exit_code=0
    local sample_count=5
    "$build_dir/java_api_test" "$sample_count" 2>&1 | tee "$tmpfile" | awk '
        /PRONOUNCE SCORE:/ {printf "    [PASS %d] score extracted\n", ++count; fflush(); next}
        /LABEL_COMPARE:|Exception|Error|SIGSEGV/ {print; fflush()}
    ' || exit_code=$?

    # 최종 결과 집계 (파일에서 카운트)
    # [2026-04-08] grep -c 결과를 안전하게 정수로 변환
    local passed
    passed=$(grep -c "PRONOUNCE SCORE:" "$tmpfile" 2>/dev/null) || passed=0
    local failed
    failed=$(grep -c -E "Killed|SIGSEGV" "$tmpfile" 2>/dev/null) || failed=0
    local label_matched
    label_matched=$(grep -c "match=true" "$tmpfile" 2>/dev/null) || label_matched=0
    local label_total
    label_total=$(grep -c "LABEL_COMPARE:" "$tmpfile" 2>/dev/null) || label_total=0

    if [[ $passed -ne $sample_count || $label_total -ne $sample_count || $exit_code -ne 0 ]]; then
        tail -n 8 "$tmpfile" >&2
        failed=$((failed + 1))
    fi
    rm -f "$tmpfile"
    TOTAL_TESTS_PASSED=$((TOTAL_TESTS_PASSED + passed))
    TOTAL_TESTS_FAILED=$((TOTAL_TESTS_FAILED + failed))

    echo -e "  ${DIM}>> Results: ${passed} passed, ${failed} failed, label match: ${label_matched}/${label_total}${NC}"

    if [[ $failed -gt 0 || $exit_code -ne 0 ]]; then
        return 1
    fi
    return 0
}

# ==============================================================================
# Run a single suite with timing and result tracking
# ==============================================================================
run_suite() {
    local name="$1"
    local func="$2"
    local index="$3"
    local total="$4"

    print_suite_header "$name" "$index" "$total"

    local start
    start=$(now_seconds)

    local rc=0
    "$func" || rc=$?

    local end
    end=$(now_seconds)
    local elapsed=$((end - start))
    local duration
    duration=$(format_duration "$elapsed")

    if [[ $rc -eq 0 ]]; then
        SUITE_RESULTS+=("${name}:PASS:${duration}")
        ((SUITES_PASSED++)) || true
        print_suite_result "$name" "PASS" "$duration"
    elif [[ $rc -eq 2 ]]; then
        SUITE_RESULTS+=("${name}:BUILD_FAIL:${duration}")
        ((SUITES_FAILED++)) || true
        print_suite_result "$name" "FAIL" "$duration"
        echo -e "  ${RED}>> Build failure (exit code 2)${NC}"
    else
        SUITE_RESULTS+=("${name}:FAIL:${duration}")
        ((SUITES_FAILED++)) || true
        print_suite_result "$name" "FAIL" "$duration"
    fi

    return $rc
}

# ==============================================================================
# Print final summary
# ==============================================================================
print_summary() {
    local overall_end
    overall_end=$(now_seconds)
    local total_elapsed=$((overall_end - OVERALL_START))
    local total_duration
    total_duration=$(format_duration "$total_elapsed")

    echo ""
    echo -e "${BOLD}${CYAN}================================================================${NC}"
    echo -e "${BOLD}${CYAN}       Test Summary${NC}"
    echo -e "${BOLD}${CYAN}================================================================${NC}"
    echo ""

    for entry in "${SUITE_RESULTS[@]}"; do
        local name="${entry%%:*}"
        local rest="${entry#*:}"
        local status="${rest%%:*}"
        local dur="${rest#*:}"

        if [[ "$status" == "PASS" ]]; then
            echo -e "  ${GREEN}[PASS]${NC}  ${name}  ${DIM}(${dur})${NC}"
        elif [[ "$status" == "SKIP" ]]; then
            echo -e "  ${YELLOW}[SKIP]${NC}  ${name}"
        else
            echo -e "  ${RED}[FAIL]${NC}  ${name}  ${DIM}(${dur})${NC}"
        fi
    done

    echo ""
    echo -e "  ${BOLD}Suites:${NC}  ${GREEN}${SUITES_PASSED} passed${NC}, ${RED}${SUITES_FAILED} failed${NC}, ${YELLOW}${SUITES_SKIPPED} skipped${NC}"
    echo -e "  ${BOLD}Tests:${NC}   ${GREEN}${TOTAL_TESTS_PASSED} passed${NC}, ${RED}${TOTAL_TESTS_FAILED} failed${NC}"
    echo -e "  ${BOLD}Time:${NC}    ${total_duration}"
    echo ""

    if [[ $SUITES_FAILED -gt 0 ]]; then
        echo -e "${RED}${BOLD}  SOME SUITES FAILED${NC}"
    else
        echo -e "${GREEN}${BOLD}  ALL SUITES PASSED${NC}"
    fi

    echo -e "${BOLD}${CYAN}================================================================${NC}"
    echo ""
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    # Parse arguments
    for arg in "$@"; do
        case "$arg" in
            --layer)    RUN_LAYER=true; RUN_ALL=false ;;
            --module)   RUN_MODULE=true; RUN_ALL=false ;;
            --api)      RUN_API=true; RUN_ALL=false ;;
            --java)     RUN_JAVA=true; RUN_ALL=false ;;
            --skip-data) OPT_SKIP_DATA=true ;;
            --regen-data) OPT_REGEN_DATA=true ;;
            --cleanup-data) OPT_CLEANUP_DATA=true ;;
            --clean)    OPT_CLEAN=true ;;
            --continue) OPT_CONTINUE=true ;;
            --help|-h)  print_usage; exit 0 ;;
            *)
                echo -e "${RED}Unknown option: ${arg}${NC}" >&2
                print_usage
                exit 1
                ;;
        esac
    done

    if [[ "$OPT_CLEAN" == "true" || "$OPT_CLEANUP_DATA" == "true" ]]; then RUNTIME_BUILD_DIR="$BUILD_ROOT/runtime"; fi

    if [[ "$RUN_ALL" == "true" ]]; then
        RUN_LAYER=true
        RUN_MODULE=true
        RUN_API=true
        RUN_JAVA=true
    fi

    # Build list of suites to run
    local suites=()
    local funcs=()
    if [[ "$RUN_LAYER" == "true" ]]; then
        suites+=("layer_test")
        funcs+=("run_layer_test")
    fi
    if [[ "$RUN_MODULE" == "true" ]]; then
        suites+=("module_test")
        funcs+=("run_module_test")
    fi
    if [[ "$RUN_API" == "true" ]]; then
        suites+=("api_test")
        funcs+=("run_api_test")
    fi
    if [[ "$RUN_JAVA" == "true" ]]; then
        suites+=("java_api_test")
        funcs+=("run_java_api_test")
    fi

    local total=${#suites[@]}

    print_banner

    OVERALL_START=$(now_seconds)

    local final_exit=0
    for i in "${!suites[@]}"; do
        local idx=$((i + 1))
        local rc=0
        run_suite "${suites[$i]}" "${funcs[$i]}" "$idx" "$total" || rc=$?

        if [[ $rc -ne 0 ]]; then
            if [[ $rc -eq 2 ]]; then
                final_exit=2
            elif [[ $final_exit -ne 2 ]]; then
                final_exit=1
            fi

            if [[ "$OPT_CONTINUE" == "false" ]]; then
                # Mark remaining suites as skipped
                for j in $(seq $((i + 1)) $((total - 1))); do
                    SUITE_RESULTS+=("${suites[$j]}:SKIP:-")
                    ((SUITES_SKIPPED++)) || true
                done
                break
            fi
        fi
    done

    print_summary

    # [2026-04-08] 테스트 후 생성 데이터 및 빌드 산출물 정리
    # --cleanup-data: .pt 데이터 파일 + build 디렉토리 모두 삭제
    if [[ "$OPT_CLEANUP_DATA" == "true" ]]; then
        echo ""
        echo -e "${YELLOW}>> Cleaning up generated test data...${NC}"
        for dir in "${LAYER_DIR}/python/binary" "${MODULE_DIR}/python/binary" "${API_DIR}/python/binary" "${JAVA_DIR}/python/binary"; do
            if [[ -d "$dir" ]]; then
                rm -rf "$dir"
                echo -e "  ${DIM}>> Removed data: ${dir}${NC}"
            fi
        done

        echo -e "${YELLOW}>> Cleaning up build artifacts...${NC}"
        for dir in "${BUILD_ROOT}/layer" "${BUILD_ROOT}/module" "$RUNTIME_BUILD_DIR"; do
            if [[ -d "$dir" ]]; then
                rm -rf "$dir"
                echo -e "  ${DIM}>> Removed build: ${dir}${NC}"
            fi
        done
        echo -e "${GREEN}>> Cleanup done${NC}"
    fi

    exit $final_exit
}

main "$@"
