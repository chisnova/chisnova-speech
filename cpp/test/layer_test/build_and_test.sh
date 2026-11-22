#!/bin/bash
#
# build_and_test.sh - Incremental build and test for Mercury layer tests.
#
# Usage:
#   ./build_and_test.sh              # Test all layers
#   ./build_and_test.sh gelu         # Test only gelu
#   ./build_and_test.sh relu sigmoid  # Test relu and sigmoid
#   ./build_and_test.sh --regen      # Force regenerate test data, then test all
#   ./build_and_test.sh --regen gelu # Force regenerate test data, then test gelu
#   ./build_and_test.sh --help       # Show usage
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#   2 - Build failed
#   3 - Test data generation failed

set -euo pipefail

# ---- Color codes ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PYTHON_DIR="${SCRIPT_DIR}/python"
BINARY_DIR="${PYTHON_DIR}/binary"
TEST_JSON="${BINARY_DIR}/layer_test.json"

ALL_LAYERS=(
    embedding batchnorm1d relu silu softmax glu
    layernorm positional conformer_positional cnn1d cnn2d
    linear sigmoid tanh rotary gelu
)

# ---- Functions ----

print_header() {
    echo -e "\n${BOLD}${CYAN}========================================${NC}"
    echo -e "${BOLD}${CYAN}  Mercury Layer Test${NC}"
    echo -e "${BOLD}${CYAN}========================================${NC}\n"
}

print_usage() {
    echo "Usage: $0 [OPTIONS] [layer_name ...]"
    echo ""
    echo "Options:"
    echo "  --regen    Force regenerate test data from Python/PyTorch"
    echo "  --clean    Clean build directory before building"
    echo "  --help     Show this help message"
    echo ""
    echo "Available layers:"
    echo "  ${ALL_LAYERS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0                  # Test all layers"
    echo "  $0 gelu             # Test only gelu"
    echo "  $0 relu sigmoid     # Test relu and sigmoid"
    echo "  $0 --regen gelu     # Regenerate data, test gelu"
}

validate_layer() {
    local name="$1"
    for l in "${ALL_LAYERS[@]}"; do
        if [[ "$l" == "$name" ]]; then
            return 0
        fi
    done
    echo -e "${RED}ERROR: Unknown layer '${name}'${NC}" >&2
    echo "Available layers: ${ALL_LAYERS[*]}" >&2
    return 1
}

generate_test_data() {
    echo -e "${YELLOW}>> Generating test data via Python/PyTorch...${NC}"
    mkdir -p "${BINARY_DIR}"
    cd "${PYTHON_DIR}"
    if ! python3 main.py 2>&1; then
        echo -e "${RED}FAILED: Test data generation failed${NC}" >&2
        exit 3
    fi
    cd "${SCRIPT_DIR}"
    echo -e "${GREEN}>> Test data generated successfully${NC}"
}

build_test() {
    local do_clean="$1"

    if [[ "$do_clean" == "true" ]]; then
        echo -e "${YELLOW}>> Cleaning build directory...${NC}"
        rm -rf "${BUILD_DIR}"
    fi

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # Run cmake only if CMakeCache.txt does not exist (first build or after clean)
    if [[ ! -f "CMakeCache.txt" ]]; then
        echo -e "${YELLOW}>> Running cmake...${NC}"
        if ! cmake .. 2>&1; then
            echo -e "${RED}FAILED: cmake configuration failed${NC}" >&2
            exit 2
        fi
    fi

    echo -e "${YELLOW}>> Building (incremental)...${NC}"
    if ! make -j"$(nproc)" 2>&1; then
        echo -e "${RED}${BOLD}========================================${NC}"
        echo -e "${RED}${BOLD}  BUILD FAILED${NC}"
        echo -e "${RED}${BOLD}========================================${NC}"
        echo -e "${RED}Check the compiler errors above.${NC}" >&2
        exit 2
    fi

    cd "${SCRIPT_DIR}"
    echo -e "${GREEN}>> Build succeeded${NC}"
}

run_tests() {
    local layers=("$@")
    local test_binary="${BUILD_DIR}/layer_tester"

    if [[ ! -x "${test_binary}" ]]; then
        echo -e "${RED}ERROR: Test binary not found at ${test_binary}${NC}" >&2
        exit 2
    fi

    echo -e "\n${BOLD}${CYAN}>> Running tests...${NC}\n"

    # Run from SCRIPT_DIR so relative paths in the binary (python/binary/...) resolve
    cd "${SCRIPT_DIR}"

    local output
    local exit_code=0
    if [[ ${#layers[@]} -eq 0 ]]; then
        output=$("${test_binary}" 2>&1) || exit_code=$?
    else
        output=$("${test_binary}" "${layers[@]}" 2>&1) || exit_code=$?
    fi

    # Parse and colorize output
    local any_fail=0
    while IFS= read -r line; do
        if [[ "$line" == RESULT:*:PASS ]]; then
            local lname="${line#RESULT:}"
            lname="${lname%:PASS}"
            echo -e "  ${GREEN}${BOLD}[PASS]${NC} ${lname}"
        elif [[ "$line" == RESULT:*:FAIL ]]; then
            local lname="${line#RESULT:}"
            lname="${lname%:FAIL}"
            echo -e "  ${RED}${BOLD}[FAIL]${NC} ${lname}"
            any_fail=1
        elif [[ "$line" == *"===== SUMMARY =====" ]]; then
            echo ""
        elif [[ "$line" == "Passed:"* ]]; then
            if [[ $any_fail -eq 1 ]]; then
                echo -e "${RED}${BOLD}${line}${NC}"
            else
                echo -e "${GREEN}${BOLD}${line}${NC}"
            fi
        elif [[ "$line" == "Failed layers:"* ]]; then
            echo -e "${RED}${line}${NC}"
        elif [[ "$line" == *"===== Testing:"* ]]; then
            # suppress verbose per-layer headers (results line is enough)
            :
        elif [[ "$line" == *"RMS ERROR"* ]] || [[ "$line" == *"infernece time"* ]]; then
            # Show detailed info in dimmer text
            echo -e "    ${line}"
        elif [[ "$line" == *"EXCEPTION"* ]] || [[ "$line" == *"ERROR"* ]]; then
            echo -e "  ${RED}${line}${NC}"
        fi
    done <<< "$output"

    echo ""
    if [[ $any_fail -eq 1 || $exit_code -ne 0 ]]; then
        echo -e "${RED}${BOLD}========================================${NC}"
        echo -e "${RED}${BOLD}  TESTS FAILED${NC}"
        echo -e "${RED}${BOLD}========================================${NC}"
        return 1
    else
        echo -e "${GREEN}${BOLD}========================================${NC}"
        echo -e "${GREEN}${BOLD}  ALL TESTS PASSED${NC}"
        echo -e "${GREEN}${BOLD}========================================${NC}"
        return 0
    fi
}

# ---- Main ----

main() {
    local do_regen=false
    local do_clean=false
    local layers=()

    # Parse arguments
    for arg in "$@"; do
        case "$arg" in
            --regen)
                do_regen=true
                ;;
            --clean)
                do_clean=true
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            -*)
                echo -e "${RED}Unknown option: ${arg}${NC}" >&2
                print_usage
                exit 1
                ;;
            *)
                validate_layer "$arg" || exit 1
                layers+=("$arg")
                ;;
        esac
    done

    print_header

    # Generate test data if missing or explicitly requested
    if [[ "$do_regen" == "true" ]]; then
        generate_test_data
    elif [[ ! -f "${TEST_JSON}" ]]; then
        echo -e "${YELLOW}>> Test data not found. Generating...${NC}"
        generate_test_data
    fi

    # Build
    build_test "$do_clean"

    # Run
    run_tests "${layers[@]}"
}

main "$@"
