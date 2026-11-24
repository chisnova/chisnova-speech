#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
build="${MERCURY_TEST_BUILD_ROOT:+$MERCURY_TEST_BUILD_ROOT/runtime}"
build="${build:-../../build}"
cmake -S ../.. -B "$build" -DMERCURY_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target java_api_test --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
