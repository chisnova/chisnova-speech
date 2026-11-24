#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
build="${MERCURY_TEST_BUILD_ROOT:+$MERCURY_TEST_BUILD_ROOT/runtime}"
build="${build:-../../build}"
exec "$build/java_api_test" "$@"
