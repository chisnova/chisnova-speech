#!/bin/bash
#
# run_test.sh - Backward-compatible wrapper.
# Delegates to run_all_tests.sh which replaces the old sequential runner.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_all_tests.sh" "$@"
