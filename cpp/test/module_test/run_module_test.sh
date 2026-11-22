#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run_all_tests.sh" --module "$@"
