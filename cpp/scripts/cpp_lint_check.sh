#!/bin/bash
# C++ Lint Check Script for Claude Code Hook
# Runs cpplint on modified .cc/.h files (excluding third_party)

PROJECT_DIR="/home/chisnova/archive2/mercury/cpp"
LINT_CMD="python3 -m cpplint"

# Get the file path from the hook input (passed via stdin as JSON)
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Extract file_path from tool input
    tool_input = data.get('tool_input', {})
    path = tool_input.get('file_path', '')
    print(path)
except:
    print('')
" 2>/dev/null)

# If no file path found, exit silently
if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Only lint .cc and .h files
if [[ ! "$FILE_PATH" =~ \.(cc|h|cpp|hpp)$ ]]; then
    exit 0
fi

# Skip third_party files
if [[ "$FILE_PATH" == *"third_party"* ]]; then
    exit 0
fi

# Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
    exit 0
fi

# Run cpplint with project-appropriate filters
# Suppress some noisy checks that may not apply to this project
RESULT=$($LINT_CMD \
    --filter=-legal/copyright,-build/header_guard,-build/include_subdir,-runtime/references,-whitespace/indent,-readability/todo \
    --linelength=120 \
    --quiet \
    "$FILE_PATH" 2>&1)

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] && [ -n "$RESULT" ]; then
    echo ""
    echo "===== C++ Lint Check Results ====="
    echo "File: $FILE_PATH"
    echo "---------------------------------"
    echo "$RESULT"
    echo "================================="
    echo ""
    echo "Please fix the lint issues above before proceeding."
    # Return non-zero to signal issues (but don't block - use as warning)
    exit 0
fi

exit 0
