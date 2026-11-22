#!/bin/bash

set -eu
cd "$(dirname "$0")"
mkdir -p java/jni_wrapper/com/chisnova/mercury/jni
swig -c++ -package com.chisnova.mercury.jni -java \
    -outdir java/jni_wrapper/com/chisnova/mercury/jni \
    -o java/jni_wrapper/asr_api_wrap.cxx asr_api.i
# Empty module class is not part of the app API.
rm -f java/jni_wrapper/com/chisnova/mercury/jni/asr_api.java
python3 - <<'PY'
from pathlib import Path
files = [Path('java/jni_wrapper/asr_api_wrap.cxx')]
files += list(Path('java/jni_wrapper/com/chisnova/mercury/jni').glob('*.java'))
for path in files:
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
PY
