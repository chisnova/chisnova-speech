#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p classes

JSON_LIB="lib/json-20250517.jar"
CP="linux_build:$JSON_LIB"

javac -d classes \
    -cp $CP \
    java/com/chisnova/mercury/audio/*.java \
    ../../api/java/jni_wrapper/com/chisnova/mercury/jni/*.java  \
    ../../api/java/jni_wrapper/com/chisnova/mercury/itf/*.java \
    java/com/chisnova/mercury/asr/ApiTest.java

echo -e "Main-Class: com.chisnova.mercury.asr.ApiTest\n" > manifest.txt
jar cfm asr_api.jar manifest.txt -C classes com 