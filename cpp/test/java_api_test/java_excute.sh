#!/bin/bash

JSON_LIB="lib/json-20250517.jar"
CP="linux_build:$JSON_LIB:asr_api.jar"
java -cp $CP -Djava.library.path=linux_build com.chisnova.mercury.asr.ApiTest
