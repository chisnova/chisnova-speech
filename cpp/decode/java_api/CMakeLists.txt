find_package(Java REQUIRED COMPONENTS Development)
include(UseJava)
set(bindings "${PROJECT_SOURCE_DIR}/api/java/jni_wrapper/com/chisnova/mercury")
set(json_jar "${CMAKE_CURRENT_SOURCE_DIR}/lib/json-20250517.jar")
configure_file("${json_jar}" "${MERCURY_OUTPUT_DIR}/json-20250517.jar" COPYONLY)
add_jar(java_api_test_jar SOURCES
    java/com/chisnova/mercury/audio/WavFile.java
    java/com/chisnova/mercury/audio/WavFileException.java
    java/com/chisnova/mercury/asr/ApiTest.java
    "${bindings}/itf/DecoderCallback.java" "${bindings}/jni/AsrAPI.java" "${bindings}/jni/asr_apiJNI.java"
    INCLUDE_JARS "${json_jar}" MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/test-manifest.mf"
    OUTPUT_DIR "${MERCURY_OUTPUT_DIR}" OUTPUT_NAME asr_api)
add_executable(java_api_test main.cc)
target_include_directories(java_api_test PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_definitions(java_api_test PRIVATE MERCURY_JAVA="${Java_JAVA_EXECUTABLE}"
    MERCURY_JAVA_DATA="${CMAKE_CURRENT_SOURCE_DIR}" MERCURY_JAVA_MODEL="${MERCURY_TEST_MODEL_DIR}")
get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(multi_config)
    target_compile_definitions(java_api_test PRIVATE MERCURY_TEST_JAR="../asr_api.jar")
else()
    target_compile_definitions(java_api_test PRIVATE MERCURY_TEST_JAR="asr_api.jar")
endif()
set_target_properties(java_api_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${MERCURY_OUTPUT_DIR}")
add_dependencies(java_api_test java_api_test_jar asr_api)
add_test(NAME java_api_test COMMAND java_api_test 5)
set_tests_properties(java_api_test PROPERTIES TIMEOUT 180)
