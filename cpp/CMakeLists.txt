cmake_minimum_required(VERSION 3.16)
option(MERCURY_EXPORT_ONLY "Configure packaging targets without a host engine or JDK" OFF)
if(MERCURY_EXPORT_ONLY)
    project(Mercury LANGUAGES NONE)
else()
    project(Mercury)
endif()

set(MERCURY_DIST_DIR "${CMAKE_CURRENT_BINARY_DIR}/dist" CACHE PATH "Staged model and platform libraries")
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT CMAKE_CROSSCOMPILING)
    include(cmake/Distribution.cmake)
endif()
if(MERCURY_EXPORT_ONLY)
    return()
endif()

# C++17 명시적 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-fpermissive)
endif()

# -march=native 는 호스트 CPU 를 가리키므로 크로스 컴파일(iOS)에서는 쓸 수 없다
if(UNIX AND NOT ANDROID AND NOT IOS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -flto")
    # -Ofast 는 -ffast-math 를 포함한다. 부동소수점 의미가 바뀌어 인식 결과가 무너지므로 쓰지 않는다
    # (실측: 같은 오디오가 "good morning how are you" -> "umsth comeundburn hopew ..." 로 깨졌다).
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -funroll-loops -march=native")
endif()

if(ANDROID)
    # Keep the runtime and its static dependencies compatible with Android's linker.
    set(android_flags "-O2 -fdata-sections -ffunction-sections -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${android_flags}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${android_flags} -fstack-protector-strong -DEIGEN_FAST_MATH -pthread")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -Wl,--icf=all -Wl,--strip-all -Wl,--as-needed")
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -mthumb")

elseif(ANDROID_ABI STREQUAL "arm64-v8a")
    # baseline armv8-a (특정 코어 튜닝 고정 없음) + Cortex-A53 erratum 835769 워크어라운드
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=armv8-a -mfix-cortex-a53-835769")
endif()
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

add_compile_options(-fPIC)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(NOT DEFINED MERCURY_OUTPUT_DIR)
    set(MERCURY_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()
add_subdirectory(third_party/sentencepiece "${MERCURY_OUTPUT_DIR}/sentencepiece_build" EXCLUDE_FROM_ALL)

set(TENSOR_SOURCES
    tensor/tensor.cc
    json_loader/json_object.cc
    logger/logger_manager.cc
)

set(LAYER_SOURCE
    layer/batchnorm1d.cc
    layer/conformer_positional.cc
    layer/convolution1d.cc
    layer/convolution2d.cc
    layer/embedding.cc
    layer/glu.cc
    layer/layernorm.cc
    layer/linear.cc
    layer/positional_encoding.cc
    layer/relu.cc
    layer/silu.cc
    layer/tanh.cc
    layer/softmax.cc
    layer/rotary_embedding.cc
    layer/gelu.cc
)

set(MODULE_SOURCE
    module/conformer_conv_module.cc
    module/conformer_subsample.cc
    module/conformer_feedforward.cc
    module/conformer_head.cc
    module/conformer_multihead.cc
    module/conformer_attention.cc
    module/conformer_encoder_layer.cc
    module/conformer_encoder.cc
    module/conformer_decoder.cc
    module/transformer_feedforward.cc
    module/transformer_head.cc
    module/transformer_multihead.cc
    module/transformer_encoder_layer.cc
    module/transformer_decoder_layer.cc
    module/transformer_encoder.cc
    module/transformer_decoder.cc
    module/conformer_stream_head.cc
    module/conformer_stream_multihead.cc
    module/conformer_stream_attention.cc
    module/conformer_stream_conv_module.cc
    module/conformer_stream_encoder_layer.cc
    module/conformer_stream_encoder.cc
    module/conformer_stream_encoderV2.cc
    module/conformer_ctc.cc
    module/transformer_transducer_layer.cc
    module/transformer_transducer.cc
)

set(DECODER_SOURCE
    decoder/asr_async_decoder.cc
    decoder/asr_stream_conformer_trf_transducer.cc
    decoder/asr_stream_conformer_trf_transducerV2.cc
    decoder/asr_stream_conformer.cc
)

set(FEATURE_SOURCE
    feature/bayesian_cmvn.cc
)

set(SCORE_SOURCE
    score/scorer.cc
)

set(UTILITY_SOURCE
    utils/thread_pool.cc
)

set(STATE_SOURCE
    state/context_manager.cc
    state/idle_state.cc
    state/init_state.cc
    state/running_state.cc
    state/finishing_state.cc
)

if(IOS)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(engine_target asr_api_objects)
    add_library(asr_api_objects OBJECT)
    set(bindings api/obj_c/MercuryAsr.mm interface/objectivec_callback.cc)
    set_source_files_properties(interface/callback_factory.cc interface/objectivec_callback.cc
        PROPERTIES LANGUAGE OBJCXX COMPILE_FLAGS "-x objective-c++")
    set_source_files_properties(api/obj_c/MercuryAsr.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set(binding_define OBJC)
else()
    set(engine_target asr_api)
    add_library(asr_api SHARED)
    set(bindings api/java/jni_wrapper/asr_api_wrap.cxx interface/java_callback.cc interface/linux_callback.cc)
    set(binding_define JAVA)
endif()

target_sources(${engine_target} PRIVATE
    api/asr_api.cc interface/callback_factory.cc ${bindings}
    ${UTILITY_SOURCE} ${STATE_SOURCE} ${DECODER_SOURCE} ${SCORE_SOURCE}
    ${MODULE_SOURCE} ${LAYER_SOURCE} ${FEATURE_SOURCE} ${TENSOR_SOURCES})
target_compile_features(${engine_target} PUBLIC cxx_std_17)
target_include_directories(${engine_target} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}" api third_party third_party/sentencepiece/src
    PRIVATE json_loader tensor module feature feature/audio_reader)
# fast-math changes the numerical results of layer normalization and attention.
target_compile_options(${engine_target} PRIVATE -O3)
target_compile_definitions(${engine_target} PRIVATE ${binding_define}
    PUBLIC USE_TRANSFORMER_POOL EIGEN_NO_DEBUG EIGEN_MPL2_ONLY
    EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
if(IOS)
    # Compile the engine once for both the static archive and the dynamic framework.
    target_link_libraries(asr_api_objects PUBLIC Threads::Threads sentencepiece-static "-framework Foundation")
    add_library(asr_api STATIC)
    target_link_libraries(asr_api PUBLIC asr_api_objects)
    add_library(MercuryAsr SHARED api/obj_c/MercuryAsr.h)
    target_link_libraries(MercuryAsr PRIVATE asr_api_objects)
    set_target_properties(MercuryAsr PROPERTIES
        FRAMEWORK TRUE
        PUBLIC_HEADER api/obj_c/MercuryAsr.h
        MACOSX_FRAMEWORK_IDENTIFIER com.chisnova.mercury.asr
        MACOSX_FRAMEWORK_BUNDLE_VERSION 1
        MACOSX_FRAMEWORK_SHORT_VERSION_STRING 1.0
        INSTALL_NAME_DIR "@rpath"
        BUILD_WITH_INSTALL_NAME_DIR TRUE
        INSTALL_RPATH ""
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED NO
        LINKER_LANGUAGE CXX)
else()
    target_link_libraries(asr_api PRIVATE Threads::Threads sentencepiece-static)
endif()

if(ANDROID)
    # The NDK supplies its own jni.h.
    target_link_libraries(asr_api PRIVATE log)
    target_link_options(asr_api PRIVATE -Wl,-plugin-opt,-import-instr-limit=30)
elseif(NOT IOS)
    find_package(JNI REQUIRED)
    target_include_directories(asr_api PRIVATE ${JNI_INCLUDE_DIRS})
endif()

set_target_properties(asr_api PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${MERCURY_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${MERCURY_OUTPUT_DIR}")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(default_tests OFF)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(default_tests ON)
    endif()
    option(MERCURY_BUILD_TESTS "Build the native and Java API test programs" ${default_tests})
    if(MERCURY_BUILD_TESTS)
        enable_testing()
        set(default_model "${MERCURY_DIST_DIR}/model_state")
        if(DEFINED ENV{MERCURY_TEST_MODEL_DIR})
            set(default_model "$ENV{MERCURY_TEST_MODEL_DIR}")
        elseif(NOT EXISTS "${default_model}/model.json" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../android/app/src/main/assets/model_state/model.json")
            set(default_model "${CMAKE_CURRENT_SOURCE_DIR}/../android/app/src/main/assets/model_state")
        endif()
        set(MERCURY_TEST_MODEL_DIR "${default_model}"
            CACHE PATH "Exported model for the API tests")
        add_subdirectory(decode/api)
        add_subdirectory(decode/java_api)
    endif()
endif()

# Stage just the artifacts for this toolchain; model preparation is independent.
if(ANDROID)
    install(TARGETS asr_api LIBRARY DESTINATION "lib/android/${ANDROID_ABI}" COMPONENT MercuryRuntime)
elseif(IOS)
    string(TOLOWER "${CMAKE_OSX_SYSROOT}" sdk_path)
    if(sdk_path MATCHES "simulator")
        set(sdk iphonesimulator)
    else()
        set(sdk iphoneos)
    endif()
    install(TARGETS MercuryAsr FRAMEWORK DESTINATION "lib/ios/${sdk}" COMPONENT MercuryRuntime)
    install(TARGETS asr_api sentencepiece-static ARCHIVE DESTINATION "lib/ios/${sdk}" COMPONENT MercuryRuntime)
    install(FILES api/obj_c/MercuryAsr.h DESTINATION lib/ios/include COMPONENT MercuryRuntime)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    install(TARGETS asr_api LIBRARY DESTINATION lib/linux COMPONENT MercuryRuntime)
endif()
add_custom_target(export_native
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --prefix "${MERCURY_DIST_DIR}" --component MercuryRuntime
    DEPENDS asr_api VERBATIM)
if(IOS)
    add_dependencies(export_native MercuryAsr)
endif()
