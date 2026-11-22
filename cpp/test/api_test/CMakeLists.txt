cmake_minimum_required (VERSION 3.0)
project (DECODER_TEST)

include_directories(.
                    ../.
                    ../../.
                    ../../api/.
                    ../../json_loader/.
                    ../../tensor/.
                    ../../third_party/.
                    ../../module/.
                    ../../feature/.
                    ../../feature/audio_reader/.
                    ../../third_party/sentencepiece/src/.
                    ../../third_party/json/single_include
                    ../../utils
                    ../../decoder/.
                    ../../interface/.
                    ../../layer/.
                    ../../logger/.
                    ../../score/.
                    ../../beam/.)

# C++17 명시적 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-fpermissive)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(SENTENCEPIECE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build)

# SentencePiece 서브디렉토리 추가 (바이너리 디렉토리 명시)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/sentencepiece
    ${SENTENCEPIECE_BINARY_DIR}
)

set(THREAD_POOL_ENABLED ON)
# add_definitions(-DDEBUG)
add_definitions(-DUSE_TRANSFORMER_POOL)

set(TENSOR_SOURCES
    ../../tensor/tensor.cc
    ../../json_loader/json_object.cc
    ../../logger/logger_manager.cc
)

set(FEATURE_SOURCE
    ../../feature/bayesian_cmvn.cc
)

set(LAYER_SOURCE 
    ../../layer/batchnorm1d.cc
    ../../layer/conformer_positional.cc
    ../../layer/convolution1d.cc
    ../../layer/convolution2d.cc
    ../../layer/embedding.cc
    ../../layer/glu.cc
    ../../layer/layernorm.cc
    ../../layer/linear.cc
    ../../layer/positional_encoding.cc
    ../../layer/relu.cc
    ../../layer/silu.cc
    ../../layer/tanh.cc
    ../../layer/softmax.cc
    ../../layer/rotary_embedding.cc
    ../../layer/gelu.cc
)

set(MODULE_SOURCE 
    ../../module/conformer_conv_module.cc
    ../../module/conformer_subsample.cc
    ../../module/conformer_feedforward.cc
    ../../module/conformer_head.cc
    ../../module/conformer_multihead.cc
    ../../module/conformer_attention.cc
    ../../module/conformer_encoder_layer.cc
    ../../module/conformer_encoder.cc
    ../../module/conformer_decoder.cc
    ../../module/transformer_feedforward.cc
    ../../module/transformer_head.cc
    ../../module/transformer_multihead.cc
    ../../module/transformer_encoder_layer.cc
    ../../module/transformer_decoder_layer.cc
    ../../module/transformer_encoder.cc
    ../../module/transformer_decoder.cc
    ../../module/conformer_stream_head.cc
    ../../module/conformer_stream_multihead.cc
    ../../module/conformer_stream_attention.cc
    ../../module/conformer_stream_conv_module.cc
    ../../module/conformer_stream_encoder_layer.cc
    ../../module/conformer_stream_encoder.cc
    ../../module/conformer_stream_encoderV2.cc
    ../../module/conformer_ctc.cc
    ../../module/transformer_transducer_layer.cc
    ../../module/transformer_transducer.cc
)


set(DECODER_SOURCE 
    # ../../decoder/asr_conformer_trf_transducer.cc
    # ../../decoder/asr_conformer_trf_transducerV2.cc
    # ../../decoder/asr_conformer.cc
    # ../../decoder/asr_decoder.cc
    ../../decoder/asr_async_decoder.cc
    # ../../decoder/asr_transformer.cc
    ../../decoder/asr_stream_conformer_trf_transducer.cc
    ../../decoder/asr_stream_conformer.cc
)

set(INTERFACE_SOURCE 
    ../../interface/callback_factory.cc
    ../../interface/linux_callback.cc
)

set(API_SOURCE 
    ../../api/asr_api.cc
)

set(STATE_SOURCE
    ../../state/context_manager.cc
    ../../state/idle_state.cc
    ../../state/init_state.cc
    ../../state/running_state.cc
    ../../state/finishing_state.cc
)

set(WAV_SOURCE
    ../../feature/audio_reader/wavreader.cc)

set(SCORE_SOURCE
    ../../score/scorer.cc
)

set(UTILITY_SOURCE
    ../../utils/thread_pool.cc
)

add_executable(api_tester api_test.cc ${UTILITY_SOURCE} ${STATE_SOURCE} ${INTERFACE_SOURCE} ${WAV_SOURCE} ${FEATURE_SOURCE} ${API_SOURCE} ${SCORE_SOURCE} ${DECODER_SOURCE} ${MODULE_SOURCE} ${TENSOR_SOURCES} ${LAYER_SOURCE})

add_executable(pronounce_score_diag pronounce_score_diag.cc ${UTILITY_SOURCE} ${STATE_SOURCE} ${INTERFACE_SOURCE} ${WAV_SOURCE} ${FEATURE_SOURCE} ${API_SOURCE} ${SCORE_SOURCE} ${DECODER_SOURCE} ${MODULE_SOURCE} ${TENSOR_SOURCES} ${LAYER_SOURCE})

add_executable(content_audit content_audit.cc ${UTILITY_SOURCE} ${STATE_SOURCE} ${INTERFACE_SOURCE} ${WAV_SOURCE} ${FEATURE_SOURCE} ${API_SOURCE} ${SCORE_SOURCE} ${DECODER_SOURCE} ${MODULE_SOURCE} ${TENSOR_SOURCES} ${LAYER_SOURCE})

add_executable(ctc_endpoint_diag ctc_endpoint_diag.cc ${UTILITY_SOURCE} ${STATE_SOURCE} ${INTERFACE_SOURCE} ${WAV_SOURCE} ${FEATURE_SOURCE} ${API_SOURCE} ${SCORE_SOURCE} ${DECODER_SOURCE} ${MODULE_SOURCE} ${TENSOR_SOURCES} ${LAYER_SOURCE})

add_executable(wav_list_compare wav_list_compare_test.cc ${UTILITY_SOURCE} ${STATE_SOURCE} ${INTERFACE_SOURCE} ${WAV_SOURCE} ${FEATURE_SOURCE} ${API_SOURCE} ${SCORE_SOURCE} ${DECODER_SOURCE} ${MODULE_SOURCE} ${TENSOR_SOURCES} ${LAYER_SOURCE})

# Debug/Release 모드에 따른 컴파일 옵션 설정
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(api_tester PRIVATE "-g" "-O0" "-DDEBUG")
    target_compile_definitions(api_tester PRIVATE DEBUG_MODE THREAD_POOL_ENABLED SPM_NO_THREADLOCAL)
else()
    target_compile_options(api_tester PRIVATE -march=native "-O3")
    target_compile_definitions(api_tester PRIVATE EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(wav_list_compare PRIVATE "-g" "-O0" "-DDEBUG")
    target_compile_definitions(wav_list_compare PRIVATE DEBUG_MODE THREAD_POOL_ENABLED SPM_NO_THREADLOCAL)
else()
    target_compile_options(wav_list_compare PRIVATE -march=native "-O3")
    target_compile_definitions(wav_list_compare PRIVATE EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
endif()
target_link_libraries(wav_list_compare PRIVATE Threads::Threads ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build/src/libsentencepiece.a)
add_dependencies(wav_list_compare sentencepiece-static)

# 검수기 — 콘텐츠 점검용. 엔진과 같은 계산을 써야 하므로 플래그도 같게 둔다.
target_compile_options(content_audit PRIVATE -march=native "-O3")
target_compile_definitions(content_audit PRIVATE EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
target_link_libraries(content_audit PRIVATE Threads::Threads ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build/src/libsentencepiece.a)

add_dependencies(ctc_endpoint_diag sentencepiece-static)
target_link_libraries(ctc_endpoint_diag PRIVATE Threads::Threads ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build/src/libsentencepiece.a)
add_dependencies(content_audit sentencepiece-static)

target_compile_options(pronounce_score_diag PRIVATE -march=native "-O3")
target_compile_definitions(pronounce_score_diag PRIVATE EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
target_link_libraries(pronounce_score_diag PRIVATE Threads::Threads ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build/src/libsentencepiece.a)
add_dependencies(pronounce_score_diag sentencepiece-static)

target_link_libraries(api_tester PRIVATE Threads::Threads sentencepiece)
target_link_libraries(api_tester PRIVATE Threads::Threads ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build/src/libsentencepiece.a)
add_dependencies(api_tester sentencepiece-static)

# Scoring regressions use tokenizer/dictionary assets; no acoustic checkpoint or inference.
add_executable(recognition_score_test ../../score/recognition_score_test.cc ${SCORE_SOURCE} ${TENSOR_SOURCES})
target_link_libraries(recognition_score_test PRIVATE Threads::Threads sentencepiece-static)
enable_testing()
set(MERCURY_TEST_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../android/app/src/main/assets/model_state"
    CACHE PATH "Directory containing the exported model and scoring assets")
add_test(NAME recognition_score_test COMMAND recognition_score_test
    "${MERCURY_TEST_MODEL_DIR}")

# Exercise lifecycle/invalid-input paths with the same engine used by the apps.
get_target_property(STABILITY_SOURCES api_tester SOURCES)
list(REMOVE_ITEM STABILITY_SOURCES api_test.cc)
add_executable(stability_test stability_test.cc ${STABILITY_SOURCES})
target_compile_definitions(stability_test PRIVATE EIGEN_NO_DEBUG EIGEN_MPL2_ONLY
    EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
target_link_libraries(stability_test PRIVATE Threads::Threads sentencepiece-static)
add_test(NAME stability_test COMMAND stability_test
    "${MERCURY_TEST_MODEL_DIR}")
set_tests_properties(stability_test PROPERTIES TIMEOUT 60)
