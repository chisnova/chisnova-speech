cmake_minimum_required (VERSION 3.0)
project (Mecury)

include_directories(.
                    ../../.
                    ../../api/.
                    ../../json_loader/.
                    ../../tensor/.
                    ../../third_party/.
                    ../../module/.
                    ../../feature/.
                    ../../feature/audio_reader/.
                    ../../third_party/rapidjson/include
                    ../../third_party/sentencepiece/src)

                    # C++17 명시적 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-fpermissive)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -march=native")

set(SENTENCEPIECE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build)
# SentencePiece 서브디렉토리 추가 (바이너리 디렉토리 명시)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/sentencepiece
    ${SENTENCEPIECE_BINARY_DIR}
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DJAVA")
find_package(JNI REQUIRED)
set(JNI_PATH1 "$ENV{JAVA_HOME}/include")
set(JNI_PATH2 "$ENV{JAVA_HOME}/include/linux")

set(THREAD_POOL_ENABLED ON)
# add_definitions(-DDEBUG)
add_definitions(-DUSE_TRANSFORMER_POOL)

set(TENSOR_SOURCES
    ../../tensor/tensor.cc
    ../../json_loader/json_object.cc
    ../../logger/logger_manager.cc
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
    # ../../decoder/asr_ns_conformer_trf_transducer.cc
    ../../decoder/asr_stream_conformer_trf_transducer.cc
    ../../decoder/asr_stream_conformer.cc
)

set(INTERFACE_SOURCE
    ../../interface/callback_factory.cc
    ../../interface/java_callback.cc
    ../../interface/linux_callback.cc
)

set(FEATURE_SOURCE
    ../../feature/bayesian_cmvn.cc
)

set(SCORE_SOURCE
    ../../score/scorer.cc
)

set(UTILITY_SOURCE
    ../../utils/thread_pool.cc
)

set(STATE_SOURCE
    ../../state/context_manager.cc
    ../../state/idle_state.cc
    ../../state/init_state.cc
    ../../state/running_state.cc
    ../../state/finishing_state.cc
)

message("Building for Linux")
add_library(asr_api SHARED ../../api/asr_api.cc ../../api/java/jni_wrapper/asr_api_wrap.cxx ${UTILITY_SOURCE} ${STATE_SOURCE} ${INTERFACE_SOURCE} ${DECODER_SOURCE} ${SCORE_SOURCE} ${MODULE_SOURCE} ${LAYER_SOURCE} ${FEATURE_SOURCE} ${TENSOR_SOURCES})
target_compile_options(asr_api PRIVATE "-O3")
target_include_directories(asr_api PRIVATE ${JNI_PATH1} ${JNI_PATH2})
target_compile_definitions(asr_api PRIVATE EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
# Link with static SentencePiece library explicitly
target_link_libraries(asr_api PRIVATE Threads::Threads ${JNI} ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build/src/libsentencepiece.a)
# Add dependency to ensure SentencePiece is built first
add_dependencies(asr_api sentencepiece-static)