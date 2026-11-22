cmake_minimum_required (VERSION 3.0)
project (MODULE_TEST)

include_directories(.
                    ../../.
                    ../../json_loader/.
                    ../../tensor/.
                    ../../third_party/.
                    ../../module/.
                    ../../feature/.
                    ../../feature/audio_reader/.
                    ../../third_party/sentencepiece/src/.
                    ../../third_party/json/single_include
                    ../../utils)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DEIGEN_STRONG_INLINE)
add_definitions(-DEIGEN_NO_AUTOMATIC_RESIZING)
add_definitions(-DEIGEN_NO_DEBUG)
add_definitions(-DEIGEN_NO_STATIC_ASSERT)
add_definitions(-DEIGEN_NO_MALLOC)
set(THREADS_PREFER_PTHREAD_FLAG ON)
add_definitions(-DEIGEN_VECTORIZE)
add_definitions(-DEIGEN_DEFAULT_ALIGN_BYTES=32)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

set(SENTENCEPIECE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build)

# SentencePiece 서브디렉토리 추가 (바이너리 디렉토리 명시)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/sentencepiece
    ${SENTENCEPIECE_BINARY_DIR}
)

set(TENSOR_SOURCES
    ../../tensor/tensor.cc
    ../../json_loader/json_object.cc
    ../../logger/logger_manager.cc)

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
    ../../layer/gelu.cc
    ../../layer/rotary_embedding.cc)

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
    ../../module/transformer_transducer_layer.cc
    ../../module/transformer_transducer.cc
    ../../module/conformer_ctc.cc
    ../../decoder/asr_stream_conformer_trf_transducer.cc)

set(UTIL_SOURCE
    ../../utils/thread_pool.cc)

add_executable(module_tester module_test.cc ${MODULE_SOURCE} ${TENSOR_SOURCES} ${LAYER_SOURCE} ${UTIL_SOURCE})
target_compile_options(module_tester PRIVATE -march=native)
target_link_libraries(module_tester PRIVATE Threads::Threads sentencepiece)

