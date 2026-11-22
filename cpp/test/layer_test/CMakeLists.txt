cmake_minimum_required (VERSION 3.0)
project (LAYER_TEST)

include_directories(.
                    ../../.
                    ../../json_loader/.
                    ../../tensor/.
                    ../../third_party/.
                    ../../module/.
                    ../../feature/.
                    ../../feature/audio_reader/.
                    ../../third_party/json/single_include)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -march=native -funroll-loops -DNDEBUG")
add_definitions(-DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE -DEIGEN_FAST_MATH)

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
    ../../layer/sigmoid.cc
    ../../layer/tanh.cc
    ../../layer/softmax.cc
    ../../layer/gelu.cc
    ../../layer/rotary_embedding.cc)


add_executable(layer_tester layer_test.cc ${TENSOR_SOURCES} ${LAYER_SOURCE})
target_compile_options(layer_tester PRIVATE -march=native)
target_link_libraries(layer_tester PRIVATE Threads::Threads)
