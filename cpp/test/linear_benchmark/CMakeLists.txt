cmake_minimum_required(VERSION 3.16)
project(linear_benchmark LANGUAGES CXX)

find_package(Threads REQUIRED)
add_executable(linear_benchmark
    linear_benchmark.cc
    ../../layer/linear.cc
    ../../tensor/tensor.cc
    ../../json_loader/json_object.cc
    ../../logger/logger_manager.cc
    ../../utils/thread_pool.cc)
target_include_directories(linear_benchmark PRIVATE ../..)
target_compile_features(linear_benchmark PRIVATE cxx_std_17)
target_compile_definitions(linear_benchmark PRIVATE
    EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_UNROLLING_LIMIT=256 THREAD_POOL_ENABLED)
target_link_libraries(linear_benchmark PRIVATE Threads::Threads)
option(LINEAR_BENCH_NATIVE "Tune for the build host (disable for cross compilation)" OFF)
if(LINEAR_BENCH_NATIVE)
    target_compile_options(linear_benchmark PRIVATE -march=native)
endif()
