# CMakeLists.txt
cmake_minimum_required(VERSION 3.28)
project(mercury_scorer_test)

include_directories(.
                    ./../.
                    ./../json_loader/.
                    ./../tensor/.
                    ./../third_party/.
                    ../third_party/sentencepiece/src/.
                    ../third_party/json/single_include)


# C++17 명시적 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-fpermissive)
endif()

set(TENSOR_SOURCES
    ../tensor/tensor.cc
    ../json_loader/json_object.cc
    ../logger/logger_manager.cc
)

set(SENTENCEPIECE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build)

# SentencePiece 서브디렉토리 추가 (바이너리 디렉토리 명시)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sentencepiece
    ${SENTENCEPIECE_BINARY_DIR}
)

# Build the scorer_test executable and link with sentencepiece shared library

add_executable(scorer_test scorer_test.cc scorer.cc ${TENSOR_SOURCES})

target_link_libraries(scorer_test PRIVATE sentencepiece-static)
enable_testing()
add_test(NAME scorer_test COMMAND scorer_test)
set_tests_properties(scorer_test PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
