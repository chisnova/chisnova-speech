cmake_minimum_required(VERSION 3.10)
project(bayesian_cmvn_test)

include_directories(.
                    ../../.
                    ../../json_loader/.
                    ../../tensor/.
                    ../../third_party/.
                    ../../module/.
                    ../../feature/.
                    ../../feature/audio_reader/.
                    ../../third_party/rapidjson/include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source files
set(SOURCES
    bayesian_cmvn_test.cc
    ../../feature/bayesian_cmvn.cc
    ../../tensor/tensor.cc
    ../../json_loader/json_object.cc
    ../../logger/logger_manager.cc
)


# Create test executable
add_executable(bayesian_cmvn_test ${SOURCES})

# Link libraries
target_link_libraries(bayesian_cmvn_test
    pthread
)
