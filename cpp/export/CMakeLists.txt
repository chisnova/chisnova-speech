cmake_minimum_required (VERSION 3.0)
project (Mercury)

include_directories(.
                    ../.
                    ../api/.
                    ../json_loader/.
                    ../tensor/.
                    ../third_party/.
                    ../module/.
                    ../feature/.
                    ../feature/audio_reader/.
                    ../third_party/rapidjson/include
                    ../third_party/sentencepiece/src)
# C++17 명시적 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-fpermissive)
endif()


# iOS 에는 JNI 가 없다. Android/Linux 빌드만 JNI 래퍼를 쓴다.
if(NOT IOS)
    find_package(JNI REQUIRED)
    set(JNI_PATH1 "$ENV{JAVA_HOME}/include")
    set(JNI_PATH2 "$ENV{JAVA_HOME}/include/linux")
endif()

# -march=native 는 호스트 CPU 를 가리키므로 크로스 컴파일(iOS)에서는 쓸 수 없다
if(UNIX AND NOT ANDROID AND NOT IOS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -flto")
    # -Ofast 는 -ffast-math 를 포함한다. 부동소수점 의미가 바뀌어 인식 결과가 무너지므로 쓰지 않는다
    # (실측: 같은 오디오가 "good morning how are you" -> "umsth comeundburn hopew ..." 로 깨졌다).
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -funroll-loops -march=native")
endif()

if(ANDROID)
    # ===== Android common optimisation flags =====
    # Compile-time flags: Thin-LTO, dead-code stripping friendly sections, hidden visibility, etc.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fdata-sections -ffunction-sections -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -fdata-sections -ffunction-sections -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -DNDEBUG")

    # Link-time optimisation & stripping
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -Wl,--icf=all -Wl,--strip-all -Wl,--as-needed")

    # ThinLTO 는 아래 CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE 로 Release 에만 켠다
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fstack-protector-strong")
    
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DEIGEN_FAST_MATH")
    
    # Thread optimization
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread")
endif()

if(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -mthumb")

    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprefetch-loop-arrays")
elseif(ANDROID_ABI STREQUAL "arm64-v8a")
    # baseline armv8-a (특정 코어 튜닝 고정 없음) + Cortex-A53 erratum 835769 워크어라운드
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=armv8-a -mfix-cortex-a53-835769")
endif()
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")



add_compile_options(-fPIC)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(SENTENCEPIECE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build)
set(SENTENCEPIECE_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
# SentencePiece 서브디렉토리 추가 (바이너리 디렉토리 명시)
if(ANDROID)
    # Android용 SentencePiece 설정
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DJAVA")
    set(SPM_CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_ANDROID_NDK}/build/cmake/android.toolchain.cmake
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${SENTENCEPIECE_LIBRARY_DIR}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_PLATFORM=${ANDROID_PLATFORM}
        -DANDROID_STL=c++_shared
        -DSPM_BUILD_SHARED=ON
        -DSPM_BUILD_TESTS=OFF
        -DSPM_BUILD_PROTO=OFF
    )
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    
    # Android용 SentencePiece 빌드
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sentencepiece
        ${SENTENCEPIECE_BINARY_DIR}
    )
elseif(IOS)
    # iOS: 툴체인 설정을 그대로 물려받아 크로스 컴파일된다. 정적 링크만 쓴다.
    set(SPM_CMAKE_ARGS
        -DSPM_BUILD_SHARED=OFF
        -DSPM_BUILD_TESTS=OFF
        -DSPM_BUILD_PROTO=OFF
    )
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sentencepiece
        ${SENTENCEPIECE_BINARY_DIR}
    )
else()
    set(SPM_CMAKE_ARGS
    -DSPM_BUILD_SHARED=ON
    -DSPM_BUILD_TESTS=OFF
    -DSPM_BUILD_PROTO=OFF
    )

    # Linux용 SentencePiece 빌드
    add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sentencepiece
    ${SENTENCEPIECE_BINARY_DIR}
    )
endif()

set(THREAD_POOL_ENABLED ON)
add_definitions(-DUSE_TRANSFORMER_POOL)

set(TENSOR_SOURCES
    ../tensor/tensor.cc
    ../json_loader/json_object.cc
    ../logger/logger_manager.cc
)

set(LAYER_SOURCE 
    ../layer/batchnorm1d.cc
    ../layer/conformer_positional.cc
    ../layer/convolution1d.cc
    ../layer/convolution2d.cc
    ../layer/embedding.cc
    ../layer/glu.cc
    ../layer/layernorm.cc
    ../layer/linear.cc
    ../layer/positional_encoding.cc
    ../layer/relu.cc
    ../layer/silu.cc
    ../layer/tanh.cc
    ../layer/softmax.cc
    ../layer/rotary_embedding.cc
    ../layer/gelu.cc
)

set(MODULE_SOURCE 
    ../module/conformer_conv_module.cc
    ../module/conformer_subsample.cc
    ../module/conformer_feedforward.cc
    ../module/conformer_head.cc
    ../module/conformer_multihead.cc
    ../module/conformer_attention.cc
    ../module/conformer_encoder_layer.cc
    ../module/conformer_encoder.cc
    ../module/conformer_decoder.cc
    ../module/transformer_feedforward.cc
    ../module/transformer_head.cc
    ../module/transformer_multihead.cc
    ../module/transformer_encoder_layer.cc
    ../module/transformer_decoder_layer.cc
    ../module/transformer_encoder.cc
    ../module/transformer_decoder.cc
    ../module/conformer_stream_head.cc
    ../module/conformer_stream_multihead.cc
    ../module/conformer_stream_attention.cc
    ../module/conformer_stream_conv_module.cc
    ../module/conformer_stream_encoder_layer.cc
    ../module/conformer_stream_encoder.cc
    ../module/conformer_stream_encoderV2.cc
    ../module/conformer_ctc.cc
    ../module/transformer_transducer_layer.cc
    ../module/transformer_transducer.cc
)

set(DECODER_SOURCE 
    # ../decoder/asr_conformer_trf_transducer.cc
    # ../decoder/asr_conformer_trf_transducerV2.cc
    # ../decoder/asr_conformer.cc
    # ../decoder/asr_decoder.cc
    ../decoder/asr_async_decoder.cc
    # ../decoder/asr_transformer.cc
    ../decoder/asr_stream_conformer_trf_transducer.cc
    ../decoder/asr_stream_conformer_trf_transducerV2.cc
    ../decoder/asr_stream_conformer.cc
)

set(INTERFACE_SOURCE
    ../interface/callback_factory.cc
    ../interface/java_callback.cc
    ../interface/linux_callback.cc
)

set(FEATURE_SOURCE
    ../feature/bayesian_cmvn.cc
)

set(SCORE_SOURCE
    ../score/scorer.cc
)

set(UTILITY_SOURCE
    ../utils/thread_pool.cc
)

set(STATE_SOURCE
    ../state/context_manager.cc
    ../state/idle_state.cc
    ../state/init_state.cc
    ../state/running_state.cc
    ../state/finishing_state.cc
)




message("Building message")
if(ANDROID)
    message("Building for Android")
    add_library(asr_api SHARED ../api/asr_api.cc ../api/java/jni_wrapper/asr_api_wrap.cxx ${UTILITY_SOURCE} ${STATE_SOURCE} ${INTERFACE_SOURCE} ${DECODER_SOURCE} ${SCORE_SOURCE} ${MODULE_SOURCE} ${LAYER_SOURCE} ${FEATURE_SOURCE} ${TENSOR_SOURCES})
    # -ffast-math 계열은 켜지 않는다. 부동소수점 의미를 바꿔(나눗셈->역수 곱, 덧셈 재결합,
    # NaN/Inf 검사 제거) layer norm 과 attention softmax 에서 오차가 커진다.
    # 같은 오디오를 macOS 엔진에 넣으면 정확한데 기기에서만 인식이 무너지던 원인이다.
    target_compile_options(asr_api PRIVATE "-O3")
    # Android 는 NDK sysroot 의 jni.h 를 써야 한다 — 데스크톱 JDK include 를 넣으면
    # 그쪽 jni.h(AttachCurrentThread 가 void**)가 먼저 잡혀 NDK용 코드가 깨진다.
    target_compile_definitions(asr_api PRIVATE EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
    # ThinLTO cross-module inlining 크기 제한 (타깃 생성 후 add_link_options 는 미적용이라 target 옵션으로)
    target_link_options(asr_api PRIVATE -Wl,-plugin-opt,-import-instr-limit=30)
    # Link with static SentencePiece library explicitly
    target_link_libraries(asr_api PRIVATE Threads::Threads ${JNI} log ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build/src/libsentencepiece.a)
    # Add dependency to ensure SentencePiece is built first
    add_dependencies(asr_api sentencepiece-static)
elseif(IOS)
    message("Building for iOS")
    # JNI 래퍼(asr_api_wrap.cxx)와 java_callback 은 빼고, Objective-C 쪽을 싣는다.
    set(IOS_INTERFACE_SOURCE
        ../interface/callback_factory.cc
        ../interface/objectivec_callback.cc
    )
    # 두 파일 모두 #ifdef __OBJC__ 로 감싼 Objective-C 코드를 갖고 있어 Obj-C++ 로 컴파일해야 한다
    set_source_files_properties(
        ${IOS_INTERFACE_SOURCE}
        PROPERTIES COMPILE_FLAGS "-x objective-c++"
    )
    # Swift 에서 부를 Objective-C 래퍼. JNI 래퍼가 api/java/jni_wrapper 에 있는 것과 대칭으로
    # api/obj_c 에 두고, 라이브러리에 함께 넣어 앱은 헤더와 .a 만 가져가면 되게 한다.
    set(IOS_WRAPPER_SOURCE ../api/obj_c/MercuryAsr.mm)
    # 앱 번들에 그대로 링크되도록 정적 라이브러리로 만든다
    add_library(asr_api STATIC ../api/asr_api.cc ${IOS_WRAPPER_SOURCE} ${UTILITY_SOURCE} ${STATE_SOURCE} ${IOS_INTERFACE_SOURCE} ${DECODER_SOURCE} ${SCORE_SOURCE} ${MODULE_SOURCE} ${LAYER_SOURCE} ${FEATURE_SOURCE} ${TENSOR_SOURCES})
    # -ffast-math / -ffp-contract=fast 는 절대 켜지 않는다.
    # 부동소수점 의미를 바꿔(나눗셈->역수 곱, 덧셈 재결합, NaN/Inf 검사 제거)
    # layer norm 과 attention softmax 에서 오차가 증폭된다. 같은 오디오로
    # feature 가 소수점 셋째 자리까지 일치하는데 인코더 출력이 4% 벌어지고
    # 인식 결과가 통째로 무너졌던 원인이다.
    target_compile_options(asr_api PRIVATE "-O3")
    target_compile_definitions(asr_api PRIVATE OBJC EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
    target_link_libraries(asr_api PRIVATE Threads::Threads ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build/src/libsentencepiece.a)
    add_dependencies(asr_api sentencepiece-static)
    # ARC 앱과 섞여도 안전하도록 래퍼는 ARC 로 컴파일한다
    set_source_files_properties(${IOS_WRAPPER_SOURCE} PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    message("Building for Linux")
    add_library(asr_api SHARED ../api/asr_api.cc ../api/java/jni_wrapper/asr_api_wrap.cxx ${UTILITY_SOURCE} ${STATE_SOURCE} ${INTERFACE_SOURCE} ${DECODER_SOURCE} ${SCORE_SOURCE} ${MODULE_SOURCE} ${LAYER_SOURCE} ${FEATURE_SOURCE} ${TENSOR_SOURCES})
    target_compile_options(asr_api PRIVATE "-O3")
    target_include_directories(asr_api PRIVATE ${JNI_PATH1} ${JNI_PATH2})
    target_compile_definitions(asr_api PRIVATE JAVA EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_UNROLLING_LIMIT=256 SPM_NO_THREADLOCAL THREAD_POOL_ENABLED)
    # Link with static SentencePiece library explicitly
    target_link_libraries(asr_api PRIVATE Threads::Threads ${JNI} ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_build/src/libsentencepiece.a)
    # Add dependency to ensure SentencePiece is built first
    add_dependencies(asr_api sentencepiece-static)
endif()


