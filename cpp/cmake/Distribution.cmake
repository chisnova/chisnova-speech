# Host-side targets. Each mobile ABI/SDK uses its own root CMake build tree.
set(MERCURY_MODEL_DIR "" CACHE PATH "Checkpoint directory containing config.json and asr_model.pt")
option(MERCURY_FORCE_EXPORT "Regenerate the model during release packaging" OFF)
function(mercury_model_target target mode)
    add_custom_target(${target}
        COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/scripts/model_assets.py" "${mode}"
            --dist "${MERCURY_DIST_DIR}" --checkpoint "${MERCURY_MODEL_DIR}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" VERBATIM)
endfunction()
mercury_model_target(export_model model)
mercury_model_target(export_score score)

function(mercury_platform_target name platform)
    add_custom_target(${name}
        COMMAND "${CMAKE_COMMAND}" "-Dplatform=${platform}" "-Dsource=${CMAKE_CURRENT_SOURCE_DIR}"
            "-Dbuild=${CMAKE_CURRENT_BINARY_DIR}/platforms/${platform}" "-Ddist=${MERCURY_DIST_DIR}"
            "-Dgenerator=${CMAKE_GENERATOR}" "-Dndk=${MERCURY_ANDROID_NDK}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildPlatform.cmake"
        USES_TERMINAL VERBATIM)
endfunction()
set(MERCURY_ANDROID_NDK "" CACHE PATH "Android NDK for export_android")
mercury_platform_target(export_android_arm64 arm64-v8a)
mercury_platform_target(export_android_armv7 armeabi-v7a)
mercury_platform_target(export_ios_simulator iphonesimulator)
mercury_platform_target(export_ios_device iphoneos)
add_custom_target(export_android DEPENDS export_android_arm64 export_android_armv7)
add_custom_target(export_ios DEPENDS export_ios_simulator export_ios_device)
if(NOT MERCURY_EXPORT_ONLY)
    include("${CMAKE_CURRENT_LIST_DIR}/AndroidNdk.cmake")
    mercury_find_android_ndk(ndk "${MERCURY_ANDROID_NDK}")
    set(android_available OFF)
    if(EXISTS "${ndk}/build/cmake/android.toolchain.cmake")
        set(android_available ON)
    endif()
    option(MERCURY_BUILD_ANDROID "Build both Android ABI libraries with the default build" ${android_available})
    set(ios_available OFF)
    if(CMAKE_HOST_APPLE)
        execute_process(COMMAND xcrun --sdk iphoneos --show-sdk-path RESULT_VARIABLE device_sdk
            OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND xcrun --sdk iphonesimulator --show-sdk-path RESULT_VARIABLE sim_sdk
            OUTPUT_QUIET ERROR_QUIET)
        if(device_sdk STREQUAL "0" AND sim_sdk STREQUAL "0")
            set(ios_available ON)
        endif()
    endif()
    option(MERCURY_BUILD_IOS "Build the iOS device and simulator libraries with the default build" ${ios_available})
    foreach(platform IN ITEMS android/arm64-v8a android/armeabi-v7a ios/iphoneos ios/iphonesimulator)
        file(MAKE_DIRECTORY "${MERCURY_DIST_DIR}/lib/${platform}")
    endforeach()
    add_custom_target(mobile_libraries ALL)
    if(MERCURY_BUILD_ANDROID)
        add_dependencies(mobile_libraries export_android)
    endif()
    if(MERCURY_BUILD_IOS)
        add_dependencies(mobile_libraries export_ios)
    endif()
    message(STATUS "Default mobile libraries: Android=${MERCURY_BUILD_ANDROID}, iOS=${MERCURY_BUILD_IOS} (${MERCURY_DIST_DIR}/lib)")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT MERCURY_EXPORT_ONLY)
    add_custom_target(export_linux DEPENDS export_native)
else()
    mercury_platform_target(export_linux linux)
endif()

if(MERCURY_FORCE_EXPORT)
    mercury_model_target(prepare_release_model force)
else()
    mercury_model_target(prepare_release_model cached-model)
endif()
add_custom_target(package_release DEPENDS prepare_release_model export_android)

foreach(item IN ITEMS model android ios)
    if(item STREQUAL "model")
        set(script android/app/src/main/assets/copy.sh)
    elseif(item STREQUAL "android")
        set(script android/app/src/main/libs/copy.sh)
    else()
        set(script ios/Engine/copy.sh)
    endif()
    add_custom_target(copy_${item}_to_app
        COMMAND "${CMAKE_COMMAND}" -E env "MERCURY_EXPORT_DIR=${MERCURY_DIST_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../${script}" VERBATIM)
endforeach()
