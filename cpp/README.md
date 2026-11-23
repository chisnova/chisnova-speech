# Mercury C++ 음성 인식 엔진

16 kHz 모노 음성을 실시간으로 인식하고 발음을 채점하는 C++17 런타임이다. 아래 명령은 `cpp/`에서 실행한다.

## 빌드 환경과 산출물

공통 도구는 CMake 3.16 이상과 Make 또는 Ninja다.

| 대상 | 필요한 환경 | 주요 산출물 |
| --- | --- | --- |
| Linux | C++17 컴파일러, JDK(JNI 헤더 포함). JDK 탐색 실패 시 `JAVA_HOME` 지정 | `build/`에 `libasr_api.so`, `api_tester`, `java_api_test`, 실행용 JAR |
| Android | Android NDK. 자동 탐색 실패 시 `ANDROID_NDK_HOME` 지정 | `build/dist/lib/android/<ABI>/libasr_api.so` (`arm64-v8a`, `armeabi-v7a`) |
| iOS | macOS, Xcode의 기기·시뮬레이터 SDK | `build/ios/dist/lib/ios/<SDK>/MercuryAsr.framework`와 정적 라이브러리 (`iphoneos`, `iphonesimulator`, 모두 arm64) |

## 빌드

Linux 기본 빌드는 NDK를 찾으면 Android 두 ABI도 함께 빌드한다.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
```

iOS 전용 빌드는 macOS에서 실행한다.

```bash
cmake -S . -B build/ios -DMERCURY_EXPORT_ONLY=ON
cmake --build build/ios --target export_ios
```

iOS 앱에 연결하려면 이어서 `cmake --build build/ios --target copy_ios_to_app`을 실행한다. 기기·시뮬레이터용 프레임워크가 `ios/Engine/lib/MercuryAsr.xcframework`로 묶인다.

## Linux 실행 및 테스트

모델 디렉터리에는 `model.json`과 가중치가 필요하다. WAV 목록에는 16 kHz 모노 16비트 PCM WAV 경로를 한 줄에 하나씩 적는다. 상대 경로는 목록 파일 기준이다.

```bash
model=../arifact/stream_conformer_mercury3_adapt_1p5s_cpp_export
./build/api_tester "$model" decode/java_api/wav.list
./build/java_api_test "$model" decode/java_api/wav.list
(cd build && ctest --output-on-failure)
```
