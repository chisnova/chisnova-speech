# Mercury C++ 음성 인식 엔진

16 kHz 모노 음성을 실시간으로 인식하고 발음을 채점하는 C++17 런타임이다. 아래 명령은 `cpp/`에서 실행한다.

## 전체 세트 측정 결과

LibriSpeech에서 기본 모델로 발화 전체를 즉시 입력한 `offline` 측정값이다. xRT는 총 음성 길이를 총 처리 시간으로 나눈 배율이며 RTF는 그 역수다. 지연 시간은 마지막 오디오 입력 직전부터 최종 전사 반환까지 측정한다.

| 세트 | 발화 수 | 2차 WER | CTC WER | xRT | RTF | 지연 시간 p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test-clean | 2,620 | 5.61% | 3.73% | 19.62× | 0.0510 | 893 ms |
| test-other | 2,939 | 8.90% | 9.35% | 19.72× | 0.0507 | 759 ms |

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
model=../arifact/stream_conformer_cpp_export
./build/api_tester "$model" decode/java_api/wav.list
./build/java_api_test "$model" decode/java_api/wav.list
(cd build && ctest --output-on-failure)
```

## LibriSpeech 벤치마크

Python 3.8 이상에서 `benchmark/run.py`가 `build/api_tester`와 `libasr_api.so`로 WER, 지연 시간, 처리 속도를 측정한다. 기본 모델은 `../arifact/stream_conformer_cpp_export`이며 `--model`로 변경할 수 있다.

```bash
./benchmark/run.py --limit 3                     # clean·other 각각 3개
./benchmark/run.py                               # 두 세트 전체, offline
./benchmark/run.py --split clean --mode realtime # clean만 100 ms 단위로 입력
```

각 세트는 `benchmark/libri_test_clean/` 또는 `benchmark/libri_test_other/`의 두 목록을 발화 ID로 연결한다. `wav.list`는 `ID<TAB>WAV경로`, `lab.list`는 `ID<TAB>정답` 형식이다. WAV는 16 kHz 모노 PCM16이며 상대 경로는 목록 파일 기준이다.

결과는 `benchmark/results/<실행 시각>/`에 저장된다. `utterances.jsonl`에는 발화별 정답과 전사가, `summary.json`에는 세트별·전체 집계가 담긴다. `hypothesis`는 2차 Transformer 디코더의 빔 검색 결과(빔 크기 3)이며, 이 전사의 WER와 CTC 전사의 WER를 각각 계산한다.

모델 로딩과 WAV 읽기는 처리 시간에서 제외한다. `realtime` 모드에서는 실제 발화 속도로 입력하므로 발화 종료 후 응답 시간은 `final_latency_ms`로 확인한다. [벤치마크 상세 안내](benchmark/README.md)
