# Mercury Speech

Mercury는 16 kHz 모노 음성을 기기에서 실시간으로 인식하고 발음을 채점하는 프로젝트다. Python에서 학습한 스트리밍 Conformer 모델을 C++ 엔진으로 내보내며, Android는 JNI, iOS는 Objective-C 브리지로 같은 엔진을 사용한다.

| 디렉터리 | 역할 |
| --- | --- |
| [model](model/README.md) | 학습 데이터 준비, PyTorch 모델 학습 및 C++ 모델 내보내기 |
| [cpp](cpp/README.md) | 공유 라이브러리, C++·Java 실행 파일과 플랫폼별 CMake 빌드 |
| [cpp/benchmark](cpp/benchmark/README.md) | LibriSpeech test-clean·test-other의 WER, 처리 속도와 지연 시간 측정 |
| [android](android/) | Kotlin 앱. 음성 인식·발음 연습, 결과 조회 및 학습 기록 |
| [ios](ios/project.yml) | Swift 앱과 XcodeGen 프로젝트. CMake로 만든 XCFramework 사용 |

## 모델 받기

학습 체크포인트가 필요하면 `stream_conformer.tar.gz`, 인식과 벤치마크에는 `stream_conformer_cpp_export.tar.gz`를 사용한다. 아래 명령은 저장소 루트에서 추론 모델을 내려받아 검증한다. [모델 Release](https://github.com/chisnova/chisnova-speech/releases/tag/stream-conformer-20261125)

```bash
mkdir -p arifact
gh release download stream-conformer-20261125 --repo chisnova/chisnova-speech \
  --pattern stream_conformer_cpp_export.tar.gz --pattern SHA256SUMS --dir arifact
(cd arifact && sha256sum --ignore-missing -c SHA256SUMS && tar -xzf stream_conformer_cpp_export.tar.gz)
```

## Linux에서 음성 인식

C++17 컴파일러, CMake 3.16 이상, Make 또는 Ninja, JNI 헤더가 포함된 JDK가 필요하다.

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DMERCURY_TEST_MODEL_DIR="$PWD/arifact/stream_conformer_cpp_export"
cmake --build cpp/build --parallel 4
cpp/build/api_tester arifact/stream_conformer_cpp_export cpp/decode/java_api/wav/conan00000.wav
```

LibriSpeech 전체 세트의 오프라인 2차 디코딩 WER는 test-clean 5.61%, test-other 8.90%다. 실행 환경, RTF와 측정 기준은 [C++ 엔진 안내](cpp/README.md)에 있다.

## 모바일 앱

두 앱의 모델을 준비하려면 내려받은 추론 패키지를 Android 자산 디렉터리에 복사한다. 앱 전용 `model.json`은 저장소에 포함되어 있으며 iOS도 같은 자산을 참조한다.

```bash
rsync -a --exclude=model.json arifact/stream_conformer_cpp_export/ android/app/src/main/assets/model_state/
```

- **Android** — JDK 17, Android SDK 34와 NDK를 사용한다. 위 CMake 구성 후 `cmake --build cpp/build --target export_android`, `cmake --build cpp/build --target copy_android_to_app`을 순서대로 실행한다. APK는 `(cd android && ./gradlew assembleDebug)`로 빌드한다.
- **iOS** — macOS와 Xcode에서 [iOS 프레임워크 빌드 및 복사](cpp/README.md#빌드)를 마친 뒤 `(cd ios && xcodegen generate)`를 실행하고 생성된 프로젝트를 연다.

앱의 학습 콘텐츠 번들과 공용 정책 리소스(`legal/`)는 별도로 준비한다.
