# 계측 테스트 (androidTest)

## 실행 전제

- **arm 실기기가 필요하다.** `app/src/main/libs` 에는 `arm64-v8a` / `armeabi-v7a` 용
  `libasr_api.so` 만 있어서, x86_64 에뮬레이터에서는 `System.loadLibrary("asr_api")` 가 실패한다.
  네이티브가 필요 없는 `ModelAssetIntegrityTest`(단, `nativeLibraryLoads` 제외),
  `ModelValidationTest`, `WavFileParsingTest` 는 에뮬레이터에서도 돌아간다.
- 앱의 `assets/model_state`가 필요하다(`assets/copy.sh`). 진단 음원은 `src/androidTest/assets/google_steve`에서 읽으며 배포 앱에는 포함하지 않는다.
- 첫 실행은 모델 약 400MB 를 `filesDir` 로 복사하므로 수 분 걸린다. 이후 실행은 재사용한다.

```bash
# 전체
./gradlew :app:connectedDebugAndroidTest

# 빠른 것만 (자산·검증·WAV 파서)
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.size=small

# 클래스 하나
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.chisnova.mercury.asr.AsrRobustnessTest
```

## 구성

| 클래스 | 무엇을 지키는가 |
|---|---|
| `asr/ModelAssetIntegrityTest` | APK 에 실린 모델 자산과 `model.json` 의 정합성 — 텐서 누락·잔여 파일·shape 불일치·vocab/phone/사전 크기 |
| `asr/ModelValidationTest` | `AsrController.validateModelFiles` 의 계약 — 손상된 config/텐서를 거부하고, 정상 모델은 통과 |
| `asr/AsrEngineLifecycleTest` | 네이티브 상태 머신 계약 — 허용되지 않은 순서의 호출이 무시되고 엔진이 계속 쓸 수 있는 상태로 남는지 |
| `asr/AsrRobustnessTest` | 비정상 오디오·라벨(무음, NaN, 클리핑, 빈 버퍼, 비ASCII 라벨)에서 죽지 않고 점수 JSON 이 온전한지 |
| `asr/AsrDecodingRegressionTest` | 인식 품질 회귀 — 기준 발화 WER, 2패스 대 1패스, 결정성, 발음 점수의 상대 순서 |
| `asr/AsrKnownHazardTest` | 프로세스를 죽이는 것으로 확인된 경로. 전부 `@Ignore` 이며 네이티브 수정 후 하나씩 켠다 |
| `jni/WavFileParsingTest` | 기준 오디오를 읽는 WAV 파서의 경계 동작 |
| `support/` | 모델 설치·공유 엔진·WAV 로딩·WER 등 공용 코드 |

## 주의

- 네이티브 엔진은 프로세스당 하나만 만들어 모든 테스트가 공유한다(`AsrTestSupport.engine()`).
  각 테스트는 `AsrTestSupport.reset()` 으로 InitState 에서 시작한다.
- `AsrKnownHazardTest` 의 시험은 assert 실패가 아니라 SIGABRT/SIGSEGV 로 끝난다.
  같은 실행에 넣으면 뒤따르는 시험이 전부 무너지므로 반드시 개별 실행한다.
- `AsrDecodingRegressionTest` 의 WER 예산은 모델 교체 시 로그의 실측값을 보고 조정한다.
