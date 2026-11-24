# LibriSpeech 음성 인식 벤치마크

저장소 루트에서 실행한다. `api_tester`는 `cpp/build/libasr_api.so`를 호출하며, 기본 모델은 `arifact/stream_conformer_cpp_export`다. Python 3.8 이상이 필요하다.

```bash
cmake --build cpp/build --target api_tester --parallel 4
cpp/benchmark/run.py --limit 3                     # clean·other에서 각각 3개
cpp/benchmark/run.py                               # 두 세트 전체
cpp/benchmark/run.py --split clean --mode realtime # 100 ms 단위 실시간 입력
```

`libri_test_{clean,other}/wav.list`와 `lab.list`를 발화 ID로 연결한다. `wav.list`의 각 줄은 `발화ID<TAB>WAV경로`, `lab.list`의 각 줄은 `발화ID<TAB>정답`이다. 16 kHz 모노 PCM16 WAV를 사용하며 상대 경로는 `wav.list`가 있는 폴더를 기준으로 읽는다. 모델은 `--model`, 빌드 디렉터리는 `--build-dir`로 바꿀 수 있다.

결과는 `results/<실행 시각>/`에 저장된다. `utterances.jsonl`에는 발화별 정답·CTC 전사·2차 전사와 측정값이, `summary.json`에는 세트별·전체 집계와 모델 로딩 시간이 들어간다.

| 지표 | 계산 |
| --- | --- |
| WER (%) | 2차 전사의 총 단어 편집거리 ÷ 총 정답 단어 수 × 100. CTC WER도 별도 기록한다. 소문자로 변환하고 하이픈은 공백으로 바꾸며 아포스트로피는 유지한다. |
| elapsed_s | 엔진 초기화 상태로 복귀하는 `clear()` 직전부터 2차 전사 반환까지 걸린 시간. 모델은 한 번 로드하며 WAV 파일 읽기는 측정 밖에서 수행한다. |
| final_latency_ms | 마지막 오디오 입력 호출 직전부터 2차 전사 반환까지의 시간. 평균·중앙값·p95를 집계한다. |
| xRT / RTF | xRT는 총 음성 길이 ÷ 총 처리 시간(속도 배율), RTF는 그 역수다. |

기본 `offline`은 발화 전체를 즉시 입력해 처리 속도를 측정한다. `realtime`은 음성이 도착하는 속도로 입력한다. 이 모드의 처리 시간에는 입력 대기가 포함되므로, 발화 종료 후 응답 지연은 `final_latency_ms`로 확인한다.
