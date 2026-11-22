# 인식 결과를 반영하는 발음 총점

발음 채점은 정답 토큰을 디코더에 입력하므로, 다른 문장에서도 일부 단어에 점수가 남을 수 있다. `recognition_weight`는 2패스 CTC 인식 결과와 라벨의 단어 일치도를 총점에 반영한다.

```text
최종 총점 = 기존 총점 × (1 - recognition_weight × (1 - similarity))
```

- `recognition_weight = 0`: 보정하지 않는다. 설정이 없는 기존 manifest의 런타임 기본값이다.
- `recognition_weight = 0.5`: 완전히 불일치하면 총점에 0.5를 곱한다. 앱 manifest와 새 exporter의 기본값이다.
- `recognition_weight = 1`: 일치도에 비례하며 완전히 불일치하면 0점이다.

단어·음소별 점수는 이 보정으로 바뀌지 않는다. 라벨이 있는데 인식 결과가 비어 있으면 일치도 0으로 처리한다. 런타임은 가중치를 0~1로 제한하고 비유한 값은 0으로 처리하며, exporter는 잘못된 설정을 오류로 거부한다.

## 일치도 계산

참조 단어 수를 기준으로 단어 편집거리에서 일치도를 구하고 0~1로 제한한다. 대소문자·문장부호·곡선 아포스트로피를 정리하고, 두 단어를 합친 철자가 같은 경우 단어 경계 차이를 허용한다. `it's`와 `its`처럼 아포스트로피가 단어를 구분하는 경우는 유지한다. 이 보정용 일치도는 콘텐츠 보고서의 원문 WER와 계산 규칙이 다르다.

숫자는 값이 같은 명시적인 읽기만 허용한다. 콘텐츠에 있는 0~10,000,000의 정수, 연도·연대, 1~99의 서수, 달러·퍼센트·`3D` 형식을 지원한다. 지원하지 않는 형식은 원래 토큰으로 비교한다.

| 라벨 | 인식 | 처리 |
|---|---|---|
| `1920s` | `nineteen twenties` | 일치 |
| `2007` | `two thousand and seven`, `twenty oh seven` | 일치 |
| `$32,000` | `thirty two thousand dollars` | 일치 |
| `5000` | `fifty`, `bananas are yellow` | 불일치 |
| `1920s` | `nineteen thirty s` | 불일치 |
| `I paid 5000 today` | `i paid today` | 숫자 누락 |

숫자가 있다는 이유로 임의 개수의 인식 단어를 정답으로 처리하지 않는다. 숫자를 생략해도 비용이 발생한다.

## 기능어 하한

기능어 하한은 원점수가 `function_word_floor / 2` 이상일 때만 적용한다. 하한이 90이면 원점수 40은 유지하고 50은 90으로 보정한다(앱 manifest와 exporter 기본값은 90, 이전에는 85). 이는 낮은 점수가 무조건 올라가지 않게 하는 기준이며, 실제 단어 누락을 확정하는 판정은 아니다.

## 고유명사 하한

지명·인명·작품명·약어·외래어는 채점 모델이 학습에서 거의 보지 못해 음소·낱말 디코더가 모두 확신을 갖지 못한다. 참조 음원(TTS)조차 60~70점대에 머무는 경우가 많아, 학습자의 발음과 무관하게 감점된다. `proper_noun_list`(기본 `proper_nouns.txt`)에 있는 낱말은 원점수가 `proper_noun_floor / 2` 이상이면 `proper_noun_floor`(기본 90)로 받쳐 준다.

- 목록 비교는 철자 그대로다(대소문자 유지, 앞뒤 문장부호만 제거). `Apple`은 걸리고 `apple`은 걸리지 않으며, 문장 첫 낱말이 목록에 없으면 영향을 받지 않는다.
- 원점수가 하한의 절반(기본 45) 미만이면 그대로 둔다. 다른 낱말을 읽은 경우 teacher forcing 원점수는 대부분 20점 미만이라 하한이 적용되지 않는다.
- 하한을 적용해도 `raw_score`(적용 전 점수)와 `floor_kind`(`function_word` / `proper_noun`)를 낱말 결과에 남긴다. 콘텐츠 검수기는 이를 `raw_score`, `score_floor` 열로 출력한다.
- 목록 파일이 없거나 `proper_noun_floor`가 0이면 적용하지 않는다. 설정이 없는 기존 manifest는 그대로 동작한다.

목록은 `python3 trace/build_proper_noun_list.py`가 콘텐츠 JSONL에서 만든다(문장 중간의 대문자 낱말 + `trace/proper_noun_manual.txt`). 콘텐츠를 바꾸면 다시 만들어 `cpp/score`, 모델 export 폴더, 앱 assets `model_state`에 복사한다.

## 채점 제외 낱말

`ungraded_word_list`(기본 `ungraded_words.txt`)의 낱말은 낱말 점수를 그대로 돌려주되 문장 평균과 최저 낱말 계산에서 빼고, 결과 JSON과 검수기 상세에 `graded=false`로 표시한다. 문장의 모든 낱말이 제외 대상이면 전체 평균으로 돌아간다. 비교는 고유명사 목록과 같이 철자 그대로다.

모델이 학습으로도 배우지 못한 형태에만 쓰는 마지막 수단이다(예: 문자명 "AA" — 관사 a 편향 때문에 라벨 교정과 추가학습 후에도 EY 확률이 19%에 그침). 넣는 만큼 학습자의 실제 오류를 놓치므로 목록은 아주 작게 유지하고 항목마다 근거를 파일에 적는다. 목록 파일이 없으면 모든 낱말을 채점한다.

ASR 오류도 총점에 영향을 줄 수 있으므로, 인식이 부정확한 환경에서는 검증 데이터로 가중치를 조정해야 한다. C++ 앱 API와 콘텐츠 검수기는 같은 보정 함수를 사용한다.

## 검증

저장소 루트에서 실행한다. C++ 회귀 테스트는 tokenizer·사전 자산만 사용하며 음향 모델 추론을 실행하지 않는다.

```bash
cmake -S cpp/test/api_test -B cpp/test/api_test/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/test/api_test/build --target recognition_score_test -j 4
ctest --test-dir cpp/test/api_test/build -R recognition_score_test --output-on-failure
PYTHONPATH=model python3 -m unittest exporter.test_stream_conformer_cpp_exporter -v
```

Android 라이브러리는 `bash cpp/export/android.sh --install`로 두 ABI를 빌드해 앱에 복사한다.
