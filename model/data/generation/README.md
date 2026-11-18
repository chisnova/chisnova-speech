# generation — 학습 데이터 만들기

주체별로 폴더가 나뉜다. 경로는 전부 [`paths.py`](paths.py) 가 파일 위치 기준으로
풀어 주므로 어느 디렉토리에서 실행해도 된다.

```
record/      wav + 전사 -> LMDB 레코드 (파이프라인의 최종 산출물)
vocab/       전사 텍스트 -> sentencepiece 어휘
g2p/         발음사전 커버리지 점검 (OOV 추출/분석, phone 규칙 검증)
artifacts/   위 스크립트들의 산출물 (코드 아님)
paths.py     공용 경로
```

## 실행 순서

```bash
# 1) 어휘 — 전사 텍스트를 뽑고 wordpiece 를 학습한다
python3 vocab/extract_text.py                 # lab -> artifacts/train_tot.txt
python3 vocab/train_wordpiece.py              # -> model/data/mercury.{model,vocab}

# 2) 발음사전 점검 — 어느 단어가 발음을 못 만드는지 (선택)
python3 g2p/get_oov_words.py                  # -> artifacts/oov_full.txt
python3 g2p/get_oov_after_rules.py            # 규칙 적용 후에도 남는 OOV
python3 g2p/validate_phone_rules.py           # phone 기호가 phone_dict 안에 있는지

# 3) 레코드 — LMDB DB 생성
./record/make_dev_db.sh
./record/make_test_db.sh
./record/make_train_db.sh                     # 샤드 ~10GB 단위, 660GB 기준 ~66 샤드
```

만들어진 DB 확인은 로더 쪽 테스트로 한다:

```bash
python3 ../loader/test/test_database.py       # dev/test/train 에서 100개씩
python3 ../loader/test/test_loader.py         # 배치 구성까지
```

## 규약 (바꾸면 DB 를 다시 만들어야 한다)

- 토큰 id: `pad=0, unk=1, SOS=2, EOS=3` — `train_wordpiece.py` 와 `data_gen.py` 가 공유
- 레코드: `{utt_id, feat (T,80) float16, word_ids, phone_ids}`
- 사전: `model/data/full.dict` (67.5만 줄)
