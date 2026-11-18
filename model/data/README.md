# 샘플 음성으로 LMDB 만들기

음성과 전사를 학습기가 읽는 레코드(LMDB)로 바꾼다. 명령은 `model/` 에서 실행한다.

## 1. 입력

```
data/sample/
  wav/              음성 100개 (LibriSpeech test, 합계 11분)
  sample.wlist      발화 id 와 음성 경로
    121-127105-0010  data/sample/wav/121-127105-0010.flac
  sample.lab        발화 id 와 전사
    121-127105-0010  SHE SENT ME THE PAGES IN QUESTION BEFORE SHE DIED

data/mercury.model  토큰 사전 (낱말 조각 1,024개)
data/full.dict      발음사전
    SHE  SH IY
```

목록 파일은 한 줄에 발화 하나이고 열은 탭으로 나눈다.

## 2. 실행

```bash
./data/make_lmdb.sh
```

인자는 없다. 안에서 아래 명령을 부른다.

```bash
python3 data/generation/record/data_gen.py \
    --wavpath  data/sample/sample.wlist \
    --txtpath  data/sample/sample.lab \
    --savepath data/sample/database \
    --vocab    data/mercury.model
```

다른 데이터로 만들 때는 이 명령의 세 경로를 바꿔 실행한다.

## 3. 출력

```
data/sample/database/
  shard_00000.lmdb     레코드 100개
  manifest.jsonl       발화마다 길이 한 줄
    {"utt_id": "121-127105-0010", "shard": 0, "feat_len": 286, "word_len": 16, "phone_len": 45}
  meta.json            "count": 100, "feat_dim": 80
  skipped.txt          버린 발화와 이유. 샘플은 비어 있다
```

레코드 하나는 발화 하나다.

```
utt_id       121-127105-0010
feat         (286, 80) float16       80차원 log-mel, 10ms 마다 한 프레임
word_ids     [2, 96, 33, 76, ..., 16, 3]      전사의 토큰 번호.  <s> ▁she ▁s ent ... ed </s>
phone_ids    [2, 35, 23, 4, ..., 13, 3]       전사의 발음 번호.  <s> SH IY _ ... D </s>
```

## 4. 확인

```bash
PYTHONPATH=$PWD python3 -c "
from data.loader.database import LmdbConn
conn = LmdbConn('data/sample/database')
manifest = conn.load_manifest()
record = conn.read_record(manifest.utt_ids[1], manifest.locators[1])
print(len(manifest), record['utt_id'], record['feat'].shape)
"
```

```
100 121-127105-0010 (286, 80)
```
