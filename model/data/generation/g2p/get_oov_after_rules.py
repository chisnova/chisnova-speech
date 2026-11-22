#!/usr/bin/env python3
"""resolveWordPhones (하이픈공백 + 아포스트로피 + 형태소 + 복합어분해) 적용 후에도
사전으로 발음을 못 만드는 단어를 빈도순으로 저장. -> G2P 폴백 / phone-optional 대상."""
import collections
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))            # generation/  -> paths
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "record"))  # -> data_gen
import data_gen as dg   # full.dict, resolveWordPhones 로드
from paths import TRAIN_LAB, artifact

LAB = TRAIN_LAB
OUT = artifact("oov_after_rules.txt")

cnt = collections.Counter()
for line in open(LAB, encoding="utf-8", errors="replace"):
    p = line.rstrip("\n").split("\t", 1)
    if len(p) < 2:
        continue
    for w in p[1].lower().replace("-", " ").split():   # refineText 와 동일 정규화
        cnt[w] += 1

oov = [(w, c) for w, c in cnt.items() if dg.resolveWordPhones(w) is None]
oov.sort(key=lambda x: (-x[1], x[0]))

with open(OUT, "w", encoding="utf-8") as f:
    for w, _c in oov:
        f.write(w + "\n")

tot_tok = sum(cnt.values())
oov_tok = sum(c for _, c in oov)
print(f"unique words   : {len(cnt):,}")
print(f"남은 OOV type  : {len(oov):,} ({100*len(oov)/len(cnt):.3f}%)")
print(f"남은 OOV token : {oov_tok:,} / {tot_tok:,} ({100*oov_tok/tot_tok:.4f}%)")
print(f"saved -> {OUT}  ({len(oov):,} lines)")
