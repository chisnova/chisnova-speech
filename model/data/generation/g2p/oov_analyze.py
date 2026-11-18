import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import FULL_DICT, TRAIN_LAB

DICT = FULL_DICT
LAB = TRAIN_LAB
suf=re.compile(r"\(\d+\)$")
keys=set()
with open(DICT,encoding="utf-8",errors="replace") as f:
    for line in f:
        line=line.rstrip("\n")
        if line: keys.add(suf.sub("",line.split("\t")[0].lower()))

cnt=collections.Counter()
with open(LAB,encoding="utf-8",errors="replace") as f:
    for line in f:
        p=line.rstrip("\n").split("\t",1)
        if len(p)<2: continue
        for w in p[1].lower().replace("-"," ").split(): cnt[w]+=1

oov={w:c for w,c in cnt.items() if w not in keys}
tot_tok=sum(cnt.values())
print(f"OOV types={len(oov):,}  OOV tokens={sum(oov.values()):,} ({100*sum(oov.values())/tot_tok:.3f}%)")

# 1) 빈도 분포: hapax(1회)가 얼마나 롱테일을 차지하나
from collections import Counter
freq_hist=Counter()
for c in oov.values():
    b = "1" if c==1 else "2-5" if c<=5 else "6-20" if c<=20 else "21-100" if c<=100 else ">100"
    freq_hist[b]+=1
print("\n[OOV 빈도 분포]")
for k in ["1","2-5","6-20","21-100",">100"]:
    n=freq_hist[k]; print(f"  {k:>7}회: {n:>8,} types ({100*n/len(oov):.1f}%)")

def in_dict(w): return w in keys

def dp_splittable(w, maxparts=3):
    # 사전 단어 2~maxparts개로 분해 가능? (최소 길이 2)
    n=len(w)
    # dp[i] = i까지 분해 가능 여부, 파트 수 추적
    dp=[None]*(n+1); dp[0]=0
    for i in range(1,n+1):
        for j in range(max(0,i-20),i):
            if dp[j] is not None and (i-j)>=2 and w[j:i] in keys:
                if dp[i] is None or dp[j]+1<dp[i]: dp[i]=dp[j]+1
    return dp[n] is not None and dp[n]>=2 and dp[n]<=maxparts

cats=collections.Counter(); cat_tok=collections.Counter()
buckets={}
for w,c in oov.items():
    if "'" in w:
        base=w.rstrip("'")
        if base.endswith("'s"): base=base[:-2]
        elif base.endswith("s'"): base=base[:-1]
        cat = "apostrophe_base_in_dict" if in_dict(w.replace("'","")) or in_dict(base) else "apostrophe_other"
    elif any(ch.isdigit() for ch in w):
        cat="has_digit"
    elif w.endswith("s") and in_dict(w[:-1]):
        cat="plural_s_base_in_dict"
    elif w.endswith("es") and in_dict(w[:-2]):
        cat="plural_es_base_in_dict"
    elif w.endswith("ing") and (in_dict(w[:-3]) or in_dict(w[:-3]+"e")):
        cat="ing_base_in_dict"
    elif w.endswith("ed") and (in_dict(w[:-2]) or in_dict(w[:-1])):
        cat="ed_base_in_dict"
    elif len(w)>=8 and dp_splittable(w):
        cat="compound_splittable"
    else:
        cat="true_oov(rare/proper)"
    cats[cat]+=1; cat_tok[cat]+=c
    buckets.setdefault(cat,[]).append((w,c))

print("\n[OOV 분류 — 줄일 수 있는 버킷]")
for cat,n in cats.most_common():
    print(f"  {cat:28s}: {n:>8,} types  {cat_tok[cat]:>9,} tokens ({100*cat_tok[cat]/tot_tok:.4f}%)")

# top-N 빈도 OOV를 사전에 추가하면 token 커버리지 얼마나 회복?
srt=sorted(oov.values(),reverse=True)
for N in [1000,5000,10000,20000]:
    cov=sum(srt[:N]); print(f"\n상위 {N:,}개 OOV 추가 시 회복 token: {cov:,} ({100*cov/sum(oov.values()):.1f}% of OOV, {100*cov/tot_tok:.4f}% of all)")
