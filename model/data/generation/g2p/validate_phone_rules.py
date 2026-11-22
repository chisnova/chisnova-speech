import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))            # generation/  -> paths
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "record"))  # -> data_gen
import data_gen as dg   # 모듈 로드 시 full.dict + phone_dict 구성됨
from paths import TRAIN_LAB

dictionary = dg.dictionary
phone_dict = dg.phone_dict
resolve = dg.resolveWordPhones
LAB = TRAIN_LAB

# 0) 사전 값의 phone 기호가 전부 phone_dict에 있는지 (없으면 그 단어 skip됨)
bad_syms = set()
for v in dictionary.values():
    for s in v.split(" "):
        if s and s not in phone_dict:
            bad_syms.add(s)
print("phone_dict 밖 기호(사전 값 기준):", sorted(bad_syms) if bad_syms else "없음")

# 1) spot check
print("\n[spot check]")
for w in ["digitalocean","softwareengineeringdaily","humanityunited","therapistfinder",
          "kubernetes","microservices","cao's","fuckin'","kubernetes'","coding","coded"]:
    print(f"  {w:26s} -> {resolve(w)}")

# 2) 전체 코퍼스: 새 발화 손실 + resolve 못한 단어 집계
import collections
tot=lost=0
unresolved=collections.Counter()
emit_bad=0
for line in open(LAB, encoding="utf-8", errors="replace"):
    p=line.rstrip("\n").split("\t",1)
    if len(p)<2: continue
    tot+=1
    words=p[1].lower().replace("-"," ").split()
    bad=False
    for w in words:
        ph=resolve(w)
        if ph is None:
            unresolved[w]+=1; bad=True
        else:
            # emit 검증: 모든 기호가 phone_dict에 있어야 함
            for s in ph.split(" "):
                if s and s not in phone_dict and s!="_":
                    emit_bad+=1; bad=True
    if bad: lost+=1

print(f"\n총 발화={tot:,}")
print(f"수정 후 버려지는 발화={lost:,} ({100*lost/tot:.3f}%)")
print(f"emit 기호 오류 건수={emit_bad}")
print(f"미해결 단어 type={len(unresolved):,}  token={sum(unresolved.values()):,}")
print("미해결 상위 15:")
for w,c in unresolved.most_common(15):
    print(f"   {c:>7,}  {w!r}")
