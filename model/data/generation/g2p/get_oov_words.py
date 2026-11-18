#!/usr/bin/env python3
"""train_tot.lab 의 단어들 중 g2p 사전에 없는 OOV 단어를 빈도순으로 뽑아 저장.

정규화는 data_gen 의 실제 조회 형태와 일치시킨다:
  - lower()
  - 하이픈은 공백으로 (refineText 의 replace("-"," ") 와 동일)
  - 공백 split
사전 키는 tokens[0].lower() 에서 변이형 접미사 "(N)" 을 제거한다.

data_gen 이 실제 로드하는 사전은 full.dict 다. --new 로 다른 사전을 주면
그것도 함께 검사해 비교한다.
출력: word<TAB>count, 빈도 내림차순.
"""
import re
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ARTIFACTS_DIR, FULL_DICT, TRAIN_LAB
import collections

VARIANT = re.compile(r"\(\d+\)$")   # CMU 변이형 발음 접미사 (2),(3) ...


def load_dict_keys(path):
    keys = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            key = line.split("\t")[0].lower()
            keys.add(VARIANT.sub("", key))
    return keys


def count_words(lab_path):
    """refineText 와 동일 정규화 후 단어 빈도. (단어 수, 총 발화 수) 반환."""
    counter = collections.Counter()
    n_utt = 0
    with open(lab_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) < 2:
                continue
            n_utt += 1
            text = parts[1].lower().replace("-", " ")   # == refineText(lower())
            for w in text.split():
                counter[w] += 1
    return counter, n_utt


def utt_loss(lab_path, keys):
    """OOV 단어를 하나라도 포함해 data_gen 에서 skip 되는 발화 수."""
    lost = 0
    n_utt = 0
    with open(lab_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) < 2:
                continue
            n_utt += 1
            text = parts[1].lower().replace("-", " ")
            if any(w not in keys for w in text.split()):
                lost += 1
    return lost, n_utt


def dump_oov(counter, keys, out_path, label):
    oov = [(w, c) for w, c in counter.items() if w not in keys]
    oov.sort(key=lambda x: (-x[1], x[0]))
    oov_tok = sum(c for _, c in oov)
    tot_tok = sum(counter.values())
    with open(out_path, "w", encoding="utf-8") as w:
        for word, _cnt in oov:
            w.write(f"{word}\n")
    print(f"\n[{label}]  dict='{out_path}'")
    print(f"  OOV type : {len(oov):,} / {len(counter):,} "
          f"({100 * len(oov) / max(len(counter), 1):.3f}%)")
    print(f"  OOV token: {oov_tok:,} / {tot_tok:,} "
          f"({100 * oov_tok / max(tot_tok, 1):.4f}%)")
    print(f"  saved    : {out_path}  ({len(oov):,} lines)")
    return len(oov)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract OOV words from a .lab against g2p dicts")
    ap.add_argument("--lab", default=TRAIN_LAB)
    ap.add_argument("--full", default=FULL_DICT)
    ap.add_argument("--new", default=None, help="비교해 볼 두 번째 사전 (선택)")
    ap.add_argument("--out_dir", default=ARTIFACTS_DIR)
    args = ap.parse_args()

    print("counting words in", args.lab, "...")
    counter, n_utt = count_words(args.lab)
    print(f"utterances={n_utt:,}  unique words={len(counter):,}  "
          f"total tokens={sum(counter.values()):,}")

    targets = [("full.dict (pipeline)", args.full, "oov_full.txt")]
    if args.new:
        targets.append(("new.dict (reference)", args.new, "oov_new.txt"))
    for name, dpath, oname in targets:
        keys = load_dict_keys(dpath)
        print(f"\ndict '{dpath}': {len(keys):,} keys")
        out = f"{args.out_dir}/{oname}"
        dump_oov(counter, keys, out, name)
        lost, _ = utt_loss(args.lab, keys)
        print(f"  발화 손실 : {lost:,} / {n_utt:,} ({100 * lost / n_utt:.3f}%)")
