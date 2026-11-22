#!/usr/bin/env python3
"""Extract transcript text from a Kaldi-style .lab file (`<id>\\t<text>`).

Writes one transcript per line (id column dropped). Blank/empty
transcripts are skipped so they don't pollute tokenizer training.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import TRAIN_LAB, artifact


def extract(src, dst):
    total = kept = 0
    with open(src, "r", encoding="utf-8", errors="replace") as fp, \
         open(dst, "w", encoding="utf-8") as wp:
        for line in fp:
            total += 1
            # split on the first tab only; text may itself contain spaces
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) < 2:
                continue
            text = parts[1].strip()
            if not text:
                continue
            wp.write(text + "\n")
            kept += 1
    return total, kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=TRAIN_LAB)
    ap.add_argument("--output", default=artifact("train_tot.txt"))
    args = ap.parse_args()

    total, kept = extract(args.input, args.output)
    print(f"lines read : {total}")
    print(f"lines kept : {kept}")
    print(f"skipped    : {total - kept}")
    print(f"written to : {args.output}")
