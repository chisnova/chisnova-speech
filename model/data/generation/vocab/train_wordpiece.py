#!/usr/bin/env python3
"""wordpiece(sentencepiece unigram) 어휘 학습 — extract_text.py 결과를 입력으로 받는다.

    python3 extract_text.py       # lab -> artifacts/train_tot.txt
    python3 train_wordpiece.py    # -> model/data/mercury.{model,vocab}

id 배치는 레코드/로더와 맞춰야 한다: pad=0, unk=1, bos(SOS)=2, eos(EOS)=3.
data_gen 이 이 규약대로 word_ids 를 SOS/EOS 로 감싸므로 바꾸면 DB 를 다시 만들어야 한다.

이전에는 모듈을 임포트만 해도 학습이 시작됐고(모듈 최상단 호출) 입력이 cwd 기준
"train_tot.txt" 였다. 인자로 받고 __main__ 안에서만 돌도록 바꿨다.
"""
import argparse
import os
import sys

import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_DIR, artifact


def trainWordPiece(txt_path, prefix, vocab_size):
    spm.SentencePieceTrainer.Train(
        input=txt_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        train_extremely_large_corpus=True,
        normalization_rule_name="nmt_nfkc_cf",
        character_coverage=1.0,
        byte_fallback=False,
        split_digits=True,
        num_threads=16,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=artifact("train_tot.txt"),
                    help="extract_text.py 가 만든 전사 텍스트")
    ap.add_argument("--prefix", default=os.path.join(DATA_DIR, "mercury"),
                    help="출력 prefix (.model / .vocab 이 붙는다)")
    ap.add_argument("--vocab_size", type=int, default=1024)
    args = ap.parse_args()

    print("input      : {}".format(args.input))
    print("output     : {}.model / {}.vocab".format(args.prefix, args.prefix))
    print("vocab_size : {}".format(args.vocab_size))
    trainWordPiece(args.input, args.prefix, args.vocab_size)
