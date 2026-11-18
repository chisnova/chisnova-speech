#!/usr/bin/env python3
"""record 케이스 — 저장소 원본 레코드 몇 개를 눈으로 확인한다."""
import sys

from common import (check, decode_phones, decode_words, print_step, print_title, 
                    run_as_pytest, run_standalone)

import numpy as np


def step_1_read_records(loader, num_records):
    """앞에서부터 num_records 개를 읽는다."""
    print_step(1, "레코드 {}개 읽기".format(num_records))
    manifest = loader.manifest
    picked = list(range(min(num_records, len(manifest))))
    records = [loader.conn.read_record(manifest.utt_ids[i], manifest.locators[i])
               for i in picked]
    print("    {}개 읽음".format(len(records)))
    return picked, records


def step_2_print_records(loader, picked, records, sp_processor):
    """feat 통계 / word_ids -> 전사 / phone_ids -> 발음."""
    print_step(2, "레코드 값")
    manifest = loader.manifest
    for utt_index, record in zip(picked, records):
        feat_arr = np.asarray(record["feat"], dtype=np.float32)
        print("    " + "-" * 70)
        print("    utt_id   : {}  (locator {})".format(record["utt_id"],
                                                       manifest.locators[utt_index]))
        print("    feat     : {} {}  min/mean/max = {:.1f}/{:.1f}/{:.1f} dB".format(
            record["feat"].shape, record["feat"].dtype,
            feat_arr.min(), feat_arr.mean(), feat_arr.max()))
        print("    feat[0,:6]: [{}]".format(
            ", ".join("{:.2f}".format(v) for v in feat_arr[0, :6])))
        print("    word_ids : {}".format([int(x) for x in record["word_ids"]]))
        print("      -> text: {!r}".format(decode_words(sp_processor,
                                                        record["word_ids"])))
        print("    phone_ids: {}".format([int(x) for x in record["phone_ids"]]))
        print("      -> phone: {}".format(decode_phones(record["phone_ids"])))


def step_3_check_against_manifest(loader, picked, records):
    """레코드가 카탈로그와 맞는지."""
    print_step(3, "카탈로그와 대조")
    manifest = loader.manifest
    for utt_index, record in zip(picked, records):
        feat_arr = np.asarray(record["feat"], dtype=np.float32)
        check(feat_arr.shape == (manifest.feat_len[utt_index], loader.feat_dim),
              "feat 모양이 카탈로그/meta 와 일치 ({}, {})".format(
                  manifest.feat_len[utt_index], loader.feat_dim))
        check(len(record["word_ids"]) == manifest.word_len[utt_index]
              and len(record["phone_ids"]) == manifest.phone_len[utt_index],
              "word/phone 길이가 카탈로그와 일치")
        check(int(record["word_ids"][0]) == 2 and int(record["word_ids"][-1]) == 3,
              "word_ids 가 SOS(2)/EOS(3) 로 감싸짐")
        check(int(record["phone_ids"][0]) == 2 and int(record["phone_ids"][-1]) == 3,
              "phone_ids 가 SOS(2)/EOS(3) 로 감싸짐")
        check(np.isfinite(feat_arr).all(), "feat 에 NaN/Inf 없음")


def run(loader, args, sp_processor):
    print_title("record — 저장소 원본 레코드 값")
    picked, records = step_1_read_records(loader, args.num_records)
    step_2_print_records(loader, picked, records, sp_processor)
    step_3_check_against_manifest(loader, picked, records)


def test_record():
    """pytest / VSCode Testing 패널 진입점."""
    run_as_pytest(run)


if __name__ == "__main__":
    sys.exit(run_standalone(run, __doc__))
