#!/usr/bin/env python3
"""manifest 케이스 — 카탈로그의 길이 분포 / 샤드 분포 / 필드 정합성."""
import sys

from common import check, print_step, print_title, run_as_pytest, run_standalone

import numpy as np


def step_1_print_layout(loader):
    """DB 레이아웃과 규모."""
    print_step(1, "레이아웃")
    manifest = loader.manifest
    print("    DB       : {}".format(loader.data_path))
    print("    샤드     : {}개".format(loader.conn.num_shards))
    print("    feat_dim : {}   meta: {}".format(loader.feat_dim,
                                                loader.conn.meta or "(없음)"))
    print("    총 발화  : {:,}   (길이필터 후 {:,})".format(len(manifest),
                                                            loader.full_data_size))
    check(len(manifest) > 0, "manifest 비어있지 않음")


def step_2_print_length_distribution(loader):
    """feat/word/phone 길이 분포 — 배치 계산의 유일한 입력."""
    print_step(2, "길이 분포")
    manifest = loader.manifest
    for field_name, length_arr in (("feat_len", manifest.feat_len),
                                   ("word_len", manifest.word_len),
                                   ("phone_len", manifest.phone_len)):
        print("    {:9s}: min {:5d}  max {:6d}  mean {:8.1f}  p50 {:5d}  p95 {:6d}".format(
            field_name, length_arr.min(), length_arr.max(), length_arr.mean(),
            int(np.percentile(length_arr, 50)), int(np.percentile(length_arr, 95))))
    total_hours = manifest.feat_len.sum() / 100.0 / 3600.0
    print("    총 프레임: {:,}  (~{:.1f} 시간, 100fps 기준)".format(
        manifest.feat_len.sum(), total_hours))
    check(int(manifest.feat_len.min()) > 0, "feat_len 전부 > 0")


def step_3_print_shard_distribution(loader):
    """locator(샤드) 분포 — 저장소 위치 힌트가 실재하는 범위인지."""
    print_step(3, "샤드(locator) 분포")
    manifest = loader.manifest
    shard_ids, shard_counts = np.unique(manifest.locators, return_counts=True)
    head = ", ".join("{}:{:,}".format(shard_id, count)
                     for shard_id, count in list(zip(shard_ids, shard_counts))[:8])
    print("    샤드별 발화 수(앞 8개): {}{}".format(
        head, " ..." if len(shard_ids) > 8 else ""))
    check(int(manifest.locators.max()) < loader.conn.num_shards,
          "locator 가 전부 실재하는 샤드 범위 안")


def step_4_check_field_consistency(loader):
    """모든 필드가 같은 길이인지 + 샘플 엔트리."""
    print_step(4, "필드 정합성")
    manifest = loader.manifest
    print("    샘플 엔트리:")
    for i in range(min(3, len(manifest))):
        print("      [{}] utt={} locator={} feat_len={} word_len={} phone_len={}".format(
            i, manifest.get_utt_id(i), manifest.locators[i], manifest.feat_len[i],
            manifest.word_len[i], manifest.phone_len[i]))
    check(manifest.locators.shape == manifest.feat_len.shape
          == manifest.word_len.shape == manifest.phone_len.shape
          == manifest.utt_ids.shape,
          "모든 필드 길이 일치 ({:,})".format(len(manifest)))


def run(loader, args, sp_processor):
    print_title("manifest — 길이 분포 / 샤드 / 필드 정합성")
    step_1_print_layout(loader)
    step_2_print_length_distribution(loader)
    step_3_print_shard_distribution(loader)
    step_4_check_field_consistency(loader)


def test_manifest():
    """pytest / VSCode Testing 패널 진입점."""
    run_as_pytest(run)


if __name__ == "__main__":
    sys.exit(run_standalone(run, __doc__))
