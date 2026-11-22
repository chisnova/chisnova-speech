#!/usr/bin/env python3
"""shuffle 케이스 — seed 재현성 / 발화 커버리지 / 에폭 재셔플."""
import sys

from common import check, print_step, print_title, run_as_pytest, run_standalone

import numpy as np

from data.loader import Sorter


def step_1_check_seed_reproducibility(loader):
    """같은 seed -> 같은 순서, 다른 seed -> 다른 순서."""
    print_step(1, "seed 재현성")
    sorter = Sorter(chunk=loader.sorter.chunk, shuffle=True)
    make = lambda seed: sorter.make_order(loader.manifest.feat_len, loader.base_index,
                                          np.random.default_rng(seed))
    order_a, order_b, order_c = make(7), make(7), make(8)
    check(np.array_equal(order_a, order_b), "같은 seed -> 같은 순서")
    check(not np.array_equal(order_a, order_c), "다른 seed -> 다른 순서")


def step_2_check_coverage(loader, tag=""):
    """usage_ratio=1 이면 전량, <1 이면 중복 없는 부분집합이어야 한다."""
    print("    {}utt_order {:,} / base_index {:,} (usage_ratio={})".format(
        tag, loader.utt_order.shape[0], loader.base_index.shape[0],
        loader.usage_ratio))
    if loader.usage_ratio >= 1.0:
        return check(np.array_equal(np.sort(loader.utt_order),
                                    np.sort(loader.base_index)),
                     tag + "모든 발화를 정확히 한 번씩 포함 (중복/누락 없음)")
    expected_size = max(1, int(loader.full_data_size * loader.usage_ratio))
    return (check(np.unique(loader.utt_order).shape[0] == loader.utt_order.shape[0],
                  tag + "부분집합에 중복 없음")
            and check(bool(np.isin(loader.utt_order, loader.base_index).all()),
                      tag + "전부 base_index 안의 발화")
            and check(loader.utt_order.shape[0] == expected_size,
                      tag + "표본 크기 == full*usage_ratio ({:,})".format(expected_size)))


def step_3_check_batch_boundaries(loader):
    """배치 경계가 발화를 빠뜨리거나 빈 배치를 만들지 않는지."""
    print_step(3, "배치 경계")
    check(int(np.diff(loader.batch_offsets).sum()) == loader.data_size,
          "배치 크기 합 == 전체 발화 수 ({:,})".format(loader.data_size))
    check(bool((np.diff(loader.batch_offsets) > 0).all()), "빈 배치 없음")


def step_4_check_epoch_reshuffle(loader):
    """에폭 경계에서 순서/구성이 다시 짜이는지."""
    print_step(4, "에폭 재셔플")
    utt_order_before = loader.utt_order.copy()
    batch_order_before = loader.batch_order.copy()
    print("    epoch {} 첫 배치 순서: {}".format(loader.epoch,
                                                 batch_order_before[:8].tolist()))
    loader.reshuffle_for_next_epoch()
    print("    epoch {} 첫 배치 순서: {}".format(loader.epoch,
                                                 loader.batch_order[:8].tolist()))
    if loader.shuffle:
        check(not np.array_equal(utt_order_before, loader.utt_order),
              "에폭마다 발화 순서 재구성")
        check(not np.array_equal(batch_order_before, loader.batch_order),
              "에폭마다 배치 순서 변경")
    else:
        check(np.array_equal(utt_order_before, loader.utt_order),
              "shuffle=False 면 순서 고정")


def step_5_print_chunk_sort_effect(loader):
    """배치 안 길이 편차 — chunk 내 정렬이 실제로 길이를 모아 주는지."""
    print_step(5, "chunk 정렬 효과")
    feat_len = loader.manifest.feat_len
    spreads = [int(feat_len[loader.utt_order[start:end]].max()
                   - feat_len[loader.utt_order[start:end]].min())
               for start, end in zip(loader.batch_offsets[:-1],
                                     loader.batch_offsets[1:])][:2000]
    print("    배치 내 feat_len 편차(max-min) 평균: {:.1f} 프레임  (chunk={:,})".format(
        float(np.mean(spreads)), loader.sorter.chunk))


def run(loader, args, sp_processor):
    print_title("shuffle — 재현성 / 커버리지 / 에폭 재셔플")
    step_1_check_seed_reproducibility(loader)
    print_step(2, "발화 커버리지")
    step_2_check_coverage(loader)
    step_3_check_batch_boundaries(loader)
    step_4_check_epoch_reshuffle(loader)
    step_2_check_coverage(loader, tag="재셔플 후 ")
    step_5_print_chunk_sort_effect(loader)


def test_shuffle():
    """pytest / VSCode Testing 패널 진입점."""
    run_as_pytest(run)


if __name__ == "__main__":
    sys.exit(run_standalone(run, __doc__))
