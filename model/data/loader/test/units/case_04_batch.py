#!/usr/bin/env python3
"""batch 케이스 — 배치 텐서의 모양·dtype·마스크 경계·패딩을 확인한다."""
import sys

from common import (check, decode_phones, decode_words, print_step, print_title, 
                    run_as_pytest, run_standalone)

import numpy as np
import torch


def find_catalog_indices(manifest, utt_ids):
    """배치의 utt_id 들이 카탈로그에서 몇 번째인지 찾는다.

    전체 역인덱스(train 이면 927만 항목 파이썬 dict)를 만들지 않는다 — 메모리도
    크고 fork 워커에서 COW 가 깨진다. 검사하는 배치는 num_batches(기본 2)개뿐이라
    배치마다 numpy 로 훑는 편이 싸다.
    """
    return [int(np.nonzero(manifest.utt_ids == utt_id.encode())[0][0])
            for utt_id in utt_ids]


def _print_batch_detail(batch, batch_index, sp_processor):
    """배치 하나의 모양/값/마스크를 찍는다. utt[0] 은 자르지 않는다."""
    feat, word, phone, feat_mask, word_mask, phone_mask, utt_ids = batch
    print("    " + "-" * 70)
    print("    [batch {}]  B={}  T={}  Lw={}  Lp={}".format(
        batch_index, feat.shape[0], feat.shape[1], word.shape[1], phone.shape[1]))
    print("    dtype/device: feat {} {} | word {} | mask {}".format(
        feat.dtype, feat.device, word.dtype, feat_mask.dtype))
    print("    feat 값     : min/mean/max = {:.1f}/{:.1f}/{:.1f} dB".format(
        feat.min().item(), feat.mean().item(), feat.max().item()))
    print("    실제 길이   : feat {} | word {} | phone {}".format(
        feat_mask.sum(1)[:5].int().tolist(), word_mask.sum(1)[:5].int().tolist(),
        phone_mask.sum(1)[:5].int().tolist()))

    real_word_len = int(word_mask[0].sum())
    real_phone_len = int(phone_mask[0].sum())
    print("    utt[0]      : {}".format(utt_ids[0]))
    print("      word_ids  : {}   (실제 {} + 패딩 {})".format(
        word[0].tolist(), real_word_len, word.shape[1] - real_word_len))
    print("      -> text   : {!r}".format(
        decode_words(sp_processor, word[0][:real_word_len].tolist())))
    print("      phone_ids : {}   (실제 {} + 패딩 {})".format(
        phone[0].tolist(), real_phone_len, phone.shape[1] - real_phone_len))
    print("      -> phone  : {}".format(
        decode_phones(phone[0][:real_phone_len].tolist())))
    print("    패딩효율    : feat {:.1f}%  word {:.1f}%  phone {:.1f}%".format(
        100 * feat_mask.mean().item(), 100 * word_mask.mean().item(),
        100 * phone_mask.mean().item()))


def _check_batch_against_catalog(batch, loader):
    """마스크가 카탈로그 길이와 맞고 패딩 자리가 0 인지."""
    feat, word, phone, feat_mask, word_mask, phone_mask, utt_ids = batch
    manifest = loader.manifest
    catalog_indices = find_catalog_indices(manifest, utt_ids)
    expected_feat_lens = torch.tensor(
        [float(manifest.feat_len[utt_index]) for utt_index in catalog_indices])
    check(torch.equal(feat_mask.sum(1).cpu(), expected_feat_lens),
          "feat_mask 합 == 카탈로그 feat_len")
    check(int(feat_mask.sum(1).max()) == feat.shape[1],
          "가장 긴 발화가 패딩 없이 T 를 채움")
    rows_with_word_padding = (word_mask[:, -1] == 0)
    check(bool((word[rows_with_word_padding][:, -1] == 0).all())
          if rows_with_word_padding.any() else True,
          "패딩 위치의 word id 는 0(<pad>)")
    check(len(utt_ids) == feat.shape[0], "utt_ids 개수 == 배치 크기")
    check(torch.isfinite(feat).all().item(), "배치 feat 에 NaN/Inf 없음")


def step_1_iterate_batches(loader, args, sp_processor):
    """배치를 순회하며 앞 몇 개는 자세히 찍고 전부에 대해 패딩효율을 집계한다."""
    print_step(1, "배치 순회 (상세 {}개 / 최대 {}개)".format(args.num_batches, args.limit))
    total_real_frames = total_cells = num_seen = 0
    for batch_index, batch in enumerate(
            loader.iter_prefetched_batches(num_workers=args.workers)):
        total_real_frames += float(batch[3].sum())
        total_cells += batch[3].numel()
        num_seen += 1
        if batch_index < args.num_batches:
            _print_batch_detail(batch, batch_index, sp_processor)
            _check_batch_against_catalog(batch, loader)
        if batch_index + 1 >= args.limit:
            break

    print_step(2, "전체 패딩효율")
    print("    {}배치  feat {:.1f}%  (실제 {:,} / 패딩포함 {:,} 프레임)".format(
        num_seen, 100 * total_real_frames / max(total_cells, 1),
        int(total_real_frames), int(total_cells)))


def run(loader, args, sp_processor):
    print_title("batch — 배치 텐서 값 / 마스크 / 패딩")
    step_1_iterate_batches(loader, args, sp_processor)


def test_batch():
    """pytest / VSCode Testing 패널 진입점."""
    run_as_pytest(run)


if __name__ == "__main__":
    sys.exit(run_standalone(run, __doc__))
