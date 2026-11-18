#!/usr/bin/env python3
"""augment 케이스 — FeatureTransform 들의 적용 전/후와 패딩 보존."""
import sys

from common import check, print_step, print_title, run_as_pytest, run_standalone

import torch

from data.loader import Cmvn, Compose, GaussianNoise, SpecAugment, SpecSubstitute


def _print_stat(tag, tensor):
    print("    {:12s}: min/mean/max = {:8.2f}/{:8.2f}/{:8.2f}".format(
        tag, float(tensor.min()), float(tensor.mean()), float(tensor.max())))


def step_1_take_batch(loader):
    """증강 전 원본 배치 하나 (디바이스 전송/변환 전 단계에서 꺼낸다)."""
    print_step(1, "원본 배치")
    batch = loader._make_batch_dataset()[int(loader.batch_order[0])]
    original_feat, feat_mask = batch[0].clone(), batch[3].clone()
    padding_mask = (feat_mask == 0).unsqueeze(-1)
    print("    배치 {} 패딩 셀 {:.1f}%".format(
        tuple(original_feat.shape), 100 * float(padding_mask.float().mean())))
    _print_stat("원본", original_feat)
    return original_feat, feat_mask, padding_mask


def step_2_check_cmvn(original_feat, feat_mask, padding_mask):
    """유효 프레임 평균 0 / 표준편차 1, 패딩은 0."""
    print_step(2, "Cmvn")
    normalized = Cmvn()(original_feat.clone())
    valid_mean = float((normalized * feat_mask.unsqueeze(-1)).sum()
                       / (feat_mask.sum() * normalized.shape[2]))
    _print_stat("Cmvn 후", normalized)
    print("    {:12s}: 유효 프레임 평균 {:.4f}".format("", valid_mean))
    check(abs(valid_mean) < 1e-3, "Cmvn 후 유효 프레임 평균 ~ 0")
    check(bool((normalized[padding_mask.expand_as(normalized)] == 0).all()),
          "Cmvn 이 패딩을 0 으로 유지")
    return normalized


def step_3_check_spec_augment(normalized, feat_mask, padding_mask):
    """마스킹이 걸리고, 발화별로 다르고, 패딩을 안 건드리는지."""
    print_step(3, "SpecAugment")
    spec_augment = SpecAugment(prob=1.0, mask_value="zero")
    masked = spec_augment(normalized.clone())
    changed = (masked != normalized)
    print("    {}".format(spec_augment))
    _print_stat("SpecAug 후", masked)
    print("    {:12s}: 마스킹된 셀 {:.1f}%".format("", 100 * float(changed.float().mean())))
    check(bool(changed.any()), "마스킹된 영역이 존재")
    check(masked.shape == normalized.shape, "마스킹 후에도 모양 유지")
    check(bool((masked[padding_mask.expand_as(masked)] == 0).all()),
          "마스킹이 패딩을 오염시키지 않음")
    if normalized.shape[0] > 1:
        _check_per_utterance_masks(normalized, feat_mask)


def _check_per_utterance_masks(normalized, feat_mask):
    """주파수/시간 마스크를 따로 떼어 발화별로 다른지, 전멸하지 않는지 본다."""
    freq_only = SpecAugment(num_freq=2, num_time=0, freq_param=27,
                            mask_value="zero", prob=1.0)
    freq_masked = freq_only(torch.ones_like(normalized))
    bands = [tuple((freq_masked[b, 0] == 0).nonzero().flatten().tolist())
             for b in range(freq_masked.shape[0])]
    print("    {:12s}: 발화별 주파수 마스크 구간 {}".format(
        "", [(band[0], band[-1]) if band else None for band in bands[:5]]))
    check(len(set(bands)) > max(1, len(bands) // 2),
          "발화마다 마스크 위치가 다름 (배치 공유 아님)")

    utt_lengths = feat_mask.sum(1).long()
    time_only = SpecAugment(num_freq=0, num_time=2, time_param=100,
                            mask_value="zero", prob=1.0)
    time_masked = time_only(torch.ones_like(normalized) * feat_mask.unsqueeze(-1))
    num_killed = torch.tensor([int((time_masked[b, :utt_lengths[b]] == 0).all(-1).sum())
                               for b in range(time_masked.shape[0])])
    check(bool((num_killed < utt_lengths).all()), "발화가 시간 마스크로 전멸하지 않음")


def step_4_check_gaussian_noise(normalized, padding_mask):
    """노이즈가 값을 바꾸되 패딩에는 새지 않는지."""
    print_step(4, "GaussianNoise")
    noised = GaussianNoise(std=0.05, prob=1.0)(normalized.clone())
    _print_stat("Noise 후", noised)
    print("    {:12s}: 평균 변화량 {:.4f}".format(
        "", float((noised - normalized).abs().mean())))
    check(not torch.equal(noised, normalized), "노이즈가 실제로 값을 바꿈")
    check(bool((noised[padding_mask.expand_as(noised)] == 0).all()),
          "노이즈가 패딩에 새지 않음")


def step_5_check_spec_substitute(normalized):
    """시간축 치환이 모양을 유지하며 값을 바꾸는지."""
    print_step(5, "SpecSubstitute")
    substituted = SpecSubstitute(prob=1.0)(normalized.clone())
    _print_stat("SpecSub 후", substituted)
    check(substituted.shape == normalized.shape, "SpecSubstitute 가 모양 유지")
    check(not torch.equal(substituted, normalized), "SpecSubstitute 가 실제로 값을 바꿈")


def step_6_check_reproducibility(normalized):
    """torch 전역 시드로 재현되는지 / prob=0 게이팅."""
    print_step(6, "재현성 / prob 게이팅")

    def augment_with_seed(seed):
        torch.manual_seed(seed)
        return SpecAugment(prob=1.0)(normalized.clone())

    check(torch.equal(augment_with_seed(7), augment_with_seed(7)),
          "같은 torch 시드 -> 같은 증강")
    check(not torch.equal(augment_with_seed(7), augment_with_seed(8)),
          "다른 시드 -> 다른 증강")
    check(torch.equal(SpecAugment(prob=0.0)(normalized.clone()), normalized),
          "prob=0 이면 원본 그대로")


def step_7_check_compose(loader, original_feat, padding_mask):
    """파이프라인 전체를 태워도 모양/패딩이 유지되는지."""
    print_step(7, "Compose 파이프라인")
    pipeline = Compose([Cmvn(), SpecAugment(mask_value="zero"), GaussianNoise()])
    pipeline_out = pipeline(original_feat.clone())
    print("    {}".format(repr(pipeline).replace("\n", "\n    ")))
    _print_stat("Compose 후", pipeline_out)
    check(pipeline_out.shape == original_feat.shape, "Compose 후에도 모양 유지")
    check(bool((pipeline_out[padding_mask.expand_as(pipeline_out)] == 0).all()),
          "Compose 전체가 패딩을 0 으로 유지")
    check(loader.transform is None if not loader.isTrain
          else loader.transform is not None,
          "isTrain={} 에 맞게 로더 변환 {}".format(
              loader.isTrain, "off" if not loader.isTrain else "on"))


def run(loader, args, sp_processor):
    print_title("augment — FeatureTransform 들의 적용 전/후")
    original_feat, feat_mask, padding_mask = step_1_take_batch(loader)
    normalized = step_2_check_cmvn(original_feat, feat_mask, padding_mask)
    step_3_check_spec_augment(normalized, feat_mask, padding_mask)
    step_4_check_gaussian_noise(normalized, padding_mask)
    step_5_check_spec_substitute(normalized)
    step_6_check_reproducibility(normalized)
    step_7_check_compose(loader, original_feat, padding_mask)


def test_augment():
    """pytest / VSCode Testing 패널 진입점."""
    run_as_pytest(run)


if __name__ == "__main__":
    sys.exit(run_standalone(run, __doc__))
