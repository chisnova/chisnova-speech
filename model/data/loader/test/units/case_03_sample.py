#!/usr/bin/env python3
"""sample 케이스 — 무작위 표본을 읽어 통계로 이상을 탐지한다.

record 케이스가 2개를 자세히 보는 것이라면 이쪽은 많이 훑는 쪽이다.
개별 값을 찍지 않고 전부 집계해서 "몇 건 어긋났는가"로만 판정한다.
"""
import sys
import time

from common import PHONES, check, print_step, print_title, run_as_pytest, run_standalone

import numpy as np

PROBLEM_NAMES = ("읽기 실패", "feat 모양 != 카탈로그", "word/phone 길이 != 카탈로그",
                 "SOS/EOS 누락", "시퀀스 안에 pad(0)", "phone id 범위 밖",
                 "feat 에 NaN/Inf", "무음 의심(std~0)")


def step_1_pick_sample(loader, sample_ratio, seed):
    """전체에서 무작위로 표본 인덱스를 뽑는다 (정렬해 두면 읽기가 조금 빠르다)."""
    num_total = len(loader.manifest)
    num_sample = max(1, int(round(num_total * sample_ratio)))
    print_step(1, "표본 추출 — {:.1%} ({:,}/{:,} 발화, seed {})".format(
        sample_ratio, num_sample, num_total, seed))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_total, size=num_sample, replace=False))


def step_2_read_and_collect(loader, sampled):
    """표본을 읽으며 발화별 스칼라와 위반 목록을 모은다."""
    print_step(2, "표본 읽기 + 집계")
    manifest = loader.manifest
    num_sample = len(sampled)
    stats = {name: np.full(num_sample, np.nan, np.float32) for name in
             ("feat_min", "feat_max", "feat_mean", "feat_std",
              "frames_per_token", "phones_per_token")}
    problems = {name: [] for name in PROBLEM_NAMES}
    num_read = 0
    started_at = time.time()
    report_every = max(1, num_sample // 10)

    for row, utt_index in enumerate(sampled):
        utt_id_bytes = manifest.utt_ids[utt_index]
        try:
            record = loader.conn.read_record(utt_id_bytes, manifest.locators[utt_index])
        except Exception as error:
            problems["읽기 실패"].append("{} ({})".format(utt_id_bytes, error))
            continue
        num_read += 1
        _inspect_record(loader, manifest, utt_index, record, row, stats, problems)

        if (row + 1) % report_every == 0:
            done_ratio = (row + 1) / num_sample
            elapsed = time.time() - started_at
            print("      ... {:>4.0%}  {:,}/{:,}  {:.0f}s 경과 / 남은 {:.0f}s 예상".format(
                done_ratio, row + 1, num_sample, elapsed,
                elapsed * (1 - done_ratio) / max(done_ratio, 1e-9)))

    elapsed = time.time() - started_at
    print("    {:,}개 읽음  {:.1f}s  ({:,.0f} 레코드/s)".format(
        num_read, elapsed, num_read / max(elapsed, 1e-9)))
    return num_read, stats, problems


def _inspect_record(loader, manifest, utt_index, record, row, stats, problems):
    """레코드 하나를 검사해 stats/problems 에 반영한다."""
    feat_arr = np.asarray(record["feat"], dtype=np.float32)
    word_ids = np.asarray(record["word_ids"], dtype=np.int64)
    phone_ids = np.asarray(record["phone_ids"], dtype=np.int64)
    utt_id = manifest.utt_ids[utt_index].decode("utf-8", "replace")

    if feat_arr.shape != (manifest.feat_len[utt_index], loader.feat_dim):
        problems["feat 모양 != 카탈로그"].append("{} {} != {}".format(
            utt_id, feat_arr.shape,
            (int(manifest.feat_len[utt_index]), loader.feat_dim)))
    if (word_ids.shape[0] != manifest.word_len[utt_index]
            or phone_ids.shape[0] != manifest.phone_len[utt_index]):
        problems["word/phone 길이 != 카탈로그"].append(utt_id)
    if not (word_ids[0] == 2 and word_ids[-1] == 3
            and phone_ids[0] == 2 and phone_ids[-1] == 3):
        problems["SOS/EOS 누락"].append(utt_id)
    if (word_ids[1:-1] == 0).any() or (phone_ids[1:-1] == 0).any():
        problems["시퀀스 안에 pad(0)"].append(utt_id)
    if phone_ids.max() >= len(PHONES) or phone_ids.min() < 0:
        problems["phone id 범위 밖"].append("{} max={}".format(utt_id, phone_ids.max()))
    if not np.isfinite(feat_arr).all():
        problems["feat 에 NaN/Inf"].append(utt_id)

    stats["feat_min"][row] = feat_arr.min()
    stats["feat_max"][row] = feat_arr.max()
    stats["feat_mean"][row] = feat_arr.mean()
    stats["feat_std"][row] = feat_arr.std()
    if stats["feat_std"][row] < 1e-3:            # 전 프레임이 같은 값 = 무음/손상
        problems["무음 의심(std~0)"].append("{} std={:.4f}".format(
            utt_id, stats["feat_std"][row]))
    # SOS/EOS 를 뺀 실토큰 기준 — 정렬이 깨진 발화를 잡는 지표
    num_tokens = max(word_ids.shape[0] - 2, 1)
    stats["frames_per_token"][row] = feat_arr.shape[0] / num_tokens
    stats["phones_per_token"][row] = (phone_ids.shape[0] - 2) / num_tokens


def step_3_print_distributions(stats):
    """발화별 통계의 분포 — 평균 하나로는 이상치가 묻힌다."""
    print_step(3, "분포")

    def describe(name, arr, unit=""):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            print("    {:16s}: (표본 없음)".format(name))
            return
        print("    {:16s}: min {:8.2f} | p1 {:8.2f} | p50 {:8.2f} | p99 {:8.2f} | "
              "max {:8.2f} {}".format(
                  name, arr.min(), np.percentile(arr, 1), np.percentile(arr, 50),
                  np.percentile(arr, 99), arr.max(), unit))

    print("    feat 값 (발화별 통계의 분포)")
    describe("발화별 min", stats["feat_min"], "dB")
    describe("발화별 mean", stats["feat_mean"], "dB")
    describe("발화별 max", stats["feat_max"], "dB")
    describe("발화별 std", stats["feat_std"], "dB")
    print("    정렬 지표 (SOS/EOS 제외한 실토큰 기준)")
    describe("프레임/토큰", stats["frames_per_token"])
    describe("phone/토큰", stats["phones_per_token"])


def step_4_print_outliers(loader, sampled, stats, num_show=3):
    """극단값은 utt_id 까지 보여준다 — 바로 들여다볼 수 있게."""
    print_step(4, "정렬 지표 극단값")
    manifest = loader.manifest
    ratio = stats["frames_per_token"]
    if not np.isfinite(ratio).any():
        print("    (표본 없음)")
        return

    def show(rows, title):
        print("    {}".format(title))
        for row in rows:
            print("      {:6.1f}  {}  (feat_len {}, word_len {})".format(
                ratio[row], manifest.get_utt_id(sampled[row]),
                manifest.feat_len[sampled[row]], manifest.word_len[sampled[row]]))

    order = np.argsort(np.where(np.isfinite(ratio), ratio, np.nan))
    show(order[:num_show], "최소 — 토큰 대비 오디오가 짧음 (전사 과다 의심)")
    show(order[-num_show:][::-1], "최대 — 토큰 대비 오디오가 김 (전사 누락 의심)")


def step_5_check_problems(num_read, num_sample, problems, num_show=5):
    """위반 건수로만 판정하고, 걸린 것은 utt_id 를 찍는다."""
    print_step(5, "판정")
    check(num_read == num_sample,
          "표본 {:,}개 전부 읽힘 (실패 {:,})".format(num_sample, num_sample - num_read))
    for name in PROBLEM_NAMES:
        if name == "읽기 실패":
            continue
        offenders = problems[name]
        if check(not offenders, "{}: {:,}건".format(name, len(offenders))):
            continue
        for offender in offenders[:num_show]:
            print("          {}".format(offender))
        if len(offenders) > num_show:
            print("          ... 외 {:,}건".format(len(offenders) - num_show))


def run(loader, args, sp_processor):
    print_title("sample — 무작위 표본을 읽어 통계로 이상 탐지")
    sampled = step_1_pick_sample(loader, args.sample_ratio, args.sample_seed)
    num_read, stats, problems = step_2_read_and_collect(loader, sampled)
    step_3_print_distributions(stats)
    step_4_print_outliers(loader, sampled, stats)
    step_5_check_problems(num_read, len(sampled), problems)


def test_sample():
    """pytest / VSCode Testing 패널 진입점."""
    run_as_pytest(run, sample_ratio=0.02)


if __name__ == "__main__":
    sys.exit(run_standalone(run, __doc__))
