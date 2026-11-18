#!/usr/bin/env python3
"""worker 케이스 — num_workers=0 과 N 의 결과가 같은지 + 처리 속도."""
import sys
import time

from common import check, print_step, print_title, run_as_pytest, run_standalone


def step_1_iterate(loader, num_workers, num_batches):
    """배치를 순회하며 발화 구성 / 모양 / 프레임 수를 모은다."""
    started_at = time.time()
    utt_id_tuples, batch_shapes, num_real_frames = [], [], 0
    for batch_index, batch in enumerate(
            loader.iter_prefetched_batches(num_workers=num_workers)):
        utt_id_tuples.append(tuple(batch[6]))
        batch_shapes.append(tuple(batch[0].shape))
        num_real_frames += int(batch[3].sum())
        if batch_index + 1 >= num_batches:
            break
    elapsed = time.time() - started_at
    print("    num_workers={:<2d}: {:.2f}s  ({:,} 프레임, {:,.0f} 프레임/s)".format(
        num_workers, elapsed, num_real_frames, num_real_frames / max(elapsed, 1e-9)))
    return utt_id_tuples, batch_shapes, num_real_frames


def step_2_compare(single, multi):
    """워커 유무가 결과를 바꾸면 안 된다 — 바뀌면 학습 재현성이 깨진다."""
    print_step(3, "두 경로 결과 비교")
    check(single[0] == multi[0], "두 경로의 발화 구성/순서 동일")
    check(single[1] == multi[1], "두 경로의 배치 모양 동일")
    check(single[2] == multi[2], "두 경로의 총 실제 프레임 수 동일")


def run(loader, args, sp_processor):
    num_workers = max(args.workers, 1)
    print_title("worker — num_workers 0 vs {} 결과 일치 / 속도".format(num_workers))
    num_batches = min(args.limit, loader.num_batches)

    print_step(1, "워커 없이 {}배치".format(num_batches))
    single = step_1_iterate(loader, 0, num_batches)
    print_step(2, "워커 {}개로 {}배치".format(num_workers, num_batches))
    multi = step_1_iterate(loader, num_workers, num_batches)
    step_2_compare(single, multi)


def test_worker():
    """pytest / VSCode Testing 패널 진입점."""
    run_as_pytest(run, limit=8)


if __name__ == "__main__":
    sys.exit(run_standalone(run, __doc__))
