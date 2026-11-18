"""학습 중 WRR 평가 스케줄러.

    PYTHONPATH=$PWD python3 util/eval_scheduler.py --model_path stream_conformer

학습이 **새 스텝을 찍을 때마다**(트레이너가 300 배치마다 로그를 남기고 체크포인트를
저장한다) **코퍼스마다 200 발화씩** 뽑아 **CPU + CTC2 greedy** 로 평가한다.
코퍼스 구성비대로 뽑으면(giga 58% / libri-clean 8%) 작은 코퍼스가 수십 개밖에 안 잡혀
그 값은 표본오차에 묻힌다. 같은 수를 뽑아야 코퍼스끼리 같은 정밀도로 비교된다. GPU 는 학습이 꽉 쓰고
있으므로 끼어들지 않고, CTC2 만 보므로 500 발화가 20초 남짓이다. 목적은 정확한
절대값이 아니라 학습이 나아지고 있는지의 **경향**이다.

시간이 아니라 스텝을 트리거로 삼는 이유: 같은 스텝을 두 번 재면 x 축이 겹쳐 곡선이
수직으로 튀고, 반대로 학습이 느려진 구간에서는 같은 가중치를 몇 번씩 재게 된다.
스텝이 바뀔 때만 재면 측정 한 점이 학습 300 배치와 정확히 대응한다.

에폭 단위 정밀 평가(전체 + attention beam)는 train_main.py 가 학습 프로세스 안에서
직접 한다 — 이미 GPU 에 올라간 모델을 그대로 쓰므로 체크포인트를 다시 읽을 필요가 없다.

결과는 <model_path>/wer_history.jsonl 에 한 줄씩 쌓인다.
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import time
from datetime import datetime

import torch

from util.asr_evaluator import AsrEvaluator


class EvalScheduler:
    """학습 중 주기적으로 표본 평가를 돌린다 (CPU, CTC2)."""

    LATEST_CHECKPOINT = "asr_model.pt"
    HISTORY_NAME = "wer_history.jsonl"
    STEP_PATTERN = re.compile(r"step\s*:\s*(\d+)\s*,\s*train loss")
    CHECKPOINT_READ_RETRIES = 3

    def __init__(self, model_path, record_path=None, label_list_path=None,
                 sample_size=500, per_corpus_size=200, min_interval_seconds=0,
                 sample_device="cpu", sample_batch_size=8, poll_seconds=20, cpu_threads=2,
                 heads=("ctc2",), primary_head="ctc2", eval_chunk_size=None):
        self.model_path = model_path
        self.record_path = record_path or AsrEvaluator.DEFAULT_RECORD
        self.label_list_path = label_list_path or AsrEvaluator.DEFAULT_LABEL_LIST
        self.sample_size = int(sample_size)
        self.per_corpus_size = int(per_corpus_size)
        # 스텝이 바뀌어도 이 시간 안에는 다시 재지 않는다 (0 이면 제한 없음).
        # 학습이 아주 빨라 로그가 촘촘해질 때의 안전장치다.
        self.min_interval_seconds = int(min_interval_seconds)
        self.sample_device = sample_device
        self.sample_batch_size = int(sample_batch_size)
        self.poll_seconds = int(poll_seconds)
        # 1패스 학습 중에는 ctc1 이 목적 지표다 — 기록의 대표 wrr 이 그 헤드가 되게 한다
        self.heads = tuple(heads)
        self.primary_head = primary_head
        # 스트림 인코더 평가에 쓸 chunk mask 크기(인코더 프레임, 1프레임=40ms).
        # None 이면 AsrEvaluator 기본(64=2.56초). 500ms 조건을 재려면 12~13 을 준다.
        self.eval_chunk_size = int(eval_chunk_size) if eval_chunk_size else None
        # 학습 로더가 워커 4개로 CPU 를 쓰고 있다 — 평가가 코어를 다 잡으면 학습이 느려진다
        if sample_device == "cpu" and cpu_threads > 0:
            torch.set_num_threads(int(cpu_threads))
        self.history_path = os.path.join(model_path, self.HISTORY_NAME)
        self.last_sample_at = 0.0

    def read_current_step(self):
        """train.log 의 마지막 스텝 — WRR 을 학습 진행도에 맞춰 그리기 위한 x 값."""
        log_path = os.path.join(self.model_path, "train.log")
        if not os.path.isfile(log_path):
            return None
        last_step = None
        mixed_epoch_batches = None
        completed_epochs = 0
        with open(log_path, errors="replace") as log_file:
            for line in log_file:
                mixed = re.search(r'\[mix\].*batches per epoch: (\d+)', line)
                if mixed:
                    mixed_epoch_batches = int(mixed.group(1))
                completed = re.search(r'Epoch(\d+) Train Loss:', line)
                if completed and mixed_epoch_batches:
                    completed_epochs = int(completed.group(1))
                    last_step = completed_epochs * mixed_epoch_batches
                match = self.STEP_PATTERN.search(line)
                if match:
                    last_step = (completed_epochs * (mixed_epoch_batches or 0)
                                 + int(match.group(1)))
        return last_step

    def snapshot_checkpoint(self):
        """평가용으로 체크포인트를 복사해 둔다.

        학습이 300 배치마다 같은 파일을 덮어쓰므로 그대로 읽으면 쓰는 중간에 걸려
        깨진 파일을 읽을 수 있다. 복사 후 한 번 열어보고, 실패하면 잠깐 뒤 재시도한다.
        """
        source_path = os.path.join(self.model_path, self.LATEST_CHECKPOINT)
        snapshot_dir = os.path.join(self.model_path, ".eval_snapshot")
        os.makedirs(snapshot_dir, exist_ok=True)
        shutil.copyfile(os.path.join(self.model_path, "config.json"),
                        os.path.join(snapshot_dir, "config.json"))
        snapshot_path = os.path.join(snapshot_dir, self.LATEST_CHECKPOINT)
        for attempt in range(self.CHECKPOINT_READ_RETRIES):
            try:
                shutil.copyfile(source_path, snapshot_path)
                torch.load(snapshot_path, map_location="cpu")
                return snapshot_dir
            except Exception as read_error:
                print("[eval] 체크포인트 읽기 재시도 {}/{}: {}".format(
                    attempt + 1, self.CHECKPOINT_READ_RETRIES, read_error))
                time.sleep(20)
        raise RuntimeError("체크포인트를 읽지 못했습니다: {}".format(source_path))

    def run_evaluation(self, scope, device, sample_size, batch_size, checkpoint_dir=None,
                       step=None):
        """평가 한 번 — 결과를 history 에 덧붙이고 요약을 돌려준다."""
        model_dir = checkpoint_dir or self.snapshot_checkpoint()
        started_at = datetime.now().replace(microsecond=0)
        print("[eval] {} 평가 시작 ({}, {}, batch {})".format(
            scope, device,
            "코퍼스당 {} 발화".format(self.per_corpus_size) if self.per_corpus_size
            else "{} 발화".format(sample_size or "전체"), batch_size))

        evaluator_kwargs = {}
        if self.eval_chunk_size:
            evaluator_kwargs["chunk_size"] = self.eval_chunk_size
        evaluator = AsrEvaluator.from_model_path(model_dir, device=device, **evaluator_kwargs)
        metrics, _, _ = evaluator.evaluate_record(
            record_path=self.record_path, label_list_path=self.label_list_path,
            sample_size=sample_size, batch_size=batch_size, progress_every=0,
            per_corpus_size=self.per_corpus_size,
            heads=self.heads, primary_head=self.primary_head)

        # 트리거가 된 스텝을 그대로 기록한다 — 평가 도는 사이 학습이 더 나갔더라도
        # 이 측정이 대응하는 가중치는 트리거 시점의 것이다.
        # 어떤 청크 조건에서 잰 수치인지 이력에 남긴다 — 500ms 파인튜닝 전후의
        # 기록(기본 64 청크)과 섞여도 구분할 수 있어야 한다
        record = dict(metrics, eval_chunk=self.eval_chunk_size or 64,
                      at=started_at.isoformat(), scope=scope,
                      step=self.read_current_step() if step is None else step)
        AsrEvaluator.append_history(self.history_path, record)
        head_summary = " / ".join("{} {:.2f}%".format(head, head_metrics["wrr"])
                                  for head, head_metrics in record["heads"].items())
        corpus_summary = " · ".join("{} {:.1f}%".format(name, corpus_metrics["wrr"])
                                    for name, corpus_metrics in record.get("corpora", {}).items())
        print("[eval] {} 완료 — {} | 발화 {:,} | {:.1f}s".format(
            scope, head_summary, record["num_utts"], record["elapsed_seconds"]))
        if corpus_summary:
            print("[eval]   코퍼스별 — {}".format(corpus_summary))
        del evaluator
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return record

    def run_forever(self):
        print("[eval] 스케줄러 시작 — 새 체크포인트 평가 ({}, {}), 청크 {}, {}초 간격 폴링".format(
            self.sample_device, ','.join(self.heads), self.eval_chunk_size or 64, self.poll_seconds))
        print("[eval] 기록: {}".format(self.history_path))
        last_evaluated_checkpoint = None
        while True:
            try:
                current_step = self.read_current_step()
                checkpoint_ready = os.path.isfile(
                    os.path.join(self.model_path, self.LATEST_CHECKPOINT))
                checkpoint_key = None
                if checkpoint_ready:
                    stat = os.stat(os.path.join(self.model_path, self.LATEST_CHECKPOINT))
                    checkpoint_key = (stat.st_mtime_ns, stat.st_size)
                throttled = (self.min_interval_seconds > 0
                             and time.time() - self.last_sample_at < self.min_interval_seconds)
                if (checkpoint_ready and current_step is not None
                        and checkpoint_key != last_evaluated_checkpoint and not throttled):
                    self.run_evaluation("sample", self.sample_device, self.sample_size,
                                        self.sample_batch_size, step=current_step)
                    last_evaluated_checkpoint = checkpoint_key
                    self.last_sample_at = time.time()
            except KeyboardInterrupt:
                print("\n[eval] 종료")
                return
            except Exception as scheduler_error:
                print("[eval][WARN] 평가 실패 — 다음 스텝에 다시 시도: {}".format(scheduler_error))
                self.last_sample_at = time.time()
            time.sleep(self.poll_seconds)


def parseArguments():
    parser = argparse.ArgumentParser(description="학습 중 WRR 평가 스케줄러")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--record', type=str, default=None)
    parser.add_argument('--lab_list', type=str, default=None)
    parser.add_argument('--sample', type=int, default=500,
                        help="per_corpus 를 0 으로 둘 때만 쓰는 전체 표본 수")
    parser.add_argument('--per_corpus', type=int, default=200,
                        help="코퍼스마다 뽑을 발화 수 (0 이면 --sample 비율 표본)")
    parser.add_argument('--min_interval', type=int, default=0,
                        help="스텝이 바뀌어도 이 시간 안에는 다시 재지 않는다(초). 0 이면 제한 없음")
    parser.add_argument('--poll', type=int, default=20, help="새 스텝을 확인하는 간격(초)")
    parser.add_argument('--cpu_threads', type=int, default=2,
                        help="평가가 쓸 CPU 스레드 (학습 로더와 코어를 나눠 쓴다)")
    parser.add_argument('--sample_device', type=str, default="cpu")
    parser.add_argument('--sample_batch_size', type=int, default=8)
    parser.add_argument('--heads', type=str, default="ctc1,ctc2",
                        help="잴 CTC 헤드 (쉼표 구분)")
    parser.add_argument('--primary', type=str, default="ctc2",
                        help="기록의 대표 wrr 로 쓸 헤드 (1패스 학습 중엔 ctc1)")
    parser.add_argument('--eval_chunk', type=int, default=0,
                        help="스트림 인코더 평가 chunk mask 크기(인코더 프레임, 1프레임=40ms). "
                             "0 이면 평가기 기본(64=2.56초). 500ms 조건은 12")
    parser.add_argument('--once', action='store_true', help="스케줄 없이 한 번만 돌린다")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArguments()
    args.model_path = os.path.realpath(args.model_path)
    # One writer and one snapshot directory per model, including manual starts.
    lock_file = open(os.path.join(args.model_path, '.eval_scheduler.lock'), 'a')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('[eval] already running for this model; exiting')
        raise SystemExit(0)
    with open(os.path.join(args.model_path, 'eval_scheduler.pid'), 'w') as pid_file:
        pid_file.write(str(os.getpid()) + '\n')
    scheduler = EvalScheduler(args.model_path, record_path=args.record,
                              label_list_path=args.lab_list, sample_size=args.sample,
                              per_corpus_size=args.per_corpus,
                              min_interval_seconds=args.min_interval,
                              sample_device=args.sample_device,
                              sample_batch_size=args.sample_batch_size,
                              poll_seconds=args.poll, cpu_threads=args.cpu_threads,
                              heads=tuple(args.heads.split(",")), primary_head=args.primary,
                              eval_chunk_size=args.eval_chunk)
    if args.once:
        scheduler.run_evaluation("sample", args.sample_device, args.sample,
                                 args.sample_batch_size)
    else:
        scheduler.run_forever()
