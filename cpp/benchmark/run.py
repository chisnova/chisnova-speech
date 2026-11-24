#!/usr/bin/env python3
"""Measure LibriSpeech word error rate, latency and throughput with libasr_api.so."""
import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CPP = HERE.parent
SPLITS = ("clean", "other")


def read_table(path):
    entries = {}
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if not line.strip():
                continue
            key, sep, value = line.strip().partition("\t")
            if not sep or not key or not value.strip() or key in entries:
                raise ValueError(f"{path}:{i}: expected a unique ID and a tab-separated value")
            entries[key] = value.strip()
    if not entries:
        raise ValueError(f"Empty list: {path}")
    return entries


def load_utterances(split, limit):
    folder = HERE / f"libri_test_{split}"
    paths = read_table(folder / "wav.list")
    labels = read_table(folder / "lab.list")
    if paths.keys() != labels.keys():
        raise ValueError(f"{folder}: IDs differ between wav.list and lab.list")

    entries = []
    for key, value in list(paths.items())[:limit]:
        wav = folder / value
        if not wav.is_file():
            raise FileNotFoundError(f"{folder}/wav.list: {key}: missing WAV: {wav}")
        entries.append((key, wav, labels[key]))
    return entries


def words(text):
    # Match model/util/asr_evaluator.py, including its handling of apostrophes.
    return text.lower().replace("-", " ").split()


def word_errors(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j-1] + (ref != hyp)))
        previous = current
    return previous[-1]


def native_record(proc):
    tail = deque(maxlen=5)
    for line in proc.stdout:
        if line.startswith("BENCH\t"):
            return json.loads(line.split("\t", 1)[1])
        tail.append(line.rstrip())
    raise RuntimeError(f"api_tester exited ({proc.wait()}): " + "\n".join(tail))


@contextmanager
def open_engine(binary, model, mode):
    command = [str(binary), "benchmark", str(model), mode]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8") as proc:
        try:
            yield proc
            proc.stdin.close()
            proc.stdout.read()
            if proc.wait() != 0:
                raise RuntimeError(f"api_tester exited ({proc.returncode})")
        finally:
            if proc.poll() is None:
                proc.kill()


def recognize_and_score(proc, split, key, wav, reference):
    ref_words = words(reference)
    if not ref_words:
        raise ValueError(f"Empty normalized reference: {key}")

    proc.stdin.write(f"{wav}\n")
    proc.stdin.flush()
    result = native_record(proc)
    errors = word_errors(ref_words, words(result["hypothesis"]))
    return {
        **result,
        "split": split,
        "id": key,
        "path": str(wav),
        "reference": reference,
        "ref_words": len(ref_words),
        "errors": errors,
        "ctc_errors": word_errors(ref_words, words(result["ctc"])),
        "wer_pct": 100 * errors / len(ref_words),
        "xrt": result["audio_s"] / result["elapsed_s"],
        "rtf": result["elapsed_s"] / result["audio_s"],
    }


def evaluate_split(proc, split, entries, results_file):
    rows = []
    for index, (key, wav, reference) in enumerate(entries, 1):
        row = recognize_and_score(proc, split, key, wav, reference)
        rows.append(row)
        results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        results_file.flush()
        if index % 100 == 0:
            print(f"{split}: {index}/{len(entries)}", file=sys.stderr)
    return rows


def summarize(rows):
    audio = sum(row["audio_s"] for row in rows)
    elapsed = sum(row["elapsed_s"] for row in rows)
    ref_words = sum(row["ref_words"] for row in rows)
    latencies = sorted(row["final_latency_ms"] for row in rows)
    return {
        "utterances": len(rows),
        "audio_s": audio,
        "elapsed_s": elapsed,
        "ref_words": ref_words,
        "wer_pct": 100 * sum(row["errors"] for row in rows) / ref_words,
        "ctc_wer_pct": 100 * sum(row["ctc_errors"] for row in rows) / ref_words,
        "final_latency_mean_ms": statistics.mean(latencies),
        "final_latency_p50_ms": statistics.median(latencies),
        "final_latency_p95_ms": latencies[math.ceil(.95 * len(rows)) - 1],
        "xrt": audio / elapsed,
        "rtf": elapsed / audio,
    }


def evaluate(args, output):
    splits = SPLITS if args.split == "both" else (args.split,)
    datasets = {split: load_utterances(split, args.limit) for split in splits}
    model = args.model.resolve()
    binary = args.build_dir.resolve() / "api_tester"
    split_summaries = {}
    all_rows = []

    with open_engine(binary, model, args.mode) as proc:
        model_load_s = native_record(proc)["model_load_s"]
        with (output / "utterances.jsonl").open("x", encoding="utf-8") as results_file:
            for split, entries in datasets.items():
                rows = evaluate_split(proc, split, entries, results_file)
                split_summaries[split] = summarize(rows)
                all_rows.extend(rows)

    return {
        "mode": args.mode,
        "model": str(model),
        "model_load_s": model_load_s,
        "splits": split_summaries,
        "overall": summarize(all_rows),
    }


def print_summary(name, metrics):
    print(f"{name}: n={metrics['utterances']}  WER={metrics['wer_pct']:.2f}%  "
          f"CTC WER={metrics['ctc_wer_pct']:.2f}%  "
          f"final latency p50/p95={metrics['final_latency_p50_ms']:.1f}/{metrics['final_latency_p95_ms']:.1f} ms  "
          f"xRT={metrics['xrt']:.2f}x  RTF={metrics['rtf']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CPP.parent / "arifact/stream_conformer_cpp_export")
    parser.add_argument("--build-dir", type=Path, default=CPP / "build")
    parser.add_argument("--split", choices=(*SPLITS, "both"), default="both")
    parser.add_argument("--mode", choices=("offline", "realtime"), default="offline")
    parser.add_argument("--limit", type=int, help="first N utterances per selected split")
    parser.add_argument("--output", type=Path, help="new or empty result directory")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not (args.model / "model.json").is_file():
        parser.error(f"model.json missing in {args.model}")
    if not (args.build_dir / "api_tester").is_file():
        parser.error("build api_tester first: cmake --build cpp/build --target api_tester")
    output = args.output or HERE / "results" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory must be empty: {output}")
    print(f"Output: {output.resolve()}", flush=True)
    report = evaluate(args, output)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for split, metrics in report["splits"].items():
        print_summary(split, metrics)
    print_summary("overall", report["overall"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as e:
        sys.exit(str(e))
