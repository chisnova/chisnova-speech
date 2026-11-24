"""Prepare model_state for the C++ runtime. Invoked by the export scripts."""
import argparse
from contextlib import contextmanager
import fcntl
from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

CPP = Path(__file__).resolve().parents[1]
MODEL = CPP.parent / "model"
SCORE_FILES = ("phone.txt", "function_words.txt", "proper_nouns.txt", "ungraded_words.txt")


@contextmanager
def stage_directory(destination):
    """Keep the previous directory until its replacement is ready."""
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "new"
        backup = Path(temporary) / "previous"
        staged.mkdir()
        yield staged
        try:
            if destination.exists():
                destination.rename(backup)
            staged.rename(destination)
        finally:
            if backup.exists() and not destination.exists():
                backup.rename(destination)


def copy_score_files(destination):
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        for name in SCORE_FILES:
            shutil.copy2(CPP / "score" / name, Path(temporary) / name)
        for name in SCORE_FILES:
            (Path(temporary) / name).replace(destination / name)


def copy_resources(destination):
    copy_score_files(destination)
    for name in ("mercury.model", "mercury.vocab", "full.dict"):
        shutil.copy2(MODEL / "data" / name, destination / name)


def prepare(action, state, checkpoint_arg):
    if action == "score":
        if not (state / "model.json").is_file():
            sys.exit("model_state/model.json이 없습니다. 먼저 export_model 타깃을 실행하세요.")
        copy_score_files(state)
        return

    checkpoint_arg = checkpoint_arg or os.environ.get("MODEL_DIR")
    checkpoint = Path(checkpoint_arg or MODEL / "stream_conformer").absolute()
    cache = checkpoint.with_name(checkpoint.name + "_cpp_export")
    reuse = action == "cached-model" and (cache / "model.json").is_file()
    if not reuse:
        if not (checkpoint / "asr_model.pt").is_file() or find_spec("torch") is None:
            if action == "model" and not checkpoint_arg:
                if (state / "model.json").is_file():
                    copy_score_files(state)
                    print(f"기존 모델 사용: {state}")
                    return
                legacy = CPP / "export" / "model_state"
                if (legacy / "model.json").is_file():
                    with stage_directory(state) as staged:
                        shutil.copytree(legacy, staged, dirs_exist_ok=True)
                        copy_score_files(staged)
                    print(f"기존 모델 복사: {legacy} -> {state}")
                    return
            sys.exit(f"모델 export에 PyTorch와 체크포인트가 필요합니다: {checkpoint}/asr_model.pt")
    # Two build directories may publish from the same checkpoint concurrently.
    parent_fd = os.open(cache.parent, os.O_RDONLY)
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        publish_model(checkpoint, cache, state, reuse)
    finally:
        os.close(parent_fd)


def publish_model(checkpoint, cache, state, reuse):
    if not reuse:
        with stage_directory(cache) as staged:
            subprocess.run(
                [sys.executable, "exporter/stream_conformer_cpp_exporter.py",
                 str(checkpoint), "--export_path", str(staged)],
                cwd=MODEL, env={**os.environ, "PYTHONPATH": str(MODEL)}, check=True,
            )
            copy_resources(staged)

    with stage_directory(state) as staged:
        shutil.copytree(cache, staged, dirs_exist_ok=True)
        if reuse:
            copy_resources(staged)
    print(f"모델 준비 완료: {state}")


def install_trees(build_tree, platforms):
    trees = [build_tree if name == "linux" else build_tree / "platforms" / name
             for name in platforms]
    for tree in trees:
        if not (tree / "cmake_install.cmake").is_file():
            sys.exit(f"Build the runtime for this toolchain first; missing install tree: {tree}")
    return trees


def main():
    platforms = {"linux": ["linux"], "android": ["arm64-v8a", "armeabi-v7a"],
                 "ios": ["iphonesimulator", "iphoneos"]}
    model_actions = ("model", "score", "cached-model", "force", "package")
    parser = argparse.ArgumentParser(description="Prepare a model or stage previously built libraries.")
    parser.add_argument("action", choices=(*model_actions, *platforms, *platforms["android"], *platforms["ios"]),
                        help="package stages the model and both Android ABIs")
    parser.add_argument("--dist", type=Path, default=Path("dist"), help="destination directory")
    parser.add_argument("--checkpoint", default="", help="checkpoint directory for model preparation")
    parser.add_argument("--build-tree", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cmake", default="cmake", help=argparse.SUPPRESS)
    parser.add_argument("--config", default="Release", help=argparse.SUPPRESS)
    args = parser.parse_args()
    dist = args.dist.absolute()
    trees = []
    if args.action == "package" or args.action not in model_actions:
        if args.build_tree is None:
            parser.error("library staging requires a configured build tree")
        platform = "android" if args.action == "package" else args.action
        trees = install_trees(args.build_tree, platforms.get(platform, [platform]))
    if args.action in model_actions:
        dist.mkdir(exist_ok=True, parents=True)
        with (dist / ".model.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            mode = "cached-model" if args.action == "package" else args.action
            prepare(mode, dist / "model_state", args.checkpoint)
    for tree in trees:
        subprocess.run([args.cmake, "--install", str(tree), "--config", args.config,
                        "--prefix", str(dist), "--component", "MercuryRuntime"], check=True)


if __name__ == "__main__":
    main()
