"""테스트 공용 — 경로 설정, 검증 집계, 출력 헬퍼, 로더 생성.

케이스 파일(case_*.py)은 맨 먼저 이 모듈을 임포트해야 한다. 임포트 시점에
sys.path 를 정리하므로 그 뒤에야 data.loader 를 임포트할 수 있다.
"""
import argparse
import os
import sys
import time

# -- 경로 (data.loader 임포트보다 먼저 실행되어야 한다) ----------------------
def _find_loader_dir(start_dir):
    """asr_data_loader.py 가 있는 조상 디렉토리를 찾는다.

    테스트 파일이 test/ 든 test/units/ 든 어디로 옮겨져도 경로가 깨지지 않게
    상대 깊이를 세지 않고 표식 파일로 찾는다.
    """
    path = start_dir
    while path != os.path.dirname(path):
        if os.path.exists(os.path.join(path, "asr_data_loader.py")):
            return path
        path = os.path.dirname(path)
    raise RuntimeError("loader 패키지를 찾지 못했습니다 (기준: {})".format(start_dir))


_HERE = os.path.dirname(os.path.abspath(__file__))
_LOADER_DIR = _find_loader_dir(_HERE)                        # .../data/loader
_MODEL_DIR = os.path.dirname(os.path.dirname(_LOADER_DIR))   # .../model
# loader/ 나 loader/database/ 가 sys.path 에 있으면 하위 패키지 database/lmdb 가
# pip 패키지 lmdb(C 바인딩)를 가려서 LmdbConn 의 `import lmdb` 가 깨진다.
_SHADOWING = {_LOADER_DIR, os.path.join(_LOADER_DIR, "database")}
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in _SHADOWING]
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from data.loader import AsrDataLoader                       # noqa: E402

# phone id -> 기호 (data_gen.py 의 phones 와 동일해야 함 — 바뀌면 함께 갱신)
PHONES = ["<pad>", "-", "<s>", "</s>", "_", "AA", "AE", "AH", "AO", "AW", "AY",
          "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IX",
          "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
          "T", "TH", "TS", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL"]

# 환경변수로 덮어쓸 수 있다 — VSCode Testing 패널에서 다른 DB 를 겨누고 싶을 때 쓴다.
DEFAULT_DB = os.environ.get("MERCURY_TEST_DB", "/home/chisnova/data/records/dev")
DEFAULT_VOCAB = os.environ.get(
    "MERCURY_TEST_VOCAB",
    os.path.join(os.path.dirname(_LOADER_DIR), "mercury.model"))

PASS, FAIL = "[ OK ]", "[FAIL]"
_check_results = []


# -- 검증 집계 ---------------------------------------------------------------
def check(cond, msg):
    """검증 결과를 찍고 기록 — summarize() 가 마지막에 합산한다."""
    print("    {} {}".format(PASS if cond else FAIL, msg))
    _check_results.append(bool(cond))
    return bool(cond)


def reset_checks():
    """검증 기록 초기화 — pytest 는 케이스마다 독립적으로 세야 한다."""
    del _check_results[:]


def failed_checks():
    return [index for index, ok in enumerate(_check_results) if not ok]


def summarize():
    """전체 결과 출력 + 종료 코드 반환."""
    print_title("요약")
    num_passed = sum(_check_results)
    print("  검증 {}/{} 통과".format(num_passed, len(_check_results)))
    if num_passed != len(_check_results):
        print("  실패한 항목이 있습니다 — 위 {} 표시 확인".format(FAIL))
    return 0 if num_passed == len(_check_results) else 1


# -- 출력 --------------------------------------------------------------------
def print_title(name):
    print("\n" + "=" * 78)
    print("### {}".format(name))
    print("=" * 78)


def print_step(index, name):
    """케이스 안의 스텝 구분 — 어디까지 진행됐는지 보이게."""
    print("\n  [step {}] {}".format(index, name))


# -- 디코딩 ------------------------------------------------------------------
def load_sentencepiece(vocab_path):
    if not vocab_path or not os.path.exists(vocab_path):
        return None
    try:
        import sentencepiece as spm
    except ImportError:
        return None
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.Load(vocab_path)
    return sp_processor


def decode_words(sp_processor, word_ids):
    if sp_processor is None:
        return "(sentencepiece 미로드)"
    return sp_processor.DecodeIds([int(x) for x in word_ids if int(x) not in (0, 2, 3)])


def decode_phones(phone_ids):
    return " ".join(PHONES[i] if 0 <= i < len(PHONES) else "?%d" % i
                    for i in phone_ids)


# -- 실행 준비 ---------------------------------------------------------------
def build_arg_parser(case_names=(), description=None):
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=DEFAULT_DB, help="레코드 디렉토리 (train/dev/test)")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB,
                        help="전사 복원용 sentencepiece .model")
    if case_names:
        parser.add_argument("--case", default="all", choices=["all"] + list(case_names),
                            help="실행할 케이스")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk", default="auto",
                        help="길이 정렬 단위. 'auto' 면 batch_size 에 맞춰 정한다")
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--usage_ratio", type=float, default=1.0,
                        help="대용량 DB 는 낮춰서 테스트")
    parser.add_argument("--train", action="store_true", help="isTrain=True (증강 on)")
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=2,
                        help="값을 자세히 출력할 배치 수")
    parser.add_argument("--num_records", type=int, default=2,
                        help="자세히 출력할 레코드 수")
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                        help="sample 케이스 표본 비율 (train 은 927만이라 낮추는 게 좋다)")
    parser.add_argument("--sample_seed", type=int, default=0, help="표본 추출 시드")
    parser.add_argument("--limit", type=int, default=20, help="순회할 최대 배치 수")
    return parser


def build_loader(args, sp_processor):
    """인자대로 AsrDataLoader 를 만들고 설정을 찍는다."""
    print("DB     : {}".format(args.db))
    print("vocab  : {}".format(args.vocab if sp_processor else "(미로드 — 전사 복원 생략)"))
    print("옵션   : batch_size={} chunk={} shuffle={} "
          "usage_ratio={} device={}".format(
              args.batch_size, args.chunk,
              not args.no_shuffle, args.usage_ratio, args.device))
    started_at = time.time()
    loader = AsrDataLoader(args.db,
                           batch_size=args.batch_size, chunk=args.chunk,
                           shuffle=not args.no_shuffle, isTrain=args.train,
                           usage_ratio=args.usage_ratio, num_workers=args.workers,
                           device=args.device)
    print("로더 초기화 {:.2f}s".format(time.time() - started_at))
    return loader


def run_standalone(run_case, description=None):
    """케이스 파일 하나를 단독 실행 — `python3 units/case_04_batch.py` 처럼 쓴다."""
    args = build_arg_parser(description=description).parse_args()
    sp_processor = load_sentencepiece(args.vocab)
    loader = build_loader(args, sp_processor)
    try:
        run_case(loader, args, sp_processor)
    finally:
        loader.close()
    return summarize()


def run_as_pytest(run_case, **arg_overrides):
    """pytest 진입점 — VSCode Testing 패널의 케이스별 실행 버튼이 이걸 부른다.

    명령행 대신 기본 인자로 돌리고(sys.argv 는 pytest 것이라 쓰면 안 된다),
    검증이 하나라도 실패하면 assert 로 떨군다. 검증 기록은 케이스마다 초기화한다.
    DB 는 MERCURY_TEST_DB 환경변수로 바꿀 수 있다.
    """
    reset_checks()
    args = build_arg_parser().parse_args([])
    for name, value in arg_overrides.items():
        setattr(args, name, value)
    sp_processor = load_sentencepiece(args.vocab)
    loader = build_loader(args, sp_processor)
    try:
        run_case(loader, args, sp_processor)
    finally:
        loader.close()
    num_failed = len(failed_checks())
    assert num_failed == 0, "검증 {}/{} 실패 — 위 {} 표시 확인".format(
        num_failed, len(_check_results), FAIL)
