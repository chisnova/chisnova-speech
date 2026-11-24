"""앱 자산(contents/**/*.jsonl)의 발음 평가 id 배열을 현재 모델 자산으로 재생성한다.

    python3 android_mercury/scripts/regenerate_pronounce_ids.py [--dry-run] [경로 ...]

각 줄의 org(원문 단어열)만 유지하고 word_idx / word_idx_group / phn_idx / phn_idx_group 을
assets/model_state 의 모델 자산으로 다시 만든다. 파일명은 하드코딩하지 않고 model.json 이
선언한 값(vocab_model / vocab_name / phone_name / dict_name)을 따르므로, 모델을 갈아끼운
뒤 그대로 다시 돌리면 된다.

왜 필요한가:
  word_idx 는 sentencepiece 서브워드 id 다. 모델을 교체하면 vocab 순서가 통째로 바뀌는데,
  앱은 이 배열을 그대로 네이티브 채점기에 넘긴다. 옛 id 를 새 모델에 넣으면 word decoder 가
  엉뚱한 토큰의 확률을 읽어 word 점수가 0 에 수렴하고, 최종 점수(0.6*phone + 0.4*word)가
  60점 부근에 갇힌다. phone id 는 phone.txt 가 그대로면 영향이 없다.

id 규약 (C++ Scorer.makeWordDec / makePhoneDec 와 동일):
  word_idx : [sos=2] + sentencepiece ids + [eos=3], group 은 org 단어 위치 (sos/eos = -1)
  phn_idx  : [sos=2] + 단어별 phone id (구분자 4) + [eos=3], group 은 org 단어 위치

숫자/기호처럼 정제 후 비는 단어는 기존 refine 필드의 철자 변환("92" -> "ninety two")을
재사용한다. 발음사전에 없는 단어는 phone 토큰 없이 그룹 자리만 유지한다 — 구 배열을 섞어
쓰면 그룹 규약이 어긋나 엉뚱한 단어에 점수가 붙는다.

원본은 android_mercury/scripts/pronounce_ids_backup/ 아래에 한 번만 백업한다.
재생성 입력은 항상 현재 파일이다. phone_overrides의 그룹별 org/phones는 우선 적용한다.
word_overrides의 그룹별 org/text는 표시 단어를 유지한 채 디코더 목표를 지정한다.
예: It's → it's, well-being → well being. 두 override는 인덱스 재생성 시에도 보존한다.
word_overrides의 join_previous=true는 health + care처럼 표시 그룹은 유지하면서
단어 디코더에서는 healthcare로 연결한다. 앞 그룹의 토큰이 바뀌는 연결은 거부한다.
word_overrides의 pieces는 text의 철자와 경계를 그대로 유지하는 SentencePiece 분절을
명시한다. 예: inessential → ["▁in", "es", "s", "ent", "ial"]. 제어·미등록 토큰은
거부하며, pieces를 지정한 그룹과 그 다음 그룹은 join_previous로 연결할 수 없다.
assets 안에 백업을 남기면 APK 에 함께 실린다.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import sentencepiece

SOS_ID, EOS_ID, PHONE_SEPARATOR_ID = 2, 3, 4

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(REPO_ROOT, "android_mercury", "app", "src", "main", "assets")
MODEL_DIR = os.path.join(ASSETS_DIR, "model_state")
# 콘텐츠 원본은 <repo>/contents 다. 앱 안의 것은 trace/sync.sh 가 만든 사본이라
# 그쪽을 고치면 다음 동기화에 덮여 사라진다 — 반드시 원본을 고친다.
CONTENTS_ROOT = os.path.join(REPO_ROOT, "contents")
CONTENTS_GLOB = os.path.join(CONTENTS_ROOT, "**", "*.jsonl")
# 백업은 assets 밖에 둔다 — assets 안에 남기면 그대로 APK 에 실린다
BACKUP_ROOT = os.path.join(REPO_ROOT, "android_mercury", "scripts", "pronounce_ids_backup")


def refine_word_for_scoring(word, pronounce_dict=None):
    """C++ Scorer.refineLabel 과 같은 규칙.

    어포스트로피는 늘 지운다("can't" -> "cant").
    하이픈은 사전을 보고 정한다:
      - 붙인 형태가 사전에 있으면 한 낱말로 둔다("counter-attacks" -> "counterattacks")
      - 없으면 가른다("low-impact" -> "low impact")

    사전에 있는 합성어까지 가르면 TTS 가 한 덩어리로 읽은 것과 낱말 경계가 어긋나
    뒤쪽이 점수를 못 받는다. 반대로 사전에 없는 것을 붙여 두면 통째로 점수에서 빠진다.

    돌려주는 값에 공백이 섞일 수 있으므로 호출부는 split() 으로 낱말을 다룬다.
    """
    lowered = word.lower()
    for drop in ("\u2019", "'"):
        lowered = lowered.replace(drop, "")
    kept = "".join(c for c in lowered
                   if c == " " or c == "-" or (c.isascii() and c.islower()))

    out = []
    for token in kept.split():
        if "-" not in token:
            out.append(token)
            continue
        joined = token.replace("-", "")
        if pronounce_dict is not None and joined in pronounce_dict:
            out.append(joined)
        else:
            out.extend(p for p in token.split("-") if p)
    return " ".join(out)


ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINAL_ONES = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
                "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
                "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
                "nineteenth"]
ORDINAL_TENS = ["", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
                "seventieth", "eightieth", "ninetieth"]


def ordinal_words(number):
    """서수를 읽는 대로 편다: 1 -> first, 19 -> nineteenth, 21 -> twenty first.

    "19th" 같은 표기는 정제하면 "th" 만 남아 낱말 행세를 한다(실제로 그 자리 점수가
    6점까지 떨어졌다). 음원은 "nineteenth" 라고 읽으므로 그 형태로 맞춘다.
    100 이상은 흔치 않아 다루지 않는다.
    """
    if number <= 0 or number >= 100:
        return ""
    if number < 20:
        return ORDINAL_ONES[number]
    tens, ones = divmod(number, 10)
    if ones == 0:
        return ORDINAL_TENS[tens]
    return TENS[tens] + " " + ORDINAL_ONES[ones]


PLURAL_TENS = {"twenty": "twenties", "thirty": "thirties", "forty": "forties",
               "fifty": "fifties", "sixty": "sixties", "seventy": "seventies",
               "eighty": "eighties", "ninety": "nineties", "ten": "tens",
               "hundred": "hundreds", "thousand": "thousands"}


def decade_words(text):
    """연도 복수형을 읽는 대로 편다: 1990s -> nineteen nineties, 1920's -> nineteen twenties.

    예전에는 정제하면 "s" 만 남아 연도가 통째로 사라졌다(그 자리 점수가 0이 됐다).
    음원은 "nineteen nineties" 라고 이어 읽으므로 그 형태로 맞춘다.
    """
    match = re.match(r"^(\d{4})'?s$", text)
    if not match:
        return ""
    year = int(match.group(1))
    if year < 1000 or year >= 3000:
        return ""
    high, low = divmod(year, 100)
    # 2000년대는 "two thousands" 로 읽는다
    if high == 20 and low == 0:
        return "two thousands"
    head = ONES[high] if high < 20 else (TENS[high // 10] + " " + ONES[high % 10]).strip()
    if low == 0:
        return head + " hundreds"
    if low < 20:
        tail = ONES[low]
    else:
        tens, ones = divmod(low, 10)
        tail = TENS[tens] if ones == 0 else TENS[tens] + " " + ONES[ones]
    last = tail.split()[-1]
    tail = tail[:len(tail) - len(last)] + PLURAL_TENS.get(last, last + "s")
    return (head + " " + tail).strip()


ORDINAL_PATTERN = re.compile(r"^(\d+)(st|nd|rd|th)$", re.IGNORECASE)


class ModelAssets(object):
    """model.json 이 선언한 이름으로 채점용 자산을 읽는다."""

    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "model.json"), errors="replace") as config_file:
            config = json.load(config_file)["model_path"]
        self.spm = sentencepiece.SentencePieceProcessor(
            model_file=os.path.join(model_dir, config["vocab_model"]))
        self.vocab = [line.split("\t", 1)[0] for line in
                      open(os.path.join(model_dir, config["vocab_name"]), errors="replace")]
        phone_tokens = open(os.path.join(model_dir, config["phone_name"]),
                            errors="replace").read().split()[:config["phone_size"]]
        self.phone_index = {phone: index for index, phone in enumerate(phone_tokens)}
        self.pronounce_dict = {}
        for line in open(os.path.join(model_dir, config["dict_name"]), errors="replace"):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                self.pronounce_dict[parts[0].lower()] = parts[1]

    def decode_words(self, word_ids):
        """word_idx 를 텍스트로 되돌린다 — 재생성 결과 검증용."""
        text = "".join(self.vocab[i] if 0 <= i < len(self.vocab) else "?" for i in word_ids)
        text = text.replace("▁", " ")
        for special in ("<s>", "</s>", "<pad>", "<unk>"):
            text = text.replace(special, "")
        return re.sub(r"\s+", " ", text).strip()


def word_target_text(record, refined_words, index):
    return record.get("word_overrides", {}).get(str(index), {}).get("text", refined_words[index])


def encode_explicit_word_pieces(override, assets, index):
    """Accept alternate segmentation only when spelling and word boundaries match."""
    pieces = override["pieces"]
    if (not isinstance(pieces, list) or not pieces or
            any(not isinstance(piece, str) or not piece or
                any(character.isspace() for character in piece) for piece in pieces)):
        raise ValueError("Invalid word pieces at group {}".format(index))
    if override.get("join_previous", False):
        raise ValueError("Explicit word pieces cannot continue a previous group")
    token_ids = [assets.spm.piece_to_id(piece) for piece in pieces]
    special_ids = {assets.spm.unk_id(), assets.spm.bos_id(),
                   assets.spm.eos_id(), assets.spm.pad_id()}
    if any(token_id in special_ids or assets.spm.is_control(token_id) or
           assets.spm.is_unknown(token_id) or assets.spm.is_unused(token_id) or
           assets.spm.id_to_piece(token_id) != piece
           for piece, token_id in zip(pieces, token_ids)):
        raise ValueError("Unknown or special word piece at group {}".format(index))
    text = override["text"]
    # Decoding alone would also accept a missing initial boundary and could join
    # this word to its predecessor. Preserve every boundary in the text as well.
    if ("".join(pieces) != "▁" + text.replace(" ", "▁") or
            assets.spm.decode(token_ids) != text or assets.decode_words(token_ids) != text):
        raise ValueError("Word pieces change text or boundaries at group {}".format(index))
    return token_ids


def encode_word_target(record, assets, refined_words, index):
    """Keep displayed groups separate when a compound has no decoder word boundary."""
    override = record.get("word_overrides", {}).get(str(index), {})
    text = word_target_text(record, refined_words, index)
    if "pieces" in override:
        return encode_explicit_word_pieces(override, assets, index)
    if not override.get("join_previous", False):
        return assets.spm.encode(text)
    if index == 0 or not refined_words[index - 1]:
        raise ValueError("Continuation requires a preceding word group")
    previous_override = record.get("word_overrides", {}).get(str(index - 1), {})
    if "pieces" in previous_override:
        raise ValueError("Cannot continue a group with explicit word pieces")
    if previous_override.get("join_previous", False):
        raise ValueError("Chained word continuations are not supported")
    previous = word_target_text(record, refined_words, index - 1)
    prefix = assets.spm.encode(previous)
    combined = assets.spm.encode(previous + text)
    # Never reassign or replace tokens belonging to the preceding display word.
    if not prefix or combined[:len(prefix)] != prefix or len(combined) <= len(prefix):
        raise ValueError("Compound tokenization changes the preceding word group")
    return combined[len(prefix):]


def expected_word_targets(record):
    text = ""
    for index, refined in enumerate(record["refine"]):
        if not refined:
            continue
        join = record.get("word_overrides", {}).get(str(index), {}).get("join_previous", False)
        text += (" " if text and not join else "") + word_target_text(record, record["refine"], index)
    return text


def regenerate_record(record, assets, oov_report):
    org_words = record["org"]
    old_refined = record.get("refine", [])
    for group, override in record.get("word_overrides", {}).items():
        if (not str(group).isdigit() or int(group) >= len(org_words) or
            not isinstance(override, dict) or override.get("org") != org_words[int(group)]):
            raise ValueError("Stale word override at group {}".format(group))
        text = override.get("text")
        if not isinstance(override.get("join_previous", False), bool):
            raise ValueError("Invalid word continuation at group {}".format(group))
        if (not isinstance(text, str) or not text.strip() or text != text.strip() or
            any(c in text for c in '\t\r\n<>')):
            raise ValueError("Invalid word override at group {}".format(group))
        if "pieces" in override:
            encode_explicit_word_pieces(override, assets, int(group))
    for group, override in record.get("phone_overrides", {}).items():
        if not str(group).isdigit() or int(group) >= len(org_words) or override.get("org") != org_words[int(group)]:
            raise ValueError("Stale phone override at group {}".format(group))

    refined_words = []
    for word_index, org_word in enumerate(org_words):
        refined = refine_word_for_scoring(org_word, assets.pronounce_dict)
        # "1990s" 같은 연도 복수형은 정제하면 "s" 만 남는다 — 읽는 대로 편다
        decade = decade_words(org_word.strip('",.!?()'))
        if decade:
            refined = decade
        # "1st", "19th" 처럼 서수는 정제하면 "st", "th" 만 남는다 — 읽는 대로 편다
        ordinal = ORDINAL_PATTERN.match(org_word.strip('",.!?()'))
        if ordinal:
            spelled = ordinal_words(int(ordinal.group(1)))
            if spelled:
                refined = spelled
        if not refined and word_index < len(old_refined):
            # 숫자/기호 단어 — 기존 refine 의 철자 변환을 재사용한다
            refined = " ".join(refine_word_for_scoring(w, assets.pronounce_dict)
                                for w in old_refined[word_index].split())
            refined = " ".join(refined.split())
        refined_words.append(refined)

    # org 는 원문 그대로 둔다 — 화면에 보이는 낱말과 같아야 한다.
    #
    # 앱은 채점 결과의 낱말 이름으로 화면 문장과 짝을 맞춰 색을 칠하고 점수를 띄운다.
    # 여기서 org 를 갈라 놓으면("Means-testing" -> means, testing) 화면의 낱말과 이름이
    # 어긋나 색이 안 칠해지고 눌러도 점수가 안 나온다(전체의 1.4%가 그랬다).
    #
    # 채점 칸 수가 어긋나지 않는 이유는, 앱이 getScore 에 원문 문장을 넘기기 때문이다.
    # Scorer 는 그 문장을 띄어쓰기로 쪼개 칸을 만들므로 칸 수 = org 수가 된다.
    # 정제 결과가 여러 낱말이어도(means testing) 그 낱말들은 한 칸 안에서 함께 평균된다.

    word_ids, word_groups = [SOS_ID], [-1]
    phone_ids, phone_groups = [SOS_ID], [-1]
    for word_index, refined in enumerate(refined_words):
        if not refined:
            continue   # 철자 변환도 불가 — 그룹 자리만 유지(점수 0 처리)
        token_ids = encode_word_target(record, assets, refined_words, word_index)
        word_ids += token_ids
        word_groups += [word_index] * len(token_ids)

        override = record.get("phone_overrides", {}).get(str(word_index))
        if override is not None:
            if override.get("org") != org_words[word_index]:
                raise ValueError("Stale phone override at group {}".format(word_index))
            selected = override["phones"].split()
            if not selected or any(p not in assets.phone_index or
                                   (assets.phone_index[p] <= 4 and
                                    not (p == "_" and assets.phone_index[p] == PHONE_SEPARATOR_ID))
                                   for p in selected):
                raise ValueError("Invalid phone override at group {}".format(word_index))
            # A spoken word boundary can occur inside one displayed org group.
            # Keep separators internal and isolated; do not allow control tokens.
            if selected[0] == "_" or selected[-1] == "_" or any(
                    a == b == "_" for a, b in zip(selected, selected[1:])):
                raise ValueError("Invalid internal phone boundary at group {}".format(word_index))
            phone_ids.extend(assets.phone_index[p] for p in selected)
            phone_groups.extend([word_index] * len(selected))
            phone_ids.append(PHONE_SEPARATOR_ID)
            phone_groups.append(word_index)
            continue
        missing = [w for w in refined.split() if w not in assets.pronounce_dict]
        if missing:
            oov_report.append((missing, " ".join(org_words[:5])))
            continue   # 발음사전 OOV — phone 토큰 없이 그룹 자리만 유지
        for sub_word in refined.split():
            for phone in assets.pronounce_dict[sub_word].split():
                phone_ids.append(assets.phone_index[phone])
                phone_groups.append(word_index)
        phone_ids.append(PHONE_SEPARATOR_ID)
        phone_groups.append(word_index)

    if phone_ids[-1] == PHONE_SEPARATOR_ID:
        phone_ids.pop()
        phone_groups.pop()
    phone_ids.append(EOS_ID)
    phone_groups.append(-1)
    word_ids.append(EOS_ID)
    word_groups.append(-1)

    record["refine"] = refined_words
    record["word_idx"] = word_ids
    record["word_idx_group"] = word_groups
    record["phn_idx"] = phone_ids
    record["phn_idx_group"] = phone_groups
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="대상 jsonl (생략하면 assets/contents 전체)")
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 결과만 확인")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()

    assets = ModelAssets(args.model_dir)
    targets = args.paths or sorted(glob.glob(CONTENTS_GLOB, recursive=True))
    # 검수 결과와 백업은 콘텐츠가 아니다 — 건드리면 되돌릴 자료가 사라진다
    targets = [t for t in targets if "audit_result" not in t.split(os.sep)]
    if not targets:
        print("대상 jsonl 이 없다", file=sys.stderr)
        return 1

    total_records = 0
    changed_files = 0
    mismatched = []
    oov_report = []
    for path in targets:
        # 백업은 콘텐츠 원본 기준 상대경로로 담는다. assets 기준으로 잡으면
        # 원본이 저장소 루트로 옮겨진 뒤 "../../.." 가 붙어 엉뚱한 곳을 가리킨다.
        backup_path = os.path.join(BACKUP_ROOT, os.path.relpath(path, CONTENTS_ROOT))

        # Backups are recovery snapshots, never regeneration inputs: loading an
        # old snapshot would silently discard current per-record phone overrides.
        source_path = path

        original = [line for line in open(source_path, errors="replace") if line.strip()]
        regenerated = []
        for line in original:
            record = regenerate_record(json.loads(line), assets, oov_report)
            total_records += 1
            # 검증: 재생성한 id 를 현재 vocab 으로 되돌리면 원문과 같아야 한다
            decoded = assets.decode_words(record["word_idx"])
            expected = expected_word_targets(record)
            if decoded != expected:
                mismatched.append((os.path.basename(path), expected[:60], decoded[:60]))
            regenerated.append(json.dumps(record, ensure_ascii=False))

        if not args.dry_run:
            if not os.path.exists(backup_path):
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copyfile(path, backup_path)
            with open(path, "w") as out_file:
                out_file.write("\n".join(regenerated) + "\n")
        changed_files += 1

    print("대상 파일 {}개 / 발화 {}건 {}".format(
        changed_files, total_records, "(dry-run)" if args.dry_run else "재생성 완료"))
    if oov_report:
        unique_oov = sorted({w for words, _ in oov_report for w in words})
        print("발음사전 OOV(phone 미부여) {}건, 고유 단어 {}개: {}".format(
            len(oov_report), len(unique_oov), unique_oov[:20]))
    if mismatched:
        print("검증 실패 {}건:".format(len(mismatched)))
        for name, expected, decoded in mismatched[:10]:
            print("  {}\n    org : {}\n    dec : {}".format(name, expected, decoded))
        return 1
    print("검증 통과: 모든 word_idx 가 현재 vocab 으로 원문과 일치")
    return 0


if __name__ == "__main__":
    sys.exit(main())
