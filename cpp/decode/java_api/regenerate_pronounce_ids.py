"""conan.jsonl 의 발음 평가 id 배열들을 현재 모델 자산으로 재생성한다.

    python3 regenerate_pronounce_ids.py

conan.jsonl 의 각 줄에서 org(원문 단어열)만 유지하고, word_idx / word_idx_group /
phn_idx / phn_idx_group 을 export 폴더(../../../model/stream_conformer_cpp_export)의
mercury.model / phone.txt / full.dict 로 다시 만든다 — 기존 배열은 구모델
(mercury5 vocab) 기준이라 새 모델의 posterior 인덱스와 맞지 않는다.

숫자 단어("92")는 구 파일의 refine 필드에 있는 철자 변환("ninety two")을 재사용해
정상 채점되게 한다. 발음사전에 없는 단어는 phone 토큰 없이 그룹 자리만 유지한다
(구모델 phn 배열을 섞어 쓰면 그룹 규약이 달라 엉뚱한 단어에 점수가 붙는다).

id 규약 (C++ Scorer.makeWordDec/makePhoneDec 와 동일):
  word_idx : [sos=2] + sentencepiece ids + [eos=3], group 은 ▁ 토큰마다 +1 (sos/eos = -1)
  phn_idx  : [sos=2] + 단어별 phone id (구분자 4) + [eos=3], group 은 구분자마다 +1

기존 파일은 conan.jsonl.mercury5.bak 으로 백업한다.
"""
import json
import os
import shutil

import sentencepiece

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "..", "model", "stream_conformer_cpp_export")
SOS_ID, EOS_ID, PHONE_SEPARATOR_ID = 2, 3, 4


def refine_word_for_scoring(word):
    """C++ Scorer.refineLabel 과 동일: 소문자화 후 [a-z] 만 남긴다."""
    return "".join(c for c in word.lower() if c.islower() and c.isascii())


def main():
    spm_processor = sentencepiece.SentencePieceProcessor(
        model_file=os.path.join(EXPORT_DIR, "mercury.model"))
    phone_tokens = open(os.path.join(EXPORT_DIR, "phone.txt"),
                        errors="replace").read().split()[:47]
    phone_index = {p: i for i, p in enumerate(phone_tokens)}
    pronounce_dict = {}
    for line in open(os.path.join(EXPORT_DIR, "full.dict"), errors="replace"):
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            pronounce_dict[parts[0].lower()] = parts[1]

    vocab_tokens = [line.split("\t", 1)[0] for line in
                    open(os.path.join(EXPORT_DIR, "mercury.vocab"), errors="replace")]

    # 원본은 항상 mercury5 백업본에서 읽는다 — 거기에만 구 파이프라인의 숫자 철자
    # 변환(refine 필드)이 남아 있다. 백업이 없으면 현재 파일이 곧 원본이다.
    source_path = ("conan.jsonl.mercury5.bak"
                   if os.path.exists("conan.jsonl.mercury5.bak") else "conan.jsonl")

    regenerated_lines = []
    phone_missing_words_total = []
    for line in open(source_path, errors="replace"):
        record = json.loads(line)
        org_words = record["org"]
        old_refined_words = record.get("refine", [])
        refined_words = []
        for word_index, org_word in enumerate(org_words):
            refined_word = refine_word_for_scoring(org_word)
            if not refined_word and word_index < len(old_refined_words):
                # 숫자/기호 단어 — 구 refine 의 철자 변환("ninety two")을 재사용
                spelled = " ".join(refine_word_for_scoring(w)
                                   for w in old_refined_words[word_index].split())
                refined_word = " ".join(spelled.split())
            refined_words.append(refined_word)

        # 그룹 인덱스는 반드시 org 단어 위치와 1:1 이어야 한다 (Scorer 가 label 을
        # org 로 split 하므로). refined_word 는 철자 변환으로 여러 단어일 수 있고,
        # 그 토큰/음소 전부가 해당 org 단어의 그룹에 귀속된다.
        word_ids, word_groups = [SOS_ID], [-1]
        phone_ids, phone_groups = [SOS_ID], [-1]
        for word_index, refined_word in enumerate(refined_words):
            if not refined_word:
                continue   # 철자 변환도 불가 — 그룹 자리만 유지 (점수 0 처리)
            token_ids = spm_processor.encode(refined_word)
            word_ids += token_ids
            word_groups += [word_index] * len(token_ids)

            missing = [w for w in refined_word.split() if w not in pronounce_dict]
            if missing:
                # 발음사전 OOV — phone 토큰 없이 그룹 자리만 유지 (혼합 규약 금지)
                phone_missing_words_total.append((missing, " ".join(org_words[:5])))
                continue
            for sub_word in refined_word.split():
                for phone in pronounce_dict[sub_word].split():
                    phone_ids.append(phone_index[phone])
                    phone_groups.append(word_index)
            phone_ids.append(PHONE_SEPARATOR_ID)
            phone_groups.append(word_index)
        if phone_ids[-1] == PHONE_SEPARATOR_ID:
            phone_ids.pop(); phone_groups.pop()
        phone_ids.append(EOS_ID)
        phone_groups.append(-1)
        word_ids.append(EOS_ID)
        word_groups.append(-1)

        record["refine"] = refined_words
        record["word_idx"] = word_ids
        record["word_idx_group"] = word_groups
        record["phn_idx"] = phone_ids
        record["phn_idx_group"] = phone_groups
        regenerated_lines.append(json.dumps(record, ensure_ascii=False))

    if not os.path.exists("conan.jsonl.mercury5.bak"):
        shutil.copyfile("conan.jsonl", "conan.jsonl.mercury5.bak")
    with open("conan.jsonl", "w") as out_file:
        out_file.write("\n".join(regenerated_lines) + "\n")
    for missing, head in phone_missing_words_total:
        print("[재생성] 발음사전 OOV(phone 미부여): {} ({})".format(missing, head))
    print("[재생성] {}발화 완료 (phone 미부여 단어 {}건) -> conan.jsonl".format(
        len(regenerated_lines), len(phone_missing_words_total)))


if __name__ == "__main__":
    main()
