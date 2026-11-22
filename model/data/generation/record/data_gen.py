import numpy as np
import sentencepiece as spm
import os
import sys
import argparse
import json
import pickle
import librosa
import lmdb
from tqdm import tqdm
try:
    from .phone_labels import load_phone_labels
except ImportError:
    from phone_labels import load_phone_labels

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import FULL_DICT, VOCAB_MODEL

phone_dict={}
phones=["<pad>", "-", "<s>", "</s>", "_","AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY","F","G","HH","IH","IX","IY","JH","K","L","M","N","NG","OW","OY","P","R","S","SH","T","TH","TS","UH","UW","V","W","Y","Z","ZH", "SIL"]
for (idx, phone) in enumerate(phones):
    phone_dict[phone]=idx

dictionary={}
_n_bad_dict = 0
with open(FULL_DICT, encoding="utf-8", errors="replace") as _df:
    for _line in _df:
        _tok = _line.rstrip("\n").split("\t")
        _key = _tok[0].strip().lower()
        _val = _tok[1].strip() if len(_tok) > 1 else ""
        if not _key or not _val:          # 키/발음 비면 스킵 (한 줄 때문에 파이프라인이 죽지 않도록)
            _n_bad_dict += 1
            continue
        dictionary[_key] = _val
print("[full.dict] {} entries from {}".format(len(dictionary), FULL_DICT))
if _n_bad_dict:
    print("[full.dict] skipped {} malformed entries (empty key or phone)".format(_n_bad_dict))


def _split_compound(word, max_parts=3, max_word_len=20):
    """붙여쓴 복합어를 사전 단어들로 분해. 파트 수가 가장 적은 분해를 반환
    (therapistfinder -> [therapist, finder], not [the, rapist, finder]).
    각 파트 길이 >= 2, 파트 수 2~max_parts. 실패 시 None."""
    n = len(word)
    if n < 4:
        return None
    dp = [None] * (n + 1)   # dp[i] = (min_parts, prev_j) for word[:i]
    dp[0] = (0, -1)
    for i in range(1, n + 1):
        best = None
        for j in range(max(0, i - max_word_len), i - 1):   # i-1 => 파트 길이 >= 2
            if dp[j] is None:
                continue
            if word[j:i] in dictionary:
                cand = (dp[j][0] + 1, j)
                if best is None or cand[0] < best[0]:
                    best = cand
        dp[i] = best
    if dp[n] is None or not (2 <= dp[n][0] <= max_parts):
        return None
    parts, i = [], n
    while i > 0:
        j = dp[i][1]
        parts.append(word[j:i])
        i = j
    parts.reverse()
    return parts


def resolveWordPhones(word):
    """단어의 발음(공백구분 phone 문자열)을 반환, 못 찾으면 None.
    순서: (1) 사전 직접 (2) 아포스트로피 정규화 (3) 규칙 형태소 (4) 복합어 분해.
    phone 라벨 생성에만 사용 — 전사문/wordpiece 경로는 건드리지 않는다.
    규칙 어미(S/Z/D/IH NG)는 정규 굴절의 근사 발음이다."""
    v = dictionary.get(word)
    if v:
        return v

    # (2) 아포스트로피
    if "'" in word:
        if word.endswith("'s"):                 # 소유격/복수소유격: cao's -> cao + S
            b = dictionary.get(word[:-2])
            if b:
                return b + " S"
        if word.endswith("s'"):                 # 복수 소유격: cats' -> cats
            b = dictionary.get(word[:-1])
            if b:
                return b
        v = dictionary.get(word.replace("'", ""))   # fuckin' -> fuckin, o'brien -> obrien
        if v:
            return v
        return None                             # 아포스트로피 단어는 복합어 분해 부적합

    # (3) 규칙 형태소 (근사 어미 발음)
    if word.endswith("ing") and len(word) > 4:
        for b in (word[:-3], word[:-3] + "e"):  # coding -> code
            v = dictionary.get(b)
            if v:
                return v + " IH NG"
    if word.endswith("ed") and len(word) > 3:
        for b in (word[:-2], word[:-1]):        # coded -> code
            v = dictionary.get(b)
            if v:
                return v + " D"
    if word.endswith("es") and len(word) > 3:
        v = dictionary.get(word[:-2])
        if v:
            return v + " IH Z"
    if word.endswith("s") and len(word) > 2:
        v = dictionary.get(word[:-1])
        if v:
            return v + " S"

    # (4) 복합어 분해 (사전 단어 사이에 _ 경계 삽입)
    parts = _split_compound(word)
    if parts is not None:
        return " _ ".join(dictionary[p] for p in parts)

    return None


class AsrDataGen():
    """Generate per-utterance ASR records (feat + word ids + phone ids) and
    store them in a single LMDB env, one record per utterance.

    Batching is intentionally NOT done here: each utterance is stored at its
    natural variable length, keyed by utt_id. Padding / bucketing / shuffling
    all happen at train time using the manifest, so batch size and grouping
    can change without regenerating the data.
    """

    def __init__(self, wav_path, label_path, vocab_model, phone_labels=None):
        self.data = self.loadWavLabel(wav_path, label_path, strict=phone_labels is not None)
        self.phone_targets, self.phone_label_provenance = None, None
        if phone_labels is not None:
            texts = {identity: self.refineText(text.lower()) for identity, _, text in self.data}
            waves = {identity: wave for identity, wave, _ in self.data}
            self.phone_targets, self.phone_label_provenance = load_phone_labels(phone_labels, texts, phone_dict, waves)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(vocab_model)

    def parseFile(self, path, strict=False):
        ret_dict = {}
        with open(path, 'r') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                token = line.split('\t')
                if strict and (len(token) != 2 or not token[0] or not token[1] or token[0] in ret_dict):
                    raise ValueError('Invalid or duplicate utterance in {}: {!r}'.format(path, line))
                ret_dict[token[0]] = token[1]
        return ret_dict

    def loadWavLabel(self, wav_path, label_path, strict=False):
        wav_dict = self.parseFile(wav_path, strict=strict)
        lab_dict = self.parseFile(label_path, strict=strict)
        if strict and set(wav_dict) != set(lab_dict):
            raise ValueError('Wave and text utterance IDs differ for --phone-labels')

        data = []
        for utt_id, wav in wav_dict.items():
            lab = lab_dict.get(utt_id)
            if lab is None:            # keep triples aligned: skip if no matching label
                continue
            data.append((utt_id, wav, lab))
        return data

    def PreEmphasis(self, wave):
        # y[n] = x[n] - 0.97*x[n-1], vectorized (was a python for-loop)
        ret = np.empty_like(wave)
        ret[0] = wave[0]
        ret[1:] = wave[1:] - 0.97 * wave[:-1]
        return ret

    def getFbankFeature(self, path):
        n_fft = 400
        win_len = 400
        n_hop = 160
        n_mel = 80

        wave_data, sample_rate = librosa.load(path, sr=16000)
        wave_data = self.PreEmphasis(wave_data)
        feat = librosa.feature.melspectrogram(y=wave_data, sr=sample_rate, n_mels=n_mel, n_fft=n_fft, hop_length=n_hop, win_length=win_len)
        feat = librosa.power_to_db(feat)
        feat = np.transpose(feat)
        return feat

    def getWordpieceFeature(self, text):
        ids = self.sp.EncodeAsIds(text)
        ids.insert(0, 2)  # SOS
        ids.append(3)  # EOS
        return ids

    def makePhoneFeature(self, text):
        text = text.replace("‘","")
        if not text.isascii():         # 비ASCII(악센트/외래어) 문자 포함 -> phone 생성 불가, 발화 skip
            raise ValueError("non-ascii text: " + text)
        words = text.split(" ")
        phone_arr=[]
        phone_arr.insert(0, 2) # SOS

        for word in words:
            if not word:
                continue
            phones = resolveWordPhones(word)
            if phones is None:
                raise KeyError(word)   # 미해결 OOV -> 발화 skip & skipped.txt 기록
            word_phone = phones + " _"
            for element in word_phone.split(" "):
                if element == "":
                    continue
                phone_arr.append(phone_dict[element])
        phone_arr = phone_arr[:-1]     # 마지막 단어 경계 "_" 제거
        phone_arr.append(3) # EOS
        return phone_arr

    def refineText(self, text):
        text = text.replace("-", " ")   # 하이픈 삭제 대신 공백: BYE-BYE -> BYE BYE (둘 다 사전 히트)
        return text

    _META = {"feat_dim": 80, "feat_dtype": "float16", "id_dtype": "int32",
             "sos": 2, "eos": 3, "pad": 0}

    def saveLMDB(self, out_dir, map_size, commit_interval=2000, shard_bytes=0):
        """utterance 당 레코드 1개를 용량 기준으로 나눈 LMDB 샤드들에 저장.

        shard_bytes > 0 : 현재 샤드에 쌓인 값 바이트가 이 값에 도달하면 새 샤드로
                          넘어감 (shard_00000.lmdb, shard_00001.lmdb, ...).
        shard_bytes <= 0: 단일 샤드(롤오버 없음).
        통합 manifest.jsonl(레코드마다 'shard' 인덱스 포함), skipped.txt,
        그리고 데이터셋 레벨 meta.json 을 함께 쓴다.
        """
        os.makedirs(out_dir, exist_ok=True)
        manifest_path = os.path.join(out_dir, "manifest.jsonl")
        skipped_path = os.path.join(out_dir, "skipped.txt")
        shard_pattern = "shard_%05d.lmdb"

        # 용량 상한이 있는 샤드는 전체 map_size 를 예약할 필요가 없다
        shard_map = min(map_size, shard_bytes * 2 + (2 << 30)) if shard_bytes > 0 else map_size

        def open_shard(i):
            return lmdb.open(os.path.join(out_dir, shard_pattern % i), map_size=shard_map,
                             subdir=True, readonly=False, meminit=False, map_async=True)

        def close_shard(env, txn, count):
            txn.put(b"__meta__", pickle.dumps(dict(self._META, count=count),
                                              protocol=pickle.HIGHEST_PROTOCOL))
            txn.commit(); env.sync(); env.close()

        shard_idx = 0
        env = open_shard(shard_idx)
        txn = env.begin(write=True)
        cur_bytes = 0          # 현재 샤드에 쌓인 값 바이트 (롤오버 판단)
        cur_count = 0          # 현재 샤드 레코드 수
        shard_counts = []
        n_written = 0
        n_skip = 0

        with open(manifest_path, "w") as mf, open(skipped_path, "w") as sf:
            for utt_id, wav_path, text in tqdm(self.data):
                text = self.refineText(text.lower())
                try:
                    feat = self.getFbankFeature(wav_path).astype(np.float16)
                    word_ids = np.asarray(self.getWordpieceFeature(text), dtype=np.int32)
                    phone_ids = np.asarray(self.phone_targets[utt_id] if self.phone_targets is not None
                                           else self.makePhoneFeature(text), dtype=np.int32)
                except Exception as e:
                    n_skip += 1
                    sf.write("{}\t{}\t{}\n".format(utt_id, wav_path, repr(e)))
                    continue

                record = {
                    "utt_id": utt_id,
                    "feat": feat,            # (T, 80) float16
                    "word_ids": word_ids,    # (Lw,)  int32
                    "phone_ids": phone_ids,  # (Lp,)  int32
                }
                payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(utt_id.encode("utf-8"), payload)
                cur_bytes += len(payload)
                cur_count += 1

                mf.write(json.dumps({
                    "utt_id": utt_id,
                    "shard": shard_idx,
                    "feat_len": int(feat.shape[0]),
                    "word_len": int(word_ids.shape[0]),
                    "phone_len": int(phone_ids.shape[0]),
                }) + "\n")

                n_written += 1
                if n_written % commit_interval == 0:
                    txn.commit()
                    txn = env.begin(write=True)

                # 용량 기준 롤오버
                if shard_bytes > 0 and cur_bytes >= shard_bytes:
                    close_shard(env, txn, cur_count)
                    shard_counts.append(cur_count)
                    shard_idx += 1
                    env = open_shard(shard_idx)
                    txn = env.begin(write=True)
                    cur_bytes = 0
                    cur_count = 0

            # 마지막(부분) 샤드 마무리
            close_shard(env, txn, cur_count)
            shard_counts.append(cur_count)

        with open(os.path.join(out_dir, "meta.json"), "w") as jf:
            json.dump(dict(self._META,
                           **({'phone_labels': self.phone_label_provenance} if self.phone_label_provenance else {}),
                           count=n_written,
                           num_shards=len(shard_counts),
                           shard_counts=shard_counts,
                           shard_pattern=shard_pattern,
                           shard_bytes_cap=shard_bytes), jf, indent=2)

        print("finish!!  written: {}  skipped: {}  shards: {}".format(
            n_written, n_skip, len(shard_counts)))
        print("dir      : {}".format(out_dir))
        print("manifest : {}".format(manifest_path))
        if n_skip:
            print("skipped log: {}".format(skipped_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASR datagen (per-utterance LMDB)')
    parser.add_argument('--wavpath', type=str, required=True, help='wav list file (utt_id<TAB>wav_path)')
    parser.add_argument('--txtpath', type=str, required=True, help='label list file (utt_id<TAB>text)')
    parser.add_argument('--savepath', type=str, required=True, help='output dir for the LMDB env + manifest')
    parser.add_argument('--vocab', type=str, default=VOCAB_MODEL)
    parser.add_argument('--phone-labels', help='Optional validated *.phone_labels.jsonl from prepare_adaptation.py.')
    parser.add_argument('--map_size_gb', type=int, default=1024, help='LMDB virtual map size in GB (sparse; safe to overshoot)')
    parser.add_argument('--commit_interval', type=int, default=2000)
    parser.add_argument('--shard_gb', type=float, default=0,
                        help='샤드당 용량 상한(GB). 0이면 단일 LMDB. 예: 6 -> ~6GB마다 새 샤드')
    args = parser.parse_args()

    saving = AsrDataGen(args.wavpath, args.txtpath, args.vocab, phone_labels=args.phone_labels)
    saving.saveLMDB(args.savepath,
                    map_size=args.map_size_gb * (1024 ** 3),
                    commit_interval=args.commit_interval,
                    shard_bytes=int(args.shard_gb * (1024 ** 3)))

# /home/chisnova/data/records/train
# /home/chisnova/data/records/dev
# /home/chisnova/data/records/test
