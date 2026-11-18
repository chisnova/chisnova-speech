"""잡음 오디오 -> log-mel dB LMDB 레코드 (SnrNoiseMix 가 학습 중에 읽어 쓴다).

    PYTHONPATH=$PWD python3 data/generation/record/noise_record_gen.py \
        --noise_list /home/chisnova/archive1/ASR_DATA/asr_list/noise.list

특징 파라미터는 data_gen.py 와 같아야 한다 — 발화 특징과 같은 축(멜 빈, dB)에
있어야 power 도메인에서 더할 수 있기 때문이다. 다만 top_db 만은 끈다: 저장된
잡음은 발화와 섞이기 전에 SNR 로 스케일되므로, 발화 피크 기준으로 바닥을
잘라 둘 이유가 없다.

레이아웃도 발화 레코드(data_gen.py)와 같은 규약을 따른다 — 클립 하나가 한 줄이다.

    <out_dir>/manifest.jsonl   클립당 한 줄: {"clip_id", "feat_len", "source"}
    <out_dir>/meta.json        스키마만: feat_dim / feat_dtype / frame_rate / count
    <out_dir>/noise.lmdb/      키 = clip_id, 값 = (T,80) float16 raw bytes
    <out_dir>/skipped.txt      건너뛴 파일과 이유

manifest 를 따로 두는 이유는 발화 레코드와 같다: 학습 중에 필요한 것은 "어떤
클립이 몇 프레임인가" 뿐이고, 그건 LMDB 를 열지 않고도 알 수 있어야 임의 클립의
임의 구간을 온라인으로 고를 수 있다.
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    import lmdb
except ImportError:
    raise SystemExit("lmdb 가 필요합니다:  python3 -m pip install lmdb")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import NOISE_DIR, NOISE_LIST, NOISE_RECORD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from data.loader.database import NoiseLmdbConn
from util.fbank_feature_extractor import FbankFeatureExtractor


class NoiseRecordGen:
    """잡음 오디오 목록을 읽어 log-mel dB LMDB 레코드로 굳힌다.

    클립 하나가 값 하나이고 manifest 한 줄이다. 학습 중에는 "임의 클립의 임의 구간
    T 프레임" 만 필요하므로 키 조회 한 번 + 구간 복사 한 번이면 끝나고, 전체를
    메모리(하물며 GPU)에 올려 둘 이유가 없다 — mmap 과 페이지 캐시가 그 일을 한다.

    float16 으로 저장한다. dB 값의 유효 범위가 -100~+22 라 float16 의 정밀도
    (이 구간에서 0.03dB 이하)로 충분하고 용량은 절반이다.
    """

    AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")
    FEATURE_DTYPE = np.float16

    def __init__(self, feature_extractor=None):
        # top_db=None: 바닥 클리핑 없이 저장 (발화와 달리 잡음은 절대 레벨로 쓰인다)
        self.feature_extractor = feature_extractor or FbankFeatureExtractor(top_db=None)

    @classmethod
    def read_noise_list(cls, list_path):
        """목록 파일 -> [(clip_id, path)]. 한 줄에 하나.

        발화 목록(train_tot.wlist)과 같은 "id<TAB>경로" 를 받고, 탭이 없으면
        파일명(확장자 제외)을 id 로 쓴다.
        """
        clips = []
        with open(list_path) as list_file:
            for line in list_file:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split("\t")
                if len(tokens) >= 2:
                    clips.append((tokens[0].strip(), tokens[1].strip()))
                else:
                    clips.append((os.path.splitext(os.path.basename(line))[0], line))
        return clips

    @classmethod
    def scan_noise_dir(cls, noise_dir):
        """디렉토리를 재귀로 훑어 [(clip_id, path)] — clip_id 는 확장자 뗀 상대경로."""
        clips = []
        for current_dir, _, file_names in os.walk(noise_dir):
            for file_name in sorted(file_names):
                if not file_name.lower().endswith(cls.AUDIO_EXTENSIONS):
                    continue
                noise_path = os.path.join(current_dir, file_name)
                clip_id = os.path.splitext(os.path.relpath(noise_path, noise_dir))[0]
                clips.append((clip_id, noise_path))
        return sorted(clips)

    @classmethod
    def write_noise_list(cls, clips, list_path):
        """훑은 결과를 목록 파일로 남긴다 — 다음 생성부터 이 목록만 쓰면 재현된다."""
        with open(list_path, "w") as list_file:
            for clip_id, noise_path in clips:
                list_file.write("{}\t{}\n".format(clip_id, noise_path))
        print("[noise] 목록 저장: {} ({} 줄)".format(list_path, len(clips)))

    def build_record(self, clips, out_dir, min_frame_count=100, map_size_gb=8):
        """특징을 뽑아 LMDB + manifest.jsonl + meta.json 으로 저장한다.

        min_frame_count 보다 짧은 클립(기본 1초)은 버린다 — 긴 발화에 붙이려면
        같은 구간을 반복해야 해서 주기적인 패턴이 생긴다.
        """
        os.makedirs(out_dir, exist_ok=True)
        manifest_path = os.path.join(out_dir, NoiseLmdbConn.MANIFEST_NAME)
        skipped_path = os.path.join(out_dir, "skipped.txt")

        written_count = 0
        skipped_count = 0
        min_db, max_db = float("inf"), float("-inf")
        env = lmdb.open(os.path.join(out_dir, NoiseLmdbConn.DB_NAME), subdir=True,
                        map_size=int(map_size_gb * 1024 ** 3))
        try:
            with open(manifest_path, "w") as manifest_file, \
                 open(skipped_path, "w") as skipped_file, \
                 env.begin(write=True) as write_txn:
                for clip_index, (clip_id, noise_path) in enumerate(clips):
                    try:
                        feature = self.feature_extractor.extract_from_wav_file(noise_path)
                    except Exception as extract_error:
                        skipped_file.write("{}\t{}\t{}\n".format(clip_id, noise_path,
                                                                repr(extract_error)))
                        skipped_count += 1
                        continue
                    if feature.size(0) < min_frame_count:
                        skipped_file.write("{}\t{}\t짧음 {} 프레임\n".format(
                            clip_id, noise_path, feature.size(0)))
                        skipped_count += 1
                        continue

                    clip = feature.numpy().astype(self.FEATURE_DTYPE)
                    write_txn.put(clip_id.encode("utf-8"), clip.tobytes())
                    manifest_file.write(json.dumps({
                        "clip_id": clip_id,
                        "feat_len": int(clip.shape[0]),
                        "source": noise_path,
                    }) + "\n")
                    written_count += 1
                    min_db = min(min_db, float(clip.min()))
                    max_db = max(max_db, float(clip.max()))
                    if (clip_index + 1) % 100 == 0:
                        print("  {}/{} 처리".format(clip_index + 1, len(clips)))
        finally:
            env.close()

        if not written_count:
            raise RuntimeError("특징을 뽑을 수 있는 잡음 파일이 없습니다")

        frame_rate = self.feature_extractor.sample_rate / self.feature_extractor.hop_length
        with open(os.path.join(out_dir, NoiseLmdbConn.META_NAME), "w") as meta_file:
            json.dump({"feat_dim": self.feature_extractor.num_mel_bins,
                       "feat_dtype": np.dtype(self.FEATURE_DTYPE).name,
                       "frame_rate": frame_rate,
                       "count": written_count}, meta_file, indent=2)

        total_frames = sum(json.loads(line)["feat_len"] for line in open(manifest_path))
        print("[noise] 클립 {} 개 (건너뜀 {}), 프레임 {:,} = {:.1f} 시간".format(
            written_count, skipped_count, total_frames, total_frames / frame_rate / 3600))
        print("[noise] dB 범위 {:.1f} ~ {:.1f}".format(min_db, max_db))
        print("[noise] 저장: {} ({:.0f} MB)".format(out_dir, self.measure_dir_size(out_dir)))
        return out_dir

    @staticmethod
    def measure_dir_size(target_dir):
        total_bytes = 0
        for current_dir, _, file_names in os.walk(target_dir):
            total_bytes += sum(os.path.getsize(os.path.join(current_dir, name))
                               for name in file_names)
        return total_bytes / 1024 ** 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="잡음 log-mel dB 레코드 생성")
    parser.add_argument('--noise_list', type=str, default=NOISE_LIST,
                        help="잡음 목록 파일 (한 줄에 하나, 'id<TAB>경로' 또는 경로만)")
    parser.add_argument('--noise_dir', type=str, default=NOISE_DIR,
                        help="목록 파일이 없을 때 훑을 디렉토리 (훑은 결과를 목록으로 저장한다)")
    parser.add_argument('--output', type=str, default=NOISE_RECORD,
                        help="레코드 디렉토리 (manifest.jsonl + meta.json + noise.lmdb)")
    parser.add_argument('--min_frame_count', type=int, default=100,
                        help="이보다 짧은 클립은 버린다 (프레임, 100fps)")
    parser.add_argument('--map_size_gb', type=int, default=8,
                        help="LMDB 가상 맵 크기 (sparse — 넉넉히 잡아도 된다)")
    args = parser.parse_args()

    record_gen = NoiseRecordGen()
    if os.path.isfile(args.noise_list):
        noise_clips = record_gen.read_noise_list(args.noise_list)
        print("[noise] 목록 {} 줄: {}".format(len(noise_clips), args.noise_list))
    else:
        noise_clips = record_gen.scan_noise_dir(args.noise_dir)
        print("[noise] {} 개 파일 발견: {}".format(len(noise_clips), args.noise_dir))
        record_gen.write_noise_list(noise_clips, args.noise_list)

    record_gen.build_record(noise_clips, args.output,
                            min_frame_count=args.min_frame_count,
                            map_size_gb=args.map_size_gb)
