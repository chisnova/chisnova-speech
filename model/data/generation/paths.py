"""생성 스크립트들이 공유하는 경로 — cwd 에 의존하지 않는다.

이전에는 각 스크립트가 "full.dict" / "train_tot.txt" 처럼 상대경로를 열어서
어느 디렉토리에서 실행하느냐에 따라 다른 파일을 읽거나 그냥 죽었다. 여기서
파일 위치 기준으로 한 번만 풀어 두고 전부 이걸 쓴다.

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import FULL_DICT, TRAIN_LAB, artifact
"""
import os

GENERATION_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(GENERATION_DIR)              # model/data
MODEL_DIR = os.path.dirname(DATA_DIR)                   # model

# -- 생성 산출물 (코드 아님) -------------------------------------------------
ARTIFACTS_DIR = os.path.join(GENERATION_DIR, "artifacts")


def artifact(name):
    """산출물 경로 — artifacts/ 아래로 모은다."""
    return os.path.join(ARTIFACTS_DIR, name)


# -- 사전 / 어휘 -------------------------------------------------------------
FULL_DICT = os.path.join(DATA_DIR, "full.dict")
VOCAB_MODEL = os.path.join(DATA_DIR, "mercury.model")

# -- 원본 오디오 목록 / 전사 -------------------------------------------------
ASR_LIST_DIR = "/home/chisnova/archive1/ASR_DATA/asr_list"
TRAIN_LAB = os.path.join(ASR_LIST_DIR, "train_tot.lab")
TRAIN_WLIST = os.path.join(ASR_LIST_DIR, "train_tot.wlist")
DEV_LAB = os.path.join(ASR_LIST_DIR, "dev_tot.lab")
DEV_WLIST = os.path.join(ASR_LIST_DIR, "dev_tot.wlist")
TEST_LAB = os.path.join(ASR_LIST_DIR, "test_tot.lab")
TEST_WLIST = os.path.join(ASR_LIST_DIR, "test_tot.wlist")

# -- 레코드 DB ---------------------------------------------------------------
RECORDS_ROOT = "/home/chisnova/data/records"

# -- 잡음 증강용 --------------------------------------------------------------
# 원본 오디오(MUSAN noise, openslr.org/17)는 아카이브 디스크에 두고, 거기서 뽑은
# 특징 레코드는 발화 레코드와 같이 RECORDS_ROOT 아래 둔다 — 학습이 매 배치 임의
# 구간을 읽으므로 원본 오디오가 아니라 이 레코드가 데이터 디스크(nvme)에 있어야 한다.
NOISE_DIR = "/home/chisnova/archive1/ASR_DATA/noise/musan/noise"       # 원본 오디오
NOISE_LIST = os.path.join(ASR_LIST_DIR, "noise.list")                  # 클립당 한 줄
NOISE_RECORD = os.path.join(RECORDS_ROOT, "musan_noise")               # 특징 레코드
