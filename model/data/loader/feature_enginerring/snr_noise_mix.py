"""SnrNoiseMix — 잡음을 목표 SNR 로 섞는다 (PMC 의 log-add)."""
import math

import numpy as np
import torch

from .feature_transform import FeatureTransform, get_padding_row_mask


class SnrNoiseMix(FeatureTransform):
    """log-mel dB 특징에 잡음 특징을 목표 SNR 로 결합한다.

    왜 로그끼리 더하면 안 되는가
      멜 필터뱅크는 파워에 대한 선형 사상이라 가산 잡음은 파워 도메인에서 더해진다:
      S_y = S_x + S_n. 그런데 저장된 특징은 이미 로그(dB)라 L_x + L_n 은 아무 의미가
      없다. 파워로 되돌려 더하고 다시 로그를 씌워야 하며, 그게 log-add 다:

          L_y = L_x + log(1 + exp(L_n - L_x))        (PMC / VTS 계열의 mismatch function)

      구현은 exp/log 를 직접 쓰지 않고 torch.logaddexp 로 한다 — 큰 쪽을 뽑아내고
      계산해 언더/오버플로가 없다. dB 는 10*log10 이므로 자연로그와의 환산
      상수(ln10/10)만 앞뒤로 곱해 준다.

    빠지는 항
      정확히는 교차항 2*sqrt(S_x*S_n)*cos(theta) 가 있다. 위상이 랜덤이라
      기대값에서는 0 이지만 프레임마다 +-2*sqrt(S_x*S_n) 범위로 요동한다.
      phase_jitter_db 로 그 요동을 프레임별 이득 지터로 흉내낸다(1차 근사).

    잡음은 어디에 있나
      LMDB 레코드(noise_record_gen.py 산출물)에 있고 배치마다 필요한 구간만
      읽는다. 6.2시간 분량이 342MB 라 통째로 GPU 에 올릴 수도 있지만 그럴 이유가
      없다 — 한 배치가 실제로 쓰는 건 (B,T,80) 뿐이고, 나머지는 mmap 과 페이지
      캐시에 맡기면 GPU 메모리를 전혀 쓰지 않는다.

    인자
      noise_conn           : NoiseLmdbConn — 잡음 레코드 커넥션. 배치마다 필요한 구간만
                             온라인으로 읽는다(전체를 메모리/GPU 에 올리지 않는다)
      snr_range            : 발화별로 뽑을 SNR(dB) 범위
      noise_count_range    : 한 발화에 겹칠 잡음 개수 범위. 현실의 잡음은 하나가
                             아니라 여러 소리가 동시에 나므로 여러 개를 파워
                             도메인에서 더한다. 개수가 늘수록 스펙트럼이 평탄해져
                             babble 에 가까워진다
      clip_level_spread_db : 겹치는 클립들의 상대 레벨 차이 (0 이면 전부 같은 비중)
      phase_jitter_db      : 프레임별 이득 지터 폭 (0 이면 끈다)
      prob            : 발화별 적용 확률 (배치 단위가 아니다 — 한 배치에 깨끗한
                        발화와 섞인 발화가 같이 있어야 모델이 둘 다 본다)

    주의: 증강이므로 train 로더에만 건다. Cmvn 을 쓴다면 반드시 이것보다 뒤에
    와야 한다 — 정규화된 값은 더 이상 dB 가 아니라 파워로 되돌릴 수 없다.
    """

    _PARAMS = ("snr_range", "noise_count_range", "clip_level_spread_db",
               "phase_jitter_db", "prob")

    DB_TO_NATURAL_LOG = math.log(10.0) / 10.0       # dB -> ln(power)
    NATURAL_LOG_TO_DB = 10.0 / math.log(10.0)       # ln(power) -> dB

    def __init__(self, noise_conn, snr_range=(0.0, 20.0), noise_count_range=(1, 5),
                 clip_level_spread_db=6.0, phase_jitter_db=3.0, prob=0.5):
        self.noise_conn = noise_conn
        self.snr_range = (float(snr_range[0]), float(snr_range[1]))
        self.noise_count_range = (int(noise_count_range[0]), int(noise_count_range[1]))
        self.clip_level_spread_db = float(clip_level_spread_db)
        self.phase_jitter_db = float(phase_jitter_db)
        self.prob = float(prob)

    @classmethod
    def from_record(cls, record_dir, **kwargs):
        """noise_record_gen.py 가 만든 레코드 디렉토리로 만든다."""
        from ..database import NoiseLmdbConn

        noise_conn = NoiseLmdbConn(record_dir)
        print("[SnrNoiseMix] 잡음 레코드 {} — 클립 {} 개 / {:,} 프레임 ({:.1f} 시간), "
              "배치마다 필요한 구간만 읽는다".format(
                  record_dir, noise_conn.num_clips, noise_conn.total_frames,
                  noise_conn.total_frames / noise_conn.frame_rate / 3600))
        return cls(noise_conn, **kwargs)

    def sample_noise_counts(self, batch_size):
        """발화마다 겹쳐 쌓을 잡음 개수 (noise_count_range 안에서 균등)."""
        return np.random.randint(self.noise_count_range[0], self.noise_count_range[1] + 1,
                                 size=batch_size)

    def read_noise_segments(self, segment_count, num_frames, device, dtype):
        """(segment_count, T, D) 잡음 dB — 행마다 임의 클립의 임의 구간을 LMDB 에서 읽는다.

        읽기는 CPU 에서 하고 완성된 덩어리만 한 번에 올린다. 클립 전체가 아니라
        구간만 복사하므로 페이지 캐시에 올라간 뒤에는 memcpy 비용뿐이다.
        """
        segments = np.empty((segment_count, num_frames, self.noise_conn.feat_dim),
                            dtype=self.noise_conn.dtype)
        clip_indexes = np.random.randint(self.noise_conn.num_clips, size=segment_count)
        for segment_index, clip_index in enumerate(clip_indexes):
            start_frame = np.random.randint(int(self.noise_conn.frame_counts[clip_index]))
            segments[segment_index] = self.noise_conn.read_segment(clip_index, start_frame,
                                                                   num_frames)
        return torch.from_numpy(segments).to(device=device, dtype=dtype, non_blocking=True)

    def sum_noise_clips_in_power(self, noise_counts, num_frames, valid, device, dtype):
        """발화별로 잡음 여러 개를 파워 도메인에서 더한 ln(power) (S, T, D).

        여기서도 로그끼리 더하면 안 되는 건 마찬가지다 — 잡음끼리도 파워가 더해지므로
        logaddexp 로 누적한다. 슬롯(0..max-1)을 돌면서 그 슬롯이 필요한 발화만
        읽어 쌓으므로, 잡음 1개짜리 발화 몫까지 5번 읽는 낭비가 없다.

        클립마다 자기 총 파워로 먼저 정규화한다. MUSAN 은 클립별 절대 레벨이
        -100~+22dB 로 제각각이라, 그대로 더하면 큰 놈 하나가 나머지를 덮어 잡음을
        몇 개 겹치든 1개와 같아진다. 정규화한 뒤 clip_level_spread_db 만큼 무작위로
        낮춰 서로 다른 비중을 준다. 전체 크기는 어차피 뒤에서 SNR 이 정하므로
        여기서의 절대 스케일은 의미가 없다.
        """
        summed_natural_log = torch.full((len(noise_counts), num_frames,
                                         self.noise_conn.feat_dim),
                                        float("-inf"), device=device, dtype=dtype)
        for slot_index in range(int(noise_counts.max())):
            rows = np.nonzero(noise_counts > slot_index)[0]
            row_index = torch.as_tensor(rows, device=device)

            clip_natural_log = self.read_noise_segments(len(rows), num_frames,
                                                        device, dtype) * self.DB_TO_NATURAL_LOG
            row_valid = valid.index_select(0, row_index)
            clip_natural_log = clip_natural_log - self.masked_total_power_log(clip_natural_log,
                                                                             row_valid)
            if self.clip_level_spread_db > 0.0:
                level_offset_db = torch.empty(len(rows), 1, 1, device=device, dtype=dtype)
                level_offset_db.uniform_(-self.clip_level_spread_db, 0.0)
                clip_natural_log = clip_natural_log + level_offset_db * self.DB_TO_NATURAL_LOG

            summed_natural_log[row_index] = torch.logaddexp(
                summed_natural_log.index_select(0, row_index), clip_natural_log)
        return summed_natural_log

    def __call__(self, feat):
        if feat.dim() != 3:
            raise ValueError("feat 은 (B,T,D) 여야 합니다: {}".format(tuple(feat.shape)))
        if self.prob <= 0.0:
            return feat

        num_frames = feat.size(1)
        # 섞을 발화를 먼저 고른다 — 버릴 발화 몫까지 LMDB 에서 읽어 올리는 것이
        # 이 변환에서 제일 비싼 부분이라, 고르고 나서 그만큼만 읽는다.
        selected_utterances = torch.nonzero(torch.rand(feat.size(0)) < self.prob,
                                            as_tuple=True)[0]
        if selected_utterances.numel() == 0:
            return feat
        selected_utterances = selected_utterances.to(feat.device)

        speech_db = feat.index_select(0, selected_utterances)           # (S,T,D)
        batch_size = speech_db.size(0)
        valid = (~get_padding_row_mask(speech_db)).unsqueeze(-1)        # (S,T,1) bool

        speech_natural_log = speech_db * self.DB_TO_NATURAL_LOG
        noise_natural_log = self.sum_noise_clips_in_power(
            self.sample_noise_counts(batch_size), num_frames, valid, feat.device, feat.dtype)

        # SNR 은 발화 전체 파워 비로 정의한다 — 패딩 프레임은 양쪽 다 빼고 센다
        # (패딩은 0dB 로 채워져 있어 그냥 세면 파워 1.0 인 프레임이 끼어든다).
        speech_power_log = self.masked_total_power_log(speech_natural_log, valid)
        noise_power_log = self.masked_total_power_log(noise_natural_log, valid)
        target_snr_db = torch.empty(batch_size, 1, 1, device=feat.device, dtype=feat.dtype)
        target_snr_db.uniform_(*self.snr_range)
        noise_gain_log = (speech_power_log - noise_power_log
                          - target_snr_db * self.DB_TO_NATURAL_LOG)

        if self.phase_jitter_db > 0.0:
            # 교차항의 프레임별 요동 근사 — 이득을 프레임마다 조금씩 흔든다
            jitter_db = torch.empty(batch_size, num_frames, 1, device=feat.device,
                                    dtype=feat.dtype)
            jitter_db.uniform_(-self.phase_jitter_db, self.phase_jitter_db)
            noise_gain_log = noise_gain_log + jitter_db * self.DB_TO_NATURAL_LOG

        mixed = torch.logaddexp(speech_natural_log,
                                noise_natural_log + noise_gain_log) * self.NATURAL_LOG_TO_DB

        # 패딩 프레임은 0 으로 되돌린 뒤 고른 발화 자리에만 되꽂는다
        return feat.index_copy_(0, selected_utterances, torch.where(valid, mixed, speech_db))

    @staticmethod
    def masked_total_power_log(natural_log_power, valid):
        """유효 프레임만 모은 ln(총 파워) (B,1,1). logsumexp 라 지수화가 없다."""
        masked = natural_log_power.masked_fill(~valid, float("-inf"))
        return torch.logsumexp(masked.flatten(1), dim=1).view(-1, 1, 1)
