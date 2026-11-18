"""log-mel filterbank 특징 추출기."""
import librosa
import numpy as np
import soundfile
import torch


class FbankFeatureExtractor:
    """wav / 파형 -> (T, 80) log-mel dB 특징.

    파라미터는 data/generation/record/data_gen.py 의 getFbankFeature 와 반드시
    같아야 한다 — 학습 특징과 한 군데라도 어긋나면 추론 결과가 무너진다.
    정규화(CMVN)는 하지 않는다: 학습 로더도 기본 파이프라인에 Cmvn 을 넣지 않아
    모델이 raw log-mel dB 분포를 그대로 보고 학습했기 때문이다.
    """

    def __init__(self, sample_rate=16000, num_mel_bins=80, fft_size=400,
                 window_length=400, hop_length=160, preemphasis_coefficient=0.97,
                 top_db=80.0):
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.fft_size = fft_size
        self.window_length = window_length
        self.hop_length = hop_length
        self.preemphasis_coefficient = preemphasis_coefficient
        # librosa.power_to_db 의 기본값 80 을 그대로 둔다 — data_gen.py 가 학습
        # 특징을 그렇게 저장했으므로 추론도 같아야 한다. 잡음 레코드를 만들 때만
        # None 으로 꺼서 바닥 클리핑 없이 뽑는다.
        self.top_db = top_db

    def apply_preemphasis(self, waveform):
        """y[n] = x[n] - 0.97*x[n-1]"""
        waveform = np.asarray(waveform, dtype=np.float32)
        emphasized = waveform.copy()
        emphasized[1:] -= self.preemphasis_coefficient * waveform[:-1]
        return emphasized

    def load_waveform(self, audio_path):
        """오디오 파일 -> 16kHz mono float32 파형.

        soundfile(libsndfile) 로 먼저 읽는다 — librosa.load 는 mp3 를 audioread
        백엔드로 넘기는데, 그게 없는 환경에서는 NoBackendError 로 죽는다.
        libsndfile 1.1+ 는 mp3 를 직접 디코딩한다.
        """
        try:
            waveform, source_sample_rate = soundfile.read(audio_path, dtype="float32",
                                                          always_2d=True)
            waveform = waveform.mean(axis=1)            # 다채널이면 mono 로 합친다
            if source_sample_rate != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=source_sample_rate,
                                            target_sr=self.sample_rate)
            return waveform
        except Exception as soundfile_error:
            print("[WARN] soundfile 로 못 읽어 librosa 로 재시도: {} ({})".format(
                audio_path, soundfile_error))
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return waveform

    def extract_from_waveform(self, waveform):
        """파형 (N,) -> 특징 (T, 80) float32 텐서."""
        emphasized = self.apply_preemphasis(waveform)
        mel_spectrogram = librosa.feature.melspectrogram(y=emphasized,
                                                         sr=self.sample_rate,
                                                         n_mels=self.num_mel_bins,
                                                         n_fft=self.fft_size,
                                                         hop_length=self.hop_length,
                                                         win_length=self.window_length)
        feature = librosa.power_to_db(mel_spectrogram, top_db=self.top_db)
        return torch.tensor(np.transpose(feature), dtype=torch.float32)

    def extract_from_wav_file(self, wav_path):
        return self.extract_from_waveform(self.load_waveform(wav_path))

    def extract_silence(self, duration_ms=300):
        """발화 끝에 붙이는 무음 특징 — 마지막 음성 프레임까지 인코더가 보게 한다."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return self.extract_from_waveform(np.zeros(num_samples, dtype=np.float32))

    def samples_per_chunk(self, chunk_duration_ms):
        return int(self.sample_rate * chunk_duration_ms / 1000)

    def frames_per_chunk(self, chunk_duration_ms):
        """chunk_duration_ms 에 해당하는 특징 프레임 수 (hop 기준)."""
        return max(1, int(self.samples_per_chunk(chunk_duration_ms) / self.hop_length))
