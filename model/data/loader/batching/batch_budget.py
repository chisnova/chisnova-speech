"""BatchBudget — 배치 하나가 넘지 말아야 할 학습 메모리 예산."""


class BatchBudget:
    """배치의 학습 피크 메모리를 카탈로그 길이만으로 예측하고 상한과 비교한다.

    배치는 레코드를 읽기 전에 짜야 하므로(길이 배열만 보고 짠다) 실제 텐서를
    만들어보고 재는 것은 불가능하다. 그래서 "배치 모양 -> 피크 바이트" 를
    선형 모델로 근사하고, 그 계수는 실제 모델로 몇 번 재서 구한다
    (trainer/batch_memory_calibrator.py 의 BatchMemoryCalibrator).

    비용 모델
        peak_bytes = btu_coef * (B * T * U) + bt_coef * (B * T) + base_bytes

      B*T*U 항  RNN-T joint (B, T', U, V) — 배치가 커질수록 지배적이다
      B*T 항    인코더 활성값 (레이어 수 x d_model 에 비례)
      base      파라미터 + gradient + optimizer state 등 배치와 무관한 고정분

    T 는 서브샘플링 전 원본 feat_len 이다 — 인코더의 시간축 축소가 선형이라
    계수에 흡수되므로, 이 클래스는 모델 구조를 몰라도 된다.

    실측 적합 결과 R^2 = 0.998, 평균오차 3.3% (Conformer-Transducer 39.9M 기준).
    """

    def __init__(self, limit_bytes, btu_coef, bt_coef, base_bytes, max_utts=None):
        self.limit_bytes = int(limit_bytes)
        if self.limit_bytes <= 0:
            raise ValueError("limit_bytes 는 양수여야 합니다: {!r}".format(limit_bytes))
        self.btu_coef = float(btu_coef)
        self.bt_coef = float(bt_coef)
        self.base_bytes = float(base_bytes)
        # 프로브로 재본 범위를 크게 벗어나면 선형 모델을 믿을 수 없다.
        # 아주 짧은 발화 구간에서 배치가 무한정 커지는 것을 막는 상한.
        self.max_utts = None if max_utts is None else int(max_utts)

    def compute_cost(self, num_utts, max_feat_len, max_word_len):
        """발화 num_utts 개를 그 길이로 패딩했을 때의 예상 피크 바이트.

        스칼라와 numpy 배열을 모두 받는다 — 배치를 채워가며 스칼라로 부르기도 하고,
        다 짠 배치들의 비용을 배열로 한 번에 구하기도 한다.
        """
        return (self.btu_coef * (num_utts * max_feat_len * max_word_len)
                + self.bt_coef * (num_utts * max_feat_len)
                + self.base_bytes)

    def fits(self, num_utts, max_feat_len, max_word_len):
        if self.max_utts is not None and num_utts > self.max_utts:
            return False
        return self.compute_cost(num_utts, max_feat_len, max_word_len) <= self.limit_bytes

    @staticmethod
    def format_bytes(num_bytes):
        for unit, divisor in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
            if abs(num_bytes) >= divisor:
                return "{:.2f}{}".format(num_bytes / divisor, unit)
        return "{:.0f}B".format(num_bytes)

    def describe(self):
        capped = "" if self.max_utts is None else " (B<={})".format(self.max_utts)
        return "peak<={}{}".format(self.format_bytes(self.limit_bytes), capped)

    def __repr__(self):
        return ("BatchBudget({}, btu_coef={:.4e}, bt_coef={:.4e}, base={})".format(
            self.describe(), self.btu_coef, self.bt_coef,
            self.format_bytes(self.base_bytes)))
