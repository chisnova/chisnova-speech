"""BatchMemoryCalibrator — 실제 모델로 학습 피크 메모리를 재서 BatchBudget 을 만든다."""
import numpy as np
import torch

from data.loader import BatchBudget


class BatchMemoryCalibrator:
    """모델을 올린 뒤 남은 GPU 메모리를 재고, 그만큼만 쓰는 배치 예산을 만든다.

    배치는 레코드를 읽기 전에 짜야 하므로 "배치 모양 -> 피크 바이트" 를 알아야 한다.
    그 관계를 이론으로 세우는 대신 실제 모델로 몇 번 재서 선형 적합한다:

        peak_bytes = btu_coef * (B*T*U) + bt_coef * (B*T) + base

      B*T*U  RNN-T joint (B, T', U, vocab) — 긴 발화에서 지배적
      B*T    인코더 활성값
      base   파라미터 + gradient + optimizer state (배치와 무관)

    T 는 원본 feat_len 이다. 인코더의 시간축 축소가 선형이라 계수가 흡수한다.

    프로브 모양이 중요하다 — 예산 배칭이 실제로 만드는 범위(짧은 발화면 B 크게,
    긴 발화면 B 작게)를 훑어야 한다. B 를 좁게만 재고 큰 배치로 외삽하면 메모리를
    크게 과소평가해 OOM 이 난다(실측: B 8~24 만 재고 B 90 을 예측하니 8GB 예측에
    실제 10.5GB). PROBE_SHAPE_RATIOS 가 그 범위를 정한다.
    """

    # (batch_size, feat_len) — 대략 같은 메모리를 쓰는 (짧고 큰) ~ (길고 작은) 조합.
    # word_len 은 manifest 에서 그 길이대의 실제 p95 로 채운다.
    PROBE_SHAPES = ((8, 1600), (16, 1000), (32, 600), (64, 400), (96, 300))

    # RNN-T 를 끄면 joint 가 빠져 같은 메모리로 배치를 훨씬 크게 잡을 수 있다.
    # 그 범위까지 재야 외삽이 안 생긴다.
    PROBE_SHAPES_NO_TRANSDUCER = ((16, 1600), (32, 1000), (64, 600), (128, 400),
                                  (256, 300))

    def __init__(self, model, loss_adapter, model_config, manifest=None, feat_dim=80,
                 phone_vocab=47, safety_ratio=0.75):
        self.model = model
        # loss_adapter(ret, target, phn_target, feat_mask, phone_mask, label_mask,
        #              model_config) -> 손실 스칼라.
        # forward 반환 튜플의 해석은 trainer 모듈의 compute_loss_from_outputs 가 갖는다.
        self.loss_adapter = loss_adapter
        self.model_config = model_config
        self.manifest = manifest
        self.feat_dim = feat_dim
        self.phone_vocab = phone_vocab
        self.safety_ratio = float(safety_ratio)

    def resolve_word_len(self, feat_len):
        """그 길이대 발화의 현실적인 word_len (없으면 대략적인 비율)."""
        if self.manifest is None:
            return max(2, feat_len // 16)
        feat_lens = np.asarray(self.manifest.feat_len)
        word_lens = np.asarray(self.manifest.word_len)
        nearby = (feat_lens >= feat_len * 0.9) & (feat_lens <= feat_len * 1.1)
        if not nearby.any():
            return max(2, feat_len // 16)
        return max(2, int(np.percentile(word_lens[nearby], 95)) + 1)

    def measure_peak_bytes(self, batch_size, feat_len, word_len, optimizer):
        """프로브 배치 하나로 forward+loss+backward+step 을 돌고 피크를 잰다."""
        phone_len = max(2, int(word_len * 2.5))     # 실측상 phone 은 word 의 2~3배
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        feat = torch.randn(batch_size, feat_len, self.feat_dim, device="cuda")
        word = torch.randint(4, self.model_config["vocab_size"],
                             (batch_size, word_len), device="cuda")
        phone = torch.randint(4, self.phone_vocab, (batch_size, phone_len), device="cuda")
        feat_mask = torch.ones(batch_size, feat_len, device="cuda")
        word_mask = torch.ones(batch_size, word_len, device="cuda")
        phone_mask = torch.ones(batch_size, phone_len, device="cuda")

        ret = self.model(word, feat, phone, dec_mask=word_mask, enc_mask=feat_mask,
                         p_dec_mask=phone_mask)
        loss = self.loss_adapter(ret, word, phone, feat_mask, phone_mask, word_mask,
                                 self.model_config)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated()

    def calibrate(self):
        """프로브를 돌려 BatchBudget 을 만든다. 학습 시작 전 한 번만 부른다."""
        was_training = self.model.training
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0)  # state 만 할당

        probe_shapes = (self.PROBE_SHAPES if getattr(self.model, "use_transducer", True)
                        else self.PROBE_SHAPES_NO_TRANSDUCER)
        shapes, peaks = [], []
        for batch_size, feat_len in probe_shapes:
            word_len = self.resolve_word_len(feat_len)
            # OOM 이면 배치를 반씩 줄여 재시도한다 — 그 길이대의 데이터 점을 아예
            # 잃으면 적합이 그 구간에서 외삽이 되어 버린다.
            while batch_size >= 2:
                try:
                    peaks.append(self.measure_peak_bytes(batch_size, feat_len, word_len,
                                                         optimizer))
                    shapes.append((batch_size, feat_len, word_len))
                    break
                except torch.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print("[calib] 프로브 T={} OOM — B={} 로 낮춰 재시도".format(
                        feat_len, batch_size))

        if len(shapes) < 3:
            raise RuntimeError(
                "메모리 프로브가 {}개만 성공했습니다 — 계수를 구할 수 없습니다. "
                "GPU 가 이 모델을 학습하기에 너무 작습니다".format(len(shapes)))

        btu_coef, bt_coef, base_bytes = self._fit_cost_model(shapes, peaks)

        del optimizer
        torch.cuda.empty_cache()
        if not was_training:
            self.model.eval()

        total_bytes = torch.cuda.get_device_properties(0).total_memory
        budget = BatchBudget(int(total_bytes * self.safety_ratio),
                             btu_coef, bt_coef, base_bytes,
                             max_utts=int(max(s[0] for s in shapes) * 1.5))
        self._print_report(shapes, peaks, budget, total_bytes)
        return budget

    def _fit_cost_model(self, shapes, peaks):
        """최소제곱 적합 후, 프로브 어디에서도 과소예측하지 않도록 계수를 부풀린다."""
        batch_sizes = np.array([s[0] for s in shapes], dtype=np.float64)
        feat_lens = np.array([s[1] for s in shapes], dtype=np.float64)
        word_lens = np.array([s[2] for s in shapes], dtype=np.float64)
        measured = np.array(peaks, dtype=np.float64)
        design = np.column_stack([batch_sizes * feat_lens * word_lens,
                                  batch_sizes * feat_lens,
                                  np.ones(len(shapes))])
        coefficients = self._solve_non_negative(design, measured)
        # 과소예측은 그대로 OOM 이므로, 가장 크게 빗나간 만큼 전체를 올려 둔다
        predicted = design @ coefficients
        safety_scale = float(np.max(measured / np.maximum(predicted, 1.0)))
        if safety_scale > 1.0:
            coefficients = coefficients * safety_scale
        return coefficients[0], coefficients[1], coefficients[2]

    @staticmethod
    def _solve_non_negative(design, measured):
        """계수가 모두 0 이상인 최소제곱 해.

        음수 계수는 "배치를 키우면 메모리가 준다" 는 뜻이라 물리적으로 말이 안 되고,
        그대로 두면 배치가 무한정 커진다. 그렇다고 푼 뒤에 음수만 0 으로 자르면
        나머지 계수가 그 몫을 떠안지 못해 적합이 통째로 망가진다(실측: 6.1GB 를
        17.7GB 로 예측). 음수가 나온 항을 빼고 남은 항으로 다시 푸는 것을
        음수가 없어질 때까지 반복한다.
        """
        active = np.ones(design.shape[1], dtype=bool)
        for _ in range(design.shape[1]):
            coefficients = np.zeros(design.shape[1])
            solved, *_ = np.linalg.lstsq(design[:, active], measured, rcond=None)
            coefficients[active] = solved
            if (solved >= 0).all():
                return coefficients
            active[np.where(active)[0][np.argmin(solved)]] = False
            if not active.any():
                break
        return np.maximum(coefficients, 0.0)

    def _print_report(self, shapes, peaks, budget, total_bytes):
        predicted = [budget.compute_cost(b, t, u) for b, t, u in shapes]
        errors = [abs(p - m) / m for p, m in zip(predicted, peaks)]
        print("[calib] 메모리 프로브 {}개 (예산 배칭이 만드는 범위를 훑는다)".format(len(shapes)))
        for (b, t, u), measured, pred in zip(shapes, peaks, predicted):
            print("        B={:3d} T={:5d} U={:3d} -> 실측 {:>8s} / 예측 {:>8s}".format(
                b, t, u, budget.format_bytes(measured), budget.format_bytes(pred)))
        print("[calib] 비용 모델 peak = {:.4e}*(B*T*U) + {:.4e}*(B*T) + {}  "
              "(평균오차 {:.1%})".format(budget.btu_coef, budget.bt_coef,
                                        budget.format_bytes(budget.base_bytes),
                                        sum(errors) / len(errors)))
        print("[calib] GPU {} 중 {:.0%} 사용 -> {}".format(
            budget.format_bytes(total_bytes), self.safety_ratio, budget.describe()))
