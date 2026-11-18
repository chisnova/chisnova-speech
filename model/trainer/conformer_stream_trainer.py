import gc
import json
import math
import os
import re
import shutil
import time
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F


ce_loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
ctc_loss_func = nn.CTCLoss(zero_infinity=True, reduction='none', blank=1)

LOSS_LOG_FORMAT = ("step : {step} , {tag} loss: {total}, wrd_loss:{wrd} phn_loss: {phn} "
                   "ctc1_loss: {ctc1} ctc2_loss: {ctc2}")
# 학습 안정성 지표 — 손실만 봐서는 안 보이는 것들을 같은 줄에 덧붙인다.
#   gnorm  클리핑 전 gradient norm 평균. max_norm 을 계속 넘으면 실효 학습률이
#          0.5/|g| 로 제멋대로 스케일링돼 손실이 정체한다
#   clip   그 구간에서 클리핑이 걸린 스텝 비율(%)
#   lr     현재 학습률 — 모듈별 오버라이드가 있으면 대표(최대) 값
#   upd    |Δw|/|w| — 0.001 근처가 건강하다. 1e-5 이하면 사실상 멈춘 것
#   lrs    (모듈별 LR 그룹이 2개 이상일 때만) group 별 LR 목록
STABILITY_LOG_FORMAT = (" | gnorm: {gnorm:.4f} clip: {clip:.0f}%@{max_norm:g} "
                        "lr: {lr:.3e} upd: {upd}{lr_groups}")


def calc_conv_subsampled_length(input_length):
    """kernel 3 / stride 2 conv 하나를 지난 뒤의 시간축 길이."""
    return torch.div(input_length - 3, 2, rounding_mode='floor') + 1


def print_with_timestamp(message):
    tm = time.localtime()
    print("{} : {}/{}/{} {}:{}:{}".format(message, tm.tm_year, tm.tm_mon, tm.tm_mday,
                                          tm.tm_hour, tm.tm_min, tm.tm_sec))


def format_loss_log(step, tag, loss_sums, loss_count, stability=None):
    """누적 손실 텐서 5개(total/wrd/phn/ctc1/ctc2)를 한 줄 로그로 만든다."""
    denominator = max(1, loss_count)
    total, wrd, phn, ctc1, ctc2 = (float(loss_sum) / denominator for loss_sum in loss_sums)
    line = LOSS_LOG_FORMAT.format(step=step, tag=tag, total=total, wrd=wrd, phn=phn,
                                  ctc1=ctc1, ctc2=ctc2)
    if stability:
        line += STABILITY_LOG_FORMAT.format(**stability)
    return line


def compute_ctc_loss(ctc_out, ctc_target, feat_mask, target_mask, time_reduction=1):
    ctc_out = torch.transpose(ctc_out, 0, 1)
    max_target_len = ctc_target.size(1)
    max_input_len = ctc_out.size(0)
    ctc_out = F.log_softmax(ctc_out, dim=-1)

    input_lengths = torch.sum(feat_mask.type(torch.int32), -1).type(torch.long)
    input_lengths = calc_conv_subsampled_length(input_lengths)
    input_lengths = calc_conv_subsampled_length(input_lengths)
    if time_reduction > 1:
        # ConformerEncoder 의 avg_pool(ceil_mode=True) 과 동일한 길이 계산
        input_lengths = torch.div(input_lengths + time_reduction - 1, time_reduction,
                                  rounding_mode='floor')
    length_overflow = input_lengths.max() - max_input_len
    if length_overflow != 0:
        input_lengths = input_lengths - length_overflow
    input_lengths = input_lengths.clamp(min=1, max=max_input_len)

    target_lengths = (torch.sum(target_mask, -1).type(torch.long) - 1).clamp(min=1, max=max_target_len)

    ctc_loss = ctc_loss_func(ctc_out, ctc_target, input_lengths, target_lengths)
    return (ctc_loss / target_lengths.float()).mean()


def compute_ce_loss(output, target, vocab_size, label_mask):
    batch_size, max_target_len, _ = output.size()
    ce_output = output.contiguous().view(-1, vocab_size).float()
    ce_target = target.contiguous().view(-1).long()

    target_lengths = (torch.sum(label_mask, -1).type(torch.int32) - 1).clamp(min=1, max=target.size(1))

    ce_loss = ce_loss_func(ce_output, ce_target)
    ce_loss = ce_loss.view(batch_size, max_target_len).sum(dim=-1) / target_lengths.float()
    return ce_loss.mean()


def compute_random_baselines(w_out, p_out, ctc1, ctc2, ctc_mid, target,
                             feat_mask, label_mask, ctc_time_reduction):
    """항별 "무작위 기준선" — 아무것도 모르는 모델이 받는 손실.

    CE 는 균등분포의 CE 인 ln(클래스 수)로 닫힌 식으로 정확하다. CTC 는 프레임
    T개 전부의 경로 확률을 곱하므로 기준선이 ~(T/U)·lnV 스케일이고 T/U 가
    배치마다 달라, zero-logit 텐서로 같은 손실 함수를 돌려 정확한 값을 쓴다
    (backward 없음 — 스텝 비용의 ~1%).
    """
    with torch.no_grad():
        baselines = [math.log(w_out.size(-1)), math.log(p_out.size(-1)),
                     float(compute_ctc_loss(torch.zeros_like(ctc1), target,
                                            feat_mask, label_mask)),
                     float(compute_ctc_loss(torch.zeros_like(ctc2), target, feat_mask,
                                            label_mask, time_reduction=ctc_time_reduction))]
        if ctc_mid is not None:
            baselines.append(float(compute_ctc_loss(torch.zeros_like(ctc_mid), target,
                                                    feat_mask, label_mask)))
    return baselines


def loss_function(ctc1, ctc2, target, phn_target, w_out, p_out, feat_mask, phone_mask, label_mask,
                  ctc_time_reduction=1, ctc_mid=None):
    """AsrStreamConformer 용 — CTC 2개 + CE 2개 (+ 선택적 intermediate CTC).

    intermediate CTC (Lee & Watanabe 2021): 인코더 중간층 출력에 같은 CTC 를 걸어
    중간층에 gradient 경로를 단축해 꽂는다.

    가중 방식: 각 항을 자기 무작위 기준선으로 나눈 무차원 진행률 L̃ 로 만든 뒤
    자기 비중(w = L̃/ΣL̃)으로 가중한다. 기준선을 안 나누면 phone CE(47클래스,
    ln=3.85)가 word CE(1024, ln=6.93)보다 자가 짧아 구조적으로 저평가된다 —
    trainer/test/loss_scale_test.py 가 이 정규화를 검증한다.
    """
    ctc1_loss = compute_ctc_loss(ctc1, target, feat_mask, label_mask)
    ctc2_loss = compute_ctc_loss(ctc2, target, feat_mask, label_mask, time_reduction=ctc_time_reduction)
    wrd_ce_loss = compute_ce_loss(w_out, target, w_out.size(-1), label_mask)
    phn_ce_loss = compute_ce_loss(p_out, phn_target[:, 1:], p_out.size(-1), phone_mask)
    terms = [wrd_ce_loss, phn_ce_loss, ctc1_loss, ctc2_loss]
    if ctc_mid is not None:
        terms.append(compute_ctc_loss(ctc_mid, target, feat_mask, label_mask))

    baselines = compute_random_baselines(w_out, p_out, ctc1, ctc2, ctc_mid, target,
                                         feat_mask, label_mask, ctc_time_reduction)
    normalized = [term / max(1e-8, baseline) for term, baseline in zip(terms, baselines)]
    loss_sum = sum(term.detach() for term in normalized)
    weights = [term.detach() / (loss_sum + 1e-8) for term in normalized]
    loss = sum(weight * term for weight, term in zip(weights, normalized))
    # 표시용 total 은 원 스케일로 — 정규화 손실(진행률 단위, ~0.1)을 그대로 찍으면
    # 개별 항(1~2)과 눈금이 어긋나 그래프가 절벽처럼 보인다. 같은 가중치를
    # 원 스케일 항에 적용한 값이라 "실효 손실" 의미는 유지된다.
    display_total = sum(weight * term.detach() for weight, term in zip(weights, terms))
    return loss, display_total, wrd_ce_loss, phn_ce_loss, ctc1_loss, ctc2_loss


def compute_loss_from_outputs(ret, target, phn_target, feat_mask, phone_mask,
                              label_mask, model_config):
    """모델 출력 튜플 -> 손실 스칼라(backward 용). 배치 메모리 캘리브레이션이 이 계약을 쓴다."""
    w_out, p_out, ctc1, ctc2, ctc_mid = ret
    return loss_function(ctc1, ctc2, target[:, 1:].to(torch.int32), phn_target,
                         w_out.float(), p_out.float(), feat_mask, phone_mask, label_mask,
                         ctc_time_reduction=model_config.get("ctc_time_reduction", 1),
                         ctc_mid=None if ctc_mid is None else ctc_mid.float())[0]


class ConformerStreamTrainer():
    SAVE_INTERVAL_BATCHES = 300
    ROLLING_CHECKPOINT_COUNT = 5
    DEFAULT_GRAD_CLIP_MAX_NORM = 5.0
    UPDATE_PROBE_PARAM_COUNT = 3      # |Δw|/|w| 를 잴 대표 파라미터 수

    # tr_method / freeze_num 은 쓰지 않지만 train_main.buildTrainer 가 모든 trainer 를
    # 같은 위치 인자로 생성하므로 시그니처를 맞춰 둔다.
    def __init__(self, model, train_data, valid_data, optimizer, scheduler, save_path,
                 vocab_size, tr_method, freeze_num):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.vocab_size = vocab_size
        self.checkpoint_counter = 0
        self.best_interim_dev_loss = float("inf")
        self.interim_stall_count = 0

        # 로드맵 옵션 (config.json 의 model 섹션에서 읽음, 없으면 기본값)
        self.ctc_time_reduction = 1
        self.num_data_workers = 4
        # 클리핑은 "가끔 튀는 스텝만 자르는" 장치다. 임계가 낮아 매 스텝 걸리면
        # 스텝 간 gradient 크기 차이가 통째로 눌린다(실측: 0.5 에서 클리핑 100%).
        self.grad_clip_max_norm = self.DEFAULT_GRAD_CLIP_MAX_NORM
        self.dev_check_ratio = 0.05
        self.dev_check_max_batches = 100
        self.adaptive_lr_factor = 0.6
        self.adaptive_lr_patience = 2
        self.adaptive_lr_min = 1e-5
        self.adaptive_lr_threshold = 0.002
        self.module_lr_overrides = {}
        self.best_interim_dev_loss = float("inf")
        self.interim_stall_count = 0
        try:
            with open(os.path.join(save_path, "config.json"), 'r') as f:
                model_config = json.load(f).get("model", {})
            self.ctc_time_reduction = int(model_config.get("ctc_time_reduction", 1))
            self.num_data_workers = int(model_config.get("num_data_workers", 4))
            self.grad_clip_max_norm = float(model_config.get(
                "grad_clip_max_norm", self.DEFAULT_GRAD_CLIP_MAX_NORM))
            # 에폭 중간 dev 점검 — 에폭(12시간+)을 기다리지 않고 과적합을 본다.
            # ratio 는 학습 배치 수 대비 간격(0.05 = 5% 마다), max_batches 는 한 번에
            # 도는 dev 배치 상한 (전체 378 중 100이면 ~5,400 발화, 1분 미만).
            self.dev_check_ratio = float(model_config.get("dev_check_ratio", 0.05))
            self.dev_check_max_batches = int(model_config.get("dev_check_max_batches", 100))
            # 적응형 LR — 에폭(12시간+)을 기다리지 않고 중간 dev 점검(40분마다)을
            # 신호로 LR 을 깎는다. patience 회 연속으로 dev 가 안 좋아지면 factor 배.
            self.adaptive_lr_factor = float(model_config.get("adaptive_lr_factor", 0.6))
            self.adaptive_lr_patience = int(model_config.get("adaptive_lr_patience", 2))
            self.adaptive_lr_min = float(model_config.get("adaptive_lr_min", 1e-5))
            self.adaptive_lr_threshold = float(model_config.get("adaptive_lr_threshold", 0.002))
            # 모듈별 LR 오버라이드 — {"최상위 모듈명": lr}. 예:
            #   {"stream_encoder": 1e-5, "ctc_linear1": 1e-5}
            # 이면 1패스는 느리게 묶고 나머지는 기본 lr 로 학습한다 (이미 조여진
            # 1패스를 보존한 채 2패스만 빠르게 재적응시키는 용도).
            self.module_lr_overrides = {
                module_name: float(lr_value)
                for module_name, lr_value
                in model_config.get("module_lr_overrides", {}).items()}
        except Exception as e:
            print("[WARN] failed to read optional config keys:", e)
        if self.module_lr_overrides:
            self.apply_module_lr_overrides()

    def setModel(self, model):
        """train_main 이 에폭마다 현재 모델을 다시 꽂아 준다."""
        self.model = model

    def apply_module_lr_overrides(self):
        """최상위 모듈별로 다른 LR 을 쓰도록 optimizer param group 을 재구성한다.

        train_main 은 전체 파라미터를 단일 group 으로 만들어 넘긴다. 파라미터
        이름의 최상위 모듈(stream_encoder, ctc_linear1, ...)이 오버라이드에
        걸리면 그 LR 의 group 으로 갈라 낸다. optimizer 생성 직후(모멘트가
        비어 있을 때) 한 번만 불러야 한다 — state 가 쌓인 뒤에 group 을
        재구성하면 param-state 연결은 유지되지만 의도가 흐려진다.
        """
        base_lr = self.optimizer.defaults["lr"]
        parameters_by_lr = {}
        for parameter_name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            top_module_name = parameter_name.split(".")[0]
            group_lr = self.module_lr_overrides.get(top_module_name, base_lr)
            parameters_by_lr.setdefault(group_lr, []).append(param)
        self.optimizer.param_groups.clear()
        for group_lr in sorted(parameters_by_lr, reverse=True):   # 대표(최대) LR 이 앞에
            self.optimizer.add_param_group({"params": parameters_by_lr[group_lr],
                                            "lr": group_lr})
        print("[module_lr] overrides {} -> lr groups: {}".format(
            self.module_lr_overrides, self.format_group_learning_rates()))

    def format_group_learning_rates(self):
        """group 별 LR 을 "5.000e-04/1.000e-05" 꼴 문자열로 만든다."""
        return "/".join("{:.3e}".format(param_group["lr"])
                        for param_group in self.optimizer.param_groups)

    GRAD_GROUP_LAYER_PATTERN = re.compile(
        r"^(stream_encoder|second_encoder|wrd_decoder|phn_decoder)\.layers\.(\d+)\.")
    GRAD_GROUP_PREFIXES = {"stream_encoder": "enc1", "second_encoder": "enc2",
                           "wrd_decoder": "dec.wrd", "phn_decoder": "dec.phn"}

    def resolve_grad_group(self, parameter_name):
        """파라미터 이름 -> 모듈 그룹 라벨 (히트맵의 행이 된다).

        레이어 단위로 갈라야 "몇 번째 레이어부터 gradient 가 죽는지" 가 보인다.
        모듈 전체로 뭉치면 vanishing 이 평균에 묻힌다.
        """
        layer_match = self.GRAD_GROUP_LAYER_PATTERN.match(parameter_name)
        if layer_match:
            return "{}.L{:02d}".format(self.GRAD_GROUP_PREFIXES[layer_match.group(1)],
                                       int(layer_match.group(2)))
        top_module = parameter_name.split(".")[0]
        if top_module == "stream_encoder":
            return "enc1.front"                    # subsample/linear/norm
        if top_module == "second_encoder":
            return "enc2.front"
        if top_module in ("wrd_decoder", "phn_decoder"):
            return self.GRAD_GROUP_PREFIXES[top_module] + ".emb"
        if top_module.startswith("ctc_linear"):
            return "head.ctc" + top_module[-1]
        return "기타"

    def measure_group_grad_norms(self):
        """그룹별 (gradient norm, weight norm) — backward 직후, step 전에 부른다.

        파라미터마다 .item() 하면 947번 동기화라, 그룹별 제곱합을 텐서로 모은 뒤
        마지막에 그룹 수(~35)만큼만 동기화한다. 로그 스텝에서만 부르므로 비용은
        스텝의 1% 미만이다.
        """
        squared_sums = {}
        for parameter_name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            group = self.resolve_grad_group(parameter_name)
            entry = squared_sums.setdefault(group, [None, None])
            grad_squared = param.grad.detach().norm() ** 2
            weight_squared = param.detach().norm() ** 2
            entry[0] = grad_squared if entry[0] is None else entry[0] + grad_squared
            entry[1] = weight_squared if entry[1] is None else entry[1] + weight_squared
        return {group: {"g": round(float(grad_sq) ** 0.5, 6),
                        "w": round(float(weight_sq) ** 0.5, 6)}
                for group, (grad_sq, weight_sq) in squared_sums.items()}

    def append_grad_norm_record(self, batch_cnt, groups):
        record = {"step": batch_cnt,
                  "at": datetime.now().replace(microsecond=0).isoformat(),
                  "max_norm": self.grad_clip_max_norm, "groups": groups}
        with open(os.path.join(self.save_path, "grad_norms.jsonl"), "a") as norm_file:
            norm_file.write(json.dumps(record) + "\n")

    def select_update_probe_params(self):
        """|Δw|/|w| 를 잴 대표 파라미터 몇 개.

        전 파라미터를 복사해 재면 168MB 를 매번 복사해야 한다. 모델 전반에 걸쳐
        고르게 몇 개만 뽑아도 업데이트 크기의 자릿수는 그대로 보인다.
        """
        weight_params = [param for param in self.model.parameters()
                         if param.requires_grad and param.dim() > 1]
        if not weight_params:
            return []
        stride = max(1, len(weight_params) // self.UPDATE_PROBE_PARAM_COUNT)
        return weight_params[::stride][:self.UPDATE_PROBE_PARAM_COUNT]

    def measure_update_ratio(self, probe_params, previous_values):
        """스텝 전후의 |Δw| / |w|."""
        update_norm = sum(float((param.detach() - previous).norm() ** 2)
                          for param, previous in zip(probe_params, previous_values)) ** 0.5
        weight_norm = sum(float(previous.norm() ** 2) for previous in previous_values) ** 0.5
        return update_norm / max(1e-12, weight_norm)

    def saveEpochModel(self, epoch_num, tr_loss, dev_loss):
        tm = time.localtime()
        record_time = "{:04d}{:02d}{:02d}T{:02d}:{:02d}".format(tm.tm_year, tm.tm_mon, tm.tm_mday,
                                                                tm.tm_hour, tm.tm_min)
        save_name = f"asr_model_{record_time}_epoch{epoch_num}_tr:{tr_loss:.3f}_dev:{dev_loss:.3f}.pt"
        torch.save(self.model.state_dict(), os.path.join(self.save_path, save_name))
        print("save: {}".format(save_name))

    def save_rolling_checkpoint(self):
        """asr_model.pt 를 쓰고 asr_model{0..4}.pt 로 돌려 가며 백업한다."""
        latest_path = os.path.join(self.save_path, "asr_model.pt")
        torch.save(self.model.state_dict(), latest_path)
        backup_index = self.checkpoint_counter % self.ROLLING_CHECKPOINT_COUNT
        shutil.copyfile(latest_path, os.path.join(self.save_path,
                                                  "asr_model{}.pt".format(backup_index)))
        self.checkpoint_counter += 1

    def compute_step_losses(self, feat, label, phone, feat_mask, lab_mask, phone_mask):
        """forward + 손실 (train/dev 공용)."""
        w_out, p_out, ctc1, ctc2, ctc_mid = self.model(label, feat, phone, dec_mask=lab_mask,
                                                       enc_mask=feat_mask,
                                                       p_dec_mask=phone_mask)
        return loss_function(
            ctc1, ctc2, label[:, 1:].to(torch.int32), phone,
            w_out.float(), p_out.float(), feat_mask, phone_mask, lab_mask,
            ctc_time_reduction=self.ctc_time_reduction,
            ctc_mid=None if ctc_mid is None else ctc_mid.float())

    def train(self, epoch_num):
        self.model.train()
        self.train_data.reshuffle_for_next_epoch()

        epoch_loss_sums = [0.0] * 5
        batch_cnt = 0
        loss_cnt = 0
        grad_norm_sum, clipped_step_count, stability_count = 0.0, 0, 0
        latest_update_ratio = None
        probe_params = self.select_update_probe_params()
        print_with_timestamp("train start time")
        print("grad clip max_norm: {}".format(self.grad_clip_max_norm))
        # 에폭 중간 dev 점검 간격 — 학습 배치 수의 dev_check_ratio 배 (로그 주기에 정렬)
        interim_dev_interval = 0
        if self.dev_check_ratio > 0:
            interim_dev_interval = max(self.SAVE_INTERVAL_BATCHES,
                                       int(len(self.train_data) * self.dev_check_ratio)
                                       // self.SAVE_INTERVAL_BATCHES
                                       * self.SAVE_INTERVAL_BATCHES)
            print("interim dev check: every {} batches ({} dev batches)".format(
                interim_dev_interval, self.dev_check_max_batches))
        self.optimizer.zero_grad(set_to_none=True)

        for batch in self.train_data.iter_prefetched_batches(num_workers=self.num_data_workers):
            feat, label, phone, feat_mask, lab_mask, phone_mask, utt_ids = batch
            batch_cnt += 1

            step_losses = self.compute_step_losses(feat, label, phone, feat_mask,
                                                   lab_mask, phone_mask)
            loss = step_losses[0]                 # backward 용 (정규화 스케일)
            logged_losses = step_losses[1:]       # 기록용 (원 스케일 total + 개별 4개)

            # fp32 학습이므로 loss scaling 없이 그대로 backward 한다.
            # 배치 크기는 메모리 예산이 정하므로(BatchMemoryCalibrator) 배치마다
            # 업데이트한다 — gradient accumulation 은 쓰지 않는다.
            loss.backward()
            # clip_grad_norm_ 이 돌려주는 total_norm 하나만 검사한다. NaN/inf loss 는
            # 반드시 NaN grad 로 이어지므로 이 한 번의 동기화로 가중치 오염을 막는다.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=self.grad_clip_max_norm)
            if not torch.isfinite(grad_norm):
                print("[WARN] Non-finite gradient; step skipped. utt_ids:", utt_ids)
                self.optimizer.zero_grad(set_to_none=True)
                continue
            # 클리핑 전 norm 은 이미 계산돼 있다 — 버리지 말고 안정성 지표로 쌓는다
            grad_norm_value = float(grad_norm)
            grad_norm_sum += grad_norm_value
            clipped_step_count += int(grad_norm_value > self.grad_clip_max_norm)
            stability_count += 1

            # |Δw|/|w| 와 그룹별 grad norm 은 로그를 찍는 스텝에서만 잰다
            should_probe = probe_params and (batch_cnt % self.SAVE_INTERVAL_BATCHES == 0)
            previous_values = ([param.detach().clone() for param in probe_params]
                               if should_probe else None)
            if should_probe:
                # backward 는 끝났고 step 전 — gradient 가 아직 살아있는 유일한 시점
                self.append_grad_norm_record(batch_cnt, self.measure_group_grad_norms())
            self.optimizer.step()
            if should_probe:
                latest_update_ratio = self.measure_update_ratio(probe_params, previous_values)
            self.optimizer.zero_grad(set_to_none=True)

            # .item() 은 GPU 를 멈춰 세우므로 텐서로 누적만 하고 실제 값 읽기는
            # 아래 출력 시점(SAVE_INTERVAL_BATCHES 마다)에 한 번만 한다.
            loss_cnt += 1
            epoch_loss_sums = [loss_sum + step_loss.detach()
                               for loss_sum, step_loss in zip(epoch_loss_sums, logged_losses)]

            if batch_cnt % self.SAVE_INTERVAL_BATCHES == 0:
                print("step : save model")
                self.save_rolling_checkpoint()
                print_with_timestamp("train print time")
                stability = {
                    "gnorm": grad_norm_sum / max(1, stability_count),
                    "clip": 100.0 * clipped_step_count / max(1, stability_count),
                    "max_norm": self.grad_clip_max_norm,
                    "lr": max(param_group["lr"]
                              for param_group in self.optimizer.param_groups),
                    "upd": ("{:.2e}".format(latest_update_ratio)
                            if latest_update_ratio is not None else "-"),
                    "lr_groups": (" lrs: " + self.format_group_learning_rates()
                                  if len(self.optimizer.param_groups) > 1 else ""),
                }
                print(format_loss_log(batch_cnt, "train", epoch_loss_sums, loss_cnt, stability))
                grad_norm_sum, clipped_step_count, stability_count = 0.0, 0, 0

            if interim_dev_interval and batch_cnt % interim_dev_interval == 0:
                self.run_interim_dev_check(batch_cnt)

        self.save_rolling_checkpoint()
        # 주기적 empty_cache()/gc.collect() 는 allocator 캐시를 버려 이후 cudaMalloc
        # 동기화를 유발하므로 epoch 끝에서만 수행한다.
        torch.cuda.empty_cache()
        gc.collect()
        print_with_timestamp("Train End at")
        mean_epoch_loss = float(epoch_loss_sums[0]) / max(1, loss_cnt)
        print("==================== Epoch{} Train Loss: {}  =========================".format(
            epoch_num, mean_epoch_loss))
        return mean_epoch_loss

    @torch.no_grad()
    def evaluate_dev(self, max_batches=None):
        """dev 를 (최대 max_batches 배치) 돌아 [total, wrd, phn, ctc1, ctc2] 합과 배치 수.

        valid() 의 에폭 전체 패스와 학습 중간 점검이 이 코어를 공유한다.
        """
        was_training = self.model.training
        self.model.eval()
        loss_sums = [0.0] * 5
        batch_count = 0
        try:
            for batch in self.valid_data.iter_prefetched_batches(
                    num_workers=self.num_data_workers):
                feat, label, phone, feat_mask, lab_mask, phone_mask, utt_ids = batch
                step_losses = self.compute_step_losses(feat, label, phone, feat_mask,
                                                       lab_mask, phone_mask)
                if not all(torch.isfinite(step_loss) for step_loss in step_losses):
                    print("[VAL][ERROR] Non-finite loss values. Skip. utt_ids:", utt_ids)
                    continue
                loss_sums = [loss_sum + step_loss.item()
                             for loss_sum, step_loss in zip(loss_sums, step_losses[1:])]
                batch_count += 1
                if max_batches and batch_count >= max_batches:
                    break
        finally:
            if was_training:
                self.model.train()
        return loss_sums, batch_count

    def run_interim_dev_check(self, train_batch_cnt):
        """학습 중간 dev 점검 — 부분 dev 패스를 돌아 기록하고, 그 결과로 LR 을 조정한다.

        로그의 step 은 dev 배치 수가 아니라 현재 학습 배치라, train/dev 곡선을
        같은 x 축에 겹쳐 볼 수 있다.
        """
        loss_sums, batch_count = self.evaluate_dev(max_batches=self.dev_check_max_batches)
        if batch_count:
            print(format_loss_log(train_batch_cnt, "dev", loss_sums, batch_count))
            self.adjust_learning_rate_on_plateau(loss_sums[0] / batch_count)

    def adjust_learning_rate_on_plateau(self, interim_dev_loss):
        """dev 가 patience 회 연속 정체하면 LR 을 factor 배로 깎는다.

        에폭 끝의 ReduceLROnPlateau 를 대체한다 — 에폭이 12시간이라 조정 주기가
        너무 길었고, 재시작이 잦으면 에폭 경계를 아예 못 밟아 LR 이 초기값에
        묶인다 (실측: 이틀간 1e-3 고정 -> dev/WRR 정체). 중간 점검(40분마다)
        기준이면 정체 감지에서 조정까지 최대 patience x 40분이다.
        """
        improved = interim_dev_loss < self.best_interim_dev_loss * (1 - self.adaptive_lr_threshold)
        if improved:
            self.best_interim_dev_loss = interim_dev_loss
            self.interim_stall_count = 0
            return
        self.interim_stall_count += 1
        if self.interim_stall_count < self.adaptive_lr_patience:
            return
        # group 마다 자기 LR 에 factor 를 곱한다 — 모듈별 오버라이드로 group 이
        # 여러 개일 때 같은 값을 대입하면 비율이 무너져 오버라이드가 사라진다.
        current_lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]
        new_lrs = [max(self.adaptive_lr_min, current_lr * self.adaptive_lr_factor)
                   for current_lr in current_lrs]
        if any(new_lr < current_lr for new_lr, current_lr in zip(new_lrs, current_lrs)):
            for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
                param_group["lr"] = new_lr
            print("[adaptive] lr {} -> {} (interim dev {}회 정체, best {:.4f})".format(
                "/".join("{:.3e}".format(current_lr) for current_lr in current_lrs),
                "/".join("{:.3e}".format(new_lr) for new_lr in new_lrs),
                self.interim_stall_count, self.best_interim_dev_loss))
        # 조정 후에는 현재 지점을 새 기준으로 — 바로 또 깎이지 않게 한다
        self.best_interim_dev_loss = interim_dev_loss
        self.interim_stall_count = 0

    @torch.no_grad()
    def valid(self, epoch_num):
        print_with_timestamp("test start time")
        epoch_loss_sums, loss_cnt = self.evaluate_dev(max_batches=None)
        torch.cuda.empty_cache()
        gc.collect()
        print_with_timestamp("Test End at")
        mean_epoch_loss = epoch_loss_sums[0] / max(1, loss_cnt)
        print("===================== Epoch{} Test Loss: {}  ==========================".format(
            epoch_num, mean_epoch_loss))
        # LR 은 중간 dev 점검의 adjust_learning_rate_on_plateau 가 관리한다 —
        # 에폭 끝 ReduceLROnPlateau 까지 겹치면 이중으로 깎인다.
        return mean_epoch_loss
