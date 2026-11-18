import os
# 예산 배칭은 배치 모양이 계속 달라져 할당자 단편화가 생긴다. 가변 크기 세그먼트를
# 쓰면 "reserved 는 남는데 allocate 실패" 하는 상황이 줄어든다. torch 임포트 전에 설정.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time

import torch
from torch import optim
from torch.optim import AdamW

from app.asr_stream_conformer import AsrStreamConformer
from data.loader import AsrDataLoader, LmdbConn
from data.loader.mixed_batch_loader import MixedBatchLoader
from data.loader.feature_enginerring import (Compose, GaussianNoise, SnrNoiseMix,
                                             SpecAugment, SpecSubstitute)
from util.model_migration import fuse_legacy_attention_state_dict
import conformer_stream_trainer
from conformer_stream_trainer import ConformerStreamTrainer
from batch_memory_calibrator import BatchMemoryCalibrator


def parseConfig(model_path, config_path=None):
    """config.json 을 model / corpus 섹션으로 나눠 준다.

    config_path 를 주면 그 파일을, 없으면 모델 디렉토리의 config.json 을 읽는다.
    """
    if config_path is None:
        config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config["model"], config["corpus"]


def loadCheckpointIfExists(model, model_path):
    """asr_model.pt 가 있으면 이어서 학습한다.

    이름과 shape 이 모두 맞는 텐서만 골라 넣는다 — 구조를 바꾼 뒤에도 겹치는
    부분은 살리고 나머지는 새로 초기화된 값을 쓰기 위해서다.
    """
    checkpoint_path = os.path.join(model_path, "asr_model.pt")
    if not os.path.isfile(checkpoint_path):
        print("New model is generated!")
        return model

    checkpoint = torch.load(checkpoint_path)
    # 구 per-head attention checkpoint 를 fused QKV 형식으로 변환
    fuse_legacy_attention_state_dict(model, checkpoint)
    model_state = model.state_dict()
    matched_state = {name: tensor for name, tensor in checkpoint.items()
                     if name in model_state and tensor.shape == model_state[name].shape}
    model_state.update(matched_state)
    model.load_state_dict(model_state)
    print("ASR model loaded ({}/{} tensors matched)".format(len(matched_state), len(model_state)))
    return model


def buildStreamConformerModel(config, model_path):
    """RNN-T 없이 CTC x2 + CE x2 만 쓰는 스트리밍 Conformer (AsrStreamConformer).

    생성 인자는 모델이 from_config 로 갖는다 — 추론(util/stream_conformer_inference.py)
    이 같은 경로를 쓰므로 학습/추론 구성이 갈라지지 않는다.
    """
    model = AsrStreamConformer.from_config(config).cuda()
    return loadCheckpointIfExists(model, model_path)


# arch -> 모델 빌더 / arch -> 손실 어댑터 / trainer 이름 -> trainer 클래스.
# 새 아키텍처를 붙일 때 이 세 곳만 채우면 된다.
MODEL_BUILDERS = {
    "conformer_stream": buildStreamConformerModel,
}

LOSS_ADAPTERS = {
    "conformer_stream": conformer_stream_trainer.compute_loss_from_outputs,
}

TRAINERS = {
    "conformer-stream": ConformerStreamTrainer,
}


def buildModel(model_path, model_config):
    print("Model Config: {}".format(model_config))
    return MODEL_BUILDERS[model_config["arch"]](model_config, model_path)


def calibrateBatchBudget(model, model_config, corpus_config, manifest):
    calibrator = BatchMemoryCalibrator(
        model, LOSS_ADAPTERS[model_config["arch"]], model_config, manifest=manifest,
        safety_ratio=float(corpus_config.get("memory_safety_ratio", 0.75)))
    return calibrator.calibrate()


def buildTrainTransform(corpus_config):
    """train 로더에 걸 특징 증강 파이프라인.

    순서가 곧 제약이다.
      SnrNoiseMix   맨 앞. feat 이 아직 dB 여야 파워로 되돌려 잡음을 더할 수 있다
      SpecSubstitute/SpecAugment  잡음까지 얹힌 뒤에 지워야 "지운 구간" 이 실제로 빈다
      GaussianNoise 마지막. 마스크 위에도 미세한 흔들림을 남긴다

    마스크 값은 "mean" 이다 — 특징이 정규화 안 된 dB(-100~-17)라 0 은 무음이 아니라
    최댓값보다 큰 값이고, 그대로 쓰면 '지우기' 가 아니라 '에너지 주입' 이 된다.
    Cmvn 을 넣게 되면 그때 "zero" 로 바꿔야 한다.
    """
    spec_augment_prob = float(corpus_config.get("spec_augment_prob", 0.7))
    spec_augment_policy = corpus_config.get("spec_augment_policy", "SM")
    spec_substitute_prob = float(corpus_config.get("spec_substitute_prob", 0.3))
    gaussian_noise_std = float(corpus_config.get("gaussian_noise_std", 0.05))

    transforms = []
    noise_record_path = corpus_config.get("noise_record")
    if noise_record_path:
        snr_range = corpus_config.get("noise_snr_range", [0.0, 20.0])
        noise_count_range = corpus_config.get("noise_count_range", [1, 5])
        transforms.append(SnrNoiseMix.from_record(
            noise_record_path,
            snr_range=(float(snr_range[0]), float(snr_range[1])),
            noise_count_range=(int(noise_count_range[0]), int(noise_count_range[1])),
            clip_level_spread_db=float(corpus_config.get("noise_clip_level_spread_db", 6.0)),
            phase_jitter_db=float(corpus_config.get("noise_phase_jitter_db", 3.0)),
            prob=float(corpus_config.get("noise_prob", 0.5))))

    if spec_substitute_prob > 0.0:
        transforms.append(SpecSubstitute(prob=spec_substitute_prob))
    if spec_augment_prob > 0.0:
        transforms.append(buildSpecAugment(spec_augment_policy, corpus_config,
                                           spec_augment_prob))
    if gaussian_noise_std > 0.0:
        # relative=True: 발화 표준편차 대비 비율. 절대값으로 주면 dB 스케일(표준편차
        # 수십 dB)에서 아무 효과가 없다.
        transforms.append(GaussianNoise(std=gaussian_noise_std, relative=True,
                                        prob=spec_augment_prob))
    return Compose(transforms) if transforms else None


def buildSpecAugment(policy_name, corpus_config, prob):
    """정책 프리셋 또는 직접 지정한 파라미터로 SpecAugment 를 만든다.

    로더 기본값(F=5x2, T=5x2)은 논문 정책의 1/5~1/14 라 사실상 안 걸린 것과 같다.
    기본을 "SM"(F=15x2, T=70x2, time_ratio=0.2)으로 올린다 — time_ratio 가 짧은
    발화를 보호하므로 길이 편차가 큰 코퍼스에 LD 보다 안전하다.
    """
    mask_value = corpus_config.get("spec_augment_mask_value", "mean")
    if policy_name:
        return SpecAugment.from_policy(policy_name, mask_value=mask_value, prob=prob)
    return SpecAugment(num_freq=int(corpus_config.get("spec_augment_num_freq", 2)),
                       freq_param=int(corpus_config.get("spec_augment_freq_param", 15)),
                       num_time=int(corpus_config.get("spec_augment_num_time", 2)),
                       time_param=int(corpus_config.get("spec_augment_time_param", 70)),
                       time_ratio=float(corpus_config.get("spec_augment_time_ratio", 0.2)),
                       mask_value=mask_value, prob=prob)


def buildDataset(corpus_config, budget=None, train_transform=None):
    print("Dataset Config: {}".format(corpus_config))
    num_data_workers = int(corpus_config.get("num_data_workers", 4))
    max_feat_len = corpus_config.get("max_feat_len")

    # 시드가 매 실행 고정(0)이면 재시작할 때마다 epoch 0 의 같은 배치 순열을
    # 반복한다 — 재시작이 잦으면 순열 앞부분만 편식하고, usage_ratio<1 이면
    # 아예 같은 부분집합만 본다. time 기반으로 매 실행 다르게 뽑되, 문제 재현이
    # 필요할 때를 위해 값을 로그에 남긴다.
    train_shuffle_seed = int(time.time())
    print("[loader] train seed: {}".format(train_shuffle_seed))
    train_data = AsrDataLoader(data_path=corpus_config["train_record"],
                               batch_size=int(corpus_config.get("train_batch_size", 16)),
                               budget=budget,
                               shuffle=True,
                               isTrain=True,
                               usage_ratio=corpus_config["train_use_ratio"],
                               max_feat_len=max_feat_len,
                               transform=train_transform,
                               seed=train_shuffle_seed,
                               num_workers=num_data_workers)

    if corpus_config.get("adaptation_record"):
        adaptation_data = AsrDataLoader(
            data_path=corpus_config["adaptation_record"],
            batch_size=int(corpus_config.get("train_batch_size", 16)),
            budget=budget, shuffle=True, isTrain=True, usage_ratio=1.0,
            max_feat_len=max_feat_len, transform=train_transform,
            seed=train_shuffle_seed + 1, num_workers=num_data_workers)
        train_data = MixedBatchLoader(
            [train_data, adaptation_data],
            batches_per_source=corpus_config.get("mix_batches", [4, 1]),
            batches_per_epoch=corpus_config.get("mix_batches_per_epoch", 1000),
            seed=train_shuffle_seed)
        print("[mix] train/adaptation ratio: {}, batches per epoch: {}".format(
            train_data.ratio, len(train_data)))

    valid_data = AsrDataLoader(data_path=corpus_config["dev_record"],
                               batch_size=int(corpus_config.get("dev_batch_size", 16)),
                               budget=budget,
                               shuffle=False,
                               isTrain=False,
                               usage_ratio=corpus_config["dev_use_ratio"],
                               max_feat_len=max_feat_len,
                               num_workers=num_data_workers)
    return train_data, valid_data


def buildTrainer(model_path, model_config, model, train_data, valid_data):
    optimizer = AdamW(params=model.parameters(), lr=model_config["lr"], eps=5e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     factor=model_config["factor"],
                                                     patience=model_config["patient"])
    return TRAINERS[model_config["trainer"]](model,
                                             train_data,
                                             valid_data,
                                             optimizer,
                                             scheduler,
                                             model_path,
                                             model_config["vocab_size"],
                                             model_config["tr_method"],
                                             model_config["encoder_freeze_num"])


def runTraining(model_path, config_path=None):
    model_config, corpus_config = parseConfig(model_path, config_path)
    model = buildModel(model_path, model_config)
    # 모델이 GPU 에 올라간 뒤에 남은 메모리를 재야 예산이 맞는다 — 순서가 중요하다.
    # 카탈로그는 프로브 길이를 현실적으로 잡는 데 쓰므로 로더보다 먼저 연다.
    train_manifest = LmdbConn(corpus_config["train_record"]).load_manifest()
    budget = calibrateBatchBudget(model, model_config, corpus_config, train_manifest)
    train_data, valid_data = buildDataset(corpus_config, budget=budget,
                                          train_transform=buildTrainTransform(corpus_config))
    trainer = buildTrainer(model_path, model_config, model, train_data, valid_data)

    epoch_num = 0
    max_epochs = int(model_config.get("max_epochs", 0))
    try:
        if model_config.get("validate_before_training", False):
            trainer.valid(0)
        while max_epochs <= 0 or epoch_num < max_epochs:
            epoch_num += 1
            trainer.setModel(model)
            train_loss = trainer.train(epoch_num)
            valid_loss = trainer.valid(epoch_num)
            trainer.saveEpochModel(epoch_num, train_loss, valid_loss)
    finally:
        train_data.close()
        valid_data.close()


def parseArguments():
    parser = argparse.ArgumentParser(description="ASR 학습")
    parser.add_argument('model_path_positional', nargs='?', default=None,
                        help="모델 디렉토리 (config.json + asr_model.pt). --model_path 와 같다")
    parser.add_argument('--model_path', type=str, default=None,
                        help="모델 디렉토리 — 체크포인트를 여기에 쓴다")
    parser.add_argument('--config_path', type=str, default=None,
                        help="config.json 경로 (기본: <model_path>/config.json)")
    args = parser.parse_args()
    model_path = args.model_path or args.model_path_positional
    if model_path is None:
        parser.error("모델 디렉토리가 필요합니다: train_main.py <model_path> "
                     "또는 --model_path <dir>")
    return model_path, args.config_path


if __name__ == "__main__":
    runTraining(*parseArguments())
