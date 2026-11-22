import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from module.conformer_stream_encoder import ConformerStreamEncoder
from module.conformer_encoder import ConformerEncoder
from module.transformer_decoder import TransformerDecoder

# 순수 모델 정의만 남긴다.
# 디코딩(beam search 등)은 util/inference_wrapper.py 의 InferenceWrapper가 담당한다.


class AsrStreamConformer(nn.Module):
    def __init__(self, d_model, enc_n_head, enc_nlayer, dec_hidden, dec_n_head, dec_nlayer, vocab_size, _k_size, num_ctc_layer, num_ctc_out, freeze_num, drop_prob,
                 dynamic_chunk=False,
                 # chunk_* 단위는 subsample 이후의 인코더 프레임이고 1프레임 = 40ms 다
                 # (feature hop 160샘플 = 10ms, 앞단 conv 두 층이 stride 2 로 4배 축소).
                 # 그래서 25 = 1.0초, 150 = 6.0초.
                 #
                 # 앱은 500ms(12~13프레임) 단위로 오디오를 넣는다
                 # (AsrAsyncDecoder::WAKE_THRESHOLD_SAMPLES). 아래 기본값은 그보다 큰
                 # 크기만 보므로 스트리밍(1패스 CTC) 인식이 불리하다. 추론에 맞춰 다시
                 # 학습하려면 config 로 chunk_min/chunk_max/chunk_sampling 을 넘긴다.
                 chunk_min=25, chunk_max=150, left_chunks=16, full_context_ratio=0.25,
                 # "uniform" 이면 균등, "log" 면 로그 균등으로 뽑는다.
                 # 균등은 범위를 넓힐수록 작은 청크가 드물어진다 — 6~50 에서 12 이하가
                 # 나올 확률이 15% 밖에 안 된다. 작은 청크일수록 문맥이 짧아 어려운
                 # 조건인데, 정작 추론에서 쓰는 크기가 거기에 있다.
                 # 로그 균등이면 같은 범위에서 12 이하가 35% 로 늘어 작은 쪽이 촘촘해진다.
                 chunk_sampling="uniform",
                 predictor_type="transformer", ctc_time_reduction=1,
                 stream_left_context=None, inter_ctc_layer=None):
        super(AsrStreamConformer, self).__init__()
        self.num_layer = dec_nlayer
        self.model_type="stream_conformer"
        self.stream_encoder= ConformerStreamEncoder(d_model=d_model, n_head=enc_n_head, n_layers=enc_nlayer, _k_size=_k_size, freeze_num=0, drop_prob=drop_prob,
                                                    stream_left_context=stream_left_context)
        self.second_encoder = ConformerEncoder(d_model=d_model, n_head=enc_n_head, n_layers=8, drop_prob=drop_prob, time_reduction=ctc_time_reduction)
        self.ctc_time_reduction = ctc_time_reduction
        self.phn_decoder = TransformerDecoder(d_model, dec_hidden, n_head=dec_n_head, n_layers=dec_nlayer, vocab_size=47, drop_prob=drop_prob)
        self.wrd_decoder = TransformerDecoder(d_model, dec_hidden, n_head=dec_n_head, n_layers=dec_nlayer, vocab_size=vocab_size, drop_prob=drop_prob)
        self.ctc_linear1 = nn.Linear(d_model, vocab_size)
        self.ctc_linear2 = nn.Linear(d_model, vocab_size)
        self.vocab_size=vocab_size
        # RNN-T 가지가 없다는 사실을 BatchMemoryCalibrator 가 읽어 간다 — joint 가
        # 없으면 같은 메모리로 배치를 훨씬 크게 잡을 수 있어 프로브 범위가 달라진다.
        self.use_transducer = False

        # intermediate CTC (Lee & Watanabe 2021): 중간층 출력에도 CTC 를 걸어
        # 중간층에 직접 gradient 를 꽂는다. head 는 ctc_linear1 을 공유한다
        # (self-conditioning 계열과 같은 선택 — 파라미터 추가 없음, 체크포인트 호환).
        self.inter_ctc_layer = inter_ctc_layer
        self.dynamic_chunk = dynamic_chunk
        self.chunk_min = chunk_min
        self.chunk_max = chunk_max
        self.left_chunks = left_chunks
        self.full_context_ratio = full_context_ratio
        self.chunk_sampling = chunk_sampling
        self.blank_id = 1
    
    @classmethod
    def from_config(cls, config):
        """config.json 의 model 섹션으로 모델을 만든다 (학습/추론이 같은 경로를 쓴다)."""
        return cls(d_model=config["dim_model"],
                   enc_n_head=config["enc_num_head"],
                   enc_nlayer=config["enc_num_layer"],
                   dec_hidden=config["dec_hidden"],
                   dec_n_head=config["dec_num_head"],
                   dec_nlayer=config["dec_num_layer"],
                   vocab_size=config["vocab_size"],
                   _k_size=config["k_size"],
                   num_ctc_layer=config["num_ctc_layer"],
                   num_ctc_out=config["num_ctc_out"],
                   freeze_num=config["encoder_freeze_num"],
                   drop_prob=config["drop_out"],
                   dynamic_chunk=config.get("dynamic_chunk", False),
                   chunk_min=config.get("chunk_min", 25),
                   chunk_max=config.get("chunk_max", 150),
                   left_chunks=config.get("left_chunks", 16),
                   full_context_ratio=config.get("full_context_ratio", 0.25),
                   chunk_sampling=config.get("chunk_sampling", "uniform"),
                   ctc_time_reduction=config.get("ctc_time_reduction", 1),
                   stream_left_context=config.get("stream_left_context", None),
                   inter_ctc_layer=config.get("inter_ctc_layer", None))

    def type_name(self):
        return "stream_conformer"

    def sample_chunk_size(self):
        """학습에 쓸 청크 크기(인코더 프레임)를 뽑는다. 1프레임 = 40ms.

        chunk_sampling="log" 면 로그 균등으로 뽑아 작은 청크를 더 자주 보게 한다.
        추론이 쓰는 크기(앱은 500ms = 12~13프레임)가 분포의 두꺼운 쪽에 오도록
        하려는 것이다. 균등으로 뽑으면 범위를 넓힐수록 그 근방이 오히려 성겨진다.
        """
        if self.chunk_sampling == "log":
            lo, hi = math.log(self.chunk_min), math.log(self.chunk_max)
            return max(self.chunk_min,
                       min(self.chunk_max, int(round(math.exp(random.uniform(lo, hi))))))
        return random.randint(self.chunk_min, self.chunk_max)

    def make2DMask(self, mask):
        B, L = mask.size()
        mask_q = mask.unsqueeze(2)          # [B, L, 1]
        mask_k = mask.unsqueeze(1)          # [B, 1, L]
        mask_2d = mask_q * mask_k           # elementwise AND (broadcasted)
        return mask_2d

    def make2DMaskV2(self, mask1, mask2):
        B, L1 = mask1.size()
        B, L2 = mask2.size()
        mask_q = mask1.unsqueeze(2)          # [B, L1, 1]
        mask_k = mask2.unsqueeze(1)          # [B, 1, L2]
        mask_2d = mask_q * mask_k           # elementwise AND (broadcasted)
        return mask_2d

    def make_enc2_mask(self, feat_mask, enc2_len):
        """입력 feat_mask 를 second_encoder 출력 길이 기준 마스크로 축소한다.

        이게 없으면 디코더 cross-attention 이 패딩 프레임까지 본다. 축소 비율은
        conv subsample 2회(각 (L-3)//2+1) x second_encoder 의 avg_pool(ceil) 이고,
        getCtcLoss 의 입력 길이 계산과 같은 식이어야 한다.
        """
        if feat_mask is None:
            return torch.ones(1, enc2_len, device=self.ctc_linear1.weight.device)
        lengths = feat_mask.sum(-1).long()
        for _ in range(2):                                  # conv subsample 2회
            lengths = torch.div(lengths - 3, 2, rounding_mode='floor') + 1
        if self.ctc_time_reduction > 1:                     # avg_pool(ceil_mode=True)
            lengths = torch.div(lengths + self.ctc_time_reduction - 1,
                                self.ctc_time_reduction, rounding_mode='floor')
        lengths = lengths.clamp(min=1, max=enc2_len)
        positions = torch.arange(enc2_len, device=feat_mask.device)
        return (positions.unsqueeze(0) < lengths.unsqueeze(1)).to(feat_mask.dtype)

    def encode_train_with_intermediate(self, enc):
        """encode_train + 중간층 출력. inter_ctc_layer 가 없으면 (enc1, None)."""
        if random.random() < self.full_context_ratio:
            chunk_size = enc.size(1)
        else:
            chunk_size = self.sample_chunk_size()
        if self.inter_ctc_layer is None:
            return self.stream_encoder(enc, chunk_size=chunk_size,
                                       left_chunks=self.left_chunks), None
        return self.stream_encoder(enc, chunk_size=chunk_size, left_chunks=self.left_chunks,
                                   intermediate_layer=self.inter_ctc_layer)

    def encode_train(self, enc):
        """Chunk mask로 full-sequence forward 1회.

        순차 stream() 루프(작은 matmul 직렬 → GPU util 저하) 대신, 청크 내부
        양방향 + 좌컨텍스트 제한 attention mask를 씌워 전체 시퀀스를 한 번에
        처리한다. 스트리밍 추론(stream(): 청크 내부 양방향)과 receptive field가
        일치하면서 배치 matmul 1회라 GPU 활용률이 크게 오른다.
        """
        if random.random() < self.full_context_ratio:
            chunk_size = enc.size(1)  # full-context(=무제한 좌컨텍스트 causal)도 섞어 학습
        else:
            chunk_size = self.sample_chunk_size()
        return self.stream_encoder(enc, chunk_size=chunk_size, left_chunks=self.left_chunks)

    def forward(self, dec, enc, p_dec, dec_mask=None, enc_mask=None, p_dec_mask=None):
        if dec_mask is None:
            dec_mask = torch.ones_like(dec, dtype=torch.float32)
        if p_dec_mask is None:
            p_dec_mask = torch.ones_like(p_dec, dtype=torch.float32)

        enc1, intermediate = self.encode_train_with_intermediate(enc)
        ctc1 = self.ctc_linear1(enc1)
        # head 공유: 같은 vocab 공간이므로 중간층도 같은 projection 을 쓴다
        ctc_mid = self.ctc_linear1(intermediate) if intermediate is not None else None

        enc2 = self.second_encoder(enc1)
        ctc2 = self.ctc_linear2(enc2)

        enc2_mask = self.make_enc2_mask(enc_mask, enc2.size(1))
        dec_2d_mask = self.make2DMask(dec_mask[:,:-1])
        p_dec_2d_mask = self.make2DMask(p_dec_mask[:,:-1])

        dec_enc2_2d_mask = self.make2DMaskV2(dec_mask[:,:-1], enc2_mask)
        p_dec_enc2_2d_mask = self.make2DMaskV2(p_dec_mask[:,:-1], enc2_mask)

        w_out = self.wrd_decoder(dec[:,:-1], enc2, dec_mask=dec_2d_mask, dec_enc_mask=dec_enc2_2d_mask)
        p_out = self.phn_decoder(p_dec[:,:-1], enc2, dec_mask=p_dec_2d_mask, dec_enc_mask=p_dec_enc2_2d_mask)

        return w_out, p_out, ctc1, ctc2, ctc_mid

    def enc_stream(self, enc):
        """스트리밍 인코딩. 이전 호출과의 연속성은 내부 캐시가 유지하며
        새 발화 시작 시 clear()를 호출해야 한다."""
        return self.stream_encoder.stream(enc)

    def ctc_stream(self, enc):
        """CTC cascade encoder를 거친 두 번째 CTC posterior."""
        enc2 = self.second_encoder(enc)
        ctc = F.softmax(self.ctc_linear2(enc2), dim=-1)
        return ctc, enc2

    def clear(self):
        self.stream_encoder.clear()
