import torch
from torch import nn

from layers.conformer_feedforward import ConformerFeedForward 
from layers.conformer_stream_conv_module import ConformerStreamConvModule
from layers.conformer_stream_multihead import ConformerStreamAttention
from layers.conformer_conv_subsample import ConformerConvSubSample
from torch.nn import LayerNorm
from layers.rotary_embedding import RotaryEmbedding
from util.saver import *
from util.model_average import *


class ConformerStreamEncoder(nn.Module):
    """청크 스트리밍 Conformer 인코더. 위치 인코딩은 RoPE 만 쓴다.

    절대 sinusoidal PE 를 걷어낸 이유
      - 같은 거리의 프레임 쌍을 절대 위치에 따라 다르게 취급한다(음성에서는 틀린 성질).
        RoPE 는 q/k 를 위치만큼 회전시켜 내적이 (n-m) 에만 의존하게 만든다.
      - 미리 만든 테이블(max_len)에 묶여 그 길이를 넘는 연속 스트림에서 깨진다.
        RoPE 는 필요한 위치의 cos/sin 을 그때그때 계산하므로 상한이 없다.
      - 테이블 상주 메모리와 학습 파라미터가 0 이다.
    다만 학습 때 본 길이를 크게 넘어서면 RoPE 도 품질이 떨어진다 —
    아주 긴 발화를 다루게 되면 ALiBi 등을 검토해야 한다.
    """

    def __init__(self, d_model, n_head, n_layers, _k_size, freeze_num, drop_prob,
                 stream_left_context=None):
        super(ConformerStreamEncoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        out_ch = d_model//4
        sub_dim = self.calKernel(self.calKernel(80))*out_ch

        self.feat_norm = LayerNorm(80)
        self.layer_norm1 = LayerNorm(d_model)
        self.feat_convsubsample = ConformerConvSubSample(1, out_ch) # medium setting # 1/4 length
        self.feat_linear = nn.Linear(sub_dim, d_model)
        self.feat_dropout = nn.Dropout(p=drop_prob)
        self.layers = nn.ModuleList([ConformerStreamEncoderLayer(d_model=d_model, n_head=n_head, _ksize=_k_size, drop_prob=drop_prob) for _ in range(n_layers)])
        self.seq_len = 0
        self.freeze_num = freeze_num

        self.rotary = RotaryEmbedding(d_model // n_head)

        # streaming 시 attention이 볼 과거 프레임 상한 (subsample 후 프레임 단위)
        self.stream_left_context = stream_left_context

    def calKernel(self, feat_dim):
        return torch.div(feat_dim-3, 2, rounding_mode='floor') + 1

    def clear(self):
        self.seq_len = 0
        self.feat_convsubsample.clear()
        for layer in self.layers:
            layer.clear()

    def make_chunk_mask(self, seq_len, chunk_size, left_chunks, device):
        """Chunk 내부는 양방향(non-causal), 미래 chunk는 차단, 과거는 left_chunks 개
        chunk까지만 보는 mask. 청크 단위로 buffering 후 처리하는 chunk 기반 스트리밍의
        표준 설계로, 추론 stream()(청크 내부 양방향)과 receptive field가 일치한다.
        left_chunks*chunk_size == stream_left_context 이면 순차 stream 루프와도 정합.
        True = masked."""
        idx = torch.arange(seq_len, device=device)
        chunk_ids = torch.div(idx, chunk_size, rounding_mode='floor')
        q_ids = chunk_ids[:, None]
        k_ids = chunk_ids[None, :]
        mask = k_ids > q_ids  # 미래 chunk 차단 (chunk 내부는 양방향)
        if left_chunks is not None and left_chunks >= 0:
            mask = mask | (k_ids < (q_ids - left_chunks))  # 좌컨텍스트: left_chunks개 이전 chunk까지
        return mask

    def rope_cos_sin(self, x, start, length):
        pos_ids = torch.arange(start, start + length, device=x.device).unsqueeze(0)
        return self.rotary(x, pos_ids)

    def forward_subsample(self, x):
        enc = self.feat_norm(x)
        enc = self.feat_convsubsample(enc)
        enc = self.feat_linear(enc)
        return enc

    def forward(self, x, mask=None, chunk_size=None, left_chunks=None,
                intermediate_layer=None):
        """Full-sequence forward.
        chunk_size가 주어지면 chunk attention mask(dynamic chunk training)를 사용하고,
        None이면 기존 per-frame causal로 동작한다.
        intermediate_layer 를 주면 그 층(0-base)을 지난 출력을 함께 돌려준다 —
        intermediate CTC 가 중간층에 직접 gradient 를 꽂는 데 쓴다.
        """
        enc = self.feat_norm(x)
        enc = self.feat_convsubsample(enc)
        enc = self.feat_linear(enc)
        enc = self.feat_dropout(enc)
        enc = self.layer_norm1(enc)

        (bs, T, dim) = enc.size()
        cos, sin = self.rope_cos_sin(enc, 0, T)

        att_mask = None
        if chunk_size is not None:
            att_mask = self.make_chunk_mask(T, chunk_size, left_chunks, enc.device)

        intermediate_output = None
        for layer_index, layer in enumerate(self.layers):
            enc = layer(enc, att_mask=att_mask, cos=cos, sin=sin)
            if intermediate_layer is not None and layer_index == intermediate_layer:
                intermediate_output = enc
        if intermediate_layer is not None:
            return enc, intermediate_output
        return enc

    def stream(self, x, mask=None):
        enc = self.feat_norm(x)
        # forward() 가 아니라 stream() 을 써야 한다 — 청크마다 독립으로 subsample 하면
        # 경계에서 프레임이 통째로 날아가고(청크가 짧으면 커널보다 짧아져 죽는다),
        # 결과도 full-sequence forward 와 어긋난다. stream() 은 conv 좌컨텍스트를
        # 캐시로 이어 붙여 full-sequence 와 같은 프레임을 낸다.
        enc = self.feat_convsubsample.stream(enc)
        if enc.size(1) == 0:        # subsample 커널을 채울 만큼 아직 안 모였다
            return enc.new_zeros(enc.size(0), 0, self.d_model)
        enc = self.feat_linear(enc)
        (bs, T, dim) = enc.size()
        enc = self.layer_norm1(enc)

        # 스트리밍은 발화 시작부터 누적된 절대 위치를 그대로 넘긴다 — RoPE 는
        # 테이블이 없어 길이 상한이 없다(clear() 로 발화 경계에서 0 으로 되돌린다).
        cos, sin = self.rope_cos_sin(enc, self.seq_len, T)
        self.seq_len = self.seq_len+ T

        for layer in self.layers:
            enc = layer.stream(enc, cos=cos, sin=sin, left_context=self.stream_left_context)
        return enc


class ConformerStreamEncoderLayer(nn.Module):
    """Conformer 블록. dropout 은 각 서브모듈이 자기 출력에 한 번만 건다.

    레이어에서 또 걸면 서브모듈 것과 겹쳐 유효 비율이 1-(1-p)^2 가 된다
    (실측: p=0.1 설정인데 FFN·attention 경로가 19% 로 동작했다).
    원조 Conformer 도 블록은 잔차 덧셈만 하고 dropout 은 서브모듈에 둔다.
    """

    def __init__(self, d_model, n_head, _ksize, drop_prob):
        super(ConformerStreamEncoderLayer, self).__init__()
        self.layer_norm1 = LayerNorm(d_model)
        self.ffn1 = ConformerFeedForward(d_model, expansion_factor=4, drop_prob=drop_prob)
        self.layer_norm2 = LayerNorm(d_model)
        self.conv_module = ConformerStreamConvModule(d_model, ksize=_ksize, dropout=drop_prob)
        self.layer_norm3 = LayerNorm(d_model)
        self.multihead = ConformerStreamAttention(d_model, n_head, drop_prob)
        self.layer_norm4 = LayerNorm(d_model)
        self.ffn2 = ConformerFeedForward(d_model, expansion_factor=4, drop_prob=drop_prob)

    def clear(self):
        self.multihead.clear()
        self.conv_module.clear()

    def forward(self, x, att_mask=None, cos=None, sin=None):
        output1 = self.layer_norm1(x)
        output1 = self.ffn1(output1)*0.5
        output1 = torch.add(output1, x)

        output2 = self.layer_norm2(output1)
        output2 = self.conv_module(output2)
        output2 = torch.add(output2, output1)

        output3 = self.layer_norm3(output2)
        output3 = self.multihead(output3, att_mask=att_mask, cos=cos, sin=sin)
        output3 = torch.add(output3,output2)

        output4 = self.layer_norm4(output3)
        output4 = self.ffn2(output4)*0.5
        output4 = torch.add(output4,output3)

        return output4

    def stream(self, x, cos=None, sin=None, left_context=None):
        output1 = self.layer_norm1(x)
        output1 = self.ffn1(output1)*0.5
        output1 = torch.add(output1, x)

        output2 = self.layer_norm2(output1)
        output2 = self.conv_module.stream(output2)
        output2 = torch.add(output2, output1)

        output3 = self.layer_norm3(output2)
        output3 = self.multihead.stream(output3, cos=cos, sin=sin, left_context=left_context)
        output3 = torch.add(output3,output2)

        output4 = self.layer_norm4(output3)
        output4 = self.ffn2(output4)*0.5
        output4 = torch.add(output4,output3)

        return output4


