import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm

from layers.conformer_positional import ConformerPositionalEncoding
from layers.conformer_feedforward import ConformerFeedForward
from layers.conformer_conv_module import ConformerConvModule
from layers.conformer_multihead import ConformerAttention


class ConformerEncoder(nn.Module):
    """Conformer 인코더 스택 — 특징을 받아 인코딩된 특징을 돌려준다.

    CTC posterior 를 내지 않는다. 이 출력에 어떤 head 를 붙일지는 호출자가 정한다
    (AsrStreamConformerTrfTransducer 는 ctc_linear2 를 붙여 두 번째 CTC 를 만든다).

    ConformerStreamEncoder(1패스, 청크 스트리밍)와 달리 마스크 없이 전체 시퀀스를
    한 번에 본다 — 1패스 인코더 뒤에 놓이는 cascade 단계다.

    time_reduction > 1 이면 입력을 avg_pool 로 먼저 줄인다(progressive downsampling).
    프레임레이트가 1/(4*time_reduction) 이 되어 뒤쪽 연산량이 그만큼 준다.
    """

    def __init__(self, d_model, n_head, n_layers, drop_prob, time_reduction=1):
        super(ConformerEncoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.pos_enc = ConformerPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            ConformerEncoderLayer(d_model=d_model, n_head=n_head, drop_prob=drop_prob)
            for _ in range(n_layers)])
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.time_reduction = time_reduction

    def calKernel(self, feat_dim):
        return torch.div(feat_dim-3, 2, rounding_mode='floor') + 1

    def clear(self):
        self.start = 0
        for layer in self.layers:
            layer.clear()

    def forward(self, enc, mask=None):
        if self.time_reduction > 1:
            enc = enc.transpose(1, 2)
            enc = F.avg_pool1d(enc, kernel_size=self.time_reduction,
                               stride=self.time_reduction, ceil_mode=True)
            enc = enc.transpose(1, 2)
        enc = self.layer_norm1(enc)
        pos_enc = self.pos_enc(enc)
        for layer in self.layers:
            enc = layer(enc, pos_enc)
        enc = self.layer_norm2(enc)
        return enc


class ConformerEncoderLayer(nn.Module):
    """Conformer 블록 하나 — FFN(1/2) -> MHSA -> Conv -> FFN(1/2), 전부 잔차 연결.

    dropout 은 각 서브모듈이 자기 출력에 한 번만 건다. 레이어에서 또 걸면
    서브모듈 것과 겹쳐 유효 비율이 1-(1-p)^2 가 된다.
    """

    def __init__(self, d_model, n_head, drop_prob):
        super(ConformerEncoderLayer, self).__init__()
        self.layer_norm1 = LayerNorm(d_model)
        self.ffn1 = ConformerFeedForward(d_model, expansion_factor=4, drop_prob=drop_prob)
        self.layer_norm2 = LayerNorm(d_model)
        self.multihead = ConformerAttention(d_model, n_head, drop_prob)
        self.layer_norm3 = LayerNorm(d_model)
        self.conv_module = ConformerConvModule(d_model, dropout=drop_prob)
        self.layer_norm4 = LayerNorm(d_model)
        self.ffn2 = ConformerFeedForward(d_model, expansion_factor=4, drop_prob=drop_prob)

    def clear(self):
        self.multihead.clear()

    def forward(self, x, pos_enc):
        output1 = self.layer_norm1(x)
        output1 = self.ffn1(output1)*0.5
        output1 = torch.add(output1, x)

        output2 = self.layer_norm2(output1)
        output2 = self.multihead(output2, pos_enc)
        output2 = torch.add(output2,output1)

        output3 = self.layer_norm3(output2)
        output3 = self.conv_module(output3)
        output3 = torch.add(output3, output2)

        output4 = self.layer_norm4(output3)
        output4 = self.ffn2(output4)*0.5
        output4 = torch.add(output4, output3)

        return output4