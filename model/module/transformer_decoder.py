import torch
from torch import nn

from layers.feedforward import FeedForward
from layers.multihead import MultiHead
from torch.nn import LayerNorm
from layers.positional import PositionalEncoding
import os
from util.saver import *
from util.model_average import *


class TransformerDecoder(nn.Module):
    """자기회귀 Transformer 디코더 — masked self-attention -> cross-attention -> FFN.
    """

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, vocab_size, drop_prob,
                 tie_embeddings=True):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.n_head = n_head
        self.n_layers = n_layers

        self.tin_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=1500)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob) for _ in range(n_layers)])
        self.layer_norm = LayerNorm(d_model)
        self.tout_emb = nn.Linear(d_model, vocab_size)
        # 입력 임베딩과 출력 projection 은 같은 (vocab, d_model) 공간이라 묶는다.
        # 파라미터가 vocab*d_model 만큼 줄고, 두 표현이 함께 학습돼 보통 같거나
        # 낫다(Press & Wolf 2017). 임베디드에서 모델 크기에 직접 이득이다.
        self.tie_embeddings = tie_embeddings
        if tie_embeddings:
            self.tout_emb.weight = self.tin_emb.weight

    def forward(self, dec, enc, dec_mask=None, dec_enc_mask=None):
        output = self.tin_emb(dec)
        output = self.pos_enc(output)
        for layer in self.layers:
            output = layer(output, enc, dec_mask=dec_mask, dec_enc_mask=dec_enc_mask)
        output = self.layer_norm(output)
        output = self.tout_emb(output)
        return output


class TransformerDecoderLayer(nn.Module):
    """dropout 은 self/cross attention 출력에만 건다.

    FeedForward 는 자체 dropout 을 갖고 있어 여기서 또 걸면 유효 비율이
    1-(1-p)^2 가 된다. MultiHead 는 자체 dropout 이 없어 여기서 건다.
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(TransformerDecoderLayer, self).__init__()
        self.layer_norm1 = LayerNorm(d_model)
        self.multihead1 = MultiHead(d_model, n_head, True)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.layer_norm2 = LayerNorm(d_model)
        self.multihead2 = MultiHead(d_model, n_head, False)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.layer_norm3 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob=drop_prob)

    def forward(self, dec, enc, dec_mask=None, dec_enc_mask=None):
        output1 = self.layer_norm1(dec)
        output1 = self.multihead1(Q=output1, K=output1, V=output1, mask=dec_mask)
        output1 = self.dropout1(output1)
        output1 = torch.add(output1, dec)

        output2 = self.layer_norm2(output1)
        output2 = self.multihead2(Q=output2, K=enc, V=enc, mask=dec_enc_mask)
        output2 = self.dropout2(output2)
        output2 = torch.add(output2, output1)

        output3 = self.layer_norm3(output2)
        output3 = self.ffn(output3)
        output3 = torch.add(output3, output2)
        return output3
