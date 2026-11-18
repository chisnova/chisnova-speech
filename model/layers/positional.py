import torch
from torch import nn
import os
import copy
from util.saver import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super(PositionalEncoding, self).__init__()

        encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # batch 차원은 broadcasting으로 처리 → (max_len, d_model) 하나면 충분.
        # persistent=False: state_dict 미포함으로 기존 checkpoint 호환.
        # buffer 등록으로 module과 함께 device 이동 → forward마다 H2D 복사 없음.
        self.register_buffer("encoding2D", encoding, persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        pos_emb = self.encoding2D[:seq_len, :].unsqueeze(0)
        x = torch.add(x, pos_emb)
        return x

    def stream(self, x, position):
        seq_len = x.size(1)
        pos_emb = self.encoding2D[position:position+seq_len, :].unsqueeze(0)
        x = torch.add(x, pos_emb)
        return x




if __name__ == "__main__":
   random = torch.rand([3, 100, 512])
   test = PositionalEncoding(512)
   print(test.encoding)
