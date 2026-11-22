import torch
from torch import nn
import os
import copy
import json
from util.saver import *
from util.model_average import *

class ConformerPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(ConformerPositionalEncoding, self).__init__()

        encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # batch 차원은 broadcasting으로 처리되므로 (max_len, d_model) 하나면 충분하다.
        # persistent=False: state_dict에 포함되지 않아 기존 checkpoint와 그대로 호환.
        # buffer 등록으로 module과 함께 device로 이동 → forward마다 H2D 복사 없음.
        self.register_buffer("encoding2D", encoding, persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding2D[:seq_len, :].unsqueeze(0)

    def stream(self, x, start):
        seq_len = x.size(1)
        return self.encoding2D[start:start + seq_len, :].unsqueeze(0)

if __name__ == "__main__":
    _in = torch.rand([1, 100, 512])
    model = ConformerPositionalEncoding(512)
    out = model(_in)
    json_save(model, "positional", ".")
    
    print(_in)
    _in = torch.reshape(_in, (100, 512))
    save_file(_in, "input.pt")


    out = torch.reshape(out, (100, 512))
    print(out)
    save_file(out, "out.pt")
