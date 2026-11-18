import torch
from torch import nn
from torch.nn import functional as F
import os
from util.saver import *
import json
from util.model_average import *

class ConformerConvModule(nn.Module):
    def __init__(self, in_ch, ksize=31, exp_factor=2, dropout=0.0):
        super(ConformerConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(in_ch)
        self.pointwise1 = nn.Conv1d(in_channels=in_ch, out_channels=in_ch*exp_factor,kernel_size=1, stride=1, padding=0,bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=ksize, stride=1, padding=(ksize-1)//2, bias=False, groups=in_ch)
        self.batch_norm = nn.BatchNorm1d(in_ch)
        self.swish = nn.SiLU()
        self.pointwise2 = nn.Conv1d(in_channels=in_ch, out_channels=in_ch,kernel_size=1, stride=1, padding=0,bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.ksize=ksize
        self.pre = None

    def clear(self):
        self.pre=None

    def forward(self, x):
        output = self.layer_norm(x)
        output = torch.transpose(output, 1,2) 
        output = self.pointwise1(output)
        output = self.glu(output)
        output = self.depthwise(output)
        output = self.batch_norm(output)
        output = self.swish(output)        
        output = self.pointwise2(output)
        output = self.dropout(output)
        output = torch.transpose(output, 1,2) 
        return output

if __name__ == "__main__":
    input = torch.rand([1, 100, 512])
    model = ConformerConvModule(512)
    output = model(input)
    json_save(model, "conv_module", ".")
    
    input = input.reshape(100, 512)
    print(input)
    print(input.size())
    save_file(input, "input.pt")

    output = output.reshape(100, 512)
    print(output)
    print(output.size())
    save_file(output, "output.pt")

    
