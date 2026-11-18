import torch
from torch import nn
from torch.nn import functional as F
import os
from util.saver import *
import json
from util.model_average import *

class ConformerStreamConvModule(nn.Module):
    def __init__(self, in_ch, ksize=31, exp_factor=2, dropout=0.0):
        super(ConformerStreamConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(in_ch)
        self.pointwise1 = nn.Conv1d(in_channels=in_ch, out_channels=in_ch*exp_factor,kernel_size=1, stride=1, padding=0,bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=ksize, stride=1, bias=False, groups=in_ch)
        self.layer_norm2 = nn.LayerNorm(in_ch)
        self.swish = nn.SiLU()
        self.pointwise2 = nn.Conv1d(in_channels=in_ch, out_channels=in_ch,kernel_size=1, stride=1, padding=0,bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.ksize=ksize
        self.pre = None


    def clear(self):
        self.pre=None

    def stream(self, x):
        output = self.layer_norm(x)
        output = torch.transpose(output, 1,2)
        output = self.pointwise1(output)
        output = self.glu(output)
        # 캐시에는 "직전 청크" 가 아니라 "패딩까지 포함한 스트림의 마지막 ksize-1 프레임"
        # 을 둔다. 직전 청크만 들고 있으면 그 청크가 ksize-1 보다 짧을 때 좌컨텍스트가
        # 모자라 depthwise conv 가 그대로 죽는다(짧은 청크 스트리밍에서 실제로 발생).
        if self.pre is not None:
            pad = torch.cat((self.pre, output), dim=-1)
        else:
            pad = F.pad(output, (self.ksize-1, 0))
        self.pre = pad[:, :, -(self.ksize-1):].detach().clone()

        output = self.depthwise(pad)
        output = torch.transpose(output, 1, 2)  # (B, C, T) -> (B, T, C)
        output = self.layer_norm2(output)
        output = torch.transpose(output, 1, 2)  # (B, T, C) -> (B, C, T)
        output = self.swish(output)        
        output = self.pointwise2(output)
        output = self.dropout(output)
        output = torch.transpose(output, 1,2)
        return output


    def forward(self, x):
        output = self.layer_norm(x)
        output = torch.transpose(output, 1,2) 
        output = self.pointwise1(output)
        output = self.glu(output)
        #output = F.pad(output, ((self.ksize-1)//2,(self.ksize-1)//2), mode="replicate")
        output = F.pad(output, (self.ksize-1,0))
        #output = F.pad(output, (0, self.ksize-1))
        output = self.depthwise(output)
        output = torch.transpose(output, 1, 2)  # (B, C, T) -> (B, T, C)
        output = self.layer_norm2(output)
        output = torch.transpose(output, 1, 2)  # (B, T, C) -> (B, C, T)
        output = self.swish(output)        
        output = self.pointwise2(output)
        output = self.dropout(output)
        output = torch.transpose(output, 1,2) 
        return output

if __name__ == "__main__":
    input = torch.rand([1, 100, 512])
    model = ConformerStreamConvModule(512)
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

    
