from torch import nn
from torch.nn import functional as F
import os
import torch
import json
from util.model_average import *
from util.saver import *

class ConformerConvSubSample(nn.Module):
    KERNEL_SIZE = 3
    STRIDE = 2

    def __init__(self, in_ch, out_ch):
        super(ConformerConvSubSample, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, groups=in_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, groups=out_ch)
        self.length1 = 0
        self.length2 = 0
        self.cache1=None
        self.cache2=None

    def clear(self):
        self.length1 = 0
        self.length2 = 0
        self.cache1 = None
        self.cache2 = None

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)         
        batch, channels, length, dim = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch, length, channels * dim)
        return x

    def calc_output_length(self, input_length):
        """kernel 3 / stride 2 conv 하나를 지난 뒤의 길이 (모자라면 0)."""
        if input_length < self.KERNEL_SIZE:
            return 0
        return (input_length - self.KERNEL_SIZE) // self.STRIDE + 1

    def calc_subsampled_dim(self, input_dim):
        """conv 2회를 지난 뒤의 마지막 차원 크기."""
        return self.calc_output_length(self.calc_output_length(input_dim))

    def stream(self, x):
        """청크 스트리밍 subsample. (B, L, D) -> (B, L', C*D')

        캐시에는 "아직 소비되지 않은 프레임" 만 남긴다. 출력 n 은 입력
        [n*stride, n*stride+kernel-1] 을 보므로, 이번 호출이 out 프레임을 냈다면
        다음 출력이 볼 시작 위치는 out*stride 다. 고정 길이(예: 2프레임)로 들고
        있으면 이미 소비한 프레임을 다시 먹여 청크 크기에 따라 총 출력 프레임 수가
        달라진다 — full-sequence forward 와 어긋나는 원인이다.

        커널을 채울 만큼 프레임이 안 모였으면 (B, 0, C*D') 을 돌려주고 캐시에만
        쌓는다. 호출자는 빈 청크를 건너뛰면 된다.
        """
        if self.cache1 is not None:
            x = torch.cat([self.cache1, x], dim=1)
        conv1_output_length = self.calc_output_length(x.size(1))
        if conv1_output_length == 0:
            self.cache1 = x.detach().clone()
            return x.new_zeros(x.size(0), 0,
                               self.conv2.out_channels * self.calc_subsampled_dim(x.size(2)))
        self.cache1 = x[:, conv1_output_length * self.STRIDE:, :].detach().clone()

        conv1_output = F.relu(self.conv1(x.unsqueeze(1)))

        if self.cache2 is not None:
            conv1_output = torch.cat([self.cache2, conv1_output], dim=2)
        conv2_output_length = self.calc_output_length(conv1_output.size(2))
        if conv2_output_length == 0:
            self.cache2 = conv1_output.detach().clone()
            return conv1_output.new_zeros(
                conv1_output.size(0), 0,
                self.conv2.out_channels * self.calc_output_length(conv1_output.size(3)))
        self.cache2 = conv1_output[:, :, conv2_output_length * self.STRIDE:, :].detach().clone()

        conv2_output = F.relu(self.conv2(conv1_output))
        batch, channels, length, dim = conv2_output.size()
        return conv2_output.permute(0, 2, 1, 3).contiguous().view(batch, length, channels * dim)
    
        
if __name__ == "__main__":
    input = torch.rand([1, 1600, 80])
    sub_sample = ConformerConvSubSample(1, 64)
    output = sub_sample(input)
    json_save(sub_sample, "conv_subsample", ".")
    
    (batch, len, dim) = input.size()
    input = input.reshape(len, dim)
    print(input)
    print(input.size())
    save_file(input, "input.pt")
    (batch, len, dim) = output.size()
    output = output.reshape(len, dim)
    print(output)
    print(output.size())
    save_file(output, "output.pt")
