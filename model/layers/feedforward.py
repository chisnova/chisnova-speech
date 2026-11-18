from torch import nn
from torch.nn import functional as F
from util.saver import *
import os
from util.model_average import *

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.0):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p=drop_prob)
        self.gelu = nn.GELU()

    def average(self, module):
        linear_average(self.linear1, module.linear1)
        linear_average(self.linear2, module.linear2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x) 
        x = self.dropout(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    positional = FeedForward(10, 10, 0.0)
    json_save(positional, "feedforward", ".")

    _in = torch.rand([3, 10])
    print(_in)
    save_file(_in, "input.pt")
    out = positional(_in)
    print(out)
    save_file(out, "out.pt")
