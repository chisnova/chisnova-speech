from torch import nn
from torch.nn import functional as F
import os
import torch
from util.saver import *
import json
from util.model_average import *

class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, expansion_factor, drop_prob):
        super(ConformerFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model*expansion_factor)
        self.swish_act = nn.SiLU()
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(d_model*expansion_factor, d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish_act(x) 
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

    def average(self, module):
        layernorm_average(self.layer_norm, module.layer_norm)
        linear_average(self.linear1, module.linear1)
        linear_average(self.linear2, module.linear2)

if __name__ == "__main__":
    input = torch.rand([1, 399, 256])
    test = ConformerFeedForward(d_model=256, expansion_factor=4, drop_prob=0.0)
    out = test(input)
    json_save(test, "feedforward", ".")
    
    (batch, len, dim) = input.size()
    input = input.reshape(len, dim)
    print(input)
    print(input.size())
    save_file(input, "input.pt")
    
    (batch, len, dim) = out.size()
    out = out.reshape(len, dim)
    print(out)
    print(out.size())
    save_file(out, "output.pt")
    
    config_dict = {}
    config_dict["num_head"] = 8
    config_dict["num_layer"] = 2
    config_dict["dim_model"] = 512
    config_dict["ffn_hidden"] = 700
    config_dict["model_path"]={}
    config_dict["model_path"]["conformer_feedforward"]=test.save("conformer_feedforward", "")
    
    with open('model.json', 'w') as json_file :
        json.dump(config_dict, json_file, ensure_ascii=False, indent="\t")
