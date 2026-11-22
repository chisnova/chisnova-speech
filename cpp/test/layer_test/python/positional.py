import torch
from torch import nn
import os
import copy
# from util.saver import *


def save_file(val, name):
    val.detach().numpy().tofile(name)

def save_tras_file(val, name):
    val.detach().numpy().T.tofile(name)

def save_int_file(val, name):
    val.detach().numpy().astype(np.int32).tofile(name)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()
        max_batch=1500

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.encoding2D = copy.deepcopy(self.encoding)
        self.encoding2D = self.encoding2D.view(max_len, d_model)
        self.encoding = torch.flatten(self.encoding)
        self.encoding = torch.tile(self.encoding, (max_batch,))
        self.encoding = self.encoding.view(max_batch, max_len, d_model)

    def forward(self, x):
        (batch_size, seq_len, dim) = x.size()
        #print("feat ", x.size())
        device = x.get_device()
        if device >= 0:
            pos_emb = self.encoding[:batch_size, :seq_len, :].to(device)
        else:
            pos_emb = self.encoding[:batch_size, :seq_len, :]

        #print("positional ", pos_emb.size())
        x = torch.add(x, pos_emb)
        return x

    def save(self, name, save_path):
        att_dict = {}
        pos_encoding_name = name + ".encoding.pt"
        pos_encoding_path = os.path.join(save_path, pos_encoding_name)
        att_dict["encoding"] = pos_encoding_name
        att_dict["shape"]=list(self.encoding2D.shape)
        save_file(self.encoding2D, pos_encoding_path)
        return att_dict


if __name__ == "__main__":
   random = torch.rand([3, 100, 512])
   test = PositionalEncoding(512)