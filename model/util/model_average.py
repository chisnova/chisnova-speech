import torch
from torch import nn
from torch.nn import functional as F
import os
import json
import numpy as np

def linear_average(linear1, linear2):
    state_dict1 = linear1.state_dict()
    state_dict2 = linear2.state_dict()

    #print(state_dict1)
    #print("linear1 average before")

    #print(state_dict2)
    #print("linear2 average before")
    for key in state_dict1:
        state_dict1[key] = state_dict1[key]*0.5 + state_dict2[key]*0.5
    #print(state_dict1)
    #print("linear1 average after")
    linear1.load_state_dict(state_dict1)

def cnn_average(cnn1, cnn2):
    state_dict1 = cnn1.state_dict()
    state_dict2 = cnn2.state_dict()

    for key in state_dict1:
        state_dict1[key] = state_dict1[key]*0.5 + state_dict2[key]*0.5

    cnn1.load_state_dict(state_dict1)

def layernorm_average(norm1, norm2):
    state_dict1 = norm1.state_dict()
    state_dict2 = norm2.state_dict()

    #print(state_dict1)
    #print("layernom average before")
    for key in state_dict1:
        state_dict1[key] = state_dict1[key]*0.5 + state_dict2[key]*0.5

    #print(state_dict1)
    #print("layernom average after")
    norm1.load_state_dict(state_dict1)

def batchnorm_average(norm1, norm2):
    state_dict1 = norm1.state_dict()
    state_dict2 = norm2.state_dict()

    #print(state_dict1)
    #print("batchnorm average before")
    for key in state_dict1:
        state_dict1[key] = state_dict1[key]*0.5 + state_dict2[key]*0.5

    #print(state_dict1)
    #print("batchnorm average after")
    norm1.load_state_dict(state_dict1)


