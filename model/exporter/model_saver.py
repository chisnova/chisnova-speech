import torch
import os
import json
import sys
from torch import nn
import numpy as np
from layers.positional import PositionalEncoding
from layers.conformer_positional import ConformerPositionalEncoding
from layers.conformer_multihead import * 
from layers.conformer_stream_multihead import * 
from layers.multihead import * 
from layers.conformer_stream_conv_module import * 
from module.conformer_encoder import ConformerEncoder
from module.transformer_decoder import * 
from module.conformer_stream_encoder import ConformerStreamEncoder


def json_save(model, name, path, vocab_name, vocab_size):    
    config_dict = {}
    config_dict["model_path"]={}
    config_dict["model_path"]["vocab_name"]=vocab_name
    config_dict["model_path"]["vocab_size"]=vocab_size
    config_dict["model_path"]["mean"]=feat_mean_saver("/home/chisnova/archive2/Mercurius/model/data", path)
    config_dict["model_path"]["std"]=feat_std_saver("/home/chisnova/archive2/Mercurius/model/data", path)
    config_dict["model_path"]["model_type"]= model.type_name()
    config_dict["model_path"][name]=model.save(name, path)
    return config_dict

def save_file(val, name):
    val.detach().numpy().tofile(name)

def save_tras_file(val, name):
    val.detach().numpy().T.tofile(name)
    
def save_int_file(val, name):
    val.detach().numpy().astype(np.int32).tofile(name)
    
def save_numpy_file(val, name):
    val.tofile(name)

def position_saver(module, name, save_path):
    att_dict = {}
    pos_encoding_name = name + ".encoding.pt"
    pos_encoding_path = os.path.join(save_path, pos_encoding_name)
    att_dict["encoding"] = pos_encoding_name
    att_dict["shape"]=list(module.encoding2D.shape)
    save_file(module.encoding2D, pos_encoding_path)
    return att_dict


def conv_module_saver(model, parents_name, save_path):
    att_dict = {}
    att_dict["ksize"]=model.ksize
    linear_json={}
    for name, module in model.named_children():
        linear_json[name] = save_model(module, parents_name+"_"+name, save_path)
    att_dict.update(linear_json)
    return att_dict

def head_saver(model, parents_name, save_path):
    att_dict = {}
    att_dict["d_model"]=model.d_model
    att_dict["n_head"]=model.n_head
    linear_json={}
    for name, module in model.named_children():
        linear_json[name] = save_model(module, parents_name+"_"+name, save_path)
    att_dict.update(linear_json)
    return att_dict

def multihead_saver(model, parents_name, save_path):
    att_dict = {}
    att_dict["n_head"]=model.n_head
    att_dict["d_model"]=model.d_model
    linear_json={}
    for name, module in model.named_children():
        linear_json[name] = save_model(module, parents_name+"_"+name, save_path)
    att_dict.update(linear_json)
    return att_dict

def enc_dec_saver(model, parents_name, save_path):
    att_dict = {}
    att_dict["num_layer"]=model.n_layers
    linear_json={}
    for name, module in model.named_children():
        linear_json[name] = save_model(module, parents_name+"_"+name, save_path)
    att_dict.update(linear_json)
    return att_dict

def stream_enc_saver(model, parents_name, save_path):
    att_dict = {}
    att_dict["num_layer"]=model.n_layers
    att_dict["num_head"]=model.n_head
    att_dict["dim_model"]=model.d_model
    linear_json={}
    for name, module in model.named_children():
        linear_json[name] = save_model(module, parents_name+"_"+name, save_path)
    att_dict.update(linear_json)
    return att_dict

def conformer_enc_saver(model, parents_name, save_path):
    att_dict = {}
    att_dict["num_layer"]=model.n_layers
    att_dict["num_head"]=model.n_head
    att_dict["dim_model"]=model.d_model
    linear_json={}
    for name, module in model.named_children():
        linear_json[name] = save_model(module, parents_name+"_"+name, save_path)
    att_dict.update(linear_json)
    return att_dict

def conformer_dec_saver(model, parents_name, save_path):
    att_dict = {}
    att_dict["num_layer"]=model.n_layers
    att_dict["num_head"]=model.n_head
    att_dict["dim_model"]=model.d_model
    att_dict["ffn_hidden"]=model.ffn_hidden
    linear_json={}
    for name, module in model.named_children():
        linear_json[name] = save_model(module, parents_name+"_"+name, save_path)
    att_dict.update(linear_json)
    return att_dict


def layernorm_saver(layernorm, name, save_path):
    model_att = {}
    model_name = name + ".weight.pt"
    weight_path = os.path.join(save_path, model_name)
    model_att["weight"]= model_name
    save_tras_file(layernorm.weight, weight_path)

    bias_name = name + ".bias.pt"
    bias_path = os.path.join(save_path, bias_name)
    model_att["bias"]= bias_name
    save_tras_file(layernorm.bias, bias_path)
    
    model_att["shape"]=[1,layernorm.weight.shape[0]]
    return model_att

def linear_saver(model, name, save_path):
    model_att = {}
    model_name = name + ".weight.pt"
    weight_path = os.path.join(save_path, model_name)
    model_att["weight"]= model_name 
    save_tras_file(model.weight, weight_path)
    
    bias_name = name + ".bias.pt"
    bias_path = os.path.join(save_path, bias_name)
    model_att["bias"]= bias_name
    save_tras_file(model.bias, bias_path)

    model_att["shape"] = list(model.weight.T.detach().numpy().shape)
    return model_att

def embedding_saver(model, name, save_path):
    model_att = {}
    model_name = name + ".weight.pt"
    weight_path = os.path.join(save_path, model_name)
    model_att["embedding"]= model_name
    save_file(model.weight, weight_path)
    model_att["shape"] = list(model.weight.detach().numpy().shape)
    return model_att


class CNN1DSaver:    
    def save_file(self, val, name):
        val.detach().numpy().tofile(name)

    def save_tras_file(self, val, name):
        val.detach().numpy().transpose(1,0,2).tofile(name)
        
    def save_numpy_file(self, val, name):
        val.tofile(name)

    def __init__(self, model):      
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels
        self.kernel_size = model.kernel_size
        self.stride=model.stride
        self.padding=model.padding
        self.groups=model.groups
        self.bias=model.bias
        self.dilation=model.dilation
        
        self.cnn = model
        self.cnn_weight = self.cnn.weight
        self.weight_shape = []
        self.cnn_bias = self.cnn.bias
    
    def cnn_saver(self, name, save_path):
        self.save_binary(name, save_path)
        
        model_att = {}
        model_att["in_channels"] = self.in_channels
        model_att["out_channels"] = self.out_channels
        model_att["kernel_size"] = self.kernel_size[0]
        model_att["stride"] = self.stride[0]
        model_att["padding"] = self.padding[0]
        model_att["dilation"] = self.dilation[0]
        model_att["groups"] = self.groups
        if self.bias != None:
            model_att["bias"] = 1
        else:
            model_att["bias"] = 0
            
        model_att["weight_path"] = name + "_weight.pt"
        model_att["bias_path"]= name + "_bias.pt"
        model_att["shape"]=list(self.weight.shape)
        return model_att
    
    def save_binary(self, name, save_path):
        self.weight = self.make_weight2D()
        weight_path = os.path.join(save_path, name +"_weight.pt")
        bias_path = os.path.join(save_path, name +"_bias.pt")
        self.save_numpy_file(self.weight, weight_path)
        if self.cnn.bias != None:
            self.save_file(self.cnn.bias, bias_path)
    
    def make_weight2D(self):
        weight = self.cnn.weight.detach().numpy()
        weight = weight
        (ou_ch, in_ch, k) = weight.shape
        weight = weight.reshape(ou_ch, in_ch*k)
        return weight


class CNN2DSaver:
    def __init__(self, model):
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels
        self.kernel_size = model.kernel_size
        self.stride=model.stride
        self.padding=model.padding
        self.groups=model.groups
        self.bias=model.bias
        self.dilation=model.dilation
        
        self.cnn = model
        self.cnn_weight = self.cnn.weight
        self.weight_shape = []
        self.cnn_bias = self.cnn.bias
        
    def cnn_saver(self, name, save_path):
        self.save_binary(name, save_path)
        
        model_att = {}        
        model_att["in_channels"] = self.in_channels
        model_att["out_channels"] = self.out_channels
        model_att["kernel_size"] = self.kernel_size[0]
        model_att["stride"] = self.stride[0]
        model_att["padding"] = self.padding[0]
        model_att["dilation"] = self.dilation[0]
        model_att["groups"] = self.groups

        if self.bias != None:
            model_att["bias"] = 1
        else:
            model_att["bias"] = 0
        model_att["weight_path"] = name + "_weight.pt"
        model_att["bias_path"]= name + "_bias.pt"
        model_att["shape"]=self.weight_shape
        return model_att
    
    def make_weight2D(self):
        weight = self.cnn.weight.detach().numpy()
        weight = weight
        (ou_ch, in_ch, k1, k2) = weight.shape
        weight = weight.reshape(ou_ch, in_ch*k1*k2)
        self.weight_shape = weight.shape
        return weight
    
    def save_binary(self, name, save_path):
        self.weight = self.make_weight2D()
        weight_path = os.path.join(save_path, name + "_weight.pt")
        bias_path = os.path.join(save_path, name + "_bias.pt")
        save_numpy_file(self.weight, weight_path)
        if self.cnn.bias != None:
            save_file(self.cnn.bias, bias_path)
            
            
class BatchSaver:
    def __init__(self, batchnorm):
        self.batchnorm = batchnorm
        self.batchnorm = self.batchnorm.eval()
        self.mean = self.batchnorm.running_mean
        self.std = torch.sqrt(self.batchnorm.running_var + self.batchnorm.eps)
        self.weight = self.batchnorm.weight
        self.bias = self.batchnorm.bias
        
    def batch_saver(self, name, save_path):
        self.save_binary(name, save_path)
        
        model_att = {}
        model_att["mean"] = name + "_mean.pt"
        model_att["std"]= name + "_std.pt"
        model_att["weight"] = name + "_weight.pt"
        model_att["bias"]= name + "_bias.pt"
        #model_att["shape"]=[1, self.weight.shape[0]]
        model_att["shape"]=[self.weight.shape[0], 1]
        return model_att
    
    def save_binary(self, name, save_path):
        mean_path = os.path.join(save_path, name + "_mean.pt")
        std_path = os.path.join(save_path, name + "_std.pt")
        weight_path = os.path.join(save_path, name + "_weight.pt")
        bias_path = os.path.join(save_path, name + "_bias.pt")
        save_file(self.mean, mean_path)
        save_file(self.std, std_path)
        save_file(self.weight, weight_path)
        save_file(self.bias, bias_path)

def save_model(model, parents_name, save_path):
    ret={}
    if isinstance(model, nn.Conv1d):
        saver = CNN1DSaver(model)
        ret=saver.cnn_saver(parents_name, save_path)
    elif isinstance(model, nn.Conv2d):
        saver = CNN2DSaver(model)
        ret=saver.cnn_saver(parents_name, save_path)
    elif isinstance(model, nn.BatchNorm1d):
        saver = BatchSaver(model)
        ret= saver.batch_saver(parents_name, save_path)
    elif isinstance(model, nn.Embedding):
        ret=embedding_saver(model, parents_name, save_path)
    elif isinstance(model, nn.Linear):
        ret=linear_saver(model, parents_name, save_path)
    elif isinstance(model, nn.LayerNorm):
        ret=layernorm_saver(model, parents_name, save_path)
    elif isinstance(model, PositionalEncoding):
        ret=position_saver(model, parents_name, save_path)
    elif isinstance(model, ConformerPositionalEncoding):
        ret=position_saver(model, parents_name, save_path)
    elif isinstance(model, ConformerMultiHead):
        ret=multihead_saver(model, parents_name, save_path)
    elif isinstance(model, ConformerStreamMultiHead):
        ret=multihead_saver(model, parents_name, save_path)
    elif isinstance(model, MultiHead):
        ret=multihead_saver(model, parents_name, save_path)
    elif isinstance(model, ConformerStreamEncoder):
        ret=stream_enc_saver(model, parents_name, save_path)
    elif isinstance(model, ConformerEncoder):
        ret=conformer_enc_saver(model, parents_name, save_path)
    elif isinstance(model, TransformerDecoder):
        ret=conformer_dec_saver(model, parents_name, save_path)
    elif isinstance(model, ConformerStreamConvModule):
        ret=conv_module_saver(model, parents_name, save_path)
    else:
        for name, module in model.named_children():
            ret[name]=save_model(module, parents_name+"_"+name, save_path)
    return ret
