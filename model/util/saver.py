import os
import numpy as np
import torch
import json

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

def feat_mean_saver(base_path, save_path):
    ret = {}
    ret['value']="mean.pt"
    load_path = os.path.join(base_path, "mean.npy")
    mean = np.load(load_path)
    ret['shape']=[1, mean.shape[0]]
    mean_path = os.path.join(save_path, "mean.pt")
    mean.tofile(mean_path)
    return ret

def feat_std_saver(base_path, save_path):
    ret = {}
    ret={}
    ret['value']="std.pt"
    load_path = os.path.join(base_path, "std.npy")
    std = np.load(load_path)
    ret['shape']=[1, std.shape[0]]
    std_path = os.path.join(save_path, "std.pt")
    std.tofile(std_path)
    return ret


def save_file(val, name):
    val.detach().numpy().tofile(name)

def save_tras_file(val, name):
    val.detach().numpy().T.tofile(name)
    
def save_int_file(val, name):
    val.detach().numpy().astype(np.int32).tofile(name)
    
def save_numpy_file(val, name):
    val.tofile(name)


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

    model_att["shape"] = model.weight.T.shape
    return model_att

def embedding_saver(model, name, save_path):
    model_att = {}
    model_name = name + ".weight.pt"
    weight_path = os.path.join(save_path, model_name)
    model_att["embedding"]= model_name
    save_file(model.weight, weight_path)
    model_att["shape"] = model.weight.shape
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
