from conformer_positional import ConformerPositionalEncoding
from positional import PositionalEncoding
import torch
from torch import nn
from torch.nn import functional as F
import json
import numpy as np
import os
import time
import functools
from rotary_embedding import RotaryEmbedding

def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function: {func.__name__}, Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def save_file(val, name):
    val.detach().numpy().tofile(name)

def save_tras_file(val, name):
    val.detach().numpy().T.tofile(name)

def save_int_file(val, name):
    val.detach().numpy().astype(np.int32).tofile(name)


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
        model_att["shape"]=[1, self.weight.shape[0]]
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

@timer_decorator
def saveBatchNorm():
    print("\nExecuting saveBatchNorm...")
    input_size = 512
    num_channel=300
    
    input = torch.rand([1, num_channel, input_size])
    batchnorm = nn.BatchNorm1d(num_channel)
    batchnorm = batchnorm.eval()
    output = batchnorm(input)
    intput = input.reshape(num_channel, input_size)
    save_file(intput, "binary/batch_1d_input.pt")
    output = output.reshape(num_channel, input_size)
    save_file(output, "binary/batch_1d_output.pt")

    saver = BatchSaver(batchnorm)
    saver.save_binary("batch_1d", "binary")

    batch_norm_dict = {}
    batch_norm_dict["input"]={}
    batch_norm_dict["input"]["value"]="batch_1d_input.pt"
    batch_norm_dict["input"]["shape"]=list(intput.shape)
    batch_norm_dict["output"]={}
    batch_norm_dict["output"]["value"]="batch_1d_output.pt"
    batch_norm_dict["output"]["shape"]=list(output.shape)
    batch_norm_dict["model"]={}
    batch_norm_dict["model"]["mean"] = "batch_1d_mean.pt"
    batch_norm_dict["model"]["std"]= "batch_1d_std.pt"
    batch_norm_dict["model"]["weight"] = "batch_1d_weight.pt"
    batch_norm_dict["model"]["bias"]= "batch_1d_bias.pt"
    batch_norm_dict["model"]["shape"]= [saver.weight.shape[0],1]
    return batch_norm_dict


class CNN1DSaver:    
    def save_file(self, val, name):
        val.detach().numpy().tofile(name)

    def save_tras_file(self, val, name):
        val.detach().numpy().transpose(1,0,2).tofile(name)
        
    def save_numpy_file(self, val, name):
        val.tofile(name)

    def __init__(self):
        self.in_channels = 256
        self.out_channels = 256*2
        # self.kernel_size = 1
        self.kernel_size = 31
        self.stride=1
        self.padding=(self.kernel_size-1)//2
        # self.padding=0
        #self.groups=128
        self.groups=1
        self.bias=True
        self.dilation=1
        
        self.cnn = nn.Conv1d(in_channels=self.in_channels, 
                             out_channels=self.out_channels, 
                             kernel_size=self.kernel_size, 
                             stride=self.stride, 
                             padding=self.padding,
                             groups=self.groups,
                             bias=self.bias,
                             dilation=self.dilation)

        
        self.input = torch.rand([1, self.in_channels, 399])
        self.cnn_weight = self.cnn.weight
        self.cnn_bias = self.cnn.bias
        self.output = self.cnn(self.input)
    
        
    def cnn_saver(self, name, save_path):
        model_att = {}
        model_att["input"]={}
        model_att["input"]["value"] = name + "_input.pt"
        model_att["input"]["shape"] = list(self.input.shape)

        model_att["output"]={}
        model_att["output"]["value"] = name + "_output.pt"
        model_att["output"]["shape"] = list(self.output.shape)

        model_att["model"]={}
        model_att["model"]["in_channels"] = self.in_channels
        model_att["model"]["out_channels"] = self.out_channels
        model_att["model"]["kernel_size"] = self.kernel_size
        model_att["model"]["stride"] = self.stride
        model_att["model"]["padding"] = self.padding
        model_att["model"]["dilation"] = self.dilation
        model_att["model"]["groups"] = self.groups
        if self.bias :
            model_att["model"]["bias"] = 1
        else:
            model_att["model"]["bias"] = 0
        model_att["model"]["weight_path"] = name + "_weight.pt"
        model_att["model"]["bias_path"]= name + "_bias.pt"
        model_att["model"]["shape"]=list(self.weight.shape)
        return model_att
    
    # def save_binary(self):
    def save_binary(self, name, save_path):
        input_path = os.path.join(save_path, name + "_input.pt")
        output_path = os.path.join(save_path, name + "_output.pt")
        weight_path = os.path.join(save_path, name + "_weight.pt")
        bias_path = os.path.join(save_path, name + "_bias.pt")

        (batch, ch, dim) = self.input.size()
        self.input = self.input.reshape(ch, dim)
        (batch, ch, dim) = self.output.size()
        self.output = self.output.reshape(ch, dim)
        self.save_file(self.input, input_path)    
        self.save_file(self.output, output_path)
        self.weight = self.make_weight2D()
        self.save_numpy_file(self.weight, weight_path)
        if self.cnn.bias != None:
            self.save_file(self.cnn.bias, bias_path)
            
    def json_saver(self, name, save_path):
        self.save_binary(name, save_path)
        return self.cnn_saver(name, save_path)
    
    def make_weight2D(self):
        weight = self.cnn.weight.detach().numpy()
        weight = weight
        (ou_ch, in_ch, k) = weight.shape
        weight = weight.reshape(ou_ch, in_ch*k)
        return weight


@timer_decorator
def saveConvolution1D():
    print("\nExecuting saveConvolution1D...")
    saver = CNN1DSaver()
    return saver.json_saver("cnn_1d", "binary")


class CNN2DSaver:    
    def save_file(self, val, name):
        val.detach().numpy().tofile(name)

    def save_tras_file(self, val, name):
        val.detach().numpy().transpose(1,0,2).tofile(name)
        
    def save_numpy_file(self, val, name):
        val.tofile(name)

    def __init__(self):
        # self.in_channels = 64
        self.in_channels = 64
        self.out_channels = 64
        self.kernel_size = 3
        self.stride=2
        self.padding=0
        #self.groups=self.in_channels
        self.groups=1
        self.bias=True
        self.dilation=1
        
        self.cnn = nn.Conv2d(in_channels=self.in_channels, 
                             out_channels=self.out_channels, 
                             kernel_size=self.kernel_size, 
                             stride=self.stride, 
                             padding=self.padding,
                             groups=self.groups,
                             bias=self.bias,
                             dilation=self.dilation).cpu()
        
        #self.input = torch.rand([1, self.in_channels, 1500, 80])
        self.input = torch.rand([1, self.in_channels, 749, 39])
        self.cnn_weight = self.cnn.weight
        self.weight_shape = []
        self.cnn_bias = self.cnn.bias
        self.output = self.cnn(self.input)
        
    def cnn_saver(self, name, save_path):
        model_att = {}
        model_att["input"]={}
        model_att["input"]["num_channel"]=self.in_channels
        for i in range(self.in_channels):
            input_ = self.input[i,:]
            save_name = name + "_input{}.pt".format(i)
            json_name = "input{}".format(i)
            model_att["input"][json_name]={}
            model_att["input"][json_name]["value"] = save_name
            model_att["input"][json_name]["shape"] = list(input_.shape)
            
        model_att["output"]={}
        model_att["output"]["num_channel"]=self.out_channels
        for i in range(self.out_channels):
            output_ = self.output[i,:]
            save_name = name + "_output{}.pt".format(i)
            json_name = "output{}".format(i)
            model_att["output"][json_name]={}
            model_att["output"][json_name]["value"] = save_name
            model_att["output"][json_name]["shape"] = list(output_.shape)

        model_att["model"] = {}
        model_att["model"]["in_channels"] = self.in_channels
        model_att["model"]["out_channels"] = self.out_channels
        model_att["model"]["kernel_size"] = self.kernel_size
        model_att["model"]["stride"] = self.stride
        model_att["model"]["padding"] = self.padding
        model_att["model"]["dilation"] = self.dilation
        model_att["model"]["groups"] = self.groups
        if self.bias :
            model_att["model"]["bias"] = 1
        else:
            model_att["model"]["bias"] = 0
        model_att["model"]["weight_path"] = name + "_weight.pt"
        model_att["model"]["bias_path"]= name + "_bias.pt"
        model_att["model"]["shape"]=self.weight_shape
        return model_att
    
    def save_binary(self, name, save_path):
        (batch, in_ch, H, W) = self.input.size()
        self.input = self.input.reshape(in_ch, H, W)
        (batch, out_ch, H, W) = self.output.size()
        self.output = self.output.reshape(out_ch, H, W)
        for i in range(in_ch):
            input_ = self.input[i,:]
            input_path = os.path.join(save_path, name + "_input{}.pt".format(i))
            self.save_file(input_, input_path)
            
        for i in range(out_ch):
            output_ = self.output[i,:]
            output_path = os.path.join(save_path, name + "_output{}.pt".format(i))
            self.save_file(output_, output_path)
            
        self.weight = self.make_weight2D()
        weight_path = os.path.join(save_path, name + "_weight.pt")
        self.save_numpy_file(self.weight, weight_path)
        if self.cnn.bias != None:
            bias_path = os.path.join(save_path, name + "_bias.pt")
            self.save_file(self.cnn.bias, bias_path)
            
    def json_saver(self, name, save_path):
        self.save_binary(name, save_path)
        return self.cnn_saver(name, save_path)
    
    def make_weight2D(self):
        weight = self.cnn.weight.detach().numpy()
        weight = weight
        (ou_ch, in_ch, k1, k2) = weight.shape
        weight = weight.reshape(ou_ch, in_ch*k1*k2)
        self.weight_shape = weight.shape
        return weight

@timer_decorator
def saveConvolution2D():
    print("\nExecuting saveConvolution2D...")
    saver = CNN2DSaver()
    return saver.json_saver("cnn2d", "binary")


@timer_decorator
def saveConformerPositional():
    print("\nExecuting saveConformerPositional...")
    _in = torch.rand([1, 100, 512])
    model = ConformerPositionalEncoding(512)
    out = model(_in)
    _in = torch.reshape(_in, (100, 512))
    save_file(_in, "binary/conformer_positional_input.pt")
    out = torch.reshape(out, (100, 512))
    save_file(out, "binary/conformer_positional_output.pt")
    model.save("conformer_positional","binary")

    conformer_positional_dict = {}
    conformer_positional_dict["input"]={}
    conformer_positional_dict["input"]["value"]="conformer_positional_input.pt"
    conformer_positional_dict["input"]["shape"]=list(_in.shape)
    conformer_positional_dict["output"]={}
    conformer_positional_dict["output"]["value"]="conformer_positional_output.pt"
    conformer_positional_dict["output"]["shape"]=list(out.shape)
    conformer_positional_dict["model"]={}
    conformer_positional_dict["model"]["encoding"] = "conformer_positional.positional.encoding.pt"
    conformer_positional_dict["model"]["shape"]= list(model.encoding2D.shape)
    return conformer_positional_dict

@timer_decorator
def saveEmbedding():
    print("\nExecuting saveEmbedding...")
    emb = nn.Embedding(4000, 512)
    _in = torch.randint(0, 4000, (1,20,))
    save_int_file(_in, "binary/embedding_input.pt")

    out = emb(_in)
    out = out.reshape(20, 512)
    save_file(out, "binary/embedding_output.pt")

    # model_att["embedding"]= weight_path
    save_file(emb.weight, "binary/embedding_weight.pt")
    # model_att["shape"] = model.weight.shape

    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="embedding_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="embedding_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    model_att["model"]={}
    model_att["model"]["embedding"] = "embedding_weight.pt"
    model_att["model"]["shape"]= list(emb.weight.shape)
    return model_att

@timer_decorator
def saveGLU():
    print("\nExecuting saveGLU...")
    # [2026-04-08] dim=1 → dim=0으로 변경: C++ GLU::forward2는 행(채널) 방향 분할
    # C++에서 Conv1D 출력 [out_channels, time]에 대해 topRows/bottomRows로 분할하므로 dim=0이 맞음
    glu = nn.GLU(dim=0)
    _in = torch.randn([50, 512])
    save_file(_in, "binary/glu_input.pt")
    out = glu(_in)
    save_file(out, "binary/glu_output.pt")
    

    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="glu_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="glu_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    return model_att


@timer_decorator
def saveLayerNorm():    
    print("\nExecuting saveLayerNorm...")
    layernorm = nn.LayerNorm(512)
    _in = torch.rand([50, 512])
    save_file(_in, "binary/layernorm_input.pt")
    save_tras_file(layernorm.weight, "binary/layernorm_weight.pt")
    save_tras_file(layernorm.bias, "binary/layernorm_bias.pt")
    out = layernorm(_in)
    save_file(out, "binary/layernorm_output.pt")
    

    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="layernorm_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="layernorm_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    model_att["model"]={}
    model_att["model"]["weight"] = "layernorm_weight.pt"
    model_att["model"]["bias"] = "layernorm_bias.pt"
    model_att["model"]["shape"]= list([1, layernorm.weight.shape[0]])
    return model_att


@timer_decorator
def saveLinear():    
    print("\nExecuting saveLinear...")
    dense = nn.Linear(4096, 8192)
    _in = torch.rand([500, 4096])

    save_file(_in, "binary/linear_input.pt")
    save_tras_file(dense.weight, "binary/linear_weight.pt")
    save_tras_file(dense.bias, "binary/linear_bias.pt")
    out = dense(_in)
    save_file(out, "binary/linear_output.pt")
    

    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="linear_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="linear_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    model_att["model"]={}
    model_att["model"]["weight"] = "linear_weight.pt"
    model_att["model"]["bias"] = "linear_bias.pt"
    model_att["model"]["shape"]= list(dense.weight.T.shape)
    return model_att

@timer_decorator
def savePositional():
    print("\nExecuting savePositional...")
    _in = torch.rand([1, 100, 512])
    test = PositionalEncoding(512)
    out = test(_in)
    _in = torch.reshape(_in, (100, 512))
    save_file(_in, "binary/positional_input.pt")

    out = torch.reshape(out, (100, 512))
    save_file(out, "binary/positional_output.pt")

    pos_encoding_path = "binary/positional_encoding.pt"
    save_file(test.encoding2D, pos_encoding_path)

    positional_dict = {}
    positional_dict["input"]={}
    positional_dict["input"]["value"]="positional_input.pt"
    positional_dict["input"]["shape"]=list(_in.shape)
    positional_dict["model"]={}
    positional_dict["model"]["encoding"] = "positional_encoding.pt"
    positional_dict["model"]["shape"]= list(test.encoding2D.shape)
    positional_dict["output"]={}
    positional_dict["output"]["value"]="positional_output.pt"
    positional_dict["output"]["shape"]=list(out.shape)
    return positional_dict


@timer_decorator
def saveRelu():    
    print("\nExecuting saveRelu...")
    _in = torch.randn([500, 512])
    save_file(_in, "binary/relu_input.pt")
    out = F.relu(_in)
    save_file(out, "binary/relu_output.pt")
    
    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="relu_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="relu_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    return model_att

@timer_decorator
def saveSilu():    
    print("\nExecuting saveSilu...")
    _in = torch.randn([50, 512])
    save_file(_in, "binary/silu_input.pt")
    silu = nn.SiLU()
    out = silu(_in)
    save_file(out, "binary/silu_output.pt")
    
    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="silu_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="silu_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    return model_att

@timer_decorator
def saveSoftmax():    
    print("\nExecuting saveSoftmax...")
    _in = torch.randn([40, 1000])
    save_file(_in, "binary/softmax_input.pt")
    out = F.softmax(_in, -1)
    save_file(out, "binary/softmax_output.pt")
    
    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="softmax_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="softmax_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    return model_att

@timer_decorator
def saveSigmoid():    
    print("\nExecuting saveSigmoid...")
    _in = torch.randn([40, 1000])
    save_file(_in, "binary/sigmoid_input.pt")
    out = torch.sigmoid(_in)
    save_file(out, "binary/sigmoid_output.pt")
    
    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="sigmoid_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="sigmoid_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    return model_att

@timer_decorator
def saveTanh():    
    print("\nExecuting saveTanh...")
    _in = torch.randn([40, 1000])
    save_file(_in, "binary/tanh_input.pt")
    out = torch.tanh(_in)
    save_file(out, "binary/tanh_output.pt")
    
    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="tanh_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="tanh_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    return model_att

@timer_decorator
def saveRotaryCosSin():
    print("\nExecuting saveRotaryCosSin...")
    # 입력 텐서 생성
    _in = torch.rand([1, 1500, 512])  # [batch_size, seq_len, dim]
    
    # 모델 초기화 및 실행
    model = RotaryEmbedding(512)  # dim=512
    # out = model(_in)
    
    # Create position IDs for the input sequence
    position_ids = torch.arange(1500).unsqueeze(0)  # [1, seq_len]
    # Get cos and sin from the model
    # cos, sin = model(_in, position_ids)
    
    rotated = model.rotate_half(_in)
    out = (_in * _in) + (rotated * _in)

    _in = torch.reshape(_in, (1500, 512))
    save_file(_in, "binary/rotary_input.pt")
    
    # 출력 텐서 저장
    out = torch.reshape(out, (1500, 512))
    save_file(out, "binary/rotary_output.pt")

    # sin = torch.reshape(sin, (1500, 512))
    # cos = torch.reshape(cos, (1500, 512))

    # # cos, sin 값 저장
    # save_file(cos, "binary/rotary_cos.pt")
    # save_file(sin, "binary/rotary_sin.pt")
    

    # JSON 설정 생성
    rotary_dict = {}
    rotary_dict["input"] = {}
    rotary_dict["input"]["value"] = "rotary_input.pt"
    rotary_dict["input"]["shape"] = list(_in.shape)
    
    rotary_dict["output"] = {}
    rotary_dict["output"]["value"] = "rotary_output.pt"
    rotary_dict["output"]["shape"] = list(out.shape)
    
    # rotary_dict["model"] = {}
    # rotary_dict["model"]["cos"] = "rotary_cos.pt"
    # rotary_dict["model"]["sin"] = "rotary_sin.pt"
    # rotary_dict["model"]["shape"] = list(cos.shape)
    
    return rotary_dict

@timer_decorator
def saveGELU():
    print("\nExecuting saveGELU...")
    _in = torch.randn([50, 512])
    save_file(_in, "binary/gelu_input.pt")
    gelu = nn.GELU()
    out = gelu(_in)
    save_file(out, "binary/gelu_output.pt")

    model_att = {}
    model_att["input"]={}
    model_att["input"]["value"]="gelu_input.pt"
    model_att["input"]["shape"]=list(_in.shape)
    model_att["output"]={}
    model_att["output"]["value"]="gelu_output.pt"
    model_att["output"]["shape"]=list(out.shape)
    return model_att

if __name__ == "__main__":
    config_dict = {}
    config_dict["layer_test"]={}
    config_dict["layer_test"]["positional"] = savePositional()
    config_dict["layer_test"]["batchnorm1d"] = saveBatchNorm()
    config_dict["layer_test"]["cnn1d"] = saveConvolution1D()
    config_dict["layer_test"]["cnn2d"] = saveConvolution2D()
    config_dict["layer_test"]["conformer_positional"] = saveConformerPositional()
    config_dict["layer_test"]["embedding"] = saveEmbedding()
    config_dict["layer_test"]["glu"] = saveGLU()
    config_dict["layer_test"]["layernorm"] = saveLayerNorm()
    config_dict["layer_test"]["linear"] = saveLinear()
    config_dict["layer_test"]["relu"] = saveRelu()
    config_dict["layer_test"]["silu"] = saveSilu()
    config_dict["layer_test"]["softmax"] = saveSoftmax()
    config_dict["layer_test"]["sigmoid"] = saveSigmoid()
    config_dict["layer_test"]["tanh"] = saveTanh()
    config_dict["layer_test"]["rotary"] = saveRotaryCosSin()
    config_dict["layer_test"]["gelu"] = saveGELU()
    
    with open('binary/layer_test.json', 'w') as json_file :
        json.dump(config_dict, json_file, ensure_ascii=False, indent="\t")
    