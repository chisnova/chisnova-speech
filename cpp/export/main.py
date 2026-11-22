import torch
from util.saver import *
from app.asr_conformer import AsrConformer
import os
from util.wav_infer import *

def getConformerModel():
    model = AsrConformer(d_model=256, enc_n_head=4, enc_nlayer=13, dec_hidden=512, dec_n_head=8, dec_nlayer=7, vocab_size=1000, drop_prob=0.0).eval()
    current_path = os.getcwd()
    model_suffix_path = "../../model/conformer_medium"
    model_path = os.path.join(current_path, model_suffix_path)
    model_path = os.path.join(model_path, "asr_model.pt")
    if os.path.isfile(model_path): 
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path,map_location=device))
        print("ASR model loaded")

    return model

if __name__ == "__main__":
    model = getConformerModel()
    model.eval()
    config_dict = json_save(model, "asr_conformer", "model_state", "mecury.vocab", 1000)
    with open('model_state/model.json', 'w') as json_file :
        json.dump(config_dict, json_file, ensure_ascii=False, indent="\t")

