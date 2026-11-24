import os
from util.model_saver import *
import librosa
import copy
import sys
import torch
import sys
from util.infer import * 
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple program to demonstrate argparse.")

    # Add arguments
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--wav_path', type=str, default="")
    parser.add_argument('--stream', type=bool, default=False)
    parser.add_argument('--mercury_path', type=str, default="")
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--chunk_size', type=int, default=16000) #ms
    parser.add_argument('--label', type=str, default="")
    parser.add_argument('--ext_lm', type=bool, default=False)
    parser.add_argument('--norm_case', type=str, default="bayesian")

    # Parse the arguments
    args = parser.parse_args()

    model, config =getModel(args.model_path)
    infer=INFERENCE(model)
    res=infer.stream_wav_run(args)

    save_mainV2(model, config, args.output_path)
    

