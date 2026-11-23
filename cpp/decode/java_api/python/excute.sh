#!/bin/bash

current_dir=$(pwd)
export PYTHONPATH=$current_dir/../../../../model
model_path=$current_dir/../../../../model/stream_conformer_transducer

python3 saveModelAndInfer.py  \
    --model_path $model_path \
    --output_path binary \
    --wav_path ../../../test/resource/StevJobs_Standford/processing/SteveJobsStanford00001.wav
