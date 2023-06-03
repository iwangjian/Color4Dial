#!/bin/bash
base_model="DialoGPT" # GPT2 | DialoGPT

# data config
dataset="DuRecDial2"
train_data="data/${dataset}/sample_train.jsonl"
dev_data="data/${dataset}/sample_dev.jsonl"
cache_dir="caches/${dataset}"
log_dir="logs/${dataset}/checkpoints_dialog/${base_model}"

# training args
num_epochs=10
batch_size=6
validate_steps=2000
lr=5e-5

python main_dialog.py --mode train \
    --base_model ${base_model} \
    --dataset ${dataset} \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    --validate_steps ${validate_steps} \
    --lr ${lr}
