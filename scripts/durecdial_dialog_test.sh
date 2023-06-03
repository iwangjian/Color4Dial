#!/bin/bash
############### Test-seen ###############
base_model="DialoGPT"  # GPT2 | DialoGPT

# data config
dataset="DuRecDial2"
test_data="data/${dataset}/sample_test_seen.jsonl"
plan_data="outputs/${dataset}/plan_test_seen.jsonl"
cache_dir="caches/${dataset}"
log_dir="logs/${dataset}/checkpoints_dialog/${base_model}"

# decoding args
infer_checkpoint="best_model.bin"
output_dir="outputs/${dataset}"
test_batch_size=4
max_dec_len=100
temperature=0.95

python main_dialog.py --mode test \
    --base_model ${base_model} \
    --dataset ${dataset} \
    --test_data ${test_data} \
    --plan_data ${plan_data} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --infer_checkpoint ${infer_checkpoint} \
    --output_dir ${output_dir} \
    --test_batch_size ${test_batch_size} \
    --max_dec_len ${max_dec_len} \
    --temperature ${temperature}

############### Test-unseen ###############
base_model="DialoGPT"  # GPT2 | DialoGPT

# data config
dataset="DuRecDial2"
test_data="data/${dataset}/sample_test_unseen.jsonl"
plan_data="outputs/${dataset}/plan_test_unseen.jsonl"
cache_dir="caches/${dataset}"
log_dir="logs/${dataset}/checkpoints_dialog/${base_model}"

# decoding args
infer_checkpoint="best_model.bin"
output_dir="outputs/${dataset}"
test_batch_size=4
max_dec_len=100
temperature=0.95

python main_dialog.py --mode test \
    --base_model ${base_model} \
    --dataset ${dataset} \
    --test_data ${test_data} \
    --plan_data ${plan_data} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --infer_checkpoint ${infer_checkpoint} \
    --output_dir ${output_dir} \
    --test_batch_size ${test_batch_size} \
    --max_dec_len ${max_dec_len} \
    --temperature ${temperature}
