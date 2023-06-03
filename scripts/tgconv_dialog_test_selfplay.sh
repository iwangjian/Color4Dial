#!/bin/bash
base_model="DialoGPT"   # GPT2 | DialoGPT

# data config
dataset="TGConv"
test_data="data/${dataset}/sample_test_selfplay.jsonl"
log_dir="logs/${dataset}/checkpoints_dialog/${base_model}"
turn_type_size=8

infer_checkpoint="best_model.bin"
output_dir="outputs/${dataset}"
max_dec_len=30
temperature=0.95

plan_log_dir="logs/${dataset}/checkpoints_planner"
infer_plan_checkpoint="planner_best_model.bin"
latent_dim=16
max_transition_number=10
easy_hard_mode="easy"     # easy | hard

python main_dialog.py --mode selfplay \
    --base_model ${base_model} \
    --dataset ${dataset} \
    --test_data ${test_data} \
    --log_dir ${log_dir} \
    --turn_type_size ${turn_type_size} \
    --infer_checkpoint ${infer_checkpoint} \
    --output_dir ${output_dir} \
    --max_dec_len ${max_dec_len} \
    --temperature ${temperature} \
    --plan_log_dir ${plan_log_dir} \
    --infer_plan_checkpoint ${infer_plan_checkpoint} \
    --latent_dim ${latent_dim} \
    --max_transition_number ${max_transition_number} \
    --easy_hard_mode ${easy_hard_mode}
