# data config
dataset="DuRecDial2"
train_data="data/${dataset}/sample_train.jsonl"
dev_data="data/${dataset}/sample_dev.jsonl"
test_seen_data="data/${dataset}/sample_test_seen.jsonl"
test_unseen_data="data/${dataset}/sample_test_unseen.jsonl"
cache_dir="caches/${dataset}"
log_dir="logs/${dataset}"

# network args
latent_dim=16
max_transition_number=8
freeze_plm="true"
use_transform="false"

# infer args
output_dir="outputs/${dataset}"
infer_use_bridge="true"
max_dec_len=80

python main_planning.py --mode infer_planner \
    --dataset ${dataset} \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --test_seen_data ${test_seen_data} \
    --test_unseen_data ${test_unseen_data} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --latent_dim ${latent_dim} \
    --max_transition_number ${max_transition_number} \
    --freeze_plm ${freeze_plm} \
    --use_transform ${use_transform} \
    --output_dir ${output_dir} \
    --infer_use_bridge ${infer_use_bridge} \
    --max_dec_len ${max_dec_len}