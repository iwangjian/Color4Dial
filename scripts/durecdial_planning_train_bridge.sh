# data config
dataset="DuRecDial2"
train_data="data/${dataset}/sample_train.jsonl"
dev_data="data/${dataset}/sample_dev.jsonl"
test_seen_data="data/${dataset}/sample_test_seen.jsonl"
test_unseen_data="data/${dataset}/sample_test_unseen.jsonl"
cache_dir="caches/${dataset}"
log_dir="logs/${dataset}"

# training args
train_batch_size_bridge=64
num_epochs=10
log_steps=100
validate_steps=800
lr=2e-4
warmup_ratio=0.0
latent_dim=16
max_transition_number=8    # set according to the dataset
freeze_plm="true"
eval_brownian_bridge="true"
use_transform="false"

python main_planning.py --mode train_bridge \
    --dataset ${dataset} \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --test_seen_data ${test_seen_data} \
    --test_unseen_data ${test_unseen_data} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --num_epochs ${num_epochs} \
    --log_steps ${log_steps} \
    --validate_steps ${validate_steps} \
    --lr ${lr} \
    --warmup_ratio ${warmup_ratio} \
    --train_batch_size_bridge ${train_batch_size_bridge} \
    --latent_dim ${latent_dim} \
    --max_transition_number ${max_transition_number} \
    --freeze_plm ${freeze_plm} \
    --eval_brownian_bridge ${eval_brownian_bridge} \
    --use_transform ${use_transform}
    