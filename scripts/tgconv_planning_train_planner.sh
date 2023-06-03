# data config
dataset="TGConv"
train_data="data/${dataset}/sample_train.jsonl"
dev_data="data/${dataset}/sample_dev.jsonl"
test_data="data/${dataset}/sample_test.jsonl"
cache_dir="caches/${dataset}"
log_dir="logs/${dataset}"
turn_type_size=8

# training args
train_batch_size_planner=16
num_epochs=10
log_steps=100
validate_steps=1000
lr=2e-4
warmup_ratio=0.1
latent_dim=16
max_transition_number=10   # set according to the dataset
freeze_plm="true"
eval_brownian_bridge="true"
use_transform="false"
trans_alpha=0.1
gen_beta=1.0
kl_gamma=1.0

python main_planning.py --mode train_planner \
    --dataset ${dataset} \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --test_data ${test_data} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --turn_type_size ${turn_type_size} \
    --train_batch_size_planner ${train_batch_size_planner} \
    --num_epochs ${num_epochs} \
    --log_steps ${log_steps} \
    --validate_steps ${validate_steps} \
    --lr ${lr} \
    --warmup_ratio ${warmup_ratio} \
    --latent_dim ${latent_dim} \
    --max_transition_number ${max_transition_number} \
    --freeze_plm ${freeze_plm} \
    --eval_brownian_bridge ${eval_brownian_bridge} \
    --use_transform ${use_transform} \
    --trans_alpha ${trans_alpha} \
    --gen_beta ${gen_beta} \
    --kl_gamma ${kl_gamma}
