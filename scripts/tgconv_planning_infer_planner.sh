# data config
dataset="TGConv"
train_data="data/${dataset}/sample_train.jsonl"
dev_data="data/${dataset}/sample_dev.jsonl"
test_data="data/${dataset}/sample_test.jsonl"
cache_dir="caches/${dataset}"
log_dir="logs/${dataset}"
turn_type_size=8

# network args
latent_dim=16
max_transition_number=10   # set according to the dataset
freeze_plm="true"
use_transform="false"

# infer args
output_dir="outputs/${dataset}"
infer_use_bridge="true"
max_dec_len=20

python main_planning.py --mode infer_planner \
    --dataset ${dataset} \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --test_data ${test_data} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --turn_type_size ${turn_type_size} \
    --latent_dim ${latent_dim} \
    --max_transition_number ${max_transition_number} \
    --freeze_plm ${freeze_plm} \
    --use_transform ${use_transform} \
    --output_dir ${output_dir} \
    --infer_use_bridge ${infer_use_bridge} \
    --max_dec_len ${max_dec_len}