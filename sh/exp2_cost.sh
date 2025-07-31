export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

DATASET="EHRSHOT"
SPLIT="held_out"
LOG_DIR="log/token_stats"

python calculate_token_stats.py \
    --dataset $DATASET \
    --data_split $SPLIT \
    --log_dir $LOG_DIR
    