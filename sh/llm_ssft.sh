#!/bin/bash
set -euo pipefail

export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

DATASET="${DATASET:-CMEDQA2}"
if [[ "$DATASET" == "CMEDQA2" ]]; then
  TASKS=("cmedqa2")
  TPE_MODEL_PATHS=(
    "data/tpe_models/Qwen2.5-1.5B-Instruct_task-cmedqa2_maxN-2_maxM-5000"
  )
  NUM_TRAIN_EPOCHS=10
elif [[ "$DATASET" == "ECTSUM" ]]; then
  TASKS=("ect_summary")
  TPE_MODEL_PATHS=(
    "data/tpe_models/Qwen2.5-1.5B-Instruct_task-ect_summary_maxN-2_maxM-5000"
  )
  NUM_TRAIN_EPOCHS=10
else
  echo "Unsupported public dataset: $DATASET. Use CMEDQA2 or ECTSUM."
  exit 1
fi

DATA_FORMAT="nl"
DATA_DIR="data/cleaned_SFT_QA"
LOG_DIR="data/"
MAX_INPUT_LEN="4k"
MAX_OUTPUT_LEN="2k"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
LR="${LR:-5e-5}"
VAL_STEPS="${VAL_STEPS:-1000}"

for TASK in "${TASKS[@]}"; do
  for TPE_MODEL_PATH in "${TPE_MODEL_PATHS[@]}"; do
    python llm_sft.py \
      --model_path "$TPE_MODEL_PATH" \
      --task "$TASK" \
      --data_dir "$DATA_DIR" \
      --log_dir "$LOG_DIR" \
      --dataset "$DATASET" \
      --data_format "$DATA_FORMAT" \
      --max_input_len "$MAX_INPUT_LEN" \
      --max_output_len "$MAX_OUTPUT_LEN" \
      --batch_size "$BATCH_SIZE" \
      --num_train_epochs "$NUM_TRAIN_EPOCHS" \
      --grad_accum_steps "$GRAD_ACCUM_STEPS" \
      --lr "$LR" \
      --val_steps "$VAL_STEPS" \
      --gpu_ids "$GPU_IDS"
  done
done
