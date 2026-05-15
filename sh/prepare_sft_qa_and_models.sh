#!/bin/bash
set -euo pipefail

export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

MODEL_PATHS=(
  "data/hf_models/Qwen--Qwen2.5-1.5B-Instruct"
)

SFT_QA_DIR="data/SFT_QA"
CLEANED_SFT_QA="data/cleaned_SFT_QA"
TPE_TOKENIZER_DIR="data/tpe_tokenizers"
TPE_MODEL_DIR="data/tpe_models"

DATASET="${DATASET:-CMEDQA2}"
if [[ "$DATASET" == "CMEDQA2" ]]; then
  TASKS=("cmedqa2")
  RAW_DATA_DIR="data/medtpe_data/cmedqa2"
elif [[ "$DATASET" == "ECTSUM" ]]; then
  TASKS=("ect_summary")
  RAW_DATA_DIR="data/medtpe_data/ectsum"
else
  echo "Unsupported public dataset: $DATASET. Use CMEDQA2 or ECTSUM."
  exit 1
fi

GPU_UTIL="${GPU_UTIL:-0.93}"
GPU_ID="${GPU_ID:-0}"
NUM_RESPONSES=1
DATA_FORMAT="nl"
MAX_INPUT_LEN="4k"
MAX_OUTPUT_LEN="2k"
DATA_SPLITS=("train" "tuning")
MAX_M=5000
MAX_N=2

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for DATA_SPLIT in "${DATA_SPLITS[@]}"; do
      python preprocessing/create_sft_QA.py \
        --gpu_util "$GPU_UTIL" \
        --output_dir "$SFT_QA_DIR" \
        --data_format "$DATA_FORMAT" \
        --task "$TASK" \
        --gpu_id "$GPU_ID" \
        --num_responses "$NUM_RESPONSES" \
        --model_path "$MODEL_PATH" \
        --max_input_len "$MAX_INPUT_LEN" \
        --max_output_len "$MAX_OUTPUT_LEN" \
        --data_split "$DATA_SPLIT" \
        --dataset "$DATASET" \
        --data_dir "$RAW_DATA_DIR"
    done

    python preprocessing/create_tpe_tokenizer.py \
      --data_format "$DATA_FORMAT" \
      --task "$TASK" \
      --model_path "$MODEL_PATH" \
      --max_m "$MAX_M" \
      --max_n "$MAX_N" \
      --dataset "$DATASET" \
      --max_input_len "$MAX_INPUT_LEN" \
      --max_output_len "$MAX_OUTPUT_LEN" \
      --log_dir "$TPE_TOKENIZER_DIR" \
      --data_dir "$SFT_QA_DIR"

    python preprocessing/llm_fit_tpe_tokenizer.py \
      --data_format "$DATA_FORMAT" \
      --task "$TASK" \
      --model_path "$MODEL_PATH" \
      --max_m "$MAX_M" \
      --max_n "$MAX_N" \
      --dataset "$DATASET" \
      --log_dir "$TPE_MODEL_DIR" \
      --tokenizer_dir "$TPE_TOKENIZER_DIR"

    for DATA_SPLIT in "${DATA_SPLITS[@]}"; do
      python preprocessing/clean_sft_QA.py \
        --data_dir "$SFT_QA_DIR" \
        --output_dir "$CLEANED_SFT_QA" \
        --task "$TASK" \
        --data_split "$DATA_SPLIT" \
        --data_format "$DATA_FORMAT" \
        --model_path "$MODEL_PATH" \
        --max_input_len "$MAX_INPUT_LEN" \
        --max_output_len "$MAX_OUTPUT_LEN" \
        --max_m "$MAX_M" \
        --max_n "$MAX_N" \
        --dataset "$DATASET" \
        --tpe_tokenizer_dir "$TPE_TOKENIZER_DIR"
    done
  done
done
