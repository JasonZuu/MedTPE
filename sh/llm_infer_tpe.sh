#!/bin/bash
set -euo pipefail

export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

MODEL_PATHS=(
  "data/hf_models/Qwen--Qwen2.5-1.5B-Instruct"
)

DATASET="${DATASET:-CMEDQA2}"
if [[ "$DATASET" == "CMEDQA2" ]]; then
  TASKS=("cmedqa2")
  DATA_DIR="data/hf_datasets/fzkuji--cMedQA2"
elif [[ "$DATASET" == "ECTSUM" ]]; then
  TASKS=("ect_summary")
  DATA_DIR="data/hf_datasets/github--ECTSum"
else
  echo "Unsupported public dataset: $DATASET. Use CMEDQA2 or ECTSUM."
  exit 1
fi

GPU_UTIL="${GPU_UTIL:-0.5}"
NUM_RESPONSES=1
DATA_FORMAT="nl"
TOKENIZER_TYPES=("tpe-sft" "bpe")
MAX_N=2
MAX_M=5000
MAX_INPUT_LEN="8k"
PE_METHODS=("raw")
GPU_IDS="${GPU_IDS:-0}"
TPE_MODEL_DIR="data/MedTPE_data/tpe_models"
SFT_MODEL_DIR="data/MedTPE_data/sft_models"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  for TOKENIZER_TYPE in "${TOKENIZER_TYPES[@]}"; do
    for PE_METHOD in "${PE_METHODS[@]}"; do
      for TASK in "${TASKS[@]}"; do
        echo "Running inference for model: $MODEL_PATH, PE method: $PE_METHOD, max input length: $MAX_INPUT_LEN, tokenizer type: $TOKENIZER_TYPE, dataset/task: $DATASET/$TASK"
        python infer_llm.py \
          --model_path "$MODEL_PATH" \
          --tokenizer_type "$TOKENIZER_TYPE" \
          --pe_method "$PE_METHOD" \
          --gpu_util "$GPU_UTIL" \
          --data_format "$DATA_FORMAT" \
          --task "$TASK" \
          --dataset "$DATASET" \
          --data_dir "$DATA_DIR" \
          --tpe_model_dir "$TPE_MODEL_DIR" \
          --sft_model_dir "$SFT_MODEL_DIR" \
          --num_responses "$NUM_RESPONSES" \
          --max_input_len "$MAX_INPUT_LEN" \
          --max_n "$MAX_N" \
          --max_m "$MAX_M" \
          --gpu_ids "$GPU_IDS"
      done
    done
  done
done
