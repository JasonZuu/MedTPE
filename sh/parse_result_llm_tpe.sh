#!/bin/bash
set -euo pipefail

export PYTHONPATH="."
export HF_HOME=./hf_cache

LLM_IDS=(
  "Qwen2.5-1.5B-Instruct"
)

DATASET="${DATASET:-CMEDQA2}"
if [[ "$DATASET" == "CMEDQA2" ]]; then
  TASKS=("cmedqa2")
elif [[ "$DATASET" == "ECTSUM" ]]; then
  TASKS=("ect_summary")
else
  echo "Unsupported public dataset: $DATASET. Use CMEDQA2 or ECTSUM."
  exit 1
fi

DATA_FORMAT="nl"
MAX_INPUT_LEN="8k"
PE_METHODS=("raw")
LOG_FNAME_POSTFIXS=("bpe" "tpe-sft")
RESPONSE_DIR="log/TPE/$DATASET"
LOG_DIR="log/TPE/metrics/$DATASET"

for LLM_ID in "${LLM_IDS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for PE_METHOD in "${PE_METHODS[@]}"; do
      for LOG_FNAME_POSTFIX in "${LOG_FNAME_POSTFIXS[@]}"; do
        python parse_result_llm.py \
          --pe_method "$PE_METHOD" \
          --data_format "$DATA_FORMAT" \
          --task "$TASK" \
          --llm_id "$LLM_ID" \
          --max_input_len "$MAX_INPUT_LEN" \
          --log_fname_postfix "$LOG_FNAME_POSTFIX" \
          --response_dir "$RESPONSE_DIR" \
          --log_dir "$LOG_DIR" \
          --dataset "$DATASET"
      done
    done
  done
done
