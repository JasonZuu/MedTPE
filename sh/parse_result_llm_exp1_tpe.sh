#!/bin/bash

export PYTHONPATH="."
export HF_HOME=./hf_cache

LLM_IDS=(
  "Meditron3-8B"
  "Qwen2.5-7B-Instruct"
  "Llama-3.1-8B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Llama-3.2-1B-Instruct"
)
DATA_FORMAT="nl"
MAX_INPUT_LEN="8k"

DATASET="EHRSHOT"
if [[ "$DATASET" == "MIMICIV" ]]; then
  TASKS=("icu_phenotyping" "icu_mortality")
elif [[ "$DATASET" == "EHRSHOT" ]]; then
  TASKS=("new_pancan" "guo_readmission")
fi

PE_METHODS=("raw" "cot")
LOG_FNAME_POSTFIXS=("bpe" "tpe-sft" "lingua2")
RESPONSE_DIR="log/TPE/$DATASET"
LOG_DIR="log/TPE/metrics/$DATASET"

for LLM_ID in "${LLM_IDS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for PE_METHOD in "${PE_METHODS[@]}"; do
      for LOG_FNAME_POSTFIX in "${LOG_FNAME_POSTFIXS[@]}"; do
        # Run the LLM inference in sequence
        python parse_result_llm.py --pe_method $PE_METHOD \
                                   --data_format $DATA_FORMAT \
                                   --task $TASK \
                                   --llm_id $LLM_ID \
                                   --max_input_len $MAX_INPUT_LEN \
                                   --log_fname_postfix $LOG_FNAME_POSTFIX \
                                   --response_dir $RESPONSE_DIR \
                                   --log_dir $LOG_DIR \
                                   --dataset $DATASET
      done
    done
  done
done