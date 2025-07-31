#!/bin/bash

export PYTHONPATH="."
export HF_HOME=./hf_cache

LLM_IDS=(
  "Qwen2.5-7B-Instruct"
  "Llama-3.1-8B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Llama-3.2-1B-Instruct"
  "Meditron3-8B"
)
DATA_FORMAT="nl"
MAX_INPUT_LENs=("500" "1k" "2k" "4k" "8k")

DATASET="MIMICIV"
if [[ "$DATASET" == "MIMICIV" ]]; then
  TASKS=("icu_phenotyping" "icu_mortality")
elif [[ "$DATASET" == "EHRSHOT" ]]; then
  TASKS=("guo_readmission" "new_pancan")
fi

PE_METHODS=("raw")
LOG_FNAME_POSTFIXS=("bpe" "tpe-sft")
RESPONSE_DIR="log/TPE_med_term/$DATASET"
LOG_DIR="log/TPE_med_term/metrics/$DATASET"

for LLM_ID in "${LLM_IDS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for PE_METHOD in "${PE_METHODS[@]}"; do
      for MAX_INPUT_LEN in "${MAX_INPUT_LENs[@]}"; do
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
done