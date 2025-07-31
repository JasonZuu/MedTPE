#!/bin/bash

export PYTHONPATH="."
export HF_HOME=./hf_cache

LLM_IDS=(
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Llama-3.2-1B-Instruct"
  "Llama-3.1-8B-Instruct"
  "Meditron3-8B"
)
DATASET="MIMICIV"
if [[ "$DATASET" == "MIMICIV" ]]; then
  TASKS=("icu_phenotyping" "icu_mortality")
elif [[ "$DATASET" == "EHRSHOT" ]]; then
  TASKS=("guo_readmission" "new_pancan")
fi

DATA_FORMAT="nl"
MAX_INPUT_LEN="2k"
PE_METHOD="raw"
LOG_FNAME_POSTFIXS=("bpe" "tpe-sft")

RESPONSE_DIR="log/TPE_tts/$DATASET"
LOG_DIR="log/TPE_tts/metrics/$DATASET"


for LLM_ID in "${LLM_IDS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for LOG_FNAME_POSTFIX in "${LOG_FNAME_POSTFIXS[@]}"; do
      if [[ "$LOG_FNAME_POSTFIX" == "bpe" ]]; then
        N_RESPONSES=(1)
      else
        N_RESPONSES=(1 3 5 10)
      fi
      for N_RESPONSE in "${N_RESPONSES[@]}"; do
        # Run the LLM inference in sequence
        python parse_result_llm.py --pe_method $PE_METHOD \
                                   --data_format $DATA_FORMAT \
                                   --task $TASK \
                                   --llm_id $LLM_ID \
                                   --max_input_len $MAX_INPUT_LEN \
                                   --n_response $N_RESPONSE \
                                   --log_dir $LOG_DIR \
                                   --response_dir $RESPONSE_DIR \
                                   --dataset $DATASET \
                                   --log_fname_postfix $LOG_FNAME_POSTFIX
      done
    done
  done
done