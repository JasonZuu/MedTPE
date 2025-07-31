export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

MODEL_PATHS=(
  "data/hf_models/meta-llama--Llama-3.2-1B-Instruct"
  "data/hf_models/Qwen--Qwen2.5-1.5B-Instruct"
  "data/hf_models/meta-llama--Llama-3.1-8B-Instruct"
  "data/hf_models/Qwen--Qwen2.5-7B-Instruct"
  "data/hf_models/OpenMeditron--Meditron3-8B"
)

DATASET="MIMICIV"
if [[ "$DATASET" == "MIMICIV" ]]; then
  TASKS=("icu_phenotyping" "icu_mortality")
elif [[ "$DATASET" == "EHRSHOT" ]]; then
  TASKS=("new_pancan" "guo_readmission")
fi

GPU_UTIL=0.9
DATA_FORMAT="nl"
TOKENIZER_TYPES=("tpe-sft" "bpe")
MAX_N=2
MAX_M=5000
MAX_INPUT_LEN="2k"
PE_METHOD="raw"
LOG_DIR="./log/TPE_tts"
NUM_WORKERS=16
GPU_ID=0
# conda activate agent

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  for TOKENIZER_TYPE in "${TOKENIZER_TYPES[@]}"; do
    if [[ "$TOKENIZER_TYPE" == "tpe-sft" ]]; then
      NUM_RESPONSES_LIST=(1 3 5)
    elif [[ "$TOKENIZER_TYPE" == "bpe" ]]; then
      NUM_RESPONSES_LIST=(1)
    fi
    for TASK in "${TASKS[@]}"; do
      for NUM_RESPONSES in "${NUM_RESPONSES_LIST[@]}"; do
        echo "Running inference for model: $MODEL_PATH, PE method: $PE_METHOD, max input length: $MAX_INPUT_LEN, tokenizer type: $TOKENIZER_TYPE, dataset (task): $DATASET ($TASK)"
        # Run the Python script with the specified parameters
        python llm_infer.py \
                --model_path "$MODEL_PATH" \
                --tokenizer_type "$TOKENIZER_TYPE" \
                --pe_method "$PE_METHOD" \
                --gpu_util "$GPU_UTIL" \
                --data_format "$DATA_FORMAT" \
                --task "$TASK" \
                --dataset "$DATASET" \
                --num_responses "$NUM_RESPONSES" \
                --max_input_len "$MAX_INPUT_LEN" \
                --max_n "$MAX_N" \
                --max_m "$MAX_M" \
                --log_dir "$LOG_DIR" \
                --num_workers "$NUM_WORKERS" \
                --gpu_id "$GPU_ID" 
      done
    done
  done
done
