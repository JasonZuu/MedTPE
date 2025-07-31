export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

DATASET="EHRSHOT"
if [[ "$DATASET" == "MIMICIV" ]]; then
    TASKS=("icu_phenotyping" "icu_mortality")
elif [[ "$DATASET" == "EHRSHOT" ]]; then
  TASKS=("new_pancan" "guo_readmission")
fi

MODEL_PATHS=(
  "data/hf_models/Qwen--Qwen2.5-1.5B-Instruct"
  "data/hf_models/Qwen--Qwen2.5-7B-Instruct"
  "data/hf_models/meta-llama--Llama-3.2-1B-Instruct"
  "data/hf_models/meta-llama--Llama-3.1-8B-Instruct"
  "data/hf_models/OpenMeditron--Meditron3-8B"
)

GPU_UTIL=0.9
NUM_RESPONSES=1
DATA_FORMAT="nl"
TOKENIZER_TYPES=("tpe-sft" "bpe" "lingua2")
MAX_N=2
MAX_M=5000
MAX_INPUT_LEN="8k"
PE_METHODS=("raw" "cot")
# conda activate agent

for MODEL_PATH in "${MODEL_PATHS[@]}"; do

  # Run inference for each PE method
  for TOKENIZER_TYPE in "${TOKENIZER_TYPES[@]}"; do
    for PE_METHOD in "${PE_METHODS[@]}"; do
      for TASK in "${TASKS[@]}"; do
        echo "Running inference for model: $MODEL_PATH, PE method: $PE_METHOD, max input length: $MAX_INPUT_LEN, tokenizer type: $TOKENIZER_TYPE, dataset (task): $DATASET ($TASK)"
        # Run the Python script with the specified parameters
        python info_cons_infer_vocab.py \
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
            --max_m "$MAX_M" 
      done
    done
  done
done
