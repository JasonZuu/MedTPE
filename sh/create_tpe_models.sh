export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

MODEL_PATHS=(
  "data/hf_models/Qwen--Qwen2.5-1.5B-Instruct"
  "data/hf_models/meta-llama--Llama-3.2-1B-Instruct"
)

TASKS=("icu_phenotyping" "icu_mortality") # choose from icu_mortality and icu_phenotyping
DATA_FORMAT="nl"
MAX_M=5000
MAX_N=3
# conda activate agent

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  for TASK in "${TASKS[@]}"; do
      # Create TPE tokenizer
      python create_tpe_tokenizer.py --data_format $DATA_FORMAT --task $TASK --model_path $MODEL_PATH --max_m $MAX_M --max_n $MAX_N
      # Fit TPE tokenizer
      python llm_fit_tpe_tokenizer.py --data_format $DATA_FORMAT --task $TASK --model_path $MODEL_PATH --max_m $MAX_M --max_n $MAX_N
  done
done


