export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

MODEL_PATHS=(
  "data/hf_models/Qwen--Qwen2.5-1.5B-Instruct"
  "data/hf_models/Qwen--Qwen2.5-7B-Instruct"
  "data/hf_models/meta-llama--Llama-3.2-1B-Instruct"
  "data/hf_models/meta-llama--Llama-3.1-8B-Instruct"
  "data/hf_models/OpenMeditron--Meditron3-8B"
)

SFT_QA_DIR="data/SFT_QA"
CLEANED_SFT_QA="data/cleaned_SFT_QA"
DATASET="MIMICIV"
if [[ $DATASET == "MIMICIV" ]]; then
  TASKS=("icu_phenotyping" "icu_mortality")
elif [[ $DATASET == "EHRSHOT" ]]; then
  TASKS=("guo_readmission" "new_pancan")
else
  echo "Unknown dataset: $DATASET"
  exit 1
fi
GPU_UTIL=0.93
NUM_RESPONSES=1
DATA_FORMAT=nl
MAX_INPUT_LEN=4k
MAX_OUTPUT_LEN=2k
DATA_SPLITS=("train" "tuning")
MAX_M=5000
MAX_N=2
# conda activate agent

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for DATA_SPLIT in "${DATA_SPLITS[@]}"; do
      # Run the LLM inference on the data split to create SFT QA data
      python create_sft_QA.py --gpu_util $GPU_UTIL --output_dir $SFT_QA_DIR \
          --data_format $DATA_FORMAT --task $TASK --gpu_id 1 --num_responses $NUM_RESPONSES \
          --model_path $MODEL_PATH --max_input_len $MAX_INPUT_LEN --max_output_len $MAX_OUTPUT_LEN \
          --data_split $DATA_SPLIT --dataset $DATASET
      # Create TPE tokenizer
      python create_tpe_tokenizer.py --data_format $DATA_FORMAT --task $TASK \
            --model_path $MODEL_PATH --max_m $MAX_M --max_n $MAX_N --dataset $DATASET \
            --max_input_len $MAX_INPUT_LEN --max_output_len $MAX_OUTPUT_LEN
      # Fit TPE tokenizer
      python llm_fit_tpe_tokenizer.py --data_format $DATA_FORMAT --task $TASK \
            --model_path $MODEL_PATH --max_m $MAX_M --max_n $MAX_N --dataset $DATASET 
      # Call the TPE tokenizer to clean the SFT QA data to get the final cleaned SFT QA data
      python clean_sft_QA.py --data_dir $SFT_QA_DIR --output_dir $CLEANED_SFT_QA \
            --task $TASK --data_split $DATA_SPLIT \
            --data_format $DATA_FORMAT --model_path $MODEL_PATH \
            --max_input_len $MAX_INPUT_LEN --max_output_len $MAX_OUTPUT_LEN \
            --data_split $DATA_SPLIT --max_m $MAX_M --max_n $MAX_N --dataset $DATASET
    done
  done
done


