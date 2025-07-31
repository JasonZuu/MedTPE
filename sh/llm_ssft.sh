export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

PHENOTYPING_TPE_MODEL_PATHS=(
  "data/tpe_models/Qwen2.5-1.5B-Instruct_task-icu_phenotyping_maxN-2_maxM-5000"
  "data/tpe_models/Llama-3.2-1B-Instruct_task-icu_phenotyping_maxN-2_maxM-5000"
  "data/tpe_models/Meditron3-8B_task-icu_phenotyping_maxN-2_maxM-5000"
  "data/tpe_models/Qwen2.5-7B-Instruct_task-icu_phenotyping_maxN-2_maxM-5000"
  "data/tpe_models/Llama-3.1-8B-Instruct_task-icu_phenotyping_maxN-2_maxM-5000"
)

MORTALITY_TPE_MODEL_PATHS=(
  "data/tpe_models/Qwen2.5-1.5B-Instruct_task-icu_mortality_maxN-2_maxM-5000"
  "data/tpe_models/Llama-3.2-1B-Instruct_task-icu_mortality_maxN-2_maxM-5000"
  "data/tpe_models/Meditron3-8B_task-icu_mortality_maxN-2_maxM-5000"
  "data/tpe_models/Qwen2.5-7B-Instruct_task-icu_mortality_maxN-2_maxM-5000"
  "data/tpe_models/Llama-3.1-8B-Instruct_task-icu_mortality_maxN-2_maxM-5000"
)

READMISSION_TPE_MODEL_PATHS=(
  "data/tpe_models/Qwen2.5-1.5B-Instruct_task-guo_readmission_maxN-2_maxM-5000"
  "data/tpe_models/Llama-3.2-1B-Instruct_task-guo_readmission_maxN-2_maxM-5000"
  "data/tpe_models/Qwen2.5-7B-Instruct_task-guo_readmission_maxN-2_maxM-5000"
  "data/tpe_models/Llama-3.1-8B-Instruct_task-guo_readmission_maxN-2_maxM-5000"
  "data/tpe_models/Meditron3-8B_task-guo_readmission_maxN-2_maxM-5000"
)

PANCAN_TPE_MODEL_PATHS=(
  "data/tpe_models/Qwen2.5-1.5B-Instruct_task-new_pancan_maxN-2_maxM-5000"
  "data/tpe_models/Llama-3.2-1B-Instruct_task-new_pancan_maxN-2_maxM-5000"
  "data/tpe_models/Qwen2.5-7B-Instruct_task-new_pancan_maxN-2_maxM-5000"
  "data/tpe_models/Llama-3.1-8B-Instruct_task-new_pancan_maxN-2_maxM-5000"
  "data/tpe_models/Meditron3-8B_task-new_pancan_maxN-2_maxM-5000"
)

MAX_INPUT_LEN="4k"  # please fix to 4k for this experiment
MAX_OUTPUT_LEN="2k"  # please fix to 2k for this experiment
DATASET="EHRSHOT"
if [[ "$DATASET" == "MIMICIV" ]]; then
  TASKS=("icu_phenotyping" "icu_mortality")
  BATCH_SIZE=2
  NUM_TRAIN_EPOCHS=1
  VAL_STEPS=1000
elif [[ "$DATASET" == "EHRSHOT" ]]; then
  TASKS=("new_pancan" "guo_readmission")
  BATCH_SIZE=2
  NUM_TRAIN_EPOCHS=10
  VAL_STEPS=1000
fi
 # choose from icu_mortality and icu_phenotyping
DATA_FORMAT="nl"

TRAIN_FN="sft"
DATA_DIR="data/cleaned_SFT_QA"
# conda activate agent

for TASK in "${TASKS[@]}"; do
  if [[ "$TASK" == "icu_phenotyping" ]]; then
    TPE_MODEL_PATHS=("${PHENOTYPING_TPE_MODEL_PATHS[@]}")
  elif [[ "$TASK" == "icu_mortality" ]]; then
    TPE_MODEL_PATHS=("${MORTALITY_TPE_MODEL_PATHS[@]}")
  elif [[ "$TASK" == "guo_readmission" ]]; then
    TPE_MODEL_PATHS=("${READMISSION_TPE_MODEL_PATHS[@]}")
  elif [[ "$TASK" == "new_pancan" ]]; then
    TPE_MODEL_PATHS=("${PANCAN_TPE_MODEL_PATHS[@]}")
  else
    echo "Unknown task: $TASK"
    exit 1
  fi
  for TPE_MODEL_PATH in "${TPE_MODEL_PATHS[@]}"; do
    python llm_sft.py \
        --model_path "$TPE_MODEL_PATH" \
        --task "$TASK" \
        --data_dir "$DATA_DIR" \
        --dataset "$DATASET" \
        --data_format "$DATA_FORMAT" \
        --max_input_len "$MAX_INPUT_LEN" \
        --max_output_len "$MAX_OUTPUT_LEN" \
        --train_fn "$TRAIN_FN" \
        --batch_size "$BATCH_SIZE" \
        --num_train_epochs "$NUM_TRAIN_EPOCHS" 
  done
done