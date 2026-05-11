# From Byte Pair to Token Pair: Efficient Prompt Compression for Large Language

![MedTPE Illustration](img/MedTPE.png)

Large language models (LLMs) are promising for clinical prediction, but real-world clinical text and EHR-derived inputs often produce long token sequences that increase computational cost and can reduce performance. Medical Token-Pair Encoding (MedTPE) is a layered extension of byte-pair encoding that merges frequent medical token pairs into composite tokens, enabling lossless prompt compression without adding a new compression module.

This public release uses CMedQA2 and ECTSum as runnable illustration datasets. The MIMIC-IV and EHRSHOT datasets used in the paper are not included in this repository. To run those datasets, first obtain the corresponding licenses and process the data following the MEDS protocol: https://github.com/Medical-Event-Data-Standard/meds.

## Getting Started

### Prerequisites
This project needs the following prerequisites:
- Python 3.12+
- CUDA 12.4+
- PyTorch
- transformers
- vLLM

### Installation
```bash
git clone <repo>
cd MedTPE
pip install -r requirements.txt
```

### 1. Prepare Public Datasets
Download and normalize the public datasets used by this release:
```bash
python preprocessing/prepare_public_datasets.py --dataset all
```

This creates:
- `data/hf_datasets/fzkuji--cMedQA2/{train,validation,test}.json`
- `data/hf_datasets/github--ECTSum/{train,val,test}.json`

To prepare only one dataset:
```bash
python preprocessing/prepare_public_datasets.py --dataset cmedqa2
python preprocessing/prepare_public_datasets.py --dataset ectsum
```

### 2. Download Pretrained LLMs
Log in to Hugging Face if your chosen model requires access, then download a model:
```bash
python preprocessing/download_llm.py --repo_id Qwen/Qwen2.5-1.5B-Instruct --local_dir data/hf_models
```

You can change `repo_id` to another compatible instruction-tuned model.

### 3. Prepare SFT QA Data and Create a MedTPE-Fitted Model
The default public workflow uses CMedQA2. This command generates SFT QA data, creates the MedTPE tokenizer, fits the LLM embeddings, and cleans the SFT QA data:
```bash
bash sh/prepare_sft_qa_and_models.sh
```

For ECTSum instead, run:
```bash
DATASET=ECTSUM bash sh/prepare_sft_qa_and_models.sh
```

The core tokenizer/model commands used by the script are:
```bash
DATA_FORMAT="nl"
DATASET="CMEDQA2"
TASK="cmedqa2"
MODEL_PATH="data/hf_models/Qwen--Qwen2.5-1.5B-Instruct"
MAX_M=5000
MAX_N=2

python preprocessing/create_tpe_tokenizer.py \
  --dataset "$DATASET" \
  --data_format "$DATA_FORMAT" \
  --task "$TASK" \
  --model_path "$MODEL_PATH" \
  --data_dir data/MedTPE_data/SFT_QA \
  --log_dir data/MedTPE_data/tpe_tokenizers \
  --max_m "$MAX_M" \
  --max_n "$MAX_N"

python preprocessing/llm_fit_tpe_tokenizer.py \
  --dataset "$DATASET" \
  --data_format "$DATA_FORMAT" \
  --task "$TASK" \
  --model_path "$MODEL_PATH" \
  --tokenizer_dir data/MedTPE_data/tpe_tokenizers \
  --log_dir data/MedTPE_data/tpe_models \
  --max_m "$MAX_M" \
  --max_n "$MAX_N"
```

### 4. Run Supervised Fine-Tuning
Fine-tune only the new MedTPE token embeddings:
```bash
bash sh/llm_ssft.sh
```

For ECTSum:
```bash
DATASET=ECTSUM bash sh/llm_ssft.sh
```

Fine-tuned checkpoints are saved under `data/MedTPE_data/sft_models`.

### 5. Run Inference Demo
With a MedTPE + SFT checkpoint ready:
```bash
python demo_inference.py \
  --model_name data/hf_models/Qwen--Qwen2.5-1.5B-Instruct \
  --sft_model_name data/MedTPE_data/sft_models/Qwen2.5-1.5B-Instruct_task-cmedqa2_maxN-2_maxM-5000
```

### 6. Evaluate the Effectiveness of MedTPE
Run CMedQA2 inference and parse metrics:
```bash
bash sh/llm_infer_tpe.sh
bash sh/parse_result_llm_tpe.sh
```

For ECTSum:
```bash
DATASET=ECTSUM bash sh/llm_infer_tpe.sh
DATASET=ECTSUM bash sh/parse_result_llm_tpe.sh
```

Text-generation evaluation uses ROUGE-1/2/L, BLEU, BERTScore, format compliance rate, output token length, and inference time.

## Citation
```bibtex
@inproceedings{
zhu2026from,
title={From Token to Token Pair: Efficient Prompt Compression for Large Language Models in Clinical Prediction},
author={Zhu, Mingcheng and Luo, Zhiyao and Liu, Yu and Zhu, Tingting},
booktitle={Forty-third International Conference on Machine Learning},
year={2026}
}
```
