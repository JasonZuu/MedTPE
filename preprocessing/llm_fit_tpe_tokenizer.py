"""
This script is to generate a new vocab and fit the LLM with the new vocab without SFT.
After running this script, there will be a new model file in the log_dir with the new vocab config and fitted model weights.
"""

import argparse
from pathlib import Path
import json
import time
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/hf_models/Qwen--Qwen2.5-1.5B-Instruct", help="LLM model path.")
    parser.add_argument("--tokenizer_dir", type=str, default="data/MedTPE_data/tpe_tokenizers", help="Output file path.")
    parser.add_argument("--log_dir", type=str, default="data/MedTPE_data/tpe_models", help="Output file path.")
    parser.add_argument("--dataset", type=str, default="CMEDQA2",
                        choices=["ECTSUM", "CMEDQA2"], help="Dataset to use for inference.")
    parser.add_argument("--task", type=str, default="cmedqa2", #
                        choices=["ect_summary", "cmedqa2"],
                        help="Task name.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"], help="Data format.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--max_m", type=int, default=5000, help="Number of new tokens to add to the vocabulary.")
    parser.add_argument("--max_n", type=int, default=2, help="Max n-gram size for vocabulary modification.")
    parser.add_argument("--vocab_mode", type=str, default="replace",
                        choices=["replace", "extend"],
                        help="Must match the mode used in create_tpe_tokenizer.py. "
                             "'replace': in-place embedding swap. "
                             "'extend': resize embeddings then initialise new rows.")
    return parser.parse_args()


if __name__ == "__main__":
    timing = {}
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    args = get_args()

    from transformers import AutoModelForCausalLM

    from tpe.embed_modify import replace_llm_embedding_for_new_token, extend_llm_embedding_for_new_token
    from tpe.tpe_tokenizer_fast import TPETokenizerFast
    from utils.misc import set_random_seed, get_sft_model_dir

    set_random_seed(args.seed)
    timing["init_args_and_seed_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    model_name = args.model_path.split("/")[-1].split("--")[-1]  # e.g., Qwen2.5-1.5B-Instruct
    tokenizer_dir = get_sft_model_dir(args.tokenizer_dir,
                                  model_name=model_name,
                                  task=args.task, max_n=args.max_n,
                                  max_m=args.max_m) # used to log the vocab and model
    with open(tokenizer_dir / "new_token_info.json", "r") as f:
        new_token_info = json.load(f)

    model_dir = get_sft_model_dir(args.log_dir,
                                  model_name=model_name,
                                  task=args.task, max_n=args.max_n,
                                  max_m=args.max_m,
                                  postfix=None) # used to log the vocab and model
    print(f"The new tokenizer and model will be saved to {model_dir}")
    if (model_dir / "config.json").exists():
        print(f"Model directory {model_dir} already exists. Skipping fit.")
        exit(0)
    timing["resolve_dirs_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()

    # adjust the model embedding
    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=False, torch_dtype="auto")
    tie_word_embeddings = model.config.tie_word_embeddings
    if args.vocab_mode == "extend":
        model = extend_llm_embedding_for_new_token(
            model,
            new_token_info,
            tie_word_embeddings=tie_word_embeddings
        )
    else:
        model = replace_llm_embedding_for_new_token(
            model,
            new_token_info,
            tie_word_embeddings=tie_word_embeddings
        )
    timing["load_model_and_fit_embedding_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    model.save_pretrained(model_dir)
    print(f"Successfully saved the model to {model_dir}")
    timing["save_model_seconds"] = time.perf_counter() - stage_start

    # initialize and save the new tokenizer, save tokenizer later to avoid overwriting by model.save_pretrained
    stage_start = time.perf_counter()
    tokenizer = TPETokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"Successfully saved the tokenizer to {model_dir}")
    timing["save_tokenizer_seconds"] = time.perf_counter() - stage_start

    timing["new_token_count"] = len(new_token_info)
    timing["total_seconds"] = time.perf_counter() - total_start
    with open(model_dir / "llm_fit_tpe_tokenizer_timing.json", "w") as f:
        json.dump(timing, f, indent=4, ensure_ascii=False)
