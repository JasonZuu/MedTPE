"""
This script is to generate a new vocab and fit the LLM with the new vocab without SFT.
After running this script, there will be a new model file in the log_dir with the new vocab config and fitted model weights.
"""

from transformers import AutoModelForCausalLM
import argparse
from pathlib import Path
import json

from tpe.tpe_tokenizer_fast import TPETokenizerFast
from utils.misc import set_random_seed, get_sft_model_dir
from tpe.embed_modify import replace_llm_embedding_for_new_token


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/hf_models/Qwen--Qwen2.5-1.5B-Instruct", help="LLM model path.")
    parser.add_argument("--tokenizer_dir", type=str, default="data/tpe_tokenizers", help="Output file path.")
    parser.add_argument("--log_dir", type=str, default="data/tpe_models", help="Output file path.")
    parser.add_argument("--dataset", type=str, default="MIMICIV",
                        choices=['MIMICIV'], help="Dataset to use for inference.")
    parser.add_argument("--task", type=str, default="icu_phenotyping", #
                        choices=["icu_mortality", "icu_phenotyping"], help="Task name.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"], help="Data format.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--max_m", type=int, default=5000, help="Number of new tokens to add to the vocabulary.")
    parser.add_argument("--max_n", type=int, default=3, help="Max n-gram size for vocabulary modification.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

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
                              max_m=args.max_m) # used to log the vocab and model
    print(f"The new tokenizer and model will be saved to {model_dir}")

    # adjust the model embedding
    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=False, torch_dtype="auto")
    tie_word_embeddings = model.config.tie_word_embeddings
    model = replace_llm_embedding_for_new_token(
        model,
        new_token_info,
        tie_word_embeddings=tie_word_embeddings
    )
    model.save_pretrained(model_dir)
    print(f"Successfully saved the model to {model_dir}")

    # initialize and save the new tokenizer, save tokenizer later to avoid overwriting by model.save_pretrained
    tokenizer = TPETokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.save_pretrained(model_dir)
    
    print(f"Successfully saved the tokenizer to {model_dir}")
