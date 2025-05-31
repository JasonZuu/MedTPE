from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
from pathlib import Path
from tokenizers import models
import json

from dataset.map_fn import tpe_sample_batch_mapping_fn
from utils.misc import set_random_seed, get_sft_qa_fname, get_tpe_tokenizer_dir, get_sft_dataset_dir
from tpe.vocab_modify_tpe import tpe_vocabulary_modify_fn
from utils.constants import deepseek_r1_new_chat_template


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/hf_models/Qwen--Qwen2.5-1.5B-Instruct", help="LLM model path.")
    parser.add_argument("--data_dir", type=str, default="data/SFT_QA", help="Data directory.")
    parser.add_argument("--log_dir", type=str, default="data/tpe_tokenizers", help="Output file path.")
    parser.add_argument("--dataset", type=str, default="MIMICIV",
                        choices=['MIMICIV'], help="Dataset to use for inference.")
    parser.add_argument("--task", type=str, default="icu_mortality", #
                        choices=["icu_mortality", "icu_phenotyping"], help="Task name.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"], help="Data format.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--max_n", type=int, default=3, help="Max n-gram size for vocabulary modification.")
    parser.add_argument("--max_m", type=int, default=8000, help="Number of new tokens to add to the vocabulary.")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum frequency for vocabulary modification.")
    parser.add_argument("--max_input_len", type=str, default="6k", choices=["2k", "4k", "6k", "8k"],
                        help="Max input length.")
    parser.add_argument("--max_output_len", type=str, default="2k", choices=["2k", "4k"],
                        help="Max output length.")
    return parser.parse_args()


def get_merges(tokenizer_json_fpath: str) -> list:
    with open(tokenizer_json_fpath, "r") as f:
        tokenizer_json = json.load(f)
        merges= tokenizer_json["model"]["merges"]
    merges = [tuple(merge.split()) for merge in merges]
    return merges

def change_chat_template(tokenizer, template: str) -> None:
    """
    Change the chat template of the tokenizer.
    """
    tokenizer.chat_template = template


if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    model_name = args.model_path.split("/")[-1].split("--")[-1]
    tpe_tokenizer_dir = get_tpe_tokenizer_dir(args.log_dir, model_name=model_name, 
                                                task=args.task, max_n=args.max_n, 
                                                max_m=args.max_m) # used to log the vocab and model
    if (tpe_tokenizer_dir/"tokenizer.json").exists():
        print(f"Tokenizer with the same name already exists in {tpe_tokenizer_dir}.")
        print("Exiting the program.")
        exit(0)

    print(f"The new tokenizer will be saved to {tpe_tokenizer_dir}")
    
    # 1. load the dataset
      # e.g., "Qwen-7B"
    dataset_dir = get_sft_dataset_dir(args.dataset,
                               max_input_len=args.max_input_len, 
                               max_output_len=args.max_output_len)
    sft_qa_fname = get_sft_qa_fname(task=args.task,
                                    data_format=args.data_format, 
                                    llm_name=model_name,
                                    data_split="train",)
    data_fpath = Path(args.data_dir) / dataset_dir / f"{sft_qa_fname}.parquet"
    data_files = {'train': str(data_fpath)}
    dataset_dict = load_dataset("parquet", data_files=data_files)
    dataset_dict = dataset_dict.map(tpe_sample_batch_mapping_fn, 
                                        batched=True,
                                        remove_columns=['message', "generated_text_0"],)

    # 2. get the new vocabulary
    tokenizer_json_fpath = Path(args.model_path) / "tokenizer.json"
    byte_merges = get_merges(tokenizer_json_fpath)

    orig_tokenizer = AutoTokenizer.from_pretrained(args.model_path) 
    vocab_new, byte_merges_new,\
        tok_merges, new_token_info = tpe_vocabulary_modify_fn(orig_tokenizer, 
                                                              dataset_dict=dataset_dict, 
                                                              max_m=args.max_m, 
                                                              max_n=args.max_n, 
                                                              byte_merges=byte_merges, 
                                                              dataset_split="train", 
                                                              min_freq=args.min_freq)
    print("Length of old vocab:", len(orig_tokenizer))
    print("Length of new vocab:", len(vocab_new))
    print("Length of old byte-level merges:", len(byte_merges))
    print("Length of new byte-level merges:", len(byte_merges_new))
    print("Length of token-level merges:", len(tok_merges))
    print("Number of new tokens:", len(new_token_info))

    # 3. Get a new merges on the dataset (medical corpus) by continuing training from the original tokenizer
    bpe_model = models.BPE(vocab=vocab_new, merges=byte_merges_new)

    orig_tokenizer.save_pretrained(tpe_tokenizer_dir)
    tokenizer = orig_tokenizer.backend_tokenizer
    tokenizer.model = bpe_model
    if "DeepSeek-R1" in model_name:
        change_chat_template(tokenizer, deepseek_r1_new_chat_template)
    tokenizer.save(str(tpe_tokenizer_dir / "tokenizer.json"))

    with open(tpe_tokenizer_dir / "new_token_info.json", "w") as f:
        json.dump(new_token_info, f, indent=4, ensure_ascii=False)
    with open(tpe_tokenizer_dir / "tok_merges.json", "w") as f:
        json.dump(tok_merges, f, indent=4, ensure_ascii=False)
