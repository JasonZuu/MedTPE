import argparse
from pathlib import Path
from functools import partial
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/hf_models/Qwen--Qwen2.5-1.5B-Instruct", help="LLM model path.")
    parser.add_argument("--data_dir", type=str, default="data/MedTPE_data/SFT_QA", help="Data directory.")
    parser.add_argument("--tpe_tokenizer_dir", type=str, default="data/MedTPE_data/tpe_tokenizers",
                        help="TPE tokenizer directory.")
    parser.add_argument("--output_dir", type=str, default="data/MedTPE_data/cleaned_SFT_QA", help="Output file path.")
    parser.add_argument("--dataset", type=str, default="CMEDQA2",
                        choices=["ECTSUM", "CMEDQA2"], help="Dataset to use for inference.")
    parser.add_argument("--task", type=str, default="cmedqa2", #
                        choices=["ect_summary", "cmedqa2"],
                        help="Task name.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"], help="Data format.")
    parser.add_argument("--data_split", type=str, default="train",
                        choices=["train", "tuning", "held_out"], help="Set name.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers to use for data loading.")
    parser.add_argument("--max_input_len", type=str, default="4k", choices=["2k", "4k", "6k", "8k"],
                        help="Max input length.")
    parser.add_argument("--max_output_len", type=str, default="2k", choices=["2k", "4k"],
                        help="Max output length.")
    parser.add_argument("--max_sft_seq_len", type=int, default=4*1024, help="Max sequence length for sft.")
    parser.add_argument("--max_n", type=int, default=2, help="Max number of generations to keep.")
    parser.add_argument("--max_m", type=int, default=5000, help="Max number of samples to keep.")
    return parser.parse_args()


def filter_by_length(example, tokenizer, max_seq_len, dataset=None, task=None):
    # 对单个样本进行判断，返回 True 则保留，False 则丢弃
    if dataset in ["ECTSUM", "CMEDQA2"]:
        return [True] * len(example["generated_text_0"])
    messages = example["message"]
    model_outputs = example["generated_text_0"]
    samples = []
    for msg, model_output in zip(messages, model_outputs):
        resp_template = {"role": "assistant", "content": model_output}
        msg.append(resp_template)
        samples.append(msg)

    samples = tokenizer.apply_chat_template(samples, tokenize=False, add_generation_prompt=False)
    samples_ids = tokenizer(samples, padding=False, truncation=False, return_attention_mask=False)["input_ids"]
    keep_mask = []
    for sample_ids, output in zip(samples_ids, model_outputs):
        is_valid_output, _ = parse_llm_output(output)
        if not is_valid_output:  # invalid output, e.g., empty or not a valid response
            keep_mask.append(False)
        elif len(sample_ids) > max_seq_len:
            keep_mask.append(False)
        else: # valid output and within max_seq_len
            keep_mask.append(True)
    return keep_mask


if __name__ == "__main__":
    args = get_args()

    from datasets import load_dataset

    from tpe.tpe_tokenizer_fast import TPETokenizerFast
    from utils.misc import get_sft_qa_fname, get_tpe_tokenizer_dir, get_sft_dataset_dir, parse_llm_output

    _dataset_dir = get_sft_dataset_dir(
        dataset=args.dataset,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len
    )

    output_dir = Path(args.output_dir) / _dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_llm_name = args.model_path.split("/")[-1].split("--")[-1]  # e.g., "Qwen-7B"
    print(f"Evaluating LLM: {log_llm_name}")
    print(f"Log dir: {output_dir}")
    print(f"data dir: {args.data_dir}")

    # check if the results already exist
    sft_qa_fname = get_sft_qa_fname(task=args.task,
                                    data_format=args.data_format,
                                    data_split=args.data_split,
                                    llm_name=log_llm_name)
    data_fpath = Path(args.data_dir) / _dataset_dir / f"{sft_qa_fname}.parquet"
    if not data_fpath.exists():
        raise FileNotFoundError(f"Data file not found: {data_fpath}. Please check the data directory and file name.")

    data_files = {args.data_split: str(data_fpath)}

    # check if the results already exist
    sft_qa_fname = get_sft_qa_fname(task=args.task,
                                    data_format=args.data_format,
                                    data_split=args.data_split,
                                    llm_name=log_llm_name)
    output_fpath = output_dir / (sft_qa_fname + ".parquet")
    print("will save to ", output_fpath)
    if output_fpath.exists():
        print(f"Results already exist at: {output_fpath}")
        exit()

    # Load the TPE tokenizer
    tpe_tokenizer_dir = get_tpe_tokenizer_dir(log_dir=args.tpe_tokenizer_dir,
                                              model_name=log_llm_name,
                                              task=args.task,
                                              max_n=args.max_n,
                                              max_m=args.max_m)  # used to log the vocab and model
    tpe_tokenizer = TPETokenizerFast.from_pretrained(tpe_tokenizer_dir)
    filter_fn = partial(filter_by_length, tokenizer=tpe_tokenizer, max_seq_len=args.max_sft_seq_len,
                                           dataset=args.dataset, task=args.task)

    dataset = load_dataset("parquet", data_files=data_files,
                            columns=['message', 'generated_text_0', "label"],
                            split=args.data_split)
    print(f"Loaded dataset with {len(dataset)} samples from {data_fpath}")
    dataset = dataset.filter(filter_fn, num_proc=args.num_workers, batched=True, load_from_cache_file=False,
                             batch_size=1000)
    print(f"Filtered dataset to {len(dataset)} samples based on sequence length.")

    # Save the filtered dataset
    dataset.to_parquet(output_fpath)
    print(f"Filtered dataset saved to {output_fpath}")
