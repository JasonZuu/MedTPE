import os
import argparse
from pathlib import Path
import time
import json
os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/hf_models/Qwen--Qwen2.5-1.5B-Instruct", help="LLM model path.")
    parser.add_argument("--data_dir", type=str, default="data/medtpe_data/cmedqa2", help="Data directory.")
    parser.add_argument("--tpe_model_dir", type=str, default="data/MedTPE_data/tpe_models", help="TPE model directory.")
    parser.add_argument("--sft_model_dir", type=str, default="data/MedTPE_data/sft_models", help="SFT model directory.")
    parser.add_argument("--log_dir", type=str, default="log/TPE", help="Output file path.")
    parser.add_argument("--dataset", type=str, default="CMEDQA2",
                        choices=["ECTSUM", "CMEDQA2"],
                        help="Dataset to use for inference.")
    parser.add_argument("--task", type=str, default="cmedqa2", #
                        choices=["ect_summary", "cmedqa2"],
                        help="Task name.")
    parser.add_argument("--tokenizer_type", type=str, default="bpe",
                        choices=["bpe", "tpe", "tpe-sft"], help="Tokenizer/model variant.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"], help="Data format.")
    parser.add_argument("--pe_method", type=str, default="raw",
                        choices=["raw", 'cot'], help="Algorithm to use for inference.")
    parser.add_argument("--set_name", type=str, default="held_out",
                        choices=["train", "tuning", "held_out"], help="Set name.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--num_responses", type=int, default=1, help="number of responses to generate.")
    parser.add_argument("--gpu_ids",  type=lambda s: [int(item) for item in s.split(',')], default=[0], help="Comma-separated list of GPU IDs to use for inference.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers to use for data loading.")
    parser.add_argument("--gpu_util", type=float, default=0.5, help="Number of workers to use for data loading.")
    parser.add_argument("--max_input_len", type=str, default="8k",
                        choices=["500", "1k","2k", "4k", "6k","8k"],
                        help="Max input length.")
    parser.add_argument("--max_n", type=int, default=2, help="Max number of results to return.")
    parser.add_argument("--max_m", type=int, default=5000, help="Max number of results to return.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    import multiprocess as mp
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoConfig
    from vllm import LLM, SamplingParams

    from utils.algo_config import LLMConfig
    from dataset.load_fn import load_ectsum_hf_dataset
    from dataset.map_fn import cmedqa2_sample_mapping_fn
    from run_fn.infer_fn import infer_llm_on_dataset
    from tpe.tpe_tokenizer_fast import TPETokenizerFast
    from utils.misc import set_random_seed, get_log_fname, get_sft_model_dir

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
    print(f"GPU IDs: {args.gpu_ids}")
    set_random_seed(args.seed)

    log_dir = Path(args.log_dir) / f"{args.dataset}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_llm_name = args.model_path.split("/")[-1].split("--")[-1]

    print(f"Log dir: {log_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"LLM model: {log_llm_name}")

    # check if the results already exist
    log_fname = get_log_fname(task=args.task,
                              data_format=args.data_format,
                              max_input_len=args.max_input_len,
                              llm_name=log_llm_name,
                              pe_method=args.pe_method,
                              n_response=args.num_responses if args.num_responses > 1 else None,
                              postfix=args.tokenizer_type)
    output_fpath = log_dir / f"{log_fname}.parquet"
    print("will save to ", output_fpath)
    if output_fpath.exists():
        print(f"Results already exist at: {output_fpath}")
        exit()


    # Initialize the LLM config
    algo_config = LLMConfig()
    algo_config.llm_name = args.model_path
    algo_config.log_dir = log_dir
    algo_config.set_name = args.set_name
    if args.max_input_len == "500":
        algo_config.max_input_len = 500
    elif args.max_input_len == "1k":
        algo_config.max_input_len = 1*1024
    elif args.max_input_len == "2k":
        algo_config.max_input_len = 2*1024
    elif args.max_input_len == "4k":
        algo_config.max_input_len = 4*1024
    elif args.max_input_len == "6k":
        algo_config.max_input_len = 6*1024
    elif args.max_input_len == "8k":
        algo_config.max_input_len = 8*1024

    # set the max_input_len and output_token_len
    orig_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    default_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    max_model_len = min(default_config.max_position_embeddings, orig_tokenizer.model_max_length)
    print(f"Model's available max model len: {max_model_len}")
    if max_model_len - algo_config.max_input_len >= 4*1024:
        algo_config.output_token_len = 4*1024
        algo_config.max_model_len = algo_config.max_input_len + algo_config.output_token_len
    elif max_model_len - algo_config.max_input_len >= 2*1024:
        algo_config.output_token_len = 2*1024
        algo_config.max_model_len = algo_config.max_input_len + algo_config.output_token_len
    else:
        raise ValueError(f"Model's max model len is too small: {max_model_len}")

    print(f"max input length: {algo_config.max_input_len}")
    print(f"output token length: {algo_config.output_token_len}")

    algo_config.device = args.device

    # Initialize and process the dataset
    if args.dataset == "ECTSUM":
        split_fname = "test.json" if args.set_name == "held_out" else ("val.json" if args.set_name == "tuning" else "train.json")
        data_fpath = Path(args.data_dir) / split_fname
        print(f"Loading ECTSum dataset from: {data_fpath}")
        if not data_fpath.exists():
            raise ValueError(f"Dataset file does not exist at: {data_fpath}")
        data_files = {args.set_name: str(data_fpath)}
        dataset = load_ectsum_hf_dataset(
            data_files,
            orig_tokenizer,
            input_max_length=algo_config.max_input_len,
            num_workers=args.num_workers,
            data_split=args.set_name,
        )
    elif args.dataset == "CMEDQA2":
        split_fname = "test.json" if args.set_name == "held_out" else ("validation.json" if args.set_name == "tuning" else "train.json")
        data_fpath = Path(args.data_dir) / split_fname
        print(f"Loading CMedQA2 dataset from: {data_fpath}")
        if not data_fpath.exists():
            raise ValueError(f"Dataset file does not exist at: {data_fpath}")
        dataset = load_dataset("json", data_files=str(data_fpath), split="train")
        dataset = dataset.map(
            cmedqa2_sample_mapping_fn,
            batched=True,
            fn_kwargs={
                "instruction": None,
                "llm_tokenizer": orig_tokenizer,
                "max_input_length": algo_config.max_input_len,
            },
            remove_columns=[col for col in dataset.column_names if col not in ("message", "label")],
            num_proc=None,
        )
    else:
        raise ValueError(f"Unsupported public dataset: {args.dataset}")
    # init the model and sampling params
    if args.tokenizer_type == "bpe":
        print("Model Path:", args.model_path)
        llm = LLM(model=args.model_path,
                  dtype="bfloat16",
                  max_model_len=algo_config.max_model_len,
                  enforce_eager=True,
                  gpu_memory_utilization=args.gpu_util,
                  tensor_parallel_size=len(args.gpu_ids),
                  trust_remote_code=True)
    elif args.tokenizer_type == "tpe":
        tpe_model_path = get_sft_model_dir(args.tpe_model_dir,
                                      model_name=log_llm_name,
                                      task=args.task,
                                      max_n=args.max_n,
                                      max_m=args.max_m)
        print("Model Path:", tpe_model_path)
        llm = LLM(model=str(tpe_model_path),
                  dtype="bfloat16",
                  max_model_len=algo_config.max_model_len,
                  enforce_eager=True,
                  gpu_memory_utilization=args.gpu_util,
                  tensor_parallel_size=len(args.gpu_ids),
                  trust_remote_code=True)
        tokenizer = TPETokenizerFast.from_pretrained(tpe_model_path)
        llm.set_tokenizer(tokenizer)
    elif args.tokenizer_type == "tpe-sft":
        sft_model_path = get_sft_model_dir(args.sft_model_dir,
                                      model_name=log_llm_name,
                                      task=args.task,
                                      max_n=args.max_n,
                                      max_m=args.max_m)
        print("Model Path:", sft_model_path)
        llm = LLM(model=str(sft_model_path),
                  dtype="bfloat16",
                  max_model_len=algo_config.max_model_len,
                  enforce_eager=True,
                  gpu_memory_utilization=args.gpu_util,
                  tensor_parallel_size=len(args.gpu_ids),
                  trust_remote_code=True)
        tokenizer = TPETokenizerFast.from_pretrained(sft_model_path)
        llm.set_tokenizer(tokenizer)
    else:
        raise ValueError(f"Unsupported tokenizer type: {args.tokenizer_type}")

    sampling_params = SamplingParams(
        n=args.num_responses,
        temperature=algo_config.temperature,
        top_p=1, top_k=-1,
        max_tokens=algo_config.output_token_len
    )

    # run the inference
    print("Running inference...")
    running_info = {}
    start_time = time.time()
    result_df = infer_llm_on_dataset(llm, dataset, sampling_params, algo_config=algo_config)
    end_time = time.time()
    running_info["inference_time"] = end_time - start_time

    # save the results
    with open(log_dir / f"running_info_{log_fname}.json", "w") as f:
        json.dump(running_info, f, indent=4)
    result_df.write_parquet(output_fpath)
