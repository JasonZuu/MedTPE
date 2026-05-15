import argparse
from pathlib import Path
import os
import json


def _configure_cuda_visible_devices(gpu_ids) -> None:
    if "CUDA_DEVICE_ORDER" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="MedTPE", help="WandB project name.")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline"])
    parser.add_argument("--model_path", type=str, default="data/MedTPE_data/tpe_models/Qwen2.5-1.5B-Instruct_task-cmedqa2_maxN-2_maxM-5000", help="LLM model path.")
    parser.add_argument("--data_dir", type=str, default="data/MedTPE_data/cleaned_SFT_QA", help="Data directory.")
    parser.add_argument("--log_dir", type=str, default="data/MedTPE_data", help="Output file path.")
    parser.add_argument("--dataset", type=str, default="CMEDQA2",
                        choices=["ECTSUM", "CMEDQA2"], help="Dataset to use for inference.")
    parser.add_argument("--train_fn", type=str, default="sft",
                        choices=["sft"], help="Training parameter to use for inference.")
    parser.add_argument("--task", type=str, default="cmedqa2", #
                        choices=["ect_summary", "cmedqa2"],
                        help="Task name.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"], help="Data format.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--gpu_ids",  type=lambda s: [int(item) for item in s.split(',')], default=[0], help="Comma-separated list of GPU IDs to use for inference.")
    parser.add_argument("--max_input_len", type=str, default="4k", choices=["2k", "4k", "6k", "8k"],
                        help="Max input length for the SFT QA dataset.")
    parser.add_argument("--max_output_len", type=str, default="2k", choices=["2k", "4k"],
                        help="Max output length for the SFT QA dataset.")
    parser.add_argument("--max_length", type=int, default=4*1024, help="Max length of input and output tokens.")
    parser.add_argument("--ds_config_path", type=str, default=None, help="Path to DeepSpeed config file.")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes to use for dataset processing.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--val_steps", type=int, default=1000, help="Validation steps.")
    parser.add_argument("--model_save_postfix", type=str, default=None)
    return parser.parse_args()


def _get_model_sft_dir(log_dir: str, model_fname:str) -> str:
    """
    Get the vocabulary path for the specified model name.
    """
    model_sft_dir = Path(log_dir) / model_fname
    model_sft_dir.mkdir(parents=True, exist_ok=True)
    return model_sft_dir


if __name__ == "__main__":
    args = get_args()
    _configure_cuda_visible_devices(args.gpu_ids)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM

    from dataset.map_fn import sft_map_postfix_fn, sft_map_template_fn
    from run_fn.llm_sft_fn import supervised_finetune_fn
    from run_fn.set_embed_fn import set_trainable_embeddings, restore_embeddings
    from tpe.tpe_tokenizer_fast import TPETokenizerFast
    from utils.constants import generation_postfixs
    from utils.misc import set_random_seed, get_sft_qa_fname, get_sft_dataset_dir

    set_random_seed(args.seed)
    print(f"GPU IDs (arg): {args.gpu_ids}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    model_fname = args.model_path.split("/")[-1]
    if args.model_save_postfix is not None and args.model_save_postfix.strip():
        model_fname = f"{model_fname}_{args.model_save_postfix.strip()}"
    log_dir = Path(args.log_dir) / f"{args.train_fn}_models"
    model_sft_dir = _get_model_sft_dir(log_dir, model_fname) # used to log the vocab and model

    if (model_sft_dir / "config.json").exists():
        print(f"Model already exists at {model_sft_dir}. Skipping SFT.")
        exit(0)

    # load the llm
    if "gemma" in model_fname:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                     attn_implementation="eager",
                                                     use_cache=False,
                                                     torch_dtype="bfloat16")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                    use_cache=False,
                                                    torch_dtype="bfloat16")

    model.gradient_checkpointing_enable()
    print(model)
    print(f"Tie word embeddings: {model.config.tie_word_embeddings}")

    # load the tokenizer
    tokenizer = TPETokenizerFast.from_pretrained(args.model_path)

    # load the train and tuning dataset
    llm_name = args.model_path.split("/")[-1].split("_")[0]  # e.g., "Qwen2.5-1.5B-Instruct"
    dataset_dir = get_sft_dataset_dir(dataset=args.dataset,
                                  max_input_len=args.max_input_len,
                                  max_output_len=args.max_output_len)
    train_data_fname = get_sft_qa_fname(
        task=args.task,
        data_format=args.data_format,
        data_split="train",
        llm_name=llm_name,
    )
    tuning_data_fname = get_sft_qa_fname(
        task=args.task,
        data_format=args.data_format,
        data_split="tuning",
        llm_name=llm_name,
    )
    data_files = {
        'train': str(Path(args.data_dir) / dataset_dir / (train_data_fname + ".parquet")),
        'tuning': str(Path(args.data_dir) / dataset_dir / (tuning_data_fname + ".parquet")),
    }
    dataset_dict = load_dataset("parquet", data_files=data_files, columns=['message', 'generated_text_0'])
    if "gemma" in model_fname:
        dataset_dict = dataset_dict.map(sft_map_template_fn,
                                        fn_kwargs={"tokenizer": tokenizer,
                                                   'max_seq_length': args.max_length},
                                        num_proc=args.num_proc,
                                        batch_size=1000,
                                        batched=True,
                                        remove_columns=['message', 'generated_text_0'],)
    else:
        dataset_dict = dataset_dict.map(sft_map_postfix_fn,
                            fn_kwargs={"tokenizer": tokenizer,
                                        'max_seq_length': args.max_length,
                                        "generation_postfix": generation_postfixs[llm_name]},
                            num_proc=args.num_proc,
                            batch_size=1000,
                            batched=True,
                            remove_columns=['message', 'generated_text_0'],)

    # Select the first 10k samples for training
    # dataset_dict = {k: v.select(range(min(args.num_train_samples, len(v)))) for k, v in dataset_dict.items()}

    # set the pad token as the eos token
    if not tokenizer.pad_token:
        tokenizer.set_pad_token(tokenizer.eos_token)

    # set the trainable parameters
    new_token_info_fpath = Path(args.model_path) / "new_token_info.json"
    if new_token_info_fpath.exists():
        with open(new_token_info_fpath, "r") as f:
            new_tok_info = json.load(f)
        new_token_ids = [info["token_id"] for info in new_tok_info.values()]
    else:
        new_token_ids = [info["token_id"] for info in tokenizer.new_tok_info.values()]
    model = set_trainable_embeddings(model, new_token_ids, freeze_params=True)

    # sft the llm with trl
    supervised_finetune_fn(
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset_dict,
        wandb_project=args.wandb_project,
        wandb_name=f"{model_fname}_{args.train_fn}",
        wandb_mode=args.wandb_mode,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        val_steps=args.val_steps,
        num_train_epochs=args.num_train_epochs,
        patience=3,
        num_proc=args.num_proc,
        ds_config_path=args.ds_config_path,
    )

    model = restore_embeddings(model, new_token_ids)

    # save the model
    model.save_pretrained(model_sft_dir)
    tokenizer.save_pretrained(model_sft_dir)
