from transformers import AutoModelForCausalLM
from datasets import load_dataset
import argparse
from pathlib import Path
import os

from run_fn.llm_sft_fn import supervised_finetune_fn
from dataset.map_fn import sft_map_postfix_fn
from tpe.tpe_tokenizer_fast import TPETokenizerFast
from utils.misc import set_random_seed,  get_sft_qa_fname, get_sft_dataset_dir
from run_fn.set_embed_fn import set_trainable_embeddings, restore_embeddings
from run_fn.set_lora_params import set_lora_on_layers
from utils.constants import generation_postfixs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="MedTPE", help="WandB project name.")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])
    parser.add_argument("--model_path", type=str, default="data/tpe_models/Qwen2.5-7B-Instruct_task-new_pancan_maxN-2_maxM-5000", help="LLM model path.")
    parser.add_argument("--data_dir", type=str, default="data/cleaned_SFT_QA", help="Data directory.")
    parser.add_argument("--log_dir", type=str, default="data", help="Output file path.")
    parser.add_argument("--dataset", type=str, default="EHRSHOT",
                        choices=['MIMICIV', "EHRSHOT"], help="Dataset to use for inference.")
    parser.add_argument("--train_fn", type=str, default="ssft",
                        choices=["ssft", "ssft-lora"], help="Training parameter to use for inference.")
    parser.add_argument("--task", type=str, default="new_pancan", #
                        choices=["icu_mortality", "icu_phenotyping", 
                                 "guo_readmission", "new_pancan"],
                        help="Task name.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"], help="Data format.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--gpu_ids",  type=lambda s: [int(item) for item in s.split(',')], default=[1], help="Comma-separated list of GPU IDs to use for inference.")
    parser.add_argument("--max_input_len", type=str, default="4k", choices=["2k", "4k", "6k", "8k"],
                        help="Max input length for the SSFT QA dataset.")
    parser.add_argument("--max_output_len", type=str, default="2k", choices=["2k", "4k"],
                        help="Max output length for the SSFT QA dataset.")
    parser.add_argument("--max_length", type=int, default=4*1024, help="Max length of input and output tokens.")
    parser.add_argument("--ds_config_path", type=str, default="config/ds_config.json", help="Path to DeepSpeed config file.")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes to use for dataset processing.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--val_steps", type=int, default=1000, help="Validation steps.")
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
    set_random_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
    print(f"GPU IDs: {args.gpu_ids}")

    model_fname = args.model_path.split("/")[-1]
    log_dir = Path(args.log_dir) / f"{args.train_fn}_models"
    model_sft_dir = _get_model_sft_dir(log_dir, model_fname) # used to log the vocab and model

    if (model_sft_dir / "config.json").exists():
        print(f"Model already exists at {model_sft_dir}. Skipping SSFT.")
        exit(0)
    
    # load the llm
    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=False, torch_dtype="auto")
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
    new_token_ids = [info["token_id"] for info in tokenizer.new_tok_info.values()]
    if args.train_fn == "ssft":
        model= set_trainable_embeddings(model, new_token_ids, freeze_params=True)
    elif args.train_fn == "ssft-lora":
        model = set_lora_on_layers(model=model,
                                    r=8, 
                                    alpha=16,
                                    dropout=0.05,
                                    train_kv_only=False,  # set to True if you want to train only KV layers
                                    )
        model= set_trainable_embeddings(model, new_token_ids, freeze_params=False)
        
    # ssft the llm with trl
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
    )

    #restore the model
    if args.train_fn == "ssft-lora":
        # set the trainable parameters
        model = model.merge_and_unload()
    model = restore_embeddings(model, new_token_ids)

    # save the model
    model.save_pretrained(model_sft_dir)
    tokenizer.save_pretrained(model_sft_dir)
    