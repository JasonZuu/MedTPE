import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from transformers import default_data_collator
from bitsandbytes.optim import Adam8bit
import wandb
from tqdm import tqdm
from datasets import DatasetDict
from typing import Optional


def supervised_finetune_fn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_dict: DatasetDict,
    # ---------------- Training parameters ----------------
    learning_rate: float = 5e-5,
    batch_size: int = 2,
    num_train_epochs: int = 1,
    max_seq_length: int = 4 * 1024,
    gradient_accumulation_steps: int = 2,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    # ---------------- DeepSpeed & precision --------------
    ds_config_path: Optional[str] = None,
    use_bf16: bool = True,
    # ---------------- Validation & Logging ---------------
    val_steps: int = 1000,
    num_proc: int = 8,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_mode: str = "online",
    # ---------------- Early stopping ---------------------
    patience: int = 3,
):
    """SFT with warm-up + exponential LR decay, periodic validation & early stopping."""
    # 1. W&B init -------------------------------------------------------------
    wandb.init(project=wandb_project, name=wandb_name, mode=wandb_mode)
    config = wandb.config
    config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_train_epochs": num_train_epochs,
        "max_seq_length": max_seq_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "max_grad_norm": max_grad_norm,
        "val_steps": val_steps,
        "patience": patience,
    }, allow_val_change=True)

    # 2. Tokenizer ------------------------------------------------------------
    tokenized_train = dataset_dict["train"]
    tokenized_val = dataset_dict["tuning"]

    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )
    val_loader = DataLoader(
        tokenized_val,
        batch_size=1,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    # 5. Accelerator ----------------------------------------------------------
    ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config_path) if ds_config_path else None
    accelerator = Accelerator(
        mixed_precision="bf16" if use_bf16 else None,
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed_plugin=ds_plugin,
    )

    # 6. Optimizer & LR Scheduler --------------------------------------------
    optimizer = Adam8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    total_steps = len(train_loader) * num_train_epochs // gradient_accumulation_steps
    warmup_steps = int(warmup_ratio * total_steps)

    # ---- LambdaLR: warm-up + exp decay ----
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 7. Prepare everything ---------------------------------------------------
    model, optimizer, train_loader, scheduler, val_loader = accelerator.prepare(
        model, optimizer, train_loader, scheduler, val_loader
    )

    # 8. Training loop --------------------------------------------------------
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    best_checkpoint = None
    no_improve_count = 0
    stop_training = False

    for epoch in range(num_train_epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()        # <-- still per step
                optimizer.zero_grad()

            wandb.log(
                {
                    "train_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],  # track LR
                },
                step=global_step,
            )
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(train_loss=loss.item())

            # ---- periodic validation ------------------------------------
            validate_now = (
                global_step % val_steps == 0 and global_step % len(train_loader) != 0
            )
            if validate_now:
                model.eval()
                total_val_loss, n_val_batches = 0.0, 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        vout = model(**vbatch)
                        total_val_loss += vout.loss.item()
                        n_val_batches += 1
                avg_val_loss = total_val_loss / n_val_batches
                wandb.log({"val_loss": avg_val_loss}, step=global_step)

                # ---- early-stopping check ------------------------------
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_checkpoint = accelerator.get_state_dict(model).copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(
                            f"Early stopping: no improvement in {patience} validations."
                        )
                        stop_training = True
                
                # post-processing: 1) free vram, 2) reset model to train mode
                del vout, vbatch
                accelerator.free_memory()
                model.train()

            if stop_training:
                break
        if stop_training:
            break

    # 9. Final validation -----------------------------------------------------
    model.eval()
    total_val_loss, n_val_batches = 0.0, 0
    with torch.no_grad():
        for vbatch in val_loader:
            vout = model(**vbatch)
            total_val_loss += vout.loss.item()
            n_val_batches += 1
    final_val_loss = total_val_loss / n_val_batches
    wandb.log({"val_loss": final_val_loss}, step=global_step)

    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_checkpoint = accelerator.get_state_dict(model).copy()
        print("Best validation loss achieved at the end of training.")

    # 10. Finalise ------------------------------------------------------------
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if best_checkpoint is not None:
        unwrapped_model.load_state_dict(best_checkpoint)
        print("Loaded best checkpoint from validation.")

    wandb.finish()
    return unwrapped_model, tokenizer
