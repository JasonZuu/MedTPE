from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np
from transformers import PreTrainedTokenizerBase
import yaml


PROMPT_DIR = Path(__file__).resolve().parent / "prompt"


@lru_cache(maxsize=None)
def load_prompt(prompt_name: str) -> dict:
    prompt_path = PROMPT_DIR / f"{prompt_name}.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = yaml.safe_load(f)
    if not isinstance(prompt, dict):
        raise ValueError(f"Prompt file must contain a mapping: {prompt_path}")
    return prompt


def get_cmedqa2_QA(question: str, answer: str):
    prompt = load_prompt("cmedqa2")["prompt"].format(question=question)
    return prompt, answer


def cmedqa2_sample_mapping_fn(sample, instruction: str, llm_tokenizer, max_input_length: int):
    prompt_cfg = load_prompt("cmedqa2")
    messages_batch = []
    labels_batch = []
    for question, answer in zip(sample["question"], sample["answer"]):
        question = question or ""
        answer = answer or ""
        formatted_question, label = get_cmedqa2_QA(question, answer)
        if instruction and instruction.strip():
            full_prompt = f"{formatted_question}\n\n{instruction}"
        else:
            full_prompt = formatted_question
        prompt_tokens = llm_tokenizer.encode(full_prompt, add_special_tokens=False)
        if len(prompt_tokens) > max_input_length:
            template_suffix = prompt_cfg["truncation_prompt"].format(question="")
            if instruction and instruction.strip():
                template_suffix += f"\n\n{instruction}"
            template_tokens = llm_tokenizer.encode(template_suffix, add_special_tokens=False)
            available_tokens = max(0, max_input_length - len(template_tokens))
            question_tokens = llm_tokenizer.encode(question, add_special_tokens=False)
            if len(question_tokens) > available_tokens:
                truncated_question_tokens = question_tokens[-available_tokens:] if available_tokens else []
                truncated_question = llm_tokenizer.decode(truncated_question_tokens)
                formatted_question, label = get_cmedqa2_QA(truncated_question, answer)
                if instruction and instruction.strip():
                    full_prompt = f"{formatted_question}\n\n{instruction}"
                else:
                    full_prompt = formatted_question
        messages_batch.append([{"role": "user", "content": full_prompt}])
        labels_batch.append(label)
    return {"message": messages_batch, "label": labels_batch}


def ectsum_sample_mapping_fn(sample, llm_tokenizer, max_input_length):
    prompt_cfg = load_prompt("ectsum")
    prompt_instruction = prompt_cfg["instruction"]
    data_prefix = prompt_cfg["data_prefix"]
    messages_batch = []
    labels_batch = []
    nodata_messages = [{"role": "user", "content": prompt_instruction}]
    nodata_tokens = llm_tokenizer.apply_chat_template(nodata_messages)
    max_data_tokens = max(0, max_input_length - len(nodata_tokens))
    texts = sample["text"]
    labels = sample["summary"]
    for text, lbl in zip(texts, labels):
        data_str = data_prefix + (text or "")
        data_tokens = llm_tokenizer.encode(data_str, add_special_tokens=True)
        if len(data_tokens) > max_data_tokens:
            data_tokens = data_tokens[-max_data_tokens:] if max_data_tokens else []
            data_str = llm_tokenizer.decode(data_tokens)
        prompt = data_str + "\n\n" + prompt_instruction
        messages_batch.append([{"role": "user", "content": prompt}])
        labels_batch.append(lbl)
    return {"message": messages_batch, "label": labels_batch}


def sft_map_postfix_fn(
    batch: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    generation_postfix: str,
) -> Dict[str, np.ndarray]:
    messages_batch: List[List[Dict[str, str]]] = batch["message"]
    outputs_batch: List[str] = batch["generated_text_0"]
    eos_id: int = tokenizer.eos_token_id
    prompt_strs: List[str] = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_enc = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    prompt_ids_batch: List[List[int]] = prompt_enc["input_ids"]

    answer_strs = [out + generation_postfix for out in outputs_batch]
    answer_enc = tokenizer(
        answer_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    answer_ids_batch: List[List[int]] = answer_enc["input_ids"]

    all_ids, all_masks, all_labels = [], [], []
    for prompt_ids, ans_ids in zip(prompt_ids_batch, answer_ids_batch):
        seq: List[int] = prompt_ids + ans_ids
        if len(seq) > max_seq_length:
            seq = seq[:max_seq_length]

        label_start = len(prompt_ids)
        attn = [1] * len(seq)
        pad_len = max_seq_length - len(seq)
        if pad_len > 0:
            seq.extend([eos_id] * pad_len)
            attn.extend([0] * pad_len)

        labels = [-100] * label_start + seq[label_start:]
        if pad_len > 0:
            labels[-pad_len:] = [-100] * pad_len

        all_ids.append(seq)
        all_masks.append(attn)
        all_labels.append(labels)

    return {
        "input_ids": np.asarray(all_ids, dtype=np.int32),
        "attention_mask": np.asarray(all_masks, dtype=np.int8),
        "labels": np.asarray(all_labels, dtype=np.int32),
    }


def sft_map_template_fn(batch, tokenizer, max_seq_length, **kwargs):
    messages_batch = batch["message"]
    outputs_batch = batch["generated_text_0"]
    eos_id = tokenizer.eos_token_id

    seqs = [m + [{"role": "assistant", "content": o}] for m, o in zip(messages_batch, outputs_batch)]
    seq_ids_list = tokenizer.apply_chat_template(
        seqs, tokenize=True, add_generation_prompt=False, skip_special_tokens=False
    )
    prompt_ids_list = tokenizer.apply_chat_template(
        messages_batch, tokenize=True, add_generation_prompt=True, skip_special_tokens=False
    )

    all_ids, all_masks, all_labels = [], [], []
    for seq_ids, prompt_ids in zip(seq_ids_list, prompt_ids_list):
        total_len = len(seq_ids)
        prompt_len = len(prompt_ids)
        if total_len - prompt_len <= 0:
            continue

        if total_len > max_seq_length:
            cut_left = total_len - max_seq_length
            seq_ids = seq_ids[cut_left:]
        else:
            cut_left = 0

        label_start = min(max(prompt_len - cut_left, 0), max_seq_length)
        if label_start >= len(seq_ids):
            continue

        attn_mask = [1] * len(seq_ids)
        pad_len = max_seq_length - len(seq_ids)
        if pad_len > 0:
            seq_ids = seq_ids + [eos_id] * pad_len
            attn_mask = attn_mask + [0] * pad_len

        labels = [-100] * max_seq_length
        right_bound = min(max_seq_length, len(seq_ids))
        for i in range(label_start, right_bound):
            if attn_mask[i] == 1:
                labels[i] = seq_ids[i]

        all_ids.append(seq_ids)
        all_masks.append(attn_mask)
        all_labels.append(labels)

    return {
        "input_ids": np.asarray(all_ids, dtype=np.int64),
        "attention_mask": np.asarray(all_masks, dtype=np.int8),
        "labels": np.asarray(all_labels, dtype=np.int64),
    }


def tpe_sample_batch_mapping_fn(samples):
    messages = samples["message"]
    labels = samples["generated_text_0"]
    messages = [message[0]["content"] for message in messages]
    inputs = [message + label for message, label in zip(messages, labels)]
    return {"input": inputs, "label": labels}


def dataset_to_Dtok_batch_mapping_fn(samples, tokenizer, filter_numbers: bool = True):
    messages = samples["input"]
    if filter_numbers:
        cleaned_messages = [re.sub(r"\d+", "", text) for text in messages]
    else:
        cleaned_messages = messages
    tokens = [tokenizer.tokenize(text) for text in cleaned_messages]
    return {"token": tokens}
