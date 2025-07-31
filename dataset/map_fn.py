from datasets import load_dataset
import json
import re
import numpy as np
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase
from llmlingua import PromptCompressor


def compress_with_lingua2_map_fn(sample,
                                 compress_rate: float):
    """
    The samples is a list of {
        "message": messages_batch,
        "label":   labels_batch
    }
    This function compresses the messages in the samples using LLMLingua2.
    return the same sample list with compressed messages.
    """
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True, # Whether to use llmlingua-2
    )
    rate = 1 - compress_rate # LLMLingua2 uses rate as the compression rate
    
    # Extract messages from samples
    prompts = []
    for message in sample["message"]:
        if isinstance(message, list):
            prompts.append(message[0]["content"])
        else:
            prompts.append(message["content"])

    # Compress prompts using LLMLingua2 by batch
    compressed_prompts = []
    for prompt in prompts:
        compressed_prompt = llm_lingua.compress_prompt(prompt,
                                                        rate=rate,
                                                        force_tokens=['\n', ':'])
        compressed_prompts.append(compressed_prompt["compressed_prompt"])

    message_batch = [
        [{"role": "user", "content": prompt}] for prompt in compressed_prompts
    ]
    labels_batch = sample["label"]
    return {
        "message": message_batch,
        "label":   labels_batch
    }


def _turn_data_dict_to_str(data_dict: dict, max_data_tokens_length: int, llm_tokenizer, data_prefix: str = "## Data\n"):
    """
    Convert a dictionary to a string.
    This function will calculate the maximum length of the data and return the string representation of the data.
    Args:
        data_dict (dict): The dictionary to be converted to a string.
        max_data_tokens_length (int): The maximum available data tokens length.
        llm_tokenizer: The tokenizer used to encode the data.
        data_prefix (str): The prefix to add before the data.
    Returns:
        str: The string representation of the dictionary.
    """
    if type(data_dict) is dict:
        # Extract the static events
        static_sentences = data_dict.pop("Static", [])
        static_sentence = "".join(static_sentences) if static_sentences else ""
        static_tokens = llm_tokenizer.encode(static_sentence, add_special_tokens=False) if static_sentence else []
        data_prefix_tokens = llm_tokenizer.encode(data_prefix, add_special_tokens=False)
        max_data_tokens_length -= len(static_tokens) + len(data_prefix_tokens)

        sentences = []
        tokens = []

        for time, str_events in reversed(data_dict.items()):
            if not str_events:
                continue
            if type(str_events) is str:
                str_events = [str_events]
            time_sentence = "".join(str_events)
            time_tokens = llm_tokenizer.encode(time_sentence, add_special_tokens=False)
            if len(tokens) + len(time_tokens) > max_data_tokens_length:
                break
            # Extend time_tokens at the beginning of the tokens list
            tokens = time_tokens + tokens
            sentences = str_events + sentences

        sentences = [data_prefix, static_sentence] + sentences
        input_prompt = "".join(sentences)
    elif type(data_dict) is str:
        data_prefix_tokens = llm_tokenizer.encode(data_prefix, add_special_tokens=False)
        max_data_tokens_length -= len(data_prefix_tokens)
        data_tokens = llm_tokenizer.encode(data_dict, add_special_tokens=False)
        if len(data_tokens) > max_data_tokens_length:
            data_tokens = data_tokens[-max_data_tokens_length:]
            input_prompt = llm_tokenizer.decode(data_tokens)
        else:
            input_prompt = data_dict
    return input_prompt


def sample_mapping_fn(sample, instruction, llm_tokenizer, max_input_length):
    """
    Batch processing version of sample_mapping_fn.
    Args:
        sample (dict): A batch of samples.
        instruction (str): The instruction to add to the prompt.
        llm_tokenizer: The tokenizer used to encode the data.
        max_input_length (int): The maximum input length for the model.
    Returns:
        dict: A dictionary containing the processed messages and labels.
    """
    labels = sample['label']
    data = sample['data']
    data = json.loads(data)
    question = sample['question']

    # Process prompt
    nodata_messages = [{"role": "user", "content": question + instruction}]
    nodata_tokens = llm_tokenizer.apply_chat_template(nodata_messages)
    max_data_tokens_length = max_input_length - len(nodata_tokens)
    data_str = _turn_data_dict_to_str(data, max_data_tokens_length, llm_tokenizer)
    prompt = data_str + question + instruction
    message = [{"role": "user", "content": prompt}]

    return {"message": message, "label": labels}


def _find_subseq(sequence: List[int], pattern: List[int]) -> int:
    """Return start index of *pattern* in *sequence* or -1 if not found."""
    pat_len = len(pattern)
    if pat_len == 0 or pat_len > len(sequence):
        return -1
    # simple sliding-window search (list.index on tuple slice is also OK)
    for i in range(len(sequence) - pat_len + 1):
        if sequence[i : i + pat_len] == pattern:
            return i
    return -1


def sft_map_prefix_fn(
    batch: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    generation_prefix: str = "<|im_start|>assistant\n",
) -> Dict[str, np.ndarray]:
    """
    Batched map function for SFT with robust response alignment.

    Parameters
    ----------
    batch : dict
        A HuggingFace Datasets batch containing:
        - "message": list of lists of dicts (conversation history)
        - "generated_text_0": the assistant’s generated outputs
    tokenizer : PreTrainedTokenizerBase
        Tokenizer configured with a chat template.
    max_seq_length : int
        Desired fixed sequence length after padding/truncation.
    generation_prefix : str, optional
        The string that always precedes the assistant’s response
        in the chat template (default: "<|im_start|>assistant\n").
        We locate its first occurrence to mark where the labels begin.

    Notes
    -----
    All tokens *before* (and including) the first generation_prefix
    are masked out in `labels` with -100; only the assistant’s reply
    (and its trailing EOS) contributes to the loss.
    """
    messages_batch = batch["message"]           # List[List[Dict[str,str]]]
    outputs_batch = batch["generated_text_0"]   # List[str]

    eos_id = tokenizer.eos_token_id
    prefix_ids = tokenizer(
        generation_prefix,
        add_special_tokens=False,
        return_attention_mask=False
    )["input_ids"]

    all_ids, all_masks, all_labels = [], [], []

    for msgs, out in zip(messages_batch, outputs_batch):
        # 1. Append the assistant’s output to the conversation history
        conv = msgs + [{"role": "assistant", "content": out}]

        # 2. Render the full chat string (we skip adding another generation prompt)
        chat_str = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        )

        # 3. Tokenize (no special tokens, since the template already includes them)
        ids = tokenizer(
            chat_str,
            add_special_tokens=False,
            return_attention_mask=False
        )["input_ids"]

        # 4. Build attention mask (1 for real tokens)
        attn = [1] * len(ids)

        # 5. Right-pad with EOS tokens up to max_seq_length
        pad_len = max_seq_length - len(ids)
        if pad_len > 0:
            ids.extend([eos_id] * pad_len)
            attn.extend([0] * pad_len)

        # 6. Find where the assistant’s response actually starts
        start = _find_subseq(ids, prefix_ids)
        if start == -1:
            # Fallback: mask everything before the last non-pad token
            start = len(ids) - pad_len - 1
        else:
            # Skip past the prefix itself
            start += len(prefix_ids)

        # 7. Create labels: mask everything before `start`
        labels = [-100] * start + ids[start:]
        if pad_len > 0:
            # Also mask all padding tokens
            labels[-pad_len:] = [-100] * pad_len

        all_ids.append(ids)
        all_masks.append(attn)
        all_labels.append(labels)

    return {
        "input_ids": np.array(all_ids, dtype=np.int32),
        "attention_mask": np.array(all_masks, dtype=np.int8),
        "labels": np.array(all_labels, dtype=np.int32),
    }


def sft_map_postfix_fn(
    batch: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    generation_postfix: str,
) -> Dict[str, np.ndarray]:
    """
    Vectorized batched map function for SFT.

    Workflow:
      1. Batch-encode all history prompts via apply_chat_template once.
      2. Batch-encode all assistant answers (each + generation_postfix) once.
      3. For each sample, concatenate prompt IDs + answer IDs,
         then truncate/pad to max_seq_length and build labels.
    """
    messages_batch = batch["message"]           # List[List[Dict[str,str]]]
    outputs_batch = batch["generated_text_0"]   # List[str]
    eos_id = tokenizer.eos_token_id

    # 1. Encode all prompts with generation prompt appended
    prompt_strs = tokenizer.apply_chat_template(
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
    prompt_ids_batch = prompt_enc["input_ids"]

    # 2. Encode all answers with the postfix
    answer_strs = [out + generation_postfix for out in outputs_batch]
    answer_enc = tokenizer(
        answer_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    answer_ids_batch = answer_enc["input_ids"]

    all_ids, all_masks, all_labels = [], [], []

    for prompt_ids, ans_ids in zip(prompt_ids_batch, answer_ids_batch):
        # Concatenate prompt + answer
        seq = prompt_ids + ans_ids
        label_start = len(prompt_ids)  # label is 0 before this

        # Build attention mask
        attn = [1] * len(seq)

        # Right-pad with EOS tokens
        pad_len = max_seq_length - len(seq)
        if pad_len > 0:
            seq.extend([eos_id] * pad_len)
            attn.extend([0] * pad_len)

        # Build labels: mask prompt region and padding
        labels = [-100] * label_start + seq[label_start:]
        if pad_len > 0:
            labels[-pad_len:] = [-100] * pad_len

        # Sanity checks
        assert len(seq) == max_seq_length
        assert len(attn) == max_seq_length
        assert len(labels) == max_seq_length

        all_ids.append(seq)
        all_masks.append(attn)
        all_labels.append(labels)

    return {
        "input_ids": np.asarray(all_ids, dtype=np.int32),
        "attention_mask": np.asarray(all_masks, dtype=np.int8),
        "labels": np.asarray(all_labels, dtype=np.int32),
    }


def tpe_sample_batch_mapping_fn(samples):
    """
    Batch processing version of sample_mapping_fn.
    Args:
        sample (dict): A batch of samples.
    """
    messages = samples['message']
    labels = samples["generated_text_0"]

    # Process prompt
    messages = [message[0]["content"] for message in messages]
    inputs = [message+label for message, label in zip(messages, labels)]

    return {"input": inputs, "label": labels}


def dataset_to_Dtok_batch_mapping_fn(samples, tokenizer):
    """
    Map the dataset to Dtok format.
    Args:
        sample (dict): A batch of samples.
        tokenizer: The tokenizer used to encode the data.

    Returns:
        dict: A dictionary containing the processed input and label.
    """
    messages = samples['input']
    cleaned_messages = [re.sub(r'\d{3,}', '', text) for text in messages]
    tokens = [tokenizer.tokenize(text) for text in cleaned_messages]
    return {"token": tokens}


def map_gender_to_index(gender:str):
    if gender in ["F"]:
        return 0
    elif gender in ["M"]:
        return 1
    else:
        raise ValueError(f"invalid gender: {gender}")
    

def map_race_to_index(race:str):
    """
    map string race into index
    needed to be updatec
    """
    race = race.lower()
    if "white" in race:
        return 0
    elif "black" in race:
        return 1
    elif "hispanic" in race:
        return 2
    elif "asian" in race:
        return 3
    elif "other" in race or "hawaiian" in race or "south american" in race:
        return 4
    else: # unknown case
        return 5
    
