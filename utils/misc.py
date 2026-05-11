import torch
from collections import Counter
import numpy as np
import random
import logging
import json
import re
try:
    import wandb
except Exception:
    wandb = None
import torch, gc
from pathlib import Path
from typing import Tuple, Union, List, Optional


def list_cuda_tensors(top_k: int = 20):
    tensors = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            tensors.append(obj)
    # dedupe
    unique = {id(t): t for t in tensors}.values()
    # sort by memory footprint
    sorted_ts = sorted(unique,
                       key=lambda t: t.element_size() * t.nelement(),
                       reverse=True)
    print(f"{'Type':16} {'Shape':20} {'Bytes':>8}")
    for t in sorted_ts[:top_k]:
        shape_str = str(tuple(t.size()))
        byte_sz = t.element_size() * t.nelement()
        print(f"{type(t).__name__:16} {shape_str:20} {byte_sz:8}")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_to_onehot(labels:torch.Tensor, num_classes:int):
    """
    turn class labels to one-hot encoding.
    :param labels: class labels.
    :return: one-hot encoding.
    """
    onehot = torch.zeros(labels.size(0), num_classes)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    return onehot


def get_log_fname(task,
                  data_format,
                  max_input_len,
                  llm_name,
                  pe_method,
                  seed=None,
                  n_response=None,
                  postfix=None):
    """
    Generate a log filename based on task, data format, and time resolution.

    Parameters:
        task (str): The task name.
        data_format (str): The data format.
        max_input_len (str): The input token context length.
        llm_name (str): The name of the LLM model.
        pe_method (str): The method used for positional encoding.

    Returns:
        str: The generated log filename.
    """
    log_fname = f"{task}_{data_format}_{max_input_len}_{llm_name}_{pe_method}"
    if seed is not None:
        log_fname += f"_seed-{seed}"
    if n_response is not None:
        log_fname += f"_nres-{n_response}"
    if postfix is not None:
        log_fname += f"_{postfix}"
    return log_fname


def get_sft_qa_fname(task, data_format, llm_name,  data_split, postfix=None):
    """
    Generate a filename for SFT QA based on task, data format, LLM name, positional encoding method, and data split.

    Parameters:
        task (str): The task name.
        data_format (str): The data format.
        llm_name (str): The name of the LLM model.
        pe_method (str): The method used for positional encoding.
        data_split (str): The data split (e.g., "train", "test").
        max_input_len (str): The maximum input length.
        max_output_len (str): The maximum output length.

    Returns:
        str: The generated filename for SFT QA.
    """
    fname = f"{task}_{data_format}_{llm_name}_set-{data_split}"
    if postfix:
        fname += f"_{postfix}"
    return fname


def majority_vote(actuals, actual_probs):
    """
    Determines the final result using majority voting and computes the probability of the selected result.

    Parameters:
    actuals (list): A list of predictions from different samples.
    actual_probs (list): A list of corresponding probabilities for each prediction.

    Returns:
    tuple: A tuple containing the final result and its aggregated probability.
    """
    # Step 1: Count the occurrences of each result
    counts = Counter(actuals)

    # Step 2: Find the result(s) with the highest count (majority candidates)
    max_count = max(counts.values())

    if max_count == 1:
        # If there are no majority candidates, return the most recent prediction
        final_result = actuals[-1]
        final_prob = actual_probs[-1]
    else:
        majority_candidates = [result for result, count in counts.items() if count == max_count]

        # Step 3: If there are multiple majority candidates, select the one with the highest probability
        if len(majority_candidates) > 1:
            candidate_probs = {
                result: sum([actual_probs[i] for i in range(len(actuals)) if actuals[i] == result]) / counts[result]
                for result in majority_candidates
            }
            # Select the result with the highest average probability
            final_result = max(candidate_probs, key=candidate_probs.get)
        else:
            # If there is a single majority candidate, select it as the final result
            final_result = majority_candidates[0]

    # Step 4: Compute the average probability for the selected result
    valid_probs = [actual_probs[i] for i in range(len(actuals)) if actuals[i] == final_result and actual_probs[i] is not None]
    final_prob = sum(valid_probs) / counts[final_result] if valid_probs else 0  # Handle empty case

    return final_result, final_prob


def calculate_prob(logprobs:list, token_ids:list):
    """
    Calculate the probability based on matching token_ids in sequence.

    Parameters:
        logprobs (list of dict): Each element is a dictionary containing token_id and its logprob.
        token_ids (list of str): The target token ID sequence to match in order.

    Returns:
        float: The calculated probability of all matched tokens. Returns 0.0 if no matches are found.
    """
    logprobs.reverse()  # Reverse the list to start from the end
    token_ids.reverse()  # Reverse the list to start from the end
    matched_logprobs = []  # Stores the logprobs of matched tokens
    index = 0  # Tracks progress in matching token_ids

    for logprob in logprobs:
        if index == len(token_ids):
                break  # Stop if all tokens have been matched
        for token_id, value in logprob.items():
            # Check if the current token matches the expected token in sequence
            if token_id == token_ids[index]:
                matched_logprobs.append(value.logprob)  # Record the logprob
                index += 1  # Move to the next token in the sequence
                break

    # Calculate the total probability
    if matched_logprobs:
        total_prob = np.exp(sum(matched_logprobs))
    else:
        total_prob = 0.0

    return total_prob


def get_task_id(dataset, task, data_format=None):
    """Get the task ID based on the dataset and task name."""
    if data_format is None:
        task_id = f"{dataset}//{task}"
    else:
        task_id = f"{dataset}//{task}//{data_format}"
    return task_id


def get_logger(name, log_file=None):
    """Get a logger with the specified name and log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def get_tpe_tokenizer_dir(log_dir: str, model_name:str, task:str,
                           max_n:int, max_m: int) -> str:
    """
    Get the vocabulary path for the specified model name.
    """
    tpe_tokenizer_dir = Path(log_dir) / f"{model_name}_task-{task}_maxN-{max_n}_maxM-{max_m}"
    tpe_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    return tpe_tokenizer_dir


def get_sft_dataset_dir(dataset: str, max_input_len: str, max_output_len: str) -> str:
    """
    Get the dataset directory based on the dataset name and max input/output lengths.
    """
    dataset_dir = f"{dataset}_max-in-{max_input_len}_max-out-{max_output_len}"
    return dataset_dir


def get_sft_model_dir(log_dir: str, model_name:str, task:str,
                      max_n:int, max_m: int, postfix: str = None) -> str:
    """
    Get the vocabulary path for the specified model name.
    """
    model_tpe_dir = Path(log_dir) / f"{model_name}_task-{task}_maxN-{max_n}_maxM-{max_m}"
    if postfix is not None:
        model_tpe_dir = Path(f"{model_tpe_dir}_{postfix}")
    model_tpe_dir.mkdir(parents=True, exist_ok=True)
    return model_tpe_dir


def get_regular_sft_model_dir(log_dir: str, model_name: str, train_fn: str = "sft") -> str:
    """
    Get the SFT model directory for regular SFT models (not TPE-based).
    This matches the directory structure used in llm_sft.py.

    Args:
        log_dir: Base log directory
        model_name: Model name (e.g., "Qwen2.5-1.5B-Instruct")
        train_fn: Training function type.

    Returns:
        Path to the SFT model directory
    """
    sft_models_dir = Path(log_dir) / f"{train_fn}_models"
    model_sft_dir = sft_models_dir / model_name
    return str(model_sft_dir)


# def parse_llm_output(generated_text: str):
#     """
#     Parses LLM-generated text to extract answer sections in various JSON-like formats.
#     Supported formats include:
#     - {answer: X}, {"answer": "X"}
#     - {Answer: [X, Y]}, {"Answer": ["X", "Y"]}
#     - {Answer: "A, B"} or unbracketed comma-separated values
#     - Handles single/double/no quotes, various spacing

#     Args:
#         generated_text: Full text generated by the LLM

#     Returns:
#         tuple: (invalid_output_flag, answer)
#             - invalid_output_flag: bool (True if output is invalid)
#             - answer: str or list or None (extracted answer(s) if valid)
#     """
#     # Pattern to capture key and value, allowing quotes or not
#     pattern = re.compile(
#         r"\{[^}]*?['\"]?(answer)['\"]?\s*:\s*(?P<value>\[[^]]*\]|(\".*?\"|'.*?'|[^,}\s]+))",
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = list(pattern.finditer(generated_text))
#     if not matches:
#         return False, None

#     # Use the last occurrence
#     raw_val = matches[-1].group('value').strip()

#     # Strip surrounding quotes/brackets for uniform processing
#     # If bracketed list, keep brackets
#     is_list = raw_val.startswith('[') and raw_val.endswith(']')

#     # Remove trailing punctuation
#     raw_val = re.sub(r"[\},]\s*$", "", raw_val).strip()

#     # If quoted string, unwrap
#     if (raw_val.startswith('"') and raw_val.endswith('"')) or \
#        (raw_val.startswith("'") and raw_val.endswith("'")):
#         raw_val = raw_val[1:-1].strip()
#         is_list = False

#     # List in brackets
#     if is_list:
#         inner = raw_val[1:-1].strip()
#         # Split on commas outside quotes
#         parts = re.split(r"\s*,\s*", inner)
#         items = [re.sub(r"^['\"]|['\"]$", "", p).strip() for p in parts if p]
#         return True, items

#     # Comma-separated values
#     if ',' in raw_val:
#         parts = re.split(r"\s*,\s*", raw_val)
#         items = [re.sub(r"^['\"]|['\"]$", "", p).strip() for p in parts if p]
#         return True, items

#     # Single token
#     single_match = re.match(r"^['\"]?(?P<v>[^'\"]+)['\"]?$", raw_val)
#     if single_match:
#         return True, single_match.group('v').strip()

#     return False, None



def get_study_name(dataset: str,
                   task: str,
                   model: str,
                   time_resolution: str = None,):
    """
    Get the database study name
    """
    study_name = f"{dataset}-{task}-{model}-tr_{time_resolution}"
    return study_name


def load_best_hparams(algo_config,
                      sweep_id,
                      args,
                      metric_name="auprc",
                      maximize=True,):
        """
        Load the best hyperparameters from the database
        """
        api = wandb.Api()
        sweep_entry = f"{args.entity}/{args.project}/{sweep_id}"
        sweep = api.sweep(sweep_entry)

        # Sort runs based on the specified metric
        runs = sorted(sweep.runs, key=lambda run: run.summary.get(metric_name, 0.0 if maximize else 1.0), reverse=maximize)
        # best_run = runs[0]
        # check runs
        best_run = None
        for run in runs:
            auroc = run.summary.get("auroc", None)
            if auroc is None: # no auroc metric
                best_run = run
                break
            elif auroc == 0.5: # skip runs with auroc=0.5 for no discrimination
                continue
            best_run = run
            break

        if best_run is None:
            raise ValueError(f"No valid run found for metric '{metric_name}'")

        best_params = best_run.config

        for key, value in best_params.items():
            setattr(algo_config, key, value)

        return algo_config


def get_sweep_id_by_name(entity, project, study_name):
    """
    Search for a sweep by its name in a given W&B project.

    Parameters:
    - entity (str): The entity name (user or team) in W&B.
    - project (str): The project name in W&B.
    - study_name (str): The name of the sweep to find.

    Returns:
    - str or None: The ID of the sweep if found, otherwise None.
    """
    # Initialize the API client
    if wandb is None:
        return None
    if wandb is None:
        raise RuntimeError("wandb is unavailable")
    api = wandb.Api()
    try:
        # Fetch all sweeps in the specified project
        project = api.project(name=project, entity=entity)
        sweeps = project.sweeps()
        # Search for the sweep with the specified name
        for sweep in sweeps:
            if sweep.config.get('name') == study_name:
                return sweep.id

        # If no sweep is found with the given name
        return None

    except Exception as e:
        print(f"An error occurred during load sweep: {e}")
        return None



def check_sweep_status(entity, project, sweep_id):
    if wandb is None:
        return "unknown"
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    return sweep.state


Candidate = Union[str, List[str]]

def parse_llm_output(generated_text: str) -> Tuple[bool, Optional[Candidate]]:
    """
    Robustly parse LLM outputs into answer(s), accepting JSON, pseudo-JSON, and loose text like:
      - {"answer": "B"}, {answer: B}, {"Answer": ["A","C"]}
      - Answer: B / final answer: B / prediction - B
      - Answer B / AnswerB / Ans: B / A, B
      - 'The answer is B', 'I choose A and C'
    Returns:
      (is_valid, answer) where is_valid=True iff parsing succeeded.
      answer is a str or list[str].
    """
    m = re.search(r'\{\s*"?[Aa]nswer"?\s*:\s*"?([A-Za-z])\b', generated_text)
    if m:
        return True, m.group(1).upper()
    text = generated_text.strip()

    # ------------ 1) JSON-first: try to find and parse the first plausible JSON object ------------
    # Grab the first {...} block and try to json.loads after mild cleanup (quote keys, etc.)
    json_obj = _extract_first_json_object(text)
    if json_obj is not None:
        # prefer common keys
        for k in ["answer", "Answer", "final_answer", "FinalAnswer", "prediction", "Prediction"]:
            if k in json_obj:
                val = _normalize_value(json_obj[k])
                if val is not None:
                    return True, val

    # ------------ 2) Braced "JSON-like" with unquoted keys/values ------------
    m = re.search(
        r"\{[^}]*?\b(answer|final[_\s]*answer|prediction)\b\s*:\s*(?P<v>\[[^\]]*\]|\".*?\"|'.*?'|[^}]+)\}",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        val = _normalize_value(m.group("v"))
        if val is not None:
            return True, val

    # ------------ 3) Key-value lines without braces ------------
    # e.g. "Answer: B", "final answer - A, C", "Prediction : ['A','B']"
    kv_patterns = [
        r"\b(final\s*answer|answer|prediction)\b\s*[:\-–—]\s*(?P<v>\[[^\]]*\]|\".*?\"|'.*?'|[A-Za-z](?:\s*,\s*[A-Za-z])+|[A-Za-z]+)",
        r"\b(final\s*answer|answer|prediction)\b\s+is\s+(?P<v>\[[^\]]*\]|\".*?\"|'.*?'|[A-Za-z](?:\s*,\s*[A-Za-z])+)"]
    for pat in kv_patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = _normalize_value(m.group("v"))
            if val is not None:
                return True, val

    # ------------ 4) Tight "AnswerB" / "Answer B" / "Ans B" ------------
    # Match concatenated or spaced label+option letters (A-D, Y/N, T/F)
    m = re.search(r"\b(ans|answer|finalanswer)\s*([:;\-–—])?\s*([A-Za-z](?:\s*,\s*[A-Za-z])*)\b", text, re.IGNORECASE)
    if m:
        val = _normalize_value(m.group(3))
        if val is not None:
            return True, val

    # Compact: "AnswerB" / "FinalAnswerC"
    m = re.search(r"\b(?:ans|answer|finalanswer)\s*([A-Za-z])\b", text, re.IGNORECASE)
    if m:
        val = _normalize_value(m.group(1))
        if val is not None:
            return True, val

    # ------------ 5) Natural phrases like "I choose A and C" / "The answer is B" ------------
    m = re.search(
        r"\b(?:choose|select|pick|is|are)\b\s+(?P<v>[A-Za-z](?:\s*(?:,|and)\s*[A-Za-z])*)\b",
        text, re.IGNORECASE
    )
    if m:
        val = _normalize_value(m.group("v"))
        if val is not None:
            return True, val

    # ------------ 6) As a last resort, try to extract the first clean JSON from anywhere ------------
    val = _extract_json_fallback(text)
    if val is not None:
        return True, val

    return False, None


# ========================== helpers ==========================

def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    Find the first {...} block and try increasingly tolerant JSON parses.
    """
    # naive brace match (first block)
    start = text.find("{")
    while start != -1:
        end = _find_matching_brace(text, start)
        if end != -1:
            block = text[start:end+1].strip()
            # try strict
            try:
                return json.loads(block)
            except Exception:
                # try: quote bare keys -> "key": value
                fixed = _quote_bare_keys(block)
                try:
                    return json.loads(fixed)
                except Exception:
                    pass
        start = text.find("{", start + 1)
    return None

def _find_matching_brace(s: str, i: int) -> int:
    depth = 0
    in_str = False
    esc = False
    for idx in range(i, len(s)):
        ch = s[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch in "\"'":
                in_str = False
        else:
            if ch in "\"'":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return idx
    return -1

def _quote_bare_keys(block: str) -> str:
    # turn {answer: B, final answer: "C"} into {"answer": "B", "final answer": "C"}
    # 1) quote keys
    tmp = re.sub(r'(\{|\s|,)([A-Za-z_][A-Za-z0-9_\s]*)(\s*):', lambda m: f'{m.group(1)}"{m.group(2).strip()}" :', block)
    # 2) quote bare alpha values up to , or }
    tmp = re.sub(r':\s*([A-Za-z]+)(\s*[,}])', r': "\1"\2', tmp)
    return tmp

def _normalize_value(raw: str) -> Optional[Candidate]:
    """
    Turn raw captured text into str or list[str], normalizing quotes/brackets and splitting on commas/and.
    Also maps common booleans/yes-no variants to canonical forms.
    """
    raw = str(raw)
    v = raw.strip()
    # strip trailing punctuation/braces
    v = re.sub(r"[}\)\]\.]\s*$", "", v).strip()

    # If looks like list
    if v.startswith("[") and v.endswith("]"):
        try:
            arr = json.loads(v.replace("'", '"'))
            return [_canonize(str(x)) for x in arr if str(x).strip()]
        except Exception:
            # fallback: split manually
            inner = v[1:-1]
            parts = _split_multi(inner)
            return [_canonize(p) for p in parts] if parts else None

    # If quoted string
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()

    # Could be comma/and separated values
    parts = _split_multi(v)
    if len(parts) > 1:
        return [_canonize(p) for p in parts]

    # single token
    return _canonize(v)

def _split_multi(s: str) -> List[str]:
    # split on commas or 'and' (outside quotes already ensured by our usage)
    # normalize separators
    s = re.sub(r"\s+(and|＆|＆and|＆AND)\s+", ",", s, flags=re.IGNORECASE)
    parts = [p.strip().strip('"').strip("'") for p in re.split(r"\s*,\s*", s) if p.strip()]
    return parts

def _canonize(s: str) -> Optional[str]:
    t = s.strip().strip('"').strip("'")
    if not t:
        return None
    # Common letter options: A-D (extend if needed)
    m = re.fullmatch(r"[A-Za-z]", t)
    if m:
        return m.group(0).upper()

    # True/False/Yes/No normalization
    tf_map = {
        "true": "True", "false": "False",
        "yes": "Yes", "no": "No",
        "y": "Yes", "n": "No",
        "t": "True", "f": "False"
    }
    low = t.lower()
    if low in tf_map:
        return tf_map[low]

    # Remove trailing punctuation like ; :
    t = re.sub(r"[:;,\.\s]+$", "", t)
    return t if t else None

def _extract_json_fallback(text: str) -> Optional[Candidate]:
    """
    Look for minimal JSON fragments like {"answer":"B"} or {"Answer":["A","C"]} anywhere.
    """
    for m in re.finditer(r"\{[^}]+\}", text, flags=re.DOTALL):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            for k in ["answer","Answer","final_answer","FinalAnswer","prediction","Prediction"]:
                if k in obj:
                    return _normalize_value(obj[k] if isinstance(obj[k], str) else json.dumps(obj[k]))
        except Exception:
            pass
    return None
