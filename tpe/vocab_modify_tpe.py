from collections import defaultdict, Counter, OrderedDict
import heapq
import json
import time
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
from copy import deepcopy

from dataset.map_fn import dataset_to_Dtok_batch_mapping_fn


def prepare_n_tokens(Dtok, max_n, min_n=2):
    """Generate all contiguous n-grams (as strings) with length from min_n to max_n

    Args:
        Dtok: Tokenized documents (list of lists of tokens)
        max_n: Maximum n-gram length
        min_n: Minimum n-gram length (default=2)

    Returns:
        List of unique n-gram strings
    """
    assert max_n >= min_n, f"max_n ({max_n}) must be >= min_n ({min_n})"

    n_tokens = set()
    pbar = tqdm(total=len(Dtok), desc="Generating n-grams", unit="doc")
    for doc in Dtok:
        tokens = doc
        length = len(tokens)
        for i in range(length):
            # Start from min_n instead of 1
            for j in range(min_n, max_n + 1):
                if i + j <= length:
                    ngram = ' '.join(tokens[i:i+j])
                    n_tokens.add(ngram)
        pbar.update(1)
    pbar.close()
    return list(n_tokens)


def cnt_freqs(Dtok, n_tokens, max_n):
    """Count the frequency of each n-gram in the corpus"""
    freq = defaultdict(int)
    ngram_set = set(n_tokens)
    pbar = tqdm(total=len(Dtok), desc="Counting n-gram frequencies", unit="doc")
    for doc in Dtok:
        tokens = doc
        doc_len = len(tokens)
        for i in range(doc_len):
            for j in range(1, max_n + 1):
                if i + j <= doc_len:
                    ngram = ' '.join(tokens[i:i+j])
                    if ngram in ngram_set:
                        freq[ngram] += 1
        pbar.update(1)
    pbar.close()
    return freq


def cnt_overlaps(n_tokens, Fn_tok, max_n):
    """
    Count the overlaps between n-grams and their prefixes.
    """
    overlaps = defaultdict(lambda: defaultdict(int))
    prefix_to_targets = defaultdict(list)

    # Build prefix-to-n-gram (t' -> list of its prefixes)
    for t_prime in n_tokens:
        tokens = t_prime.split()
        for k in range(1, min(len(tokens), max_n)):
            prefix = ' '.join(tokens[:k])
            prefix_to_targets[prefix].append(t_prime)

    # Match each n-gram t to all t' it is a prefix of
    for t in n_tokens:
        for t_prime in prefix_to_targets.get(t, []):
            overlaps[t][t_prime] = Fn_tok[t_prime]

    return overlaps


def count_old_token_frequencies(
    tokenized_docs: List[List[str]],
    vocab_old: Dict[str, int],
    disable_progress: bool = False
) -> Dict[str, int]:
    """
    Efficiently counts frequencies of tokens present in original vocabulary

    Args:
        tokenized_docs: List of tokenized documents (list of lists)
        vocab_old: Original vocabulary {token: id}
        disable_progress: Whether to hide progress bar

    Returns:
        Dictionary of {token: count} for tokens in original vocabulary
    """
    # Precompute set for faster lookups
    vocab_set = set(vocab_old.keys())
    freq_counts = defaultdict(int)

    # Use manual progress bar control for better performance
    with tqdm(total=len(tokenized_docs),
             desc="Counting token frequencies",
             unit="doc",
             disable=disable_progress) as pbar:

        # Process documents in batches for better memory efficiency
        batch_size = 1024
        for i in range(0, len(tokenized_docs), batch_size):
            batch = tokenized_docs[i:i+batch_size]

            # Flatten and count using Counter's C-optimized implementation
            batch_counts = Counter(
                token
                for doc in batch
                for token in doc
                if token in vocab_set
            )

            # Merge counts
            for token, count in batch_counts.items():
                freq_counts[token] += count

            pbar.update(len(batch))
    return freq_counts


def deduplicate_keep_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def insert_one_merge(
    merges: List[Tuple[str, str]],
    merge_to_insert: Tuple[str, str],
    position: int
) -> None:
    """
    Insert a single merge at the specified position if it does not already exist.
    Args:
        merges: List of existing merges (modified in place).
        merge_to_insert: The merge tuple (a, b) to insert.
        position: Index at which to insert the new merge.
    """
    if merge_to_insert in merges:
        return
    # Ensure position is within bounds
    pos = max(0, min(position, len(merges)))
    merges.insert(pos, merge_to_insert)


def validate_merge_path(merge_path, vocab):
    """Check if the merge path is valid in the vocabulary.
    which means all tokens in the path exist in the vocabulary.
    """
    for merge in merge_path:
        if len(merge) != 2:
            return False
        a, b = merge
        if a not in vocab or b not in vocab:
            return False
    return True


def get_all_merge_paths(subtokens: List[str]) -> List[List[Tuple[str, str]]]:
    """
    Enumerate all full merge paths for a list of subtokens.

    Args:
        subtokens: List of subtokens, e.g. ["a", "b", "c", "d"]

    Returns:
        A list of merge-sequences. Each merge-sequence is a list of tuples,
        where each tuple is (left, right) tokens that were merged at that step.
        The sequence has length len(subtokens) - 1.
    """
    # 递归基：只有一个 token 时，不用 merge，返回包含一个空路径的列表
    if len(subtokens) <= 1:
        return [[]]

    all_merge_paths = []

    # 对每一个可能的相邻位置 i，进行一次 merge
    for i in range(len(subtokens) - 1):
        left = subtokens[i]
        right = subtokens[i + 1]
        merged = left + right

        # 构造下一层的 token 列表
        next_tokens = (
            subtokens[:i] +         # 左侧不变部分
            [merged] +              # 放入新合并的 token
            subtokens[i + 2:]       # 右侧不变部分
        )

        # 递归获取剩余的 merge-paths
        for suffix_path in get_all_merge_paths(next_tokens):
            # 把当前这一步 (left,right) prepend 到后续路径上
            all_merge_paths.append([(left, right)] + suffix_path)

    return all_merge_paths


def build_tok_merges(ngram_toks, ngram_score):
    """
    基于给定的 token-level n-gram 频次，自动构建 tok-level merges 列表，支持任意长度的 n-gram，但始终以二元合并步骤输出，去除重复的合并对。

    参数:
    - ngram_score: dict，键为 tuple(token1, token2, ...)，值为对应的分数
    - ngram_toks: iterable，包含所有候选的 n-gram tuple（长度>=2），将按照它们的频次排序。

    返回:
    - merges: list of tuple(tuple, tuple)，每一项为一次二元合并操作，按频次优先级排列且无重复：
        ((t1,), (t2,)) 表示先合并 t1 和 t2，
        ((t1, t2), (t3,)) 表示再合并结果 (t1,t2) 与 t3，以此类推。

    备注:
    - 不需要指定 num_merges，所有传入的 ngram_toks 都会被考虑。
    - 输出的 merges 仅包含长度为2的合并操作，且若同一操作在多个 n-gram 中出现，仅保留首次。
    """
    # 1. 过滤并排序候选 n-grams
    candidates = [(tuple(ngram.split()), ngram_score.get(ngram, 0))
                  for ngram in ngram_toks]
    # 按频次降序，频次相同时按字典序排序
    candidates.sort(key=lambda x: (-x[1], x[0]))

    # 2. 生成二元合并操作列表，去重
    merges = []
    seen = set()
    for ngram, _ in candidates:
        prev = ngram[0]
        for tok in ngram[1:]:
            curr = tok
            pair = (prev, curr)
            if pair not in seen:
                merges.append(pair)
                seen.add(pair)
            # 更新前缀为已合并的结果
            prev = prev + curr
    return merges


from typing import Dict, Tuple, List
from collections import defaultdict

def repair_vocab_new(
    vocab_new: Dict[str, int],
    vocab_old: Dict[str, int]
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    修复 vocab_new 中的两类问题：
      1) 重复 ID：只保留每个 ID 对应的第一个 token，并统计被移除的重复 token
      2) 缺失 ID（holes）：用 vocab_old 对应的 token 填回

    Args:
        vocab_new:  经过增删改后可能有“洞”或重复 ID 的词表 mapping token->id
        vocab_old:  原始完整词表 mapping token->id

    Returns:
        repaired:   修补完毕的 vocab_new mapping token->id
        duplicates: 重复 ID 的统计 dict，形如 {id: [tok_removed1, tok_removed2, ...], ...}
    """
    # 1. 反向映射 vocab_new: id -> [tokens]
    id_to_tokens = defaultdict(list)
    for tok, tid in vocab_new.items():
        id_to_tokens[tid].append(tok)

    repaired = {}
    duplicates: Dict[int, List[str]] = {}

    # 2. 对每个 id，保留第一遇到的 token，记录其余为重复
    for tid, toks in id_to_tokens.items():
        # 保留第一个
        first_tok = toks[0]
        repaired[first_tok] = tid
        # 如果有多余，就记录这些被移除的 duplicates
        if len(toks) > 1:
            duplicates[tid] = toks[1:]

    # 3. 找出缺失 ID（holes）并用 vocab_old 补回
    #    只在 [0..max_old_id] 范围内补
    if vocab_old:
        # 建立反向 old 映射
        id_to_old = {tid: tok for tok, tid in vocab_old.items()}
        max_old_id = max(id_to_old.keys())

        used_ids = set(repaired.values())
        holes = [i for i in range(max_old_id + 1) if i not in used_ids]

        for hid in holes:
            old_tok = id_to_old.get(hid)
            if old_tok and old_tok not in repaired:
                repaired[old_tok] = hid

    return repaired



def tpe_vocabulary_modify_fn(tokenizer: PreTrainedTokenizerFast,
                            dataset_dict: DatasetDict,  # Changed to HF Dataset
                            max_m: int,
                            max_n: int,
                            filter_numbers: bool,
                            byte_merges: list,
                            vocab_size: int,
                            dataset_split: str="train",
                            min_freq: int=2,
                            timing_fpath: Optional[str]=None,
                            ) -> Tuple[Dict[str, int], List[Tuple[str, str]], List[Tuple[str, str]], Dict[int, Dict[str, str]]]:
    """
    Modify the vocabulary: replace up to max_m lowest-frequency old tokens with top max_m n-grams

    Key notes:
        1) skip special tokens and tokens with len(str) == 1 from replacement
        2) skip tokens in the merge paths in an recursive manner
        3) keeps all the old tokens that are not replaced by new tokens
    Args:
        tokenizer: PreTrainedTokenizerFast object
        dataset_dict: HF DatasetDict object
        max_m: Maximum number of new tokens to add
        max_n: Maximum n-gram length
        byte_merges: List of tuples representing byte level merges
        dataset_split: Dataset split to use (e.g., "train", "test")
        min_freq: Minimum frequency for n-grams to be considered (default=2)
    Returns:
        vocab_new: New vocabulary
        byte_merges: Updated byte merges
        tok_merges: List of tuples representing token merges
        new_token_info: Information about new tokens added
    """
    total_start = time.perf_counter()
    stage_timing = {}

    # Initialize old and new vocabularies
    stage_start = time.perf_counter()
    _vocab_old = {k: v for k, v in tokenizer.vocab.items() if v < vocab_size}  # Limit to vocab_size
    vocab_old = deepcopy(_vocab_old)  # Copy the original vocabulary
    vocab_new = {}
    stage_timing["init_vocab_seconds"] = time.perf_counter() - stage_start

    # 1. Tokenize the corpus from HF Dataset
    stage_start = time.perf_counter()
    dataset = dataset_dict[dataset_split]  # Extract text column
    dataset = dataset.map(dataset_to_Dtok_batch_mapping_fn,
                          fn_kwargs={"tokenizer": tokenizer, "filter_numbers": filter_numbers},
                          batched=True,
                          desc="Tokenizing dataset",
                          batch_size=1000,
                          num_proc=8)  # Use multiple processes for efficiency
    stage_timing["tokenize_dataset_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    Dtok = [item["token"] for item in dataset]  # Assuming 'data' is the text column
    stage_timing["materialize_tokenized_docs_seconds"] = time.perf_counter() - stage_start

    # 2. Extract n-grams and calculate their score
    stage_start = time.perf_counter()
    ngram_tokens = prepare_n_tokens(Dtok, max_n)
    stage_timing["prepare_ngrams_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    Fn_tok = cnt_freqs(Dtok, ngram_tokens, max_n)
    stage_timing["count_ngram_freq_seconds"] = time.perf_counter() - stage_start

    # Calculate overlap relationships
    stage_start = time.perf_counter()
    Foverlaps = cnt_overlaps(ngram_tokens, Fn_tok, max_n)
    stage_timing["count_overlap_seconds"] = time.perf_counter() - stage_start

    # Calculate initial scores S = frequency × length
    stage_start = time.perf_counter()
    ngram_score = {t: Fn_tok[t] * len(t.split()) for t in ngram_tokens if Fn_tok[t] >= min_freq} # remove low freq n-grams
    S = OrderedDict(sorted(ngram_score.items(), key=lambda x: x[1], reverse=True))  # Sort by score descending
    ngram_tokens = list(S.keys())
    stage_timing["build_ranked_ngram_scores_seconds"] = time.perf_counter() - stage_start

    # 4. Set token list that won't be replaced
    # Special tokens
    stage_start = time.perf_counter()
    special_tokens = set()
    for key, token in tokenizer.special_tokens_map.items():
        if token is None:
            continue
        if key == "additional_special_tokens":
            special_tokens.update(set(token))
        else:
            special_tokens.add(token)
    for token, token_id in tokenizer.added_tokens_encoder.items():
        if token is None:
            continue
        special_tokens.add(token)
    stage_timing["collect_special_tokens_seconds"] = time.perf_counter() - stage_start

    # Reserve tokens on the merge paths for new toks
    merges_dict = {f"{parts[0]}{parts[1]}": parts for parts in byte_merges}
    def _decompose_token(t: str, vocab) -> List[str]:
        """递归分解token直到所有子token在merges_dict存在"""
        if t in merges_dict:
            t1, t2 = merges_dict[t]
            return [t] + _decompose_token(t1, vocab) + _decompose_token(t2, vocab)
        elif t in vocab:
            return [t]
        else:
            raise ValueError(f"Token {t} not in vocab")

    subtokens = set()
    stage_start = time.perf_counter()
    pbar = tqdm(total=len(ngram_tokens), desc="Decomposing tokens", unit="token")
    for tok in ngram_tokens:
        for sub_tok in tok.split():
            if sub_tok not in vocab_old: # make sure all sub-tokens are in vocab_old
                S.pop(tok)
                break
            subtokens.update(_decompose_token(sub_tok, vocab_old))
        pbar.update(1)
    pbar.close()
    stage_timing["decompose_and_filter_subtokens_seconds"] = time.perf_counter() - stage_start

    # 3. create the new vocab by replacing old tokens with new tokens
    # Build min-heap (sorted by frequency ascending) of old tokens
    stage_start = time.perf_counter()
    old_token_freq = count_old_token_frequencies(
        Dtok, vocab_old, disable_progress=False
    )
    stage_timing["count_old_token_freq_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    heap = []

    for token_str, token_id in vocab_old.items():
        # Skip special tokens and tokens with length 1
        if token_str in special_tokens or len(token_str) == 1:
            continue
        elif token_str in subtokens:
            continue
        freq = old_token_freq.get(token_str, 0)
        heapq.heappush(heap, (freq, token_str, token_id))
    stage_timing["build_old_token_heap_seconds"] = time.perf_counter() - stage_start

    # Determine actual number of replacements
    stage_start = time.perf_counter()
    if max_m == -1:
        actual_replacements = len(S)
    else:
        actual_replacements = min(max_m, len(S))
    stage_timing["compute_replacement_budget_seconds"] = time.perf_counter() - stage_start
    new_token_info = {}

    # Replace tokens and collect tok_merges
    stage_start = time.perf_counter()
    pbar = tqdm(total=actual_replacements, desc="Replacing tokens", unit="token")
    for _ in range(actual_replacements):
        if not heap or not S:
            break

        new_token_ngram = max(S, key=lambda k: S[k])
        sub_tokens = new_token_ngram.split(" ")
        if not all(sub_token in vocab_old for sub_token in sub_tokens):
            print(f"Token {new_token_ngram} contains unknown sub-tokens. Skipping.")
            S.pop(new_token_ngram)
            pbar.update(1)
            continue

        # Create new token string and sub-token IDs
        sub_token_ids = [vocab_old[sub_token] for sub_token in sub_tokens]
        new_token_str = new_token_ngram.replace(" ", "")  # only drop the space caused by the split and keep the original space

        # Check if new token already exists in old vocab
        if new_token_str in vocab_old:
            print(f"Token {new_token_str} already exists in old vocab. Skipping.")
            S.pop(new_token_ngram)  # Already in old vocab
            pbar.update(1)
            continue

        # Pop the lowest frequency token from the heap of old tokens but skip the ones in dependent_toks
        freq, old_token_str, token_id = heapq.heappop(heap)

        # 1) Reserve the ID by inserting the new token immediately…
        vocab_new[new_token_str] = token_id
        # 2) …then remove the old
        vocab_old.pop(old_token_str)

        # add new token info
        new_token_info[new_token_str] = {
            "token_id": token_id,
            "token": new_token_str,
            "token_ngram": new_token_ngram,
            "sub_token_ids": sub_token_ids,
            "sub_tokens": sub_tokens,
        }

        # Decrease the score of the n-gram with the new token as prefix
        for t_prime in Foverlaps.get(new_token_ngram, {}):
            if t_prime in S:
                S[t_prime] -= Foverlaps[new_token_ngram][t_prime] * len(new_token_ngram.split())

        S.pop(new_token_ngram)  # Remove the n-gram from S
        pbar.update(1)
    pbar.close()
    stage_timing["replace_tokens_loop_seconds"] = time.perf_counter() - stage_start

    # Merge remaining vocabulary
    stage_start = time.perf_counter()
    vocab_new.update(vocab_old)
    stage_timing["merge_vocab_seconds"] = time.perf_counter() - stage_start

    # Repair any ID‐holes with old toks
    stage_start = time.perf_counter()
    vocab_new = repair_vocab_new(vocab_new, _vocab_old)
    stage_timing["repair_vocab_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    new_token_info = {new_tok: info for new_tok, info in new_token_info.items() if new_tok in vocab_new}
    stage_timing["filter_new_token_info_seconds"] = time.perf_counter() - stage_start

    # 6. remove invalid merges for the new vocab
    stage_start = time.perf_counter()
    byte_merges_new = []
    for merge in byte_merges:
        if merge[0]+merge[1] in vocab_new and all(part in vocab_new and part != "" for part in merge):
            byte_merges_new.append((merge[0], merge[1]))
    stage_timing["filter_byte_merges_seconds"] = time.perf_counter() - stage_start

    # 7. build tok_merges
    stage_start = time.perf_counter()
    new_ngram = [info["token_ngram"] for info in new_token_info.values()]
    tok_merges = build_tok_merges(ngram_toks=new_ngram, ngram_score=ngram_score)
    stage_timing["build_tok_merges_seconds"] = time.perf_counter() - stage_start
    stage_timing["total_seconds"] = time.perf_counter() - total_start
    stage_timing["n_new_tokens"] = len(new_token_info)
    stage_timing["n_vocab_old"] = len(_vocab_old)
    stage_timing["n_vocab_new"] = len(vocab_new)
    stage_timing["n_dataset_rows"] = len(Dtok)
    stage_timing["n_candidate_ngrams"] = len(ngram_tokens)
    stage_timing["n_filtered_ngrams"] = len(S)
    stage_timing["n_byte_merges_old"] = len(byte_merges)
    stage_timing["n_byte_merges_new"] = len(byte_merges_new)
    stage_timing["n_tok_merges_new"] = len(tok_merges)
    stage_timing["max_n"] = max_n
    stage_timing["max_m"] = max_m
    stage_timing["min_freq"] = min_freq
    stage_timing["dataset_split"] = dataset_split

    if timing_fpath is not None:
        timing_path = Path(timing_fpath)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(timing_fpath, "w") as f:
            json.dump(stage_timing, f, indent=4, ensure_ascii=False)

    return vocab_new, byte_merges_new, tok_merges, new_token_info


def tpe_vocabulary_extend_fn(tokenizer: PreTrainedTokenizerFast,
                              dataset_dict: DatasetDict,
                              max_m: int,
                              max_n: int,
                              filter_numbers: bool,
                              byte_merges: list,
                              vocab_size: int,
                              dataset_split: str = "train",
                              min_freq: int = 2,
                              ) -> Tuple[Dict[str, int], List[Tuple[str, str]], List[Tuple[str, str]], Dict[int, Dict[str, str]]]:
    """
    Extend the vocabulary: append up to max_m new n-gram tokens WITHOUT removing any original tokens.
    New token IDs start at vocab_size and increment from there.

    Args: same as tpe_vocabulary_modify_fn
    Returns: same signature as tpe_vocabulary_modify_fn
    """
    # Keep all original tokens
    vocab_old = {k: v for k, v in tokenizer.vocab.items() if v < vocab_size}

    # 1. Tokenize the corpus
    dataset = dataset_dict[dataset_split]
    dataset = dataset.map(dataset_to_Dtok_batch_mapping_fn,
                          fn_kwargs={"tokenizer": tokenizer, "filter_numbers": filter_numbers},
                          batched=True,
                          desc="Tokenizing dataset",
                          batch_size=1000,
                          num_proc=8)
    Dtok = [item["token"] for item in dataset]

    # 2. Extract n-grams and calculate scores
    ngram_tokens = prepare_n_tokens(Dtok, max_n)
    Fn_tok = cnt_freqs(Dtok, ngram_tokens, max_n)
    Foverlaps = cnt_overlaps(ngram_tokens, Fn_tok, max_n)
    ngram_score = {t: Fn_tok[t] * len(t.split()) for t in ngram_tokens if Fn_tok[t] >= min_freq}
    S = OrderedDict(sorted(ngram_score.items(), key=lambda x: x[1], reverse=True))
    ngram_tokens = list(S.keys())

    # 3. Identify sub-tokens that must stay in vocab (merge path protection — same as replace mode)
    merges_dict = {f"{parts[0]}{parts[1]}": parts for parts in byte_merges}

    def _decompose_token(t: str, vocab) -> List[str]:
        if t in merges_dict:
            t1, t2 = merges_dict[t]
            return [t] + _decompose_token(t1, vocab) + _decompose_token(t2, vocab)
        elif t in vocab:
            return [t]
        else:
            raise ValueError(f"Token {t} not in vocab")

    pbar = tqdm(total=len(ngram_tokens), desc="Decomposing tokens", unit="token")
    for tok in ngram_tokens:
        for sub_tok in tok.split():
            if sub_tok not in vocab_old:
                S.pop(tok)
                break
            _decompose_token(sub_tok, vocab_old)  # validate merge path exists
        pbar.update(1)
    pbar.close()

    # 4. Extend vocabulary — no heap, no removal
    actual_additions = min(max_m, len(S)) if max_m != -1 else len(S)
    new_token_info = {}

    pbar = tqdm(total=actual_additions, desc="Extending tokens", unit="token")
    for _ in range(actual_additions):
        if not S:
            break

        new_token_ngram = max(S, key=lambda k: S[k])
        sub_tokens = new_token_ngram.split(" ")
        new_token_str = new_token_ngram.replace(" ", "")

        if new_token_str in vocab_old:
            S.pop(new_token_ngram)
            pbar.update(1)
            continue
        if not all(sub_token in vocab_old for sub_token in sub_tokens):
            S.pop(new_token_ngram)
            pbar.update(1)
            continue

        new_token_id = vocab_size + len(new_token_info)
        sub_token_ids = [vocab_old[sub_token] for sub_token in sub_tokens]
        vocab_old[new_token_str] = new_token_id  # extend in-place

        new_token_info[new_token_str] = {
            "token_id": new_token_id,
            "token": new_token_str,
            "token_ngram": new_token_ngram,
            "sub_token_ids": sub_token_ids,
            "sub_tokens": sub_tokens,
        }

        # Decrease scores of overlapping n-grams
        for t_prime in Foverlaps.get(new_token_ngram, {}):
            if t_prime in S:
                S[t_prime] -= Foverlaps[new_token_ngram][t_prime] * len(new_token_ngram.split())

        S.pop(new_token_ngram)
        pbar.update(1)
    pbar.close()

    vocab_new = vocab_old  # all original tokens + new ones

    # All original byte-level merges remain valid (no tokens were removed)
    byte_merges_new = [
        (m[0], m[1]) for m in byte_merges
        if m[0] + m[1] in vocab_new and all(p in vocab_new and p != "" for p in m)
    ]

    new_ngrams = [info["token_ngram"] for info in new_token_info.values()]
    tok_merges = build_tok_merges(ngram_toks=new_ngrams, ngram_score=ngram_score)

    return vocab_new, byte_merges_new, tok_merges, new_token_info
