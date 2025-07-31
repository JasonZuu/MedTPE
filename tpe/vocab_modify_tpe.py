from collections import defaultdict, Counter, OrderedDict
import heapq
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
    if len(subtokens) <= 1:
        return [[]]

    all_merge_paths = []

    # try merging each adjacent pair of subtokens
    for i in range(len(subtokens) - 1):
        left = subtokens[i]
        right = subtokens[i + 1]
        merged = left + right

        # construct the next subtokens list
        next_tokens = (
            subtokens[:i] +         # the unchanged prefix
            [merged] +              # newly merged token
        subtokens[i + 2:]           # the unchanged suffix
        )

        # recursively get all merge paths for the next tokens
        for suffix_path in get_all_merge_paths(next_tokens):
            # put together the current merge with the suffix path
            all_merge_paths.append([(left, right)] + suffix_path)

    return all_merge_paths


def build_tok_merges(ngram_toks, ngram_score):
    """
    Automatically build a list of token-level merges based on given token-level n-gram frequencies.
    Supports arbitrary n-gram lengths, but always outputs binary-merge steps and removes duplicate merge pairs.

    Parameters:
    - ngram_score: dict, where each key is a tuple (token1, token2, ...), and each value is its corresponding score.
    - ngram_toks: iterable of candidate n-gram tuples (length >= 2), which will be sorted by their frequencies.

    Returns:
    - merges: list of tuple(tuple, tuple), where each entry represents one binary merge operation, ordered by frequency priority with no duplicates:
        ((t1,), (t2,)) means merging t1 and t2 first,
        ((t1, t2), (t3,)) means merging the result (t1, t2) with t3 next, and so on.

    Notes:
    - There is no need to specify num_merges; all passed-in ngram_toks will be considered.
    - The output merges contain only length-2 merge operations. If the same operation appears in multiple n-grams, only the first occurrence is kept.
    """
    # 1. Filter and sort candidate n-grams
    candidates = [(tuple(ngram.split()), ngram_score.get(ngram, 0))
                  for ngram in ngram_toks]
    # Sort by descending score, and lexicographically among ties
    candidates.sort(key=lambda x: (-x[1], x[0]))

    # 2. Generate binary merge operations list and remove duplicates
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
            # Update prefix to the merged result
            prev = prev + curr
    return merges


def repair_vocab_new(
    vocab_new: Dict[str, int],
    vocab_old: Dict[str, int]
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    Fix two types of issues in vocab_new:
      1) Duplicate IDs: keep only the first token for each ID and record any removed duplicates.
      2) Missing IDs (holes): fill in missing IDs using vocab_old.

    Args:
        vocab_new:  The modified vocabulary mapping token -> id, which may have "holes" or duplicate IDs.
        vocab_old:  The original complete vocabulary mapping token -> id.

    Returns:
        repaired:   The repaired vocab_new mapping token -> id.
        duplicates: A dictionary recording duplicate IDs, formatted as {id: [removed_tok1, removed_tok2, ...], ...}.
    """
    # 1. Build reverse mapping from ID to a list of tokens in vocab_new
    id_to_tokens = defaultdict(list)
    for tok, tid in vocab_new.items():
        id_to_tokens[tid].append(tok)

    repaired = {}
    duplicates: Dict[int, List[str]] = {}

    # 2. For each ID, keep the first encountered token, and record the rest as duplicates
    for tid, toks in id_to_tokens.items():
        first_tok = toks[0]
        repaired[first_tok] = tid
        if len(toks) > 1:
            duplicates[tid] = toks[1:]

    # 3. Find missing IDs (holes) within [0..max_old_id] and fill them using vocab_old
    if vocab_old:
        # Build reverse mapping from ID to token in the old vocabulary
        id_to_old = {tid: tok for tok, tid in vocab_old.items()}
        max_old_id = max(id_to_old.keys())

        used_ids = set(repaired.values())
        holes = [i for i in range(max_old_id + 1) if i not in used_ids]

        for hid in holes:
            old_tok = id_to_old.get(hid)
            if old_tok and old_tok not in repaired:
                repaired[old_tok] = hid

    return repaired, duplicates


def tpe_vocabulary_modify_fn(tokenizer: PreTrainedTokenizerFast, 
                            dataset_dict: DatasetDict,  # Changed to HF Dataset
                            max_m: int, 
                            max_n: int,
                            byte_merges: list,
                            dataset_split: str="train",
                            min_freq: int=2) -> Tuple[Dict[str, int], List[Tuple[str, str]], List[Tuple[str, str]], Dict[int, Dict[str, str]]]:
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
    # Initialize old and new vocabularies
    _vocab_old = tokenizer.vocab
    vocab_old = deepcopy(_vocab_old)  # Copy the original vocabulary
    vocab_new = {}
    
    # 1. Tokenize the corpus from HF Dataset
    dataset = dataset_dict[dataset_split]  # Extract text column
    dataset = dataset.map(dataset_to_Dtok_batch_mapping_fn,
                          fn_kwargs={"tokenizer": tokenizer},
                          batched=True,
                          desc="Tokenizing dataset",
                          batch_size=1000,
                          num_proc=8)  # Use multiple processes for efficiency
    Dtok = [item["token"] for item in dataset]  # Assuming 'data' is the text column
    
    # 2. Extract n-grams and calculate their score
    ngram_tokens = prepare_n_tokens(Dtok, max_n)
    Fn_tok = cnt_freqs(Dtok, ngram_tokens, max_n)
    
    # Calculate overlap relationships
    Foverlaps = cnt_overlaps(ngram_tokens, Fn_tok, max_n)
    
    # Calculate initial scores S = frequency × length
    ngram_score = {t: Fn_tok[t] * len(t.split()) for t in ngram_tokens if Fn_tok[t] >= min_freq} # remove low freq n-grams
    S = OrderedDict(sorted(ngram_score.items(), key=lambda x: x[1], reverse=True))  # Sort by score descending
    ngram_tokens = list(S.keys())

    # 4. Set token list that won't be replaced
    # Special tokens
    special_tokens = set()
    for key, token in tokenizer.special_tokens_map.items():
        if token is None:
            continue
        if key == "additional_special_tokens":
            special_tokens.update(set(token))
        else:
            special_tokens.add(token)
        
    # Reserve tokens on the merge paths for new toks
    merges_dict = {f"{parts[0]}{parts[1]}": parts for parts in byte_merges}
    def _decompose_token(t: str, vocab) -> List[str]:
        """iteratively decompose a token into its sub-tokens"""   
        if t in merges_dict:
            t1, t2 = merges_dict[t]
            return [t] + _decompose_token(t1, vocab) + _decompose_token(t2, vocab)
        elif t in vocab:
            return [t]
        else:   
            raise ValueError(f"Token {t} not in vocab")
        
    subtokens = set()
    pbar = tqdm(total=len(ngram_tokens), desc="Decomposing tokens", unit="token")
    for tok in ngram_tokens:
        for sub_tok in tok.split():
            if sub_tok not in vocab_old: # make sure all sub-tokens are in vocab_old
                S.pop(tok)
                break
            subtokens.update(_decompose_token(sub_tok, vocab_old))
        pbar.update(1)
    pbar.close()

    # 3. create the new vocab by replacing old tokens with new tokens
    # Build min-heap (sorted by frequency ascending) of old tokens
    old_token_freq = count_old_token_frequencies(
        Dtok, vocab_old, disable_progress=False
    )
    heap = []

    for token_str, token_id in vocab_old.items():
        # Skip special tokens and tokens with length 1
        if token_str in special_tokens or len(token_str) == 1:
            continue
        elif token_str in subtokens:
            continue
        freq = old_token_freq.get(token_str, 0)
        heapq.heappush(heap, (freq, token_str, token_id))
    
    # Determine actual number of replacements
    if max_m == -1:
        actual_replacements = len(S)
    else:
        actual_replacements = min(max_m, len(S))
    new_token_info = {}
    
    # Replace tokens and collect tok_merges
    pbar = tqdm(total=actual_replacements, desc="Replacing tokens", unit="token")
    for _ in range(actual_replacements):
        if not heap or not S:
            break
        
        new_token_ngram = max(S, key=lambda k: S[k])
        sub_tokens = new_token_ngram.split()
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

    # Merge remaining vocabulary
    vocab_new.update(vocab_old)

    # Repair any ID‐holes with old toks
    vocab_new = repair_vocab_new(vocab_new, _vocab_old)
    
    new_token_info = {new_tok: info for new_tok, info in new_token_info.items() if new_tok in vocab_new}

    # 6. remove invalid merges for the new vocab
    byte_merges_new = []
    for merge in byte_merges:
        if merge[0]+merge[1] in vocab_new and all(part in vocab_new and part != "" for part in merge):
            byte_merges_new.append((merge[0], merge[1]))

    # 7. build tok_merges
    new_ngram = [info["token_ngram"] for info in new_token_info.values()]
    tok_merges = build_tok_merges(ngram_toks=new_ngram, ngram_score=ngram_score)

    return vocab_new, byte_merges_new, tok_merges, new_token_info

