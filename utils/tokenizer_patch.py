import json


def get_merge_table(tokenizer_config_fpath: str) -> dict:
    """Load and parse the merge table from tokenizer config file
    
    Args:
        tokenizer_config_fpath: Path to tokenizer.json config file
    
    Returns:
        Dictionary mapping merged tokens to their constituent parts
    """
    # Load merge rules
    with open(tokenizer_config_fpath, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
        merges = tokenizer_config["model"]["merges"]
    
    # Construct merge_table (parent token → child token pairs)
    merge_table = {}
    for merge in merges:
        t1, t2 = merge.split(" ")
        merged = t1 + t2
        merge_table[merged] = (t1, t2)

    return merge_table


def decomp(t: str, Vocab_new: set, merge_table: dict) -> list:
    """Recursively decompose token until all sub-tokens exist in Vocab_new
    
    Args:
        t: Token to decompose
        Vocab_new: Set of tokens in the new vocabulary
        merge_table: Merge rules dictionary
    
    Returns:
        List of decomposed sub-tokens
    """
    if t in Vocab_new:
        return [t]
    if t in merge_table:
        t1, t2 = merge_table[t]
        return decomp(t1, Vocab_new, merge_table) + decomp(t2, Vocab_new, merge_table)
    else:
        # If cannot decompose and t not in Vocab_new, keep as-is
        # (requires Vocab_new to cover all possible subwords)
        return [t]


def tokenization_patching_fn(text: str, tokenizer, merge_table: dict, 
                             Vocab_new: dict, max_n: int) -> list:
    """Adapt original tokenization results to new vocabulary Vocab_new
    
    Args:
        text: Input text to tokenize
        tokenizer: Original tokenizer instance
        merge_table: Merge rules dictionary
        Vocab_new: Set of tokens in the new vocabulary
        max_n: Maximum number of tokens to consider for matching (the same with n-gram for tokenizer_modify)
    
    Returns:
        List of tokens compatible with the new vocabulary
    """
    # Step 1: Decompose original tokens into base units (tokens_orig)
    tokens_from_tokenizer = tokenizer.encode(text).tokens
    tokens_orig = []
    for t in tokens_from_tokenizer:
        tokens_orig.extend(decomp(t, Vocab_new, merge_table))

    # Step 2: Recombine using maximum prefix matching
    tokens_new = []
    i = 0
    while i < len(tokens_orig):
        matched = False
        for k in range(min(max_n, len(tokens_orig) - i), 0, -1):
            candidate = " ".join(tokens_orig[i:i+k])
            if candidate in Vocab_new:
                tokens_new.append(candidate)
                i += k
                matched = True
                break
        if not matched:
            # If no match found, add current subword as-is
            tokens_new.append(tokens_orig[i])
            i += 1
    return tokens_new


if __name__ == "__main__":
    from tokenizers import Tokenizer

    # Initialize tokenizer and test data
    tokenizer_config_fpath = "data/hf_models/Qwen--Qwen2.5-7B-Instruct/tokenizer.json"
    original_tokenizer = Tokenizer.from_file(tokenizer_config_fpath)
    merge_table = get_merge_table(tokenizer_config_fpath)
    
    # Create a more realistic test vocabulary
    new_vocab = {
        "hello": 1001,
        "world": 1002,
        "hell": 1003,
        "o": 1004,
        "wor": 1005,
        "ld": 1006,
        ".": 1007,
        "hello world": 1008  # Added multi-word token for testing
    }
    
    # Test cases
    test_cases = [
        "hello world.",  # Should match "hello world" if in vocab
        "hello there",   # Should fall back to "hello" + "there"
        "helloworld",    # Should decompose using merge rules
        "worried"        # Should handle partial matches
    ]
    
    # Run tests
    for text in test_cases:
        print(f"\nInput text: '{text}'")
        tokens = tokenization_patching_fn(original_tokenizer, merge_table, new_vocab, text)
        print(f"Tokenized: {tokens}")
        print(f"Token IDs: {[new_vocab.get(t, 'OOV') for t in tokens]}")