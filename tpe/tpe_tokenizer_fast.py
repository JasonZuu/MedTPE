import json
from typing import Dict, List, Union, Optional, Tuple
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding
import os
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
from collections import defaultdict
from itertools import chain


class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.priority: Optional[int] = None  # BPE priority (lower value = higher merge priority)


class TrieMerger:
    def __init__(self, merges: List[Tuple[str, ...]]):
        """
        merges: List where each element is an n-gram tuple (length >= 2), 
                e.g. ('Ġhe', 'llo'), ('Ġwor', 'ld'), ...
        The list order defines priority—lower index means higher priority.
        """
        self.root = TrieNode()
        # Build the Trie
        for prio, ng in enumerate(merges):
            node = self.root
            for tok in ng:
                node = node.children.setdefault(tok, TrieNode())
            # Store priority at terminal node
            node.priority = prio
        # Mark if any rules exist
        self.has_rules = bool(merges)

    def tokenize(self, tokens: List[str]) -> List[str]:
        """
        Perform batch matching and merging of tokens using the Trie.

        - If a subsequence in `tokens` matches a rule in `merges`, merge it into one new token.
          Choose the longest match with the highest priority (lowest priority value).
        - If a token does not match any rule, output it as is.

        Returns the merged token list.
        """
        # If there are no rules, return a copy of the original tokens
        if not self.has_rules:
            return list(tokens)

        out: List[str] = []
        i = 0
        N = len(tokens)
        while i < N:
            node = self.root
            best_prio = None
            best_end = None
            j = i
            # Try to match from position i onward
            while j < N and tokens[j] in node.children:
                node = node.children[tokens[j]]
                if node.priority is not None:
                    # Update best match: higher priority or first match
                    if best_prio is None or node.priority < best_prio:
                        best_prio = node.priority
                        best_end = j
                j += 1
            # If a match is found, merge tokens[i:best_end+1]
            if best_end is not None:
                merged_tok = ''.join(tokens[i:best_end+1])
                out.append(merged_tok)
                i = best_end + 1
            else:
                # No match, output single token
                out.append(tokens[i])
                i += 1
        return out


class TPETokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, original_tokenizer, tok_merges, new_tok_info: dict, max_workers: int = 8, **kwargs):
        """
        Args:
            original_tokenizer: Original tokenizer instance.
            tok_merges: List of tuples representing token merges.
            new_tok_info: Information about newly added tokens.
            max_workers: Number of worker threads for parallel operations.
        """
        super().__init__(
            tokenizer_object=original_tokenizer.backend_tokenizer,
            **original_tokenizer.init_kwargs  # Pass other necessary parameters
        )
        self.encode_special_tokens = original_tokenizer.backend_tokenizer.encode_special_tokens
        self.original_tokenizer = original_tokenizer
        self.new_tok_info = new_tok_info
        self.max_workers = max_workers
        # Preserve truncation and padding settings
        setattr(self.original_tokenizer, "truncation", self.original_tokenizer.backend_tokenizer.truncation)
        setattr(self.original_tokenizer, "padding", self.original_tokenizer.backend_tokenizer.padding)
        self.id_to_token = {v: k for k, v in self.vocab.items()}  # Reverse mapping from ID to token

        # Load token merges
        self.tok_merges = tok_merges
        self.tok_merger = TrieMerger(tok_merges)

    def _add_unk_token(self, unk_token: str = "<|unk_token|>") -> None:
        """
        Ensure the tokenizer has a valid unk_token / unk_token_id.

        Procedure:
        1. If 'unk_token' already exists in special_tokens_map, use it.
        2. Otherwise, generate a new unique token name based on `unk_token`,
           add it via `add_special_tokens`, and update the ID mapping.
        """
        # 1. If the original tokenizer already has <unk>
        if "unk_token" in self.special_tokens_map and self.special_tokens_map["unk_token"]:
            self.unk_token = self.special_tokens_map["unk_token"]
            self.unk_token_id = self.convert_tokens_to_ids(self.unk_token)
            return

        # 2. Create a new <UNK>, ensuring no name conflict with existing vocab
        if unk_token in self.vocab:  # If the token name already exists
            base = unk_token.rstrip("_")
            idx = 1
            while unk_token in self.vocab:  # Append suffix until unique
                unk_token = f"{base}_{idx}"
                idx += 1

        # Actually add it to the vocabulary (updates vocab, id_to_token, special_tokens_map)
        self.add_special_tokens({"unk_token": unk_token})

        # 3. Save to instance attributes for quick access
        self._unk_token = unk_token
        self._unk_token_id = self.vocab[unk_token]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Core tokenization method."""
        # 1. Tokenize with the original tokenizer
        orig_tokens = self.original_tokenizer.tokenize(text, **kwargs)

        # 2. Apply the token merger
        toks = self.tok_merger.tokenize(orig_tokens)
        return toks

    def set_pad_token(self, pad_token: str) -> None:
        """Set pad token."""
        self.original_tokenizer.pad_token = pad_token

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
    ) -> BatchEncoding:
        # 1. Validation & configuration
        if not isinstance(batch_text_or_text_pairs, (tuple, list)):
            raise TypeError(
                f"batch_text_or_text_pairs must be a list or tuple (got {type(batch_text_or_text_pairs)})"
            )
        # Set truncation and padding
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
        )
        # Control special token splitting
        self._tokenizer.encode_special_tokens = split_special_tokens

        # 2. Call original encode_batch to get fast encodings
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )

        # 3. Post-process each encoding: apply the token merger
        all_input_ids = []
        for encoding in encodings:
            # Get original token list (including special tokens)
            orig_tokens = encoding.tokens
            # Apply our merger
            merged_tokens = self.tok_merger.tokenize(orig_tokens)
            # Convert to IDs
            ids = self.convert_tokens_to_ids(merged_tokens)
            # Pad/truncate to max_length
            if max_length is not None:
                if len(ids) > max_length:
                    ids = ids[:max_length]
                elif len(ids) < max_length:
                    pad_len = max_length - len(ids)
                    ids = ids + [self.pad_token_id] * pad_len
            all_input_ids.append(ids)

        # 4. Build attention masks and other outputs
        encode_data = defaultdict(list)
        for ids in all_input_ids:
            encode_data["input_ids"].append(ids)
            if return_attention_mask:
                attn = [1] * min(len(ids), max_length or len(ids))
                if max_length and len(ids) < max_length:
                    attn += [0] * (max_length - len(ids))
                encode_data["attention_mask"].append(attn)
            if return_token_type_ids:
                encode_data["token_type_ids"].append([0] * len(ids))
            if return_special_tokens_mask:
                # Assume special tokens are in tokenizer.special_tokens_map
                mask = [
                    1 if tok in self.encode_special_tokens else 0
                    for tok in self.convert_ids_to_tokens(ids)
                ]
                encode_data["special_tokens_mask"].append(mask)
            if return_length:
                encode_data["length"].append(len(ids))

        # 5. Return BatchEncoding
        return BatchEncoding(encode_data, tensor_type=return_tensors)

    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self.id_to_token.get(index)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        idx = self.vocab.get(token)
        if idx is not None:
            return idx

        # 2) Byte-fallback: encode without adding special tokens
        ids = self.original_tokenizer.encode(token, add_special_tokens=False)
        return ids

    def convert_tokens_to_ids(self, tokens):
        """
        Single token ⇒ int
        Multiple tokens ⇒ flat List[int]
        """
        # ------ Fast path for single token ------
        if isinstance(tokens, str):
            idx = self.vocab.get(tokens)
            return idx if idx is not None else \
                self.original_tokenizer.encode(tokens, add_special_tokens=False)

        # ------ Multiple tokens: all in a C-level loop ------
        vocab_get = self.vocab.get                    # Localize attribute lookup
        encode = self.original_tokenizer.encode       # Same as above

        # 1) Generate a nested list/int generator (*list comprehension iterates at C layer*)
        nested = (
            ([idx] if (idx := vocab_get(tok)) is not None
             else encode(tok, add_special_tokens=False))
            for tok in tokens
        )

        # 2) chain.from_iterable is a pure C implementation, almost zero-cost flattening
        return list(chain.from_iterable(nested))

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        elif isinstance(ids, list):
            return [self._convert_id_to_token(i) for i in ids]

    def save_pretrained(self, save_directory, **kwargs):
        """Save tokenizer."""
        # Save the original tokenizer configuration
        self.original_tokenizer.save_pretrained(save_directory)

        with open(os.path.join(save_directory, "tok_merges.json"), "w", encoding="utf-8") as f:
            json.dump(self.tok_merges, f, ensure_ascii=False)
        with open(os.path.join(save_directory, "new_token_info.json"), "w", encoding="utf-8") as f:
            json.dump(self.new_tok_info, f, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load from a saved directory."""
        original_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path)

        with open(os.path.join(pretrained_model_name_or_path, "tok_merges.json"), "r", encoding="utf-8") as f:
            tok_merges = json.load(f)

        with open(os.path.join(pretrained_model_name_or_path, "new_token_info.json"), "r", encoding="utf-8") as f:
            new_tok_info = json.load(f)

        return cls(
            original_tokenizer=original_tokenizer,
            tok_merges=tok_merges,
            new_tok_info=new_tok_info,
            **kwargs
        )

    def __getattr__(self, name):
        # Directly access instance dictionary to avoid triggering __getattr__
        if name == "tokenize":
            return self._tokenize
        original_tokenizer = self.__dict__.get("original_tokenizer")
        if original_tokenizer is not None:
            return getattr(original_tokenizer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
