import json
from typing import Dict, List, Union
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding
import os
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
from collections import defaultdict
from typing import Optional, Tuple
from itertools import chain
import numpy as np


class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.priority: Optional[int] = None  # BPE 优先级（越小越先合并）

class TrieMerger:
    def __init__(self, merges: List[Tuple[str, ...]]):
        """
        merges: 列表中每个元素是一个 n-gram tuple（长度>=2），例如 ('Ġhe','llo'), ('Ġwor','ld'), ...
        列表顺序即优先级——索引越小优先级越高。
        """
        self.root = TrieNode()
        # 构建 Trie
        for prio, ng in enumerate(merges):
            node = self.root
            for tok in ng:
                node = node.children.setdefault(tok, TrieNode())
            # 终端节点记录优先级
            node.priority = prio
        # 标记是否有任何规则
        self.has_rules = bool(merges)

    def tokenize(self, tokens: List[str]) -> List[str]:
        """
        使用 Trie 批量匹配并合并 tokens 中的 n-gram。

        - 若 tokens 中的子序列匹配到 merges 里的规则，则合并为一个新 token，
          选择优先级最高（最低 priority）的最长匹配。
        - 若某个 tok 不在任何规则中，直接原样输出。

        返回合并后的 token 列表。
        """
        # 若无任何规则，直接返回原始 tokens 的拷贝
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
            # 从位置 i 尝试向后匹配
            while j < N and tokens[j] in node.children:
                node = node.children[tokens[j]]
                if node.priority is not None:
                    # 更新最优匹配：优先级更高或首次匹配
                    if best_prio is None or node.priority < best_prio:
                        best_prio = node.priority
                        best_end = j
                j += 1
            # 如果找到匹配，则合并 tokens[i:best_end+1]
            if best_end is not None:
                merged_tok = ''.join(tokens[i:best_end+1])
                out.append(merged_tok)
                i = best_end + 1
            else:
                # 无匹配则输出单个 token
                out.append(tokens[i])
                i += 1
        return out


class TPETokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, original_tokenizer,
                 tok_merges,
                  new_tok_info: dict,
                  max_workers: int = 8, **kwargs):
        """
        Args:
            original_tokenizer: Original tokenizer instance
            tok_merges: List of tuples representing token merges
        """

        super().__init__(
            tokenizer_object=original_tokenizer.backend_tokenizer,
            **original_tokenizer.init_kwargs  # 传递其他必要参数,
        )
        self.encode_special_tokens = original_tokenizer.backend_tokenizer.encode_special_tokens
        self.original_tokenizer = original_tokenizer
        self.new_tok_info = new_tok_info
        self.max_workers = max_workers
        setattr(self.original_tokenizer, "truncation", self.original_tokenizer.backend_tokenizer.truncation)
        setattr(self.original_tokenizer, "padding", self.original_tokenizer.backend_tokenizer.padding)
        self.id_to_token = {v: k for k, v in self.vocab.items()}  # Reverse mapping

        # load tok_merges
        self.tok_merges = tok_merges
        # self._valid_tok_merges()  # Check validity of tok_merges
        self.tok_merger = TrieMerger(tok_merges)


    def _add_unk_token(self, unk_token: str = "<|unk_token|>") -> None:
        """确保 tokenizer 拥有有效的 unk_token / unk_token_id。

        规律：
        1. 若 special_tokens_map 已经有 'unk_token'，直接取来用；
        2. 否则以 `candidate` 为模板，新建一个不会与现有 vocab 冲突的 token，
        通过 `add_special_tokens` 注入，并同步 id 映射。
        """
        # ① 老 tokenizer本来就带 <unk>
        if "unk_token" in self.special_tokens_map and self.special_tokens_map["unk_token"]:
            self.unk_token    = self.special_tokens_map["unk_token"]
            self.unk_token_id = self.convert_tokens_to_ids(self.unk_token)
            return

        # ② 为它造一个 <UNK>，确保不跟现有 token 撞名
        if unk_token in self.vocab:          # 已存在同名 token
            base = unk_token.rstrip("_")
            idx  = 1
            while unk_token in self.vocab:   # 递增后缀直到独一无二
                unk_token = f"{base}_{idx}"
                idx += 1

        # 真正把它塞进词表（会同时更新 vocab / id_to_token / special_tokens_map）
        self.add_special_tokens({"unk_token": unk_token})

        # ③ 保存到实例属性，后续快速访问
        self._unk_token    = unk_token
        self._unk_token_id = self.vocab[unk_token]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """核心分词方法"""
        # 1. 用原始tokenizer分词
        orig_tokens = self.original_tokenizer.tokenize(text, **kwargs)

        # 2. tok_merger分词
        toks = self.tok_merger.tokenize(orig_tokens)
        return toks

    def set_pad_token(self, pad_token: str) -> None:
        """设置pad token"""
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
        # 1. validation & config
        if not isinstance(batch_text_or_text_pairs, (tuple, list)):
            raise TypeError(f"batch_text_or_text_pairs has to be a list or a tuple (got {type(batch_text_or_text_pairs)})")
        # set truncation/padding
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
        )
        # control special token splitting
        self._tokenizer.encode_special_tokens = split_special_tokens

        # 2. call original encode_batch to get fast encodings
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )

        # 3. post-process each encoding: apply tok_merger
        all_input_ids = []
        for encoding in encodings:
            # get original token list (including special tokens)
            orig_tokens = encoding.tokens
            # apply our merger
            merged_tokens = self.tok_merger.tokenize(orig_tokens)
            # convert to ids
            ids = self.convert_tokens_to_ids(merged_tokens)

            # pad/truncate
            if max_length is not None:
                if len(ids) > max_length:
                    ids = ids[:max_length]
                elif len(ids) < max_length:
                    pad_len = max_length - len(ids)
                    ids = ids + [self.pad_token_id] * pad_len
            all_input_ids.append(ids)

        # 4. build attention masks and other outputs
        encode_data = defaultdict(list)
        for ids in all_input_ids:
            encode_data["input_ids"].append(ids)
            if return_attention_mask:
                attn = [1]*min(len(ids), max_length or len(ids))
                if max_length and len(ids) < max_length:
                    attn += [0]*(max_length - len(ids))
                encode_data["attention_mask"].append(attn)
            if return_token_type_ids:
                encode_data["token_type_ids"].append([0]*len(ids))
            if return_special_tokens_mask:
                # assume special tokens in tokenizer.special_tokens_map
                mask = [1 if tok in self.all_special_tokens else 0 for tok in
                        self.convert_ids_to_tokens(ids)]
                encode_data["special_tokens_mask"].append(mask)
            if return_length:
                encode_data["length"].append(len(ids))

        # 5. return BatchEncoding
        return BatchEncoding(encode_data, tensor_type=return_tensors)

    def _convert_id_to_token(self, index: int) -> str:
        """将ID转换为token"""
        return self.id_to_token.get(index)

    def _convert_token_to_id(self, token: str) -> int:
        """将token转换为ID"""
        idx = self.vocab.get(token)
        if idx is not None:
            return idx

        # 2) byte-fallback：encode 不加 special tokens
        ids = self.original_tokenizer.encode(token, add_special_tokens=False)
        return ids

    def convert_tokens_to_ids(self, tokens):
        """
        单 token ⇒ int
        多 token ⇒ 扁平 List[int]
        """
        # ------ 单 token 最快路径 ------
        if isinstance(tokens, str):
            idx = self.vocab.get(tokens)
            return idx if idx is not None else \
                self.original_tokenizer.encode(tokens, add_special_tokens=False)

        # ------ 多 token：全部在 C 循环里跑 ------
        vocab_get = self.vocab.get                    # 本地化属性查找
        encode     = self.original_tokenizer.encode   # 同上

        # 1) 生成一个「嵌套 list/int」的生成器（*推导式在 C 层迭代*）
        nested = (
            ([idx] if (idx := vocab_get(tok)) is not None
            else encode(tok, add_special_tokens=False))
            for tok in tokens
        )

        # 2) chain.from_iterable 纯 C 实现，几乎零开销扁平化
        return list(chain.from_iterable(nested))

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        elif isinstance(ids, list):
            return [self._convert_id_to_token(i) for i in ids]

    def save_pretrained(self, save_directory, **kwargs):
        """保存tokenizer"""
        # 保存原始tokenizer配置
        self.original_tokenizer.save_pretrained(save_directory)

        with open(os.path.join(save_directory, "tok_merges.json"), "w", encoding="utf-8") as f:
            json.dump(self.tok_merges, f, ensure_ascii=False)
        with open(os.path.join(save_directory, "new_token_info.json"), "w", encoding="utf-8") as f:
            json.dump(self.new_tok_info, f, ensure_ascii=False)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """从保存的目录加载"""
        original_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path)

        with open(os.path.join(pretrained_model_name_or_path, "tok_merges.json"), "r", encoding="utf-8") as f:
            tok_merges = json.load(f)

        with open(os.path.join(pretrained_model_name_or_path, "new_token_info.json"), "r", encoding="utf-8") as f:
            new_tok_info = json.load(f)

        return cls(original_tokenizer=original_tokenizer,
                   tok_merges=tok_merges,
                   new_tok_info=new_tok_info,
                   **kwargs)

    # 代理其他必要方法
    def __getattr__(self, name):
        # 直接访问实例字典，避免触发 __getattr__
        if name == "tokenize":
            return self._tokenize
        original_tokenizer = self.__dict__.get("original_tokenizer")
        if original_tokenizer is not None:
            return getattr(original_tokenizer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")



if __name__ == "__main__":
    print("TPETokenizerFast is intended to be loaded with from_pretrained().")
