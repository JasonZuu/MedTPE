import torch
from torch import nn
from typing import List
from transformers import AutoModelForCausalLM
import torch.nn.functional as F


class SplitEmbedding(nn.Module):
    """
    仅为 tune_ids 对应的行保留可训练权重，其余行冻结为 buffer。
    """
    def __init__(self, base_emb: nn.Embedding, tune_ids: List[int], shared_train_weight: nn.Parameter = None):
        super().__init__()
        self.register_buffer("base_weight", base_emb.weight.data)  # (V, d), register_buffer to put it in state_dict
        tune_ids_tensor = torch.as_tensor(tune_ids, dtype=torch.int32)
        self.register_buffer("tune_ids", tune_ids_tensor) # (n,) 
        self.id2row = {tid.item(): i for i, tid in enumerate(self.tune_ids)}

        if shared_train_weight is None:
            self.train_weight = nn.Parameter(self.base_weight[self.tune_ids].clone())  # (n, d)
        else:
            self.train_weight = shared_train_weight

        self.embedding_dim = base_emb.embedding_dim
        self.padding_idx = base_emb.padding_idx

        # freeze base_weight and tune_ids
        self.base_weight.requires_grad = False
        self.tune_ids.requires_grad = False

    def forward(self, input_ids):
        out = F.embedding(input_ids, self.base_weight, padding_idx=self.padding_idx)
        mask = torch.isin(input_ids, self.tune_ids)
        if mask.any():
            rows = self.train_weight[torch.tensor([self.id2row[int(t)] for t in input_ids[mask]],
                                                  device=self.train_weight.device)]
            out[mask] = rows
        return out

    @property
    def weight(self):
        # 动态拼一份完整权重  (V, d)  —— 仅在 save / tie 时用到
        full = self.base_weight.clone()
        full[self.tune_ids] = self.train_weight
        return full

    @weight.setter
    def weight(self, new_w):
        # tie_weights() 会走到这里，把指针换成输入层同一块
        with torch.no_grad():
            self.base_weight.copy_(new_w)                 # 冻结行
            self.train_weight.copy_(new_w[self.tune_ids]) # 可训练行


class SplitLinear(nn.Module):
    """
    仅对 tune_ids 列使用可训练权重；其余列固定。
    shape: in=(B,L,d) → out=(B,L,V)
    """
    def __init__(self, base_lin: nn.Linear, tune_ids: List[int], shared_train_weight: nn.Parameter = None):
        super().__init__()
        self.register_buffer("base_weight", base_lin.weight.data)  # (V, d)
        tune_ids_tensor = torch.as_tensor(tune_ids, dtype=torch.int32)
        self.register_buffer("tune_ids", tune_ids_tensor)  # (n,)

        if shared_train_weight is None:
            self.train_weight = nn.Parameter(self.base_weight[self.tune_ids].clone())  # (n, d)
        else:
            self.train_weight = shared_train_weight            # 与输入层共用

        if base_lin.bias is None:
            self.bias = None
        else:
            self.register_parameter("bias", nn.Parameter(base_lin.bias.data.clone()))
        
        # freeze base_weight and tune_ids
        self.base_weight.requires_grad = False
        self.tune_ids.requires_grad = False

    def forward(self, hidden_states):                          # (B,L,d)
        logits = hidden_states.matmul(self.base_weight.t())    # (B,L,V) 冻结部分
        # 仅更新 tune_ids 这一小片列
        logits_tune = hidden_states.matmul(self.train_weight.t())  # (B,L,n)
        logits[..., self.tune_ids] = logits_tune
        if self.bias is not None:
            logits = logits + self.bias
        return logits



def set_trainable_embeddings(model: AutoModelForCausalLM,
                             new_tok_ids: List[int],
                             freeze_params: bool=False) -> None:
    # 0) 记录最初是否 tie
    was_tied = getattr(model.config, "tie_word_embeddings", False)

    # 1) 全冻结
    if freeze_params:
        for p in model.parameters():
            p.requires_grad = False

    # 2) 构造 SplitEmbedding（输入层）
    old_in = model.get_input_embeddings()
    split_in = SplitEmbedding(old_in, new_tok_ids)          # 可训练权重在 split_in.train_weight
    split_in.to(device=model.device, dtype=model.dtype)  # 确保在同一设备上
    model.set_input_embeddings(split_in)

    # 3) 构造输出层 —— 形状可能是 Embedding 或 Linear
    old_out = model.get_output_embeddings()
    if isinstance(old_out, nn.Embedding):
        split_out = SplitEmbedding(old_out, new_tok_ids,    # 共享权重！
                                   shared_train_weight=split_in.train_weight)
    else:
        split_out = SplitLinear(old_out, new_tok_ids,
                                 shared_train_weight=split_in.train_weight)
    split_out.to(device=model.device, dtype=model.dtype)  # 确保在同一设备上
    model.set_output_embeddings(split_out)

    # 4) 如果模型原本是 tie 的，保留 flag 并重新调用 tie_weights()
    if was_tied:
        model.config.tie_word_embeddings = True
        model.tie_weights()          # 让 HF 保存/加载过程继续认为是 tie

    # 5) 打印可训练参数
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Embedding] trainable params: {trainable}/{total} = {100*trainable/total:.4f}%")

    # 6) get the trainable layer name
    trainable_layer_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layer_names.append(name)
    return model


def restore_embeddings(
    model: AutoModelForCausalLM,
    new_tok_ids: List[int]
) -> None:
    """
    将 model 中的 SplitEmbedding/SplitLinear 层，恢复为原生的
    nn.Embedding + nn.Linear（或 nn.Embedding）结构，数值保持训练后的权重。

    1) 拼 full_input_weight = base_weight.clone(); full_input_weight[tune_ids] = train_weight
    2) 用 full_input_weight 构造新的 nn.Embedding，并 set_input_embeddings
    3) 同理构造新的 lm_head（nn.Linear 或 nn.Embedding），并 set_output_embeddings
    4) 如果原本 tie，则重新 tie_weights()
    """
    # 0) 记录是否 tie
    was_tied = getattr(model.config, "tie_word_embeddings", False)

    # 1) 处理输入 embedding
    split_in = model.get_input_embeddings()
    # 从 split_in 拿 base_weight, train_weight, tune_ids
    base_w = split_in.base_weight            # buffer
    t_ids  = split_in.tune_ids               # buffer
    train_w = split_in.train_weight          # Parameter

    # 拼 full
    full_in_w = base_w.clone()
    full_in_w[t_ids] = train_w

    # 新建标准 embedding
    new_in = nn.Embedding(
        num_embeddings=full_in_w.size(0),
        embedding_dim=full_in_w.size(1),
        padding_idx=split_in.padding_idx,
    )
    new_in.weight.data.copy_(full_in_w)
    model.set_input_embeddings(new_in)

    # 2) 处理输出层
    split_out = model.get_output_embeddings()
    if isinstance(split_out, SplitEmbedding):
        # 输出层是 Embedding
        base_w2 = split_out.base_weight
        t_ids2  = split_out.tune_ids
        train_w2 = split_out.train_weight
        full_out_w = base_w2.clone()
        full_out_w[t_ids2] = train_w2

        new_out = nn.Embedding(
            num_embeddings=full_out_w.size(0),
            embedding_dim=full_out_w.size(1),
            padding_idx=split_out.padding_idx,
        )
        new_out.weight.data.copy_(full_out_w)
    else:
        # 输出层是 Linear
        base_w2 = split_out.base_weight    # shape (V, d)
        t_ids2  = split_out.tune_ids
        train_w2 = split_out.train_weight
        full_out_w = base_w2.clone()
        full_out_w[t_ids2] = train_w2
        # bias
        b = split_out.bias.data if split_out.bias is not None else None

        # new linear: in_features=d, out_features=V
        new_out = nn.Linear(
            in_features=full_out_w.size(1),
            out_features=full_out_w.size(0),
            bias=(b is not None),
        )
        new_out.weight.data.copy_(full_out_w)
        if b is not None:
            new_out.bias.data.copy_(b)

    model.set_output_embeddings(new_out)

    # 3) 恢复 tie flag
    if was_tied:
        model.config.tie_word_embeddings = True
        model.tie_weights()

    return model