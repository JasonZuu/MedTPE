import torch
from torch import nn
from typing import List
from transformers import AutoModelForCausalLM
import torch.nn.functional as F


class SplitEmbedding(nn.Module):
    """
    Keep only the rows corresponding to tune_ids trainable; freeze all other rows as buffers.
    """
    def __init__(self, base_emb: nn.Embedding, tune_ids: List[int], shared_train_weight: nn.Parameter = None):
        super().__init__()
        # Store the full embedding weight as a non-trainable buffer
        self.register_buffer("base_weight", base_emb.weight.data)  # shape (V, d)
        tune_ids_tensor = torch.as_tensor(tune_ids, dtype=torch.int32)
        self.register_buffer("tune_ids", tune_ids_tensor)         # shape (n,)
        # Map from token ID to index in the trainable subset
        self.id2row = {tid.item(): i for i, tid in enumerate(self.tune_ids)}

        # Create trainable weights for the chosen IDs (or reuse a shared parameter)
        if shared_train_weight is None:
            self.train_weight = nn.Parameter(self.base_weight[self.tune_ids].clone())  # shape (n, d)
        else:
            self.train_weight = shared_train_weight

        self.embedding_dim = base_emb.embedding_dim
        self.padding_idx = base_emb.padding_idx

        # Freeze the buffers
        self.base_weight.requires_grad = False
        self.tune_ids.requires_grad = False

    def forward(self, input_ids):
        # Look up all embeddings from the frozen base
        out = F.embedding(input_ids, self.base_weight, padding_idx=self.padding_idx)
        # Replace the subset corresponding to tune_ids with trainable rows
        mask = torch.isin(input_ids, self.tune_ids)
        if mask.any():
            rows = self.train_weight[
                torch.tensor([self.id2row[int(t)] for t in input_ids[mask]],
                             device=self.train_weight.device)
            ]
            out[mask] = rows
        return out

    @property
    def weight(self):
        # Dynamically assemble a full weight matrix (V, d) for saving or tying
        full = self.base_weight.clone()
        full[self.tune_ids] = self.train_weight
        return full

    @weight.setter
    def weight(self, new_w):
        # Used by tie_weights(): copy into both frozen and trainable rows
        with torch.no_grad():
            self.base_weight.copy_(new_w)                      # frozen rows
            self.train_weight.copy_(new_w[self.tune_ids])      # trainable rows


class SplitLinear(nn.Module):
    """
    Only allow trainable weights for the columns in tune_ids; keep all other columns fixed.
    Input shape: (B, L, d) → Output shape: (B, L, V)
    """
    def __init__(self, base_lin: nn.Linear, tune_ids: List[int], shared_train_weight: nn.Parameter = None):
        super().__init__()
        # Store the full linear weight as a non-trainable buffer
        self.register_buffer("base_weight", base_lin.weight.data)  # shape (V, d)
        tune_ids_tensor = torch.as_tensor(tune_ids, dtype=torch.int32)
        self.register_buffer("tune_ids", tune_ids_tensor)         # shape (n,)

        # Create trainable subset (or reuse shared parameter)
        if shared_train_weight is None:
            self.train_weight = nn.Parameter(self.base_weight[self.tune_ids].clone())  # shape (n, d)
        else:
            self.train_weight = shared_train_weight

        # Handle bias if present
        if base_lin.bias is None:
            self.bias = None
        else:
            self.register_parameter("bias", nn.Parameter(base_lin.bias.data.clone()))

        # Freeze buffers
        self.base_weight.requires_grad = False
        self.tune_ids.requires_grad = False

    def forward(self, hidden_states):                          # shape (B, L, d)
        # Compute logits for frozen columns
        logits = hidden_states.matmul(self.base_weight.t())    # shape (B, L, V)
        # Compute logits for trainable columns only
        logits_tune = hidden_states.matmul(self.train_weight.t())  # shape (B, L, n)
        # Replace those columns in the full logits
        logits[..., self.tune_ids] = logits_tune
        if self.bias is not None:
            logits = logits + self.bias
        return logits


def set_trainable_embeddings(
    model: AutoModelForCausalLM,
    new_tok_ids: List[int],
    freeze_params: bool = False
) -> AutoModelForCausalLM:
    """
    Replace the input and output embeddings of the model so that only new_tok_ids remain trainable.
    Optionally freeze all other parameters.
    """
    # 0) Save whether embeddings were originally tied
    was_tied = getattr(model.config, "tie_word_embeddings", False)

    # 1) Optionally freeze all model parameters
    if freeze_params:
        for p in model.parameters():
            p.requires_grad = False

    # 2) Build SplitEmbedding for the input layer
    old_in = model.get_input_embeddings()
    split_in = SplitEmbedding(old_in, new_tok_ids)
    split_in.to(device=model.device, dtype=model.dtype)
    model.set_input_embeddings(split_in)

    # 3) Build SplitEmbedding or SplitLinear for the output layer
    old_out = model.get_output_embeddings()
    if isinstance(old_out, nn.Embedding):
        split_out = SplitEmbedding(
            old_out,
            new_tok_ids,
            shared_train_weight=split_in.train_weight
        )
    else:
        split_out = SplitLinear(
            old_out,
            new_tok_ids,
            shared_train_weight=split_in.train_weight
        )
    split_out.to(device=model.device, dtype=model.dtype)
    model.set_output_embeddings(split_out)

    # 4) If embeddings were tied before, restore that behavior
    if was_tied:
        model.config.tie_word_embeddings = True
        model.tie_weights()

    # 5) Print counts of trainable parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Embedding] trainable params: {trainable}/{total} = {100*trainable/total:.4f}%")

    # 6) (Optional) Return the modified model
    return model


def restore_embeddings(
    model: AutoModelForCausalLM,
    new_tok_ids: List[int]
) -> AutoModelForCausalLM:
    """
    Restore the original nn.Embedding + nn.Linear (or nn.Embedding) layers after tuning,
    keeping the trained weights.

    Steps:
      1) Assemble full_input_weight = base_weight.clone(); full_input_weight[tune_ids] = train_weight
      2) Create a new nn.Embedding with that weight and set_input_embeddings
      3) Do the same for the output layer (Linear or Embedding) and set_output_embeddings
      4) If embeddings were originally tied, call tie_weights() again
    """
    # 0) Save original tie flag
    was_tied = getattr(model.config, "tie_word_embeddings", False)

    # 1) Handle input embedding
    split_in = model.get_input_embeddings()
    base_w = split_in.base_weight      # buffer
    t_ids = split_in.tune_ids          # buffer
    train_w = split_in.train_weight    # Parameter

    # Build the full embedding weight
    full_in_w = base_w.clone()
    full_in_w[t_ids] = train_w

    # Create and set a fresh nn.Embedding
    new_in = nn.Embedding(
        num_embeddings=full_in_w.size(0),
        embedding_dim=full_in_w.size(1),
        padding_idx=split_in.padding_idx,
    )
    new_in.weight.data.copy_(full_in_w)
    model.set_input_embeddings(new_in)

    # 2) Handle output layer
    split_out = model.get_output_embeddings()
    if isinstance(split_out, SplitEmbedding):
        # Output is an Embedding
        base_w2 = split_out.base_weight
        t_ids2 = split_out.tune_ids
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
        # Output is a Linear layer
        base_w2 = split_out.base_weight    # shape (V, d)
        t_ids2 = split_out.tune_ids
        train_w2 = split_out.train_weight
        full_out_w = base_w2.clone()
        full_out_w[t_ids2] = train_w2
        b = split_out.bias.data if split_out.bias is not None else None

        new_out = nn.Linear(
            in_features=full_out_w.size(1),
            out_features=full_out_w.size(0),
            bias=(b is not None),
        )
        new_out.weight.data.copy_(full_out_w)
        if b is not None:
            new_out.bias.data.copy_(b)

    model.set_output_embeddings(new_out)

    # 3) Restore tie flag if needed
    if was_tied:
        model.config.tie_word_embeddings = True
        model.tie_weights()

    return model
