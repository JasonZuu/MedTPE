import torch
import torch.nn as nn
from tqdm import tqdm


def init_exp_embedding(emb_layer, sub_token_ids, sign,
                        global_norm, scale):
    """计算带指数权重的组合嵌入
    """
    k = len(sub_token_ids)
    assert k > 1, "Sub-token IDs must contain at least 2 items."

    # 获取所有sub-token的嵌入
    embeddings = emb_layer.weight.data[sub_token_ids]  # (k, hidden_size)

    # 生成位置权重（1-based）
    device = embeddings.device
    positions = torch.arange(1, k+1, dtype=torch.float32, device=device)

    # 计算指数权重
    exponents = 2 * positions if sign == '+' else -2 * positions
    weights = torch.exp(exponents)
    weights /= weights.sum()  # 归一化

    # 加权组合
    _init_embedding = torch.sum(weights.view(-1, 1) * embeddings, dim=0)  # (hidden_size,)
    init_embedding = _init_embedding / _init_embedding.norm() * global_norm * scale # (hidden_size,)
    return init_embedding


def init_mean_embedding(emb_layer: torch.nn.Embedding,
                        sub_token_ids: list[int],
                        global_norm: float,
                        scale,
                        **kwargs) -> torch.Tensor:
    """
    Parameters
    ----------
    emb_layer : nn.Embedding
        模型的输入嵌入层（input embeddings）
    sub_token_ids : List[int]
        组成 n-gram 的若干子 token id，长度 ≥ 2

    Returns
    -------
    torch.Tensor
        初始化好的 n-gram 向量（shape = [hidden_size]）
    """
    assert len(sub_token_ids) >= 2, "Need at least two subtokens"

    # ① 简单平均
    sub_embs = emb_layer.weight.data[sub_token_ids]          # (k, d)
    e_avg    = sub_embs.mean(dim=0)                          # (d,)

    # ② 尺度校准：让范数与全局平均范数一致
    new_emb     = e_avg / e_avg.norm() * global_norm * scale  # (d,)

    return new_emb


def adapted_embedding(
    input_embeds: nn.Embedding,
    output_embeds: nn.Module,  # 可以是Linear或Embedding
    sub_token_ids: list,
    tie_word_embeddings: bool,
    global_norm: float,
    scale: float = 0.5
):
    """
    生成适配后的输入输出嵌入

    Args:
        input_embeds: 输入嵌入层 (nn.Embedding)
        output_embeds: 输出嵌入层 (nn.Linear/nn.Embedding)
        sub_token_ids: 组成新token的sub-token IDs列表
        tie_word_embeddings: 是否共享输入输出嵌入权重
        global_norm: 全局平均范数，用于缩放嵌入向量
        scale: 缩放因子，默认值为0.5. It is an empirical value for not influcing direct inference.
      and fintuning steps.
    Returns:
        adapted_input: 适配后的输入嵌入
        adapted_output: 适配后的输出嵌入
    """
    # 计算输入嵌入（递增权重）
    adapted_input_embeds = init_mean_embedding(emb_layer=input_embeds,
                                               sub_token_ids=sub_token_ids,
                                                sign='+',
                                                global_norm=global_norm,
                                                scale=scale)

    # 计算输出嵌入
    if tie_word_embeddings:
        adapted_output_embeds = adapted_input_embeds.clone()
    else:
        if not hasattr(output_embeds, 'weight'):
            raise ValueError("输出嵌入层必须包含weight属性")

        adapted_output_embeds = init_mean_embedding(emb_layer=output_embeds,
                                                    sub_token_ids=sub_token_ids,
                                                    global_norm=global_norm,
                                                    scale=scale)
    return adapted_input_embeds, adapted_output_embeds


def extend_llm_embedding_for_new_token(model, new_token_info, tie_word_embeddings=False):
    """Extend model embeddings for new tokens appended beyond the original vocab.

    Unlike replace_llm_embedding_for_new_token, this function first resizes the
    embedding matrix to accommodate the new token IDs, then initialises the new
    rows using the same sub-token mean-embedding strategy.
    """
    new_vocab_size = max(info["token_id"] for info in new_token_info.values()) + 1
    model.resize_token_embeddings(new_vocab_size)

    input_emb = model.get_input_embeddings()
    global_norm = input_emb.weight.data.norm(dim=1).mean()
    output_emb = model.get_output_embeddings()

    pbar = tqdm(total=len(new_token_info), desc="Extending embeddings", unit="token")
    for new_tok, info in new_token_info.items():
        adapted_input, adapted_output = adapted_embedding(
            input_emb,
            output_emb,
            info["sub_token_ids"],
            tie_word_embeddings=tie_word_embeddings,
            global_norm=global_norm,
        )
        input_emb.weight.data[info["token_id"]] = adapted_input
        output_emb.weight.data[info["token_id"]] = adapted_output
        pbar.update(1)
    pbar.close()
    return model


def replace_llm_embedding_for_new_token(model, new_token_info, tie_word_embeddings=False):
    # get the model's input and output embedding layers
    input_emb = model.get_input_embeddings()
    global_norm = input_emb.weight.data.norm(dim=1).mean()  # global norm of the input embedding
    output_emb = model.get_output_embeddings()
    eos_token_id = model.config.eos_token_id
    bos_token_id = model.config.bos_token_id
    special_token_ids = []
    if isinstance(eos_token_id, list):
        special_token_ids.extend(eos_token_id)
    else:
        special_token_ids.append(eos_token_id)
    if isinstance(bos_token_id, list):
        special_token_ids.extend(bos_token_id)
    else:
        special_token_ids.append(bos_token_id)

    pbar = tqdm(total=len(new_token_info), desc="Replacing embeddings", unit="token")
    for new_tok, info in new_token_info.items():
        sub_token_ids = info['sub_token_ids']
        new_token_id = info['token_id']
        if new_token_id in special_token_ids:
            print(f"Skipping special token ID: {new_token_id}")
            pbar.update(1)
            continue

        # 计算适配后的嵌入
        adapted_input, adapted_output = adapted_embedding(
            input_emb,
            output_emb,
            sub_token_ids,
            tie_word_embeddings=tie_word_embeddings,
            global_norm=global_norm
        )

        # 替换嵌入
        input_emb.weight.data[new_token_id] = adapted_input
        output_emb.weight.data[new_token_id] = adapted_output

        pbar.update(1)
    pbar.close()
    return model
