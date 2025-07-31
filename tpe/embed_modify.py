import torch
import torch.nn as nn
from tqdm import tqdm


def init_mean_embedding(emb_layer: torch.nn.Embedding,
                        sub_token_ids: list[int],
                        global_norm: float,
                        scale,
                        **kwargs) -> torch.Tensor:
    """
    Parameters
    ----------
    emb_layer : nn.Embedding
        Input embedding layer of the model.
    sub_token_ids : List[int]
        List of sub-token IDs forming an n-gram, length ≥ 2.

    Returns
    -------
    torch.Tensor
        The initialized n-gram vector (shape = [hidden_size]).
    """
    assert len(sub_token_ids) >= 2, "Need at least two subtokens"

    # 1. Simple mean
    sub_embs = emb_layer.weight.data[sub_token_ids]          # (k, d)
    e_avg    = sub_embs.mean(dim=0)                           # (d,)

    # 2. Scale calibration: match the norm to the global average norm
    new_emb     = e_avg / e_avg.norm() * global_norm * scale  # (d,)

    return new_emb


def adapted_embedding(
    input_embeds: nn.Embedding,
    output_embeds: nn.Module,  # Can be Linear or Embedding
    sub_token_ids: list,
    tie_word_embeddings: bool,
    global_norm: float,
    scale: float = 0.5
):
    """
    Generate adapted input and output embeddings.
    
    Args:
        input_embeds: Input embedding layer (nn.Embedding).
        output_embeds: Output embedding layer (nn.Linear or nn.Embedding).
        sub_token_ids: List of sub-token IDs composing the new token.
        tie_word_embeddings: Whether to tie input and output embedding weights.
        global_norm: Global average norm used to scale embedding vectors.
        scale: Scale factor, default is 0.5. It is an empirical value to avoid influencing
               direct inference and finetuning steps.
    Returns:
        adapted_input: Adapted input embedding.
        adapted_output: Adapted output embedding.
    """
    # Compute input embedding (mean weighting)
    adapted_input_embeds = init_mean_embedding(
        emb_layer=input_embeds, 
        sub_token_ids=sub_token_ids, 
        global_norm=global_norm,
        scale=scale
    )
    
    # Compute output embedding
    if tie_word_embeddings:
        adapted_output_embeds = adapted_input_embeds.clone()
    else:
        # Verify output layer type
        if not hasattr(output_embeds, 'weight'):
            raise ValueError("Output embedding layer must have a weight attribute")
        
        # Compute output embedding (exponential weighting)
        adapted_output_embeds = init_mean_embedding(
            emb_layer=output_embeds.weight, 
            sub_token_ids=sub_token_ids, 
            global_norm=global_norm,
            scale=scale
        )
    return adapted_input_embeds, adapted_output_embeds


def replace_llm_embedding_for_new_token(model, new_token_info, tie_word_embeddings=False):
    # Get the model's input and output embedding layers
    input_emb = model.get_input_embeddings()
    global_norm = input_emb.weight.data.norm(dim=1).mean()  # Global norm of the input embedding
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
        
        # Compute adapted embeddings
        adapted_input, adapted_output = adapted_embedding(
            input_emb,
            output_emb,
            sub_token_ids,
            tie_word_embeddings=tie_word_embeddings,
            global_norm=global_norm
        )
        
        # Replace embeddings
        input_emb.weight.data[new_token_id] = adapted_input
        output_emb.weight.data[new_token_id] = adapted_output

        pbar.update(1)
    pbar.close()
    return model
