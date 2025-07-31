from torch import nn
from peft import LoraConfig, get_peft_model


def set_lora_on_layers(
    model: nn.Module,
    layer_idx: list[int] | None = None,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    train_kv_only: bool = False
) -> nn.Module:
    """
    Inject LoRA adapters into specified Transformer layers of a loaded Hugging Face model,
    freezing all other parameters.

    - If layer_idx is None, LoRA is applied to all layers; otherwise, only to layers in the list.
    - If train_kv_only=True, only the key and value projection matrices receive LoRA; otherwise,
      all nn.Linear modules in the layer are targeted.

    Args:
        model: A loaded transformers model instance (e.g., Qwen2Model).
        layer_idx: List of layer indices (0-based) to apply LoRA, or None for all layers.
        r: Rank parameter for LoRA.
        alpha: Scaling factor (alpha) for LoRA.
        dropout: Dropout rate for LoRA adapters.
        train_kv_only: Whether to apply LoRA only to k_proj and v_proj modules.

    Returns:
        A PEFT-wrapped model with only LoRA parameters trainable.
    """
    # Locate the Transformer layers stored in a ModuleList
    layers = None
    prefix = None
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            layers, prefix = child, name
            break
    if layers is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                layers, prefix = module, name
                break
    if layers is None:
        raise ValueError("Unable to locate Transformer layers. Please check the model structure.")

    num_layers = len(layers)
    # Determine which layer indices to target
    if layer_idx is None:
        target_idxs = list(range(num_layers))
    else:
        # Support negative indexing
        target_idxs = [i if i >= 0 else num_layers + i for i in layer_idx]

    # Build the list of Linear module names to target
    target_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Check if this Linear belongs to a targeted layer
        for idx in target_idxs:
            layer_prefix = f"{prefix}.{idx}"
            if name.startswith(layer_prefix):
                if train_kv_only:
                    # Only target key and value projections
                    if "k_proj" in name or "v_proj" in name:
                        target_modules.append(name)
                else:
                    target_modules.append(name)
                break

    if not target_modules:
        raise ValueError(
            "No matching Linear submodules found. Check layer_idx and model structure."
        )

    # Configure LoRA
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True
    )

    # Wrap the model with PEFT LoRA
    peft_model = get_peft_model(model, lora_cfg)

    # Report trainable parameter percentage
    total = sum(p.numel() for p in peft_model.parameters())
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(
        f"Trainable params after adding LoRA: {trainable}/{total}"
        f" = {100 * trainable / total:.4f}%"
    )

    return peft_model
