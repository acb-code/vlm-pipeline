from typing import Dict, Any
from peft import LoraConfig, get_peft_model


def apply_lora(model, peft_cfg: Dict[str, Any]):
    """
    Wrap a base causal LM in a LoRA PEFT adapter.

    Args:
        model: base HuggingFace model (e.g., Qwen3VLForConditionalGeneration)
        peft_cfg: config["peft"] dict

    Returns:
        LoRA-wrapped model with trainable adapter params.
    """
    if not peft_cfg.get("enabled", False):
        print("[LoRA] PEFT disabled, returning base model.")
        return model

    lora_config = LoraConfig(
        r=peft_cfg["lora_r"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        target_modules=peft_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("[LoRA] Applying LoRA with config:")
    print(lora_config)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model
