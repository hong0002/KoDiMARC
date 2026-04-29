from __future__ import annotations

import torch



def get_torch_dtype(dtype_str: str):
    dtype_str = (dtype_str or "").lower()
    if "bf16" in dtype_str:
        return torch.bfloat16
    if "16" in dtype_str:
        return torch.float16
    if "32" in dtype_str:
        return torch.float32
    return None



def build_plain_sft_model(model_cfg: dict, lora_cfg: dict):
    """Build the Step1 response-only SFT model with optional LoRA."""
    base_model_name = model_cfg["base_model"]
    init_ckpt = model_cfg.get("init_ckpt")
    model_name = init_ckpt if init_ckpt else base_model_name

    quant_type = (model_cfg.get("quant_type") or "").lower()
    max_seq_len = int(model_cfg.get("max_length", 512))
    load_in_4bit = quant_type == "4bit"
    load_in_8bit = quant_type == "8bit"

    if load_in_4bit or load_in_8bit:
        dtype = None
    else:
        dtype = get_torch_dtype(model_cfg.get("dtype", "bfloat16"))

    device_map = model_cfg.get("device_map", "auto")
    print(f"[plain_sft] loading base model: {model_name}")
    print(
        f"[plain_sft] quant_type={quant_type}, load_in_4bit={load_in_4bit}, "
        f"load_in_8bit={load_in_8bit}, dtype={dtype}, device_map={device_map}"
    )

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError("unsloth is required for Step1 generator training.") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device_map=device_map,
        trust_remote_code=True,
    )

    if lora_cfg.get("enable", True):
        target_modules = lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        print("[plain_sft] applying LoRA adapters.")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("rank", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.print_trainable_parameters()
    else:
        print("[plain_sft] LoRA disabled; full fine-tuning may require substantially more GPU memory.")

    return model, tokenizer
