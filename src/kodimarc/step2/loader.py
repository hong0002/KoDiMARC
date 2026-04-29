import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# ============================================================
# 8) Model loader
# ============================================================
def _parse_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).lower()
    if "bf16" in s or "bfloat16" in s:
        return torch.bfloat16
    if "fp32" in s or "float32" in s:
        return torch.float32
    return torch.float16


def sync_tokenizer_model_embeddings(tokenizer, model) -> bool:
    target_size = len(tokenizer)
    try:
        current_size = int(model.get_input_embeddings().num_embeddings)
    except Exception:
        return False
    if current_size == target_size:
        return False
    model.resize_token_embeddings(target_size)
    return True


def ensure_additional_special_tokens(tokenizer, model, tokens: List[str]) -> int:
    desired = [str(t) for t in tokens if str(t).strip()]
    if not desired:
        return 0

    current = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged = current + [tok for tok in desired if tok not in current]
    added = tokenizer.add_special_tokens({"additional_special_tokens": merged})
    sync_tokenizer_model_embeddings(tokenizer, model)
    return int(added)

def load_encoder_and_tokenizer(model_cfg: Dict[str, Any]):
    model_name = model_cfg["name"]
    quant_type = str(model_cfg.get("quant_type", "8bit")).lower()
    dtype = _parse_dtype(model_cfg.get("dtype", "bfloat16"))
    device_map = model_cfg.get("device_map", "auto")
    use_fast = bool(model_cfg.get("use_fast", True))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if quant_type in ["8bit", "int8", "8"]:
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError(
                "8bit quantization requires `peft` to be installed because "
                "`prepare_model_for_kbit_training` is used during setup."
            ) from exc
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        base = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base)
        try:
            base.gradient_checkpointing_enable()
        except Exception:
            pass
        return base, tokenizer

    elif quant_type in ["none", "fp16", "bf16", "full"]:
        base = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        base.config.use_cache = False
        try:
            base.gradient_checkpointing_enable()
        except Exception:
            pass
        return base, tokenizer

    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}. Use 8bit/none.")

def apply_lora(base_model, model_cfg: Dict[str, Any]):
    r = int(model_cfg.get("lora_r", 0))
    if r <= 0:
        return base_model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA is enabled in the config, but `peft` is not installed."
        ) from exc

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=int(model_cfg.get("lora_alpha", 16)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.0)),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=list(model_cfg.get("target_modules", [])),
    )
    m = get_peft_model(base_model, lora_cfg)
    return m
