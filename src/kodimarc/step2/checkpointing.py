import os
import json
from typing import Dict, Any
import torch

from kodimarc.common.io import ensure_dir
from kodimarc.step2.model import Step2EncoderClassifier

# ============================================================
# 7) Save best
# ============================================================
def _save_tokenizer_compat(tokenizer, save_dir: str):
    try:
        tokenizer.save_pretrained(save_dir)
        return
    except TypeError as exc:
        if "filename_prefix" not in str(exc):
            raise

    # Compatibility fallback for tokenizers such as KoBERT whose
    # save_vocabulary() does not accept the newer filename_prefix argument.
    tokenizer.save_vocabulary(save_dir)

    tokenizer_config_path = os.path.join(save_dir, "tokenizer_config.json")
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.init_kwargs, f, ensure_ascii=False, indent=2)

    special_tokens_map_path = os.path.join(save_dir, "special_tokens_map.json")
    with open(special_tokens_map_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.special_tokens_map, f, ensure_ascii=False, indent=2)

    added_vocab = tokenizer.get_added_vocab()
    if added_vocab:
        added_tokens_path = os.path.join(save_dir, "added_tokens.json")
        with open(added_tokens_path, "w", encoding="utf-8") as f:
            json.dump(added_vocab, f, ensure_ascii=False, indent=2)


def save_best(
    exp_dir: str,
    model: Step2EncoderClassifier,
    tokenizer,
    logic_label2id: Dict[str, int],
    best_score: float,
    cfg_pack: Dict[str, Any]
):
    best_dir = os.path.join(exp_dir, "best")
    ensure_dir(best_dir)

    model.lm.save_pretrained(best_dir)
    _save_tokenizer_compat(tokenizer, best_dir)

    torch.save(
        {
            "nli_head": model.nli_head.state_dict() if hasattr(model, "nli_head") else None,
            "logic_head": model.logic_head.state_dict() if hasattr(model, "logic_head") else None,
            "mrel_head": model.mrel_head.state_dict(),
            "use_marker_aware_head": bool(getattr(model, "use_marker_aware_head", False)),
            "use_base_delta_head": bool(getattr(model, "use_base_delta_head", False)),
            "use_category_distribution_feature": bool(getattr(model, "use_category_distribution_feature", False)),
            "num_marker_categories": int(getattr(model, "num_marker_categories", 0)),
            "pooling_strategy": str(getattr(model, "pooling_strategy", "last_token")),
            "delta_scale_nli": float(getattr(model, "delta_scale_nli", 1.0)),
            "delta_scale_logic": float(getattr(model, "delta_scale_logic", 1.0)),
            "use_learned_marker_gate": bool(getattr(model, "use_learned_marker_gate", False)),
            "marker_gate_hidden_size": int(getattr(model, "marker_gate_hidden_size", 0)),
            "marker_gate_init_bias": float(getattr(model, "marker_gate_init_bias", 1.0)),
            "use_marker_compatibility_head": bool(getattr(model, "use_marker_compatibility_head", False)),
            "use_compatibility_for_delta": bool(getattr(model, "use_compatibility_for_delta", False)),
            "compatibility_hidden_size": int(getattr(model, "compatibility_hidden_size", 0)),
            "compatibility_init_bias": float(getattr(model, "compatibility_init_bias", 0.0)),
            "marker_proj": model.marker_proj.state_dict() if getattr(model, "use_marker_aware_head", False) else None,
            "cat_dist_proj": model.cat_dist_proj.state_dict() if getattr(model, "use_category_distribution_feature", False) else None,
            "marker_gate": model.marker_gate.state_dict() if getattr(model, "use_learned_marker_gate", False) else None,
            "compatibility_proj": model.compatibility_proj.state_dict() if getattr(model, "use_marker_compatibility_head", False) else None,
            "compatibility_head": model.compatibility_head.state_dict() if getattr(model, "use_marker_compatibility_head", False) else None,
            "base_nli_head": model.base_nli_head.state_dict() if getattr(model, "use_base_delta_head", False) else None,
            "base_logic_head": model.base_logic_head.state_dict() if getattr(model, "use_base_delta_head", False) else None,
            "delta_proj": model.delta_proj.state_dict() if getattr(model, "use_base_delta_head", False) else None,
            "delta_nli_head": model.delta_nli_head.state_dict() if getattr(model, "use_base_delta_head", False) else None,
            "delta_logic_head": model.delta_logic_head.state_dict() if getattr(model, "use_base_delta_head", False) else None,
            "logic_label2id": logic_label2id,
            "best_score": best_score,
            "cfg_pack": cfg_pack,
        },
        os.path.join(best_dir, "heads_and_meta.pt"),
    )
