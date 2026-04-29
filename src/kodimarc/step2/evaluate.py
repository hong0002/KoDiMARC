import json
import os
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from kodimarc.common.io import ensure_dir, read_jsonl, write_json
from kodimarc.common.markers import CAT2ID, ID2NLI, relation_special_tokens
from kodimarc.step2.dataset import PadCollator, Step2Dataset
from kodimarc.step2.loader import ensure_additional_special_tokens
from kodimarc.step2.model import Step2EncoderClassifier
from kodimarc.step2.eval_utils import eval_collect, save_confusion_and_report, save_transition_analysis


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).lower()
    if "bf16" in s or "bfloat16" in s:
        return torch.bfloat16
    if "fp16" in s or "float16" in s:
        return torch.float16
    return torch.float32


def resolve_step2_best_dir(config_path: str, cfg: Dict[str, Any], exp_name: Optional[str] = None) -> tuple[str, str]:
    run_dir = os.path.dirname(os.path.abspath(config_path))
    if os.path.isdir(os.path.join(run_dir, "experiments")):
        if exp_name is None:
            summary_path = os.path.join(run_dir, "summary.jsonl")
            if os.path.exists(summary_path):
                best_row = None
                with open(summary_path, "r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        if best_row is None or float(row.get("best_score", -1.0)) > float(best_row.get("best_score", -1.0)):
                            best_row = row
                if best_row is not None:
                    exp_name = best_row["exp_name"]
        if exp_name is None:
            exp_name = cfg["experiments"][0]["name"]
        return run_dir, os.path.join(run_dir, "experiments", exp_name, "best")

    if exp_name is None:
        exp_name = cfg["experiments"][0]["name"]
    best_dir = os.path.join(cfg["paths"]["outputs_root"], "experiments", exp_name, "best")
    return cfg["paths"]["outputs_root"], best_dir


def load_step2_model(best_dir: str, cfg: Dict[str, Any], device: torch.device):
    meta = torch.load(os.path.join(best_dir, "heads_and_meta.pt"), map_location="cpu")
    logic_label2id = meta["logic_label2id"]
    dtype = _parse_dtype(cfg["model"].get("dtype", "float32"))
    prompt_cfg = cfg.get("prompt", {}) or {}
    special_tokens = []
    if str(prompt_cfg.get("pair_marker_representation", "raw_text")).strip().lower() == "category_special":
        special_tokens = relation_special_tokens(
            template=str(prompt_cfg.get("pair_category_special_template", "[REL_{cat}]")),
            include_none=bool(prompt_cfg.get("pair_category_special_include_none", False)),
            include_unk=bool(prompt_cfg.get("pair_category_special_include_unk", True)),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        best_dir,
        use_fast=bool(cfg["model"].get("use_fast", True)),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if os.path.exists(os.path.join(best_dir, "adapter_config.json")):
        from peft import PeftModel

        base = AutoModel.from_pretrained(
            cfg["model"]["name"],
            dtype=dtype,
            trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        )
        ensure_additional_special_tokens(tokenizer, base, special_tokens)
        base_model = PeftModel.from_pretrained(base, best_dir)
    else:
        base_model = AutoModel.from_pretrained(
            best_dir,
            dtype=dtype,
            trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        )
        ensure_additional_special_tokens(tokenizer, base_model, special_tokens)

    with torch.no_grad():
        tmp = tokenizer("dtype_check", return_tensors="pt")
        out = base_model(
            input_ids=tmp["input_ids"],
            attention_mask=tmp["attention_mask"],
            return_dict=True,
            use_cache=False,
        )
        head_dtype = out.last_hidden_state.dtype

    model = Step2EncoderClassifier(
        base_model=base_model,
        hidden_size=base_model.config.hidden_size,
        num_logic=len(logic_label2id),
        dropout=float(cfg["train"].get("head_dropout", 0.1)),
        head_dtype=head_dtype,
        use_marker_aware_head=bool(meta.get("use_marker_aware_head", cfg["model"].get("use_marker_aware_head", False))),
        use_base_delta_head=bool(meta.get("use_base_delta_head", cfg["model"].get("use_base_delta_head", False))),
        pooling_strategy=str(meta.get("pooling_strategy", cfg["model"].get("pooling_strategy", "last_token"))),
        delta_scale_nli=float(meta.get("delta_scale_nli", cfg["model"].get("delta_scale_nli", 1.0))),
        delta_scale_logic=float(meta.get("delta_scale_logic", cfg["model"].get("delta_scale_logic", 1.0))),
        use_category_distribution_feature=bool(meta.get("use_category_distribution_feature", cfg["model"].get("use_category_distribution_feature", False))),
        num_marker_categories=int(meta.get("num_marker_categories", len(CAT2ID))),
        use_learned_marker_gate=bool(meta.get("use_learned_marker_gate", cfg["model"].get("use_learned_marker_gate", False))),
        marker_gate_hidden_size=int(meta.get("marker_gate_hidden_size", cfg["model"].get("marker_gate_hidden_size", 0))),
        marker_gate_init_bias=float(meta.get("marker_gate_init_bias", cfg["model"].get("marker_gate_init_bias", 1.0))),
        use_marker_compatibility_head=bool(meta.get("use_marker_compatibility_head", cfg["model"].get("use_marker_compatibility_head", False))),
        use_compatibility_for_delta=bool(meta.get("use_compatibility_for_delta", cfg["model"].get("use_compatibility_for_delta", False))),
        compatibility_hidden_size=int(meta.get("compatibility_hidden_size", cfg["model"].get("compatibility_hidden_size", 0))),
        compatibility_init_bias=float(meta.get("compatibility_init_bias", cfg["model"].get("compatibility_init_bias", 0.0))),
    ).to(device)
    if bool(meta.get("use_marker_aware_head", False)) and meta.get("marker_proj") is not None:
        model.marker_proj.load_state_dict(meta["marker_proj"])
    if bool(meta.get("use_category_distribution_feature", False)) and meta.get("cat_dist_proj") is not None:
        model.cat_dist_proj.load_state_dict(meta["cat_dist_proj"])
    if bool(meta.get("use_learned_marker_gate", False)) and meta.get("marker_gate") is not None:
        model.marker_gate.load_state_dict(meta["marker_gate"])
    if bool(meta.get("use_marker_compatibility_head", False)) and meta.get("compatibility_proj") is not None:
        model.compatibility_proj.load_state_dict(meta["compatibility_proj"])
    if bool(meta.get("use_marker_compatibility_head", False)) and meta.get("compatibility_head") is not None:
        model.compatibility_head.load_state_dict(meta["compatibility_head"])
    if bool(meta.get("use_base_delta_head", False)):
        model.base_nli_head.load_state_dict(meta["base_nli_head"])
        model.base_logic_head.load_state_dict(meta["base_logic_head"])
        model.delta_proj.load_state_dict(meta["delta_proj"])
        model.delta_nli_head.load_state_dict(meta["delta_nli_head"])
        model.delta_logic_head.load_state_dict(meta["delta_logic_head"])
    else:
        model.nli_head.load_state_dict(meta["nli_head"])
        model.logic_head.load_state_dict(meta["logic_head"])
    model.mrel_head.load_state_dict(meta["mrel_head"])
    model.eval()
    return model, tokenizer, logic_label2id


def evaluate_step2_checkpoint(
    config_path: str,
    exp_name: Optional[str] = None,
    output_subdir: str = "test_eval",
):
    cfg = load_yaml(config_path)
    run_dir, best_dir = resolve_step2_best_dir(config_path, cfg, exp_name)
    device = pick_device(cfg.get("device", "auto"))
    model, tokenizer, logic_label2id = load_step2_model(best_dir, cfg, device)
    id2logic = {v: k for k, v in logic_label2id.items()}

    paths = cfg["paths"]
    nli_path = paths.get("test_nli_jsonl") or paths["dev_kornli_jsonl"]
    logic_path = paths.get("test_logic_jsonl") or paths["dev_ai_jsonl"]
    nli_rows = read_jsonl(nli_path)
    logic_rows = read_jsonl(logic_path)

    marker_cfg = cfg["marker"]
    eval_cfg = cfg.get("evaluation", {})
    confidence_gating_cfg = dict(marker_cfg.get("confidence_gating", {}) or {})
    collator = PadCollator(tokenizer, pad_to_multiple_of=8)

    def make_ds(rows, mode: str):
        return Step2Dataset(
            rows=rows,
            tokenizer=tokenizer,
            logic_label2id=logic_label2id,
            max_len=int(cfg["model"].get("max_seq_length", 512)),
            is_train=False,
            seed=int(cfg["seed"]),
            marker_temperature=float(marker_cfg["topk_temperature"]),
            dropout_nli_start=0.0,
            dropout_nli_end=0.0,
            dropout_logic_start=0.0,
            dropout_logic_end=0.0,
            dropout_unk_boost=0.0,
            corrupt_prob_nli=0.0,
            corrupt_prob_logic=0.0,
            logic_forbidden_categories=marker_cfg.get("logic_forbidden_categories", {}),
            mode=mode,
            enable_corruption=False,
            enable_dropout=False,
            prompt_style=str(cfg.get("prompt", {}).get("style", "legacy")),
            prompt_input_style=str(cfg.get("prompt", {}).get("input_style", "prompt")),
            pair_marker_placement=str(cfg.get("prompt", {}).get("pair_marker_placement", "hypothesis_prefix")),
            pair_marker_prefix=str(cfg.get("prompt", {}).get("pair_marker_prefix", "[M] ")),
            pair_marker_representation=str(cfg.get("prompt", {}).get("pair_marker_representation", "raw_text")),
            pair_category_special_template=str(cfg.get("prompt", {}).get("pair_category_special_template", "[REL_{cat}]")),
            pair_distribution_temperature=cfg.get("prompt", {}).get("pair_distribution_temperature"),
            marker_source=str(marker_cfg.get("source", "predicted")),
            confidence_gating_enabled=bool(confidence_gating_cfg.get("enabled", False)),
            confidence_gating_apply_train=bool(confidence_gating_cfg.get("apply_train", True)),
            confidence_gating_apply_eval=bool(confidence_gating_cfg.get("apply_eval", True)),
            confidence_gating_temperature=float(confidence_gating_cfg.get("temperature", marker_cfg.get("topk_temperature", 1.0))),
            confidence_gating_min_top1_prob=float(confidence_gating_cfg.get("min_top1_prob", 0.0)),
            confidence_gating_min_top1_gap=float(confidence_gating_cfg.get("min_top1_gap", 0.0)),
            strong_wrong_nli=bool(eval_cfg.get("strong_wrong_nli", True)),
            wrong_nli_exclude_topk=bool(eval_cfg.get("wrong_nli_exclude_topk", True)),
            wrong_nli_exclude_same_category=bool(eval_cfg.get("wrong_nli_exclude_same_category", True)),
        )

    out_dir = os.path.join(run_dir, output_subdir)
    ensure_dir(out_dir)

    def run_one(tag: str, rows, task: str, view: str, mode: str, class_names):
        ds = make_ds(rows, mode)
        loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, collate_fn=collator)
        collected = eval_collect(model, loader, device, task, view, len(class_names), False, None)
        save_confusion_and_report(out_dir, tag, len(class_names), class_names, collected)
        return collected

    no_nli = run_one("test_no_nli_v1", nli_rows, "nli", "v1", "eval_no_marker", [ID2NLI[i] for i in range(3)])
    with_nli = run_one("test_with_nli_v2", nli_rows, "nli", "v2", "eval_with_marker", [ID2NLI[i] for i in range(3)])
    no_logic = run_one("test_no_logic_v1", logic_rows, "logic", "v1", "eval_no_marker", [id2logic[i] for i in range(len(id2logic))])
    with_logic = run_one("test_with_logic_v2", logic_rows, "logic", "v2", "eval_with_marker", [id2logic[i] for i in range(len(id2logic))])

    wrong_nli = None
    wrong_logic = None
    if bool(cfg.get("evaluation", {}).get("enable_wrong_marker_eval", True)):
        wrong_nli = run_one("test_wrong_nli_v2", nli_rows, "nli", "v2", "eval_wrong_marker", [ID2NLI[i] for i in range(3)])
        wrong_logic = run_one("test_wrong_logic_v2", logic_rows, "logic", "v2", "eval_wrong_marker", [id2logic[i] for i in range(len(id2logic))])

    save_transition_analysis(out_dir, "test_nli", nli_rows, "nli", no_nli, with_nli, wrong_nli, ID2NLI)
    save_transition_analysis(out_dir, "test_logic", logic_rows, "logic", no_logic, with_logic, wrong_logic, id2logic)

    def flatten(prefix: str, collected: Dict[str, Any]) -> Dict[str, float]:
        return {
            f"{prefix}_acc": float(collected["acc"]),
            f"{prefix}_macro_precision": float(collected["macro_precision"]),
            f"{prefix}_macro_recall": float(collected["macro_recall"]),
            f"{prefix}_macro_f1": float(collected["macro_f1"]),
        }

    metrics = {}
    metrics.update(flatten("test_no_nli", no_nli))
    metrics.update(flatten("test_with_nli", with_nli))
    metrics.update(flatten("test_no_logic", no_logic))
    metrics.update(flatten("test_with_logic", with_logic))
    if wrong_nli is not None and wrong_logic is not None:
        metrics.update(flatten("test_wrong_nli", wrong_nli))
        metrics.update(flatten("test_wrong_logic", wrong_logic))
    write_json(os.path.join(out_dir, "metrics.json"), metrics)
    return metrics
