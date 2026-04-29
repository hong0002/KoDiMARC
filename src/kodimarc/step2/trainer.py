import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from kodimarc.common.io import ensure_dir, read_jsonl
from kodimarc.common.markers import (
    CAT2ID,
    ID2NLI,
    LOGIC_KO_TO_CAT,
    NLI_TO_CAT,
    relation_special_tokens,
)
from kodimarc.common.utils import (
    build_logic_label_vocab,
    cuda_cleanup,
    get_lr,
    pick_device,
    set_seed,
    split_train_rows_by_task,
)
from kodimarc.step2.dataset import PadCollator, Step2Dataset
from kodimarc.step2.loader import (
    apply_lora,
    ensure_additional_special_tokens,
    load_encoder_and_tokenizer,
)
from kodimarc.step2.losses import (
    masked_bce_with_logits,
    kl_consistency_loss,
    masked_ce,
    masked_ce_with_mask,
    masked_focal_ce,
    masked_margin_between_views,
    supcon_loss_with_queue,
)
from kodimarc.step2.memory_bank import MemoryBank
from kodimarc.step2.model import Step2EncoderClassifier
from kodimarc.step2.checkpointing import save_best
from kodimarc.step2.eval_utils import (
    eval_collect,
    save_confusion_and_report,
    save_transition_analysis,
)


class InfiniteLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def _infer_head_dtype(base_model, tokenizer, device: torch.device, use_autocast: bool, amp_dtype):
    try:
        base_device = next(base_model.parameters()).device
    except StopIteration:
        base_device = device
    with torch.no_grad():
        tmp = tokenizer("dtype_check", return_tensors="pt")
        tmp_ids = tmp["input_ids"].to(base_device)
        tmp_mask = tmp["attention_mask"].to(base_device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast and base_device.type == "cuda"):
            out = base_model(
                input_ids=tmp_ids,
                attention_mask=tmp_mask,
                return_dict=True,
                use_cache=False,
            )
    return out.last_hidden_state.dtype


def _zero_result() -> Dict[str, Any]:
    return {
        "acc": 0.0,
        "macro_f1": 0.0,
        "row_id": [],
        "y": [],
        "pred": [],
        "cat": [],
        "forb": [],
        "top1_marker_text": [],
        "v2_marker_text": [],
        "topk_markers_text": [],
        "topk_scores_text": [],
    }


def _pair_metric(result_a: Dict[str, Any], result_b: Dict[str, Any], metric_key: str) -> float:
    return 0.5 * float(result_a.get(metric_key, 0.0)) + 0.5 * float(result_b.get(metric_key, 0.0))


def _build_marker_compat_targets(
    y_nli: torch.Tensor,
    y_logic: torch.Tensor,
    marker_cat_id: torch.Tensor,
    logic_id2label: Dict[int, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros_like(marker_cat_id, dtype=torch.float32)
    mask = marker_cat_id != CAT2ID["NONE"]

    nli_valid = (y_nli != -100) & mask
    if nli_valid.any():
        for label_name, label_id in [("entailment", 0), ("neutral", 1), ("contradiction", 2)]:
            gold_cat = NLI_TO_CAT.get(label_name, "UNK")
            gold_cat_id = CAT2ID.get(gold_cat, CAT2ID["UNK"])
            label_mask = nli_valid & (y_nli == label_id)
            if label_mask.any():
                target[label_mask] = (marker_cat_id[label_mask] == gold_cat_id).float()

    logic_valid = (y_logic != -100) & mask
    if logic_valid.any():
        for logic_id, logic_label in logic_id2label.items():
            gold_cat = LOGIC_KO_TO_CAT.get(logic_label, "UNK")
            gold_cat_id = CAT2ID.get(gold_cat, CAT2ID["UNK"])
            label_mask = logic_valid & (y_logic == int(logic_id))
            if label_mask.any():
                target[label_mask] = (marker_cat_id[label_mask] == gold_cat_id).float()

    return target, mask


def _build_logic_class_weights(
    rows: List[Dict[str, Any]],
    logic_label2id: Dict[str, int],
    mode: str,
    smoothing: float,
    device: torch.device,
    overrides: Dict[str, float] | None = None,
) -> torch.Tensor | None:
    mode_l = str(mode).lower()
    if mode_l in ("", "none", "off", "false"):
        return None

    counts = torch.full((len(logic_label2id),), float(smoothing), dtype=torch.float32)
    for row in rows:
        lab = row.get("logic_label")
        if lab is None or lab not in logic_label2id:
            continue
        counts[logic_label2id[lab]] += 1.0

    if mode_l in ("inverse", "inv"):
        weights = counts.reciprocal()
    else:
        weights = counts.rsqrt()

    for label, mult in (overrides or {}).items():
        if label not in logic_label2id:
            continue
        weights[logic_label2id[label]] *= float(mult)

    weights = weights / weights.mean().clamp_min(1e-12)
    return weights.to(device=device, dtype=torch.float32)


def run_experiment(
    cfg: Dict[str, Any],
    exp: Dict[str, Any],
    run_dir: str,
    train_script_path: str,
    used_yaml_path: str,
) -> Dict[str, Any]:
    seed = int(cfg["seed"])
    set_seed(seed)
    device = pick_device(cfg.get("device", "auto"))

    model_cfg = cfg["model"]
    marker_cfg = cfg["marker"]
    prompt_cfg = cfg.get("prompt", {}) or {}
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("evaluation", {})

    exp_name = exp["name"]
    exp_dir = f"{run_dir}/experiments/{exp_name}"
    tb_dir = f"{run_dir}/tb/{exp_name}"
    eval_dir = f"{exp_dir}/eval"
    ensure_dir(exp_dir)
    ensure_dir(tb_dir)
    ensure_dir(eval_dir)

    writer = SummaryWriter(log_dir=tb_dir)

    enable_kl = bool(exp.get("enable_kl", True))
    enable_dropout = bool(exp.get("enable_dropout", True))
    enable_corruption = bool(exp.get("enable_corruption", True))
    enable_mrel = bool(exp.get("enable_mrel", True))
    enable_supcon = bool(exp.get("enable_supcon", True))

    mp = str(train_cfg.get("mixed_precision", "bf16")).lower()
    use_autocast = mp in ["bf16", "fp16"]
    amp_dtype = torch.bfloat16 if mp == "bf16" else (torch.float16 if mp == "fp16" else None)
    use_scaler = mp == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    base, tokenizer = load_encoder_and_tokenizer(model_cfg)
    if str(prompt_cfg.get("pair_marker_representation", "raw_text")).strip().lower() == "category_special":
        ensure_additional_special_tokens(
            tokenizer,
            base,
            relation_special_tokens(
                template=str(prompt_cfg.get("pair_category_special_template", "[REL_{cat}]")),
                include_none=bool(prompt_cfg.get("pair_category_special_include_none", False)),
                include_unk=bool(prompt_cfg.get("pair_category_special_include_unk", True)),
            ),
        )
    base = apply_lora(base, model_cfg)
    hidden = base.config.hidden_size
    head_dtype = _infer_head_dtype(base, tokenizer, device, use_autocast, amp_dtype)

    paths = cfg["paths"]
    train_rows = read_jsonl(paths["train_jsonl"])
    dev_k_rows = read_jsonl(paths["dev_kornli_jsonl"])
    dev_a_rows = read_jsonl(paths["dev_ai_jsonl"])

    train_nli_rows, train_logic_rows = split_train_rows_by_task(train_rows)
    logic_label2id = build_logic_label_vocab(train_rows)
    if not logic_label2id:
        raise RuntimeError("No logic_label examples were found in train_jsonl.")
    id2logic = {v: k for k, v in logic_label2id.items()}

    model = Step2EncoderClassifier(
        base_model=base,
        hidden_size=hidden,
        num_logic=len(logic_label2id),
        dropout=float(train_cfg.get("head_dropout", 0.1)),
        head_dtype=head_dtype,
        use_marker_aware_head=bool(model_cfg.get("use_marker_aware_head", True)),
        use_base_delta_head=bool(model_cfg.get("use_base_delta_head", False)),
        pooling_strategy=str(model_cfg.get("pooling_strategy", "last_token")),
        delta_scale_nli=float(model_cfg.get("delta_scale_nli", 1.0)),
        delta_scale_logic=float(model_cfg.get("delta_scale_logic", 1.0)),
        use_category_distribution_feature=bool(model_cfg.get("use_category_distribution_feature", False)),
        num_marker_categories=len(CAT2ID),
        use_learned_marker_gate=bool(model_cfg.get("use_learned_marker_gate", False)),
        marker_gate_hidden_size=int(model_cfg.get("marker_gate_hidden_size", 0)),
        marker_gate_init_bias=float(model_cfg.get("marker_gate_init_bias", 1.0)),
        use_marker_compatibility_head=bool(model_cfg.get("use_marker_compatibility_head", False)),
        use_compatibility_for_delta=bool(model_cfg.get("use_compatibility_for_delta", False)),
        compatibility_hidden_size=int(model_cfg.get("compatibility_hidden_size", 0)),
        compatibility_init_bias=float(model_cfg.get("compatibility_init_bias", 0.0)),
    ).to(device)

    max_seq_length = int(model_cfg.get("max_seq_length", 512))
    logic_forbidden = marker_cfg.get("logic_forbidden_categories", {}) or {}
    if "override_logic_forbidden_categories" in exp:
        logic_forbidden = exp["override_logic_forbidden_categories"] or {}

    strong_wrong_nli = bool(eval_cfg.get("strong_wrong_nli", True))
    wrong_nli_exclude_topk = bool(eval_cfg.get("wrong_nli_exclude_topk", True))
    wrong_nli_exclude_same_category = bool(eval_cfg.get("wrong_nli_exclude_same_category", True))
    confidence_gating_cfg = dict(marker_cfg.get("confidence_gating", {}) or {})
    confidence_gating_enabled = bool(confidence_gating_cfg.get("enabled", False))
    confidence_gating_apply_train = bool(confidence_gating_cfg.get("apply_train", True))
    confidence_gating_apply_eval = bool(confidence_gating_cfg.get("apply_eval", True))
    confidence_gating_temperature = float(
        confidence_gating_cfg.get("temperature", marker_cfg.get("topk_temperature", 1.0))
    )
    confidence_gating_min_top1_prob = float(confidence_gating_cfg.get("min_top1_prob", 0.0))
    confidence_gating_min_top1_gap = float(confidence_gating_cfg.get("min_top1_gap", 0.0))
    train_sampling_mode = str(marker_cfg.get("train_sampling_mode", "temperature"))
    if bool(marker_cfg.get("train_use_top1", False)):
        train_sampling_mode = "top1"
    compat_loss_cfg = dict(loss_cfg.get("compatibility", {}) or {})
    lambda_compat_hint = float(compat_loss_cfg.get("weight", loss_cfg.get("lambda_compat", 0.0)))
    compat_use_wrong_view = bool(compat_loss_cfg.get("use_wrong_view", True))

    def make_ds(rows, is_train: bool, mode: str):
        margin_cfg_local = dict(loss_cfg.get("with_wrong_margin", {}) or {})
        include_wrong_view = is_train and float(
            margin_cfg_local.get("weight", loss_cfg.get("with_wrong_margin_weight", 0.0))
        ) > 0.0
        if is_train and lambda_compat_hint > 0.0 and compat_use_wrong_view:
            include_wrong_view = True
        return Step2Dataset(
            rows=rows,
            tokenizer=tokenizer,
            logic_label2id=logic_label2id,
            max_len=max_seq_length,
            is_train=is_train,
            seed=seed,
            marker_temperature=float(marker_cfg["topk_temperature"]),
            dropout_nli_start=float(marker_cfg["dropout_nli_start"]) if is_train else 0.0,
            dropout_nli_end=float(marker_cfg["dropout_nli_end"]) if is_train else 0.0,
            dropout_logic_start=float(marker_cfg["dropout_logic_start"]) if is_train else 0.0,
            dropout_logic_end=float(marker_cfg["dropout_logic_end"]) if is_train else 0.0,
            dropout_unk_boost=float(marker_cfg["dropout_unk_boost"]) if is_train else 0.0,
            corrupt_prob_nli=float(marker_cfg.get("corrupt_prob_nli_start", marker_cfg.get("corrupt_prob_nli", 0.0))) if is_train else 0.0,
            corrupt_prob_logic=float(marker_cfg.get("corrupt_prob_logic_start", marker_cfg.get("corrupt_prob_logic", 0.0))) if is_train else 0.0,
            corrupt_prob_nli_end=float(marker_cfg.get("corrupt_prob_nli_end", marker_cfg.get("corrupt_prob_nli", 0.0))) if is_train else 0.0,
            corrupt_prob_logic_end=float(marker_cfg.get("corrupt_prob_logic_end", marker_cfg.get("corrupt_prob_logic", 0.0))) if is_train else 0.0,
            logic_forbidden_categories=logic_forbidden,
            mode=mode,
            enable_corruption=enable_corruption if is_train else False,
            enable_dropout=enable_dropout if is_train else False,
            include_wrong_view=include_wrong_view,
            prompt_style=str(prompt_cfg.get("style", "legacy")),
            prompt_input_style=str(prompt_cfg.get("input_style", "prompt")),
            pair_marker_placement=str(prompt_cfg.get("pair_marker_placement", "hypothesis_prefix")),
            pair_marker_prefix=str(prompt_cfg.get("pair_marker_prefix", "[M] ")),
            pair_marker_representation=str(prompt_cfg.get("pair_marker_representation", "raw_text")),
            pair_category_special_template=str(prompt_cfg.get("pair_category_special_template", "[REL_{cat}]")),
            pair_distribution_temperature=prompt_cfg.get("pair_distribution_temperature"),
            train_sampling_mode=train_sampling_mode,
            marker_source=str(marker_cfg.get("source", "predicted")),
            confidence_gating_enabled=confidence_gating_enabled,
            confidence_gating_apply_train=confidence_gating_apply_train,
            confidence_gating_apply_eval=confidence_gating_apply_eval,
            confidence_gating_temperature=confidence_gating_temperature,
            confidence_gating_min_top1_prob=confidence_gating_min_top1_prob,
            confidence_gating_min_top1_gap=confidence_gating_min_top1_gap,
            strong_wrong_nli=strong_wrong_nli,
            wrong_nli_exclude_topk=wrong_nli_exclude_topk,
            wrong_nli_exclude_same_category=wrong_nli_exclude_same_category,
        )

    train_nli_ds = make_ds(train_nli_rows, is_train=True, mode="train")
    train_logic_ds = make_ds(train_logic_rows, is_train=True, mode="train")
    dev_no_k_ds = make_ds(dev_k_rows, is_train=False, mode="eval_no_marker")
    dev_no_a_ds = make_ds(dev_a_rows, is_train=False, mode="eval_no_marker")
    dev_with_k_ds = make_ds(dev_k_rows, is_train=False, mode="eval_with_marker")
    dev_with_a_ds = make_ds(dev_a_rows, is_train=False, mode="eval_with_marker")

    enable_wrong = bool(eval_cfg.get("enable_wrong_marker_eval", True))
    if enable_wrong:
        dev_wrong_k_ds = make_ds(dev_k_rows, is_train=False, mode="eval_wrong_marker")
        dev_wrong_a_ds = make_ds(dev_a_rows, is_train=False, mode="eval_wrong_marker")

    selection_cfg = dict(cfg.get("selection_weights", {}) or {})
    selection_cfg.update(exp.get("selection_weights", {}) or {})
    selection_weight_no = float(selection_cfg.get("no", 0.5))
    selection_weight_with = float(selection_cfg.get("with", 0.3))
    selection_weight_wrong = float(selection_cfg.get("wrong", 0.2))
    selection_weight_sum = selection_weight_no + selection_weight_with + selection_weight_wrong
    if selection_weight_sum <= 0:
        selection_weight_no, selection_weight_with, selection_weight_wrong = 0.5, 0.3, 0.2
        selection_weight_sum = 1.0
    selection_weight_no /= selection_weight_sum
    selection_weight_with /= selection_weight_sum
    selection_weight_wrong /= selection_weight_sum

    collator = PadCollator(tokenizer, pad_to_multiple_of=8)
    bs = int(train_cfg["batch_size"])
    ratio_cfg = train_cfg.get("task_ratio", {}) or {}
    ratio_nli = int(ratio_cfg.get("nli", 3))
    ratio_logic = int(ratio_cfg.get("logic", 1))
    ratio_sum = max(1, ratio_nli + ratio_logic)

    train_nli_loader = DataLoader(train_nli_ds, batch_size=bs, shuffle=True, collate_fn=collator)
    train_logic_loader = DataLoader(train_logic_ds, batch_size=bs, shuffle=True, collate_fn=collator)
    nli_inf = InfiniteLoader(train_nli_loader)
    logic_inf = InfiniteLoader(train_logic_loader)

    dev_no_k_loader = DataLoader(dev_no_k_ds, batch_size=bs, shuffle=False, collate_fn=collator)
    dev_no_a_loader = DataLoader(dev_no_a_ds, batch_size=bs, shuffle=False, collate_fn=collator)
    dev_with_k_loader = DataLoader(dev_with_k_ds, batch_size=bs, shuffle=False, collate_fn=collator)
    dev_with_a_loader = DataLoader(dev_with_a_ds, batch_size=bs, shuffle=False, collate_fn=collator)
    if enable_wrong:
        dev_wrong_k_loader = DataLoader(dev_wrong_k_ds, batch_size=bs, shuffle=False, collate_fn=collator)
        dev_wrong_a_loader = DataLoader(dev_wrong_a_ds, batch_size=bs, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    params = [p for p in model.parameters() if p.requires_grad]

    epochs = int(train_cfg["epochs"])
    grad_accum = int(train_cfg["grad_accum_steps"])
    nli_batches = max(1, len(train_nli_loader))
    microsteps_per_epoch = int(math.ceil(nli_batches * (ratio_sum / max(1, ratio_nli))))
    optim_steps_per_epoch = int(math.ceil(microsteps_per_epoch / max(1, grad_accum)))
    total_optim_steps = optim_steps_per_epoch * epochs

    warmup_steps = int(total_optim_steps * float(train_cfg["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    log_every = int(train_cfg.get("log_every_steps", 25))
    eval_every = int(train_cfg.get("eval_every_steps", 500))

    patience = int(train_cfg.get("early_stop_patience", 10))
    min_delta = float(train_cfg.get("early_stop_min_delta", 0.001))
    default_early_start = max(warmup_steps, int(total_optim_steps * 0.10))
    early_stop_start_optim_step = int(train_cfg.get("early_stop_start_optim_step", default_early_start))
    no_improve_count = 0
    early_stopped = False

    lambda_kl = float(loss_cfg.get("lambda_kl", 0.0)) if enable_kl else 0.0
    gamma_mrel = float(loss_cfg.get("gamma_mrel", 0.0)) if enable_mrel else 0.0
    beta_supcon = float(loss_cfg.get("beta_supcon", 0.0)) if enable_supcon else 0.0
    supcon_T = float(loss_cfg.get("supcon_temperature", 0.1))
    forbidden_neg_weight = float(loss_cfg.get("supcon_forbidden_neg_weight", 2.0))
    w_nli = float(loss_cfg.get("task_weight_nli", 1.0))
    w_logic = float(loss_cfg.get("task_weight_logic", 1.0))
    alpha_v2_ce = float(loss_cfg.get("alpha_v2_ce", 0.0))
    v2_ce_only_clean = bool(loss_cfg.get("v2_ce_only_clean", True))
    margin_cfg = dict(loss_cfg.get("with_wrong_margin", {}) or {})
    with_wrong_margin_weight = float(margin_cfg.get("weight", loss_cfg.get("with_wrong_margin_weight", 0.0)))
    with_wrong_margin_value = float(margin_cfg.get("margin", loss_cfg.get("with_wrong_margin_value", 0.2)))
    with_wrong_margin_only_clean = bool(
        margin_cfg.get("only_clean", loss_cfg.get("with_wrong_margin_only_clean", True))
    )
    compat_cfg = dict(loss_cfg.get("compatibility", {}) or {})
    lambda_compat = float(compat_cfg.get("weight", loss_cfg.get("lambda_compat", 0.0)))
    compat_positive_weight = float(compat_cfg.get("positive_weight", loss_cfg.get("compat_positive_weight", 1.0)))
    compat_use_with_view = bool(compat_cfg.get("use_with_view", True))
    compat_use_wrong_view = bool(compat_cfg.get("use_wrong_view", True))

    qcfg = train_cfg.get("supcon_queue", {}) or {}
    queue_size_nli = int(qcfg.get("nli_size", 1024))
    queue_size_logic = int(qcfg.get("logic_size", 2048))
    queue_warmup_steps = int(qcfg.get("warmup_steps", 0))
    nli_bank = MemoryBank(queue_size_nli, hidden, device=device, dtype=head_dtype)
    logic_bank = MemoryBank(queue_size_logic, hidden, device=device, dtype=head_dtype)

    logic_class_weight_mode = str(loss_cfg.get("logic_class_weight_mode", "inverse_sqrt"))
    logic_class_weight_smoothing = float(loss_cfg.get("logic_class_weight_smoothing", 1.0))
    logic_class_weight_overrides = dict(loss_cfg.get("logic_class_weight_overrides", {}) or {})
    logic_focal_gamma = float(loss_cfg.get("logic_focal_gamma", 0.0))
    logic_class_weights = _build_logic_class_weights(
        train_logic_rows,
        logic_label2id,
        logic_class_weight_mode,
        logic_class_weight_smoothing,
        device,
        logic_class_weight_overrides,
    )

    ce_nli = nn.CrossEntropyLoss(ignore_index=-100)
    ce_logic = nn.CrossEntropyLoss(ignore_index=-100, weight=logic_class_weights)
    ce_mrel = nn.CrossEntropyLoss()

    best_score = -1.0
    global_step = 0
    optim_step = 0
    pbar = tqdm(total=total_optim_steps, desc=f"Train[{exp_name}](optim_steps)", leave=True)

    selection_cfg_full = dict(cfg.get("selection", {}) or {})
    selection_cfg_full.update(exp.get("selection", {}) or {})
    selection_metric = str(selection_cfg_full.get("metric", "with_only")).lower()
    selection_base_metric = str(selection_cfg_full.get("base_metric", "macro_f1")).lower()
    selection_gap_weight = float(selection_cfg_full.get("gap_weight", 0.0))
    if selection_metric == "f1_with_gap" and "gap_weight" not in selection_cfg_full:
        selection_gap_weight = 0.5
    if selection_base_metric in ("f1", "macrof1", "macro_f1"):
        selection_base_metric = "macro_f1"
    elif selection_base_metric in ("precision", "macro_precision"):
        selection_base_metric = "macro_precision"
    elif selection_base_metric in ("recall", "macro_recall"):
        selection_base_metric = "macro_recall"
    else:
        selection_base_metric = "acc"

    def run_one_eval_and_free(
        tag: str,
        loader: DataLoader,
        task: str,
        view: str,
        num_classes: int,
        class_names: List[str],
        global_step_: int,
    ) -> Dict[str, Any]:
        c = eval_collect(model, loader, device, task, view, num_classes, use_autocast, amp_dtype)
        writer.add_scalar(f"dev/{tag}_acc", c["acc"], global_step_)
        writer.add_scalar(f"dev/{tag}_macro_precision", c["macro_precision"], global_step_)
        writer.add_scalar(f"dev/{tag}_macro_recall", c["macro_recall"], global_step_)
        writer.add_scalar(f"dev/{tag}_macro_f1", c["macro_f1"], global_step_)
        save_confusion_and_report(eval_dir, tag, num_classes, class_names, c)
        cuda_cleanup()
        return dict(c)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for _epoch in range(1, epochs + 1):
        for _ in range(microsteps_per_epoch):
            global_step += 1
            progress = min(1.0, float(optim_step) / max(total_optim_steps, 1))
            train_nli_ds.set_progress(progress)
            train_logic_ds.set_progress(progress)

            use_nli_batch = ((global_step - 1) % ratio_sum) < ratio_nli
            batch = nli_inf.next() if use_nli_batch else logic_inf.next()

            v1_ids = batch.v1_input_ids.to(device)
            v1_mask = batch.v1_attention_mask.to(device)
            v1_type_ids = batch.v1_token_type_ids.to(device)
            v1_cat_dist = batch.v1_marker_cat_dist.to(device)
            v2_ids = batch.v2_input_ids.to(device)
            v2_mask = batch.v2_attention_mask.to(device)
            v2_type_ids = batch.v2_token_type_ids.to(device)
            v2_cat_dist = batch.v2_marker_cat_dist.to(device)
            v2_marker_start = batch.v2_marker_start.to(device)
            v2_marker_end = batch.v2_marker_end.to(device)
            v_wrong_ids = batch.v_wrong_input_ids.to(device)
            v_wrong_mask = batch.v_wrong_attention_mask.to(device)
            v_wrong_type_ids = batch.v_wrong_token_type_ids.to(device)
            v_wrong_cat_dist = batch.v_wrong_marker_cat_dist.to(device)
            v_wrong_marker_start = batch.v_wrong_marker_start.to(device)
            v_wrong_marker_end = batch.v_wrong_marker_end.to(device)
            y_nli = batch.y_nli.to(device)
            y_logic = batch.y_logic.to(device)
            v2_corrupt = batch.v2_is_corrupt.to(device)
            v2_forbidden = batch.v2_marker_is_forbidden.to(device)
            v2_cat_id = batch.v2_marker_cat_id.to(device)
            v_wrong_available = batch.v_wrong_available.to(device)
            v2_is_clean = (v2_corrupt == 0) & (v2_cat_id != CAT2ID["NONE"])
            need_aux_outputs = lambda_compat > 0.0

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast and device.type == "cuda"):
                model_out_v1 = model(
                    v1_ids,
                    v1_mask,
                    token_type_ids=v1_type_ids,
                    marker_category_distribution=v1_cat_dist,
                    return_aux=need_aux_outputs,
                )
                if need_aux_outputs:
                    rep1, logit1_nli, logit1_logic, _, aux_v1 = model_out_v1
                else:
                    rep1, logit1_nli, logit1_logic, _ = model_out_v1
                    aux_v1 = {}
                model_out_v2 = model(
                    v2_ids,
                    v2_mask,
                    token_type_ids=v2_type_ids,
                    marker_start=v2_marker_start,
                    marker_end=v2_marker_end,
                    marker_category_distribution=v2_cat_dist,
                    return_aux=need_aux_outputs,
                )
                if need_aux_outputs:
                    rep2, logit2_nli, logit2_logic, logit2_mrel, aux_v2 = model_out_v2
                else:
                    rep2, logit2_nli, logit2_logic, logit2_mrel = model_out_v2
                    aux_v2 = {}
                logit_wrong_nli = None
                logit_wrong_logic = None
                aux_wrong = {}
                if with_wrong_margin_weight > 0:
                    model_out_wrong = model(
                        v_wrong_ids,
                        v_wrong_mask,
                        token_type_ids=v_wrong_type_ids,
                        marker_start=v_wrong_marker_start,
                        marker_end=v_wrong_marker_end,
                        marker_category_distribution=v_wrong_cat_dist,
                        return_aux=need_aux_outputs,
                    )
                    if need_aux_outputs:
                        _, logit_wrong_nli, logit_wrong_logic, _, aux_wrong = model_out_wrong
                    else:
                        _, logit_wrong_nli, logit_wrong_logic, _ = model_out_wrong
                elif need_aux_outputs and compat_use_wrong_view:
                    model_out_wrong = model(
                        v_wrong_ids,
                        v_wrong_mask,
                        token_type_ids=v_wrong_type_ids,
                        marker_start=v_wrong_marker_start,
                        marker_end=v_wrong_marker_end,
                        marker_category_distribution=v_wrong_cat_dist,
                        return_aux=True,
                    )
                    _, _, _, _, aux_wrong = model_out_wrong

                loss_nli_v1 = masked_ce(ce_nli, logit1_nli, y_nli)
                if logic_focal_gamma > 0:
                    loss_logic_v1 = masked_focal_ce(
                        logit1_logic,
                        y_logic,
                        gamma=logic_focal_gamma,
                        weight=logic_class_weights,
                    )
                else:
                    loss_logic_v1 = masked_ce(ce_logic, logit1_logic, y_logic)
                loss_nli_v2 = rep1.new_tensor(0.0)
                loss_logic_v2 = rep1.new_tensor(0.0)
                if alpha_v2_ce > 0:
                    if v2_ce_only_clean:
                        loss_nli_v2 = masked_ce_with_mask(ce_nli, logit2_nli, y_nli, v2_is_clean)
                        if logic_focal_gamma > 0:
                            loss_logic_v2 = masked_focal_ce(
                                logit2_logic,
                                y_logic,
                                gamma=logic_focal_gamma,
                                weight=logic_class_weights,
                                extra_mask=v2_is_clean,
                            )
                        else:
                            loss_logic_v2 = masked_ce_with_mask(ce_logic, logit2_logic, y_logic, v2_is_clean)
                    else:
                        loss_nli_v2 = masked_ce(ce_nli, logit2_nli, y_nli)
                        if logic_focal_gamma > 0:
                            loss_logic_v2 = masked_focal_ce(
                                logit2_logic,
                                y_logic,
                                gamma=logic_focal_gamma,
                                weight=logic_class_weights,
                            )
                        else:
                            loss_logic_v2 = masked_ce(ce_logic, logit2_logic, y_logic)

                loss_cls_v1 = (w_nli * loss_nli_v1) + (w_logic * loss_logic_v1)
                loss_cls_v2 = (w_nli * loss_nli_v2) + (w_logic * loss_logic_v2)
                loss_cls = loss_cls_v1 + (alpha_v2_ce * loss_cls_v2)
                loss_margin_nli = rep1.new_tensor(0.0)
                loss_margin_logic = rep1.new_tensor(0.0)
                loss_margin = rep1.new_tensor(0.0)
                if with_wrong_margin_weight > 0 and logit_wrong_nli is not None and logit_wrong_logic is not None:
                    margin_mask = v_wrong_available == 1
                    if with_wrong_margin_only_clean:
                        margin_mask = margin_mask & v2_is_clean
                    loss_margin_nli = masked_margin_between_views(
                        logit2_nli,
                        logit_wrong_nli,
                        y_nli,
                        with_wrong_margin_value,
                        margin_mask,
                    )
                    loss_margin_logic = masked_margin_between_views(
                        logit2_logic,
                        logit_wrong_logic,
                        y_logic,
                        with_wrong_margin_value,
                        margin_mask,
                    )
                    loss_margin = (w_nli * loss_margin_nli) + (w_logic * loss_margin_logic)

                loss_compat_v2 = rep1.new_tensor(0.0)
                loss_compat_wrong = rep1.new_tensor(0.0)
                loss_compat = rep1.new_tensor(0.0)
                if lambda_compat > 0.0 and need_aux_outputs:
                    compat_target_v2, compat_mask_v2 = _build_marker_compat_targets(
                        y_nli,
                        y_logic,
                        v2_cat_id,
                        id2logic,
                    )
                    compat_target_wrong, compat_mask_wrong = _build_marker_compat_targets(
                        y_nli,
                        y_logic,
                        batch.v_wrong_marker_cat_id.to(device),
                        id2logic,
                    )
                    compat_nli_v2 = aux_v2.get("compat_nli_logits")
                    compat_logic_v2 = aux_v2.get("compat_logic_logits")
                    compat_nli_wrong = aux_wrong.get("compat_nli_logits")
                    compat_logic_wrong = aux_wrong.get("compat_logic_logits")

                    if compat_use_with_view and compat_nli_v2 is not None and compat_logic_v2 is not None:
                        mask_nli_v2 = compat_mask_v2 & (y_nli != -100)
                        mask_logic_v2 = compat_mask_v2 & (y_logic != -100)
                        loss_compat_v2 = (
                            w_nli * masked_bce_with_logits(
                                compat_nli_v2,
                                compat_target_v2,
                                mask_nli_v2,
                                pos_weight=compat_positive_weight,
                            )
                            + w_logic * masked_bce_with_logits(
                                compat_logic_v2,
                                compat_target_v2,
                                mask_logic_v2,
                                pos_weight=compat_positive_weight,
                            )
                        )

                    if compat_use_wrong_view and compat_nli_wrong is not None and compat_logic_wrong is not None:
                        mask_wrong_base = compat_mask_wrong & (v_wrong_available == 1)
                        mask_nli_wrong = mask_wrong_base & (y_nli != -100)
                        mask_logic_wrong = mask_wrong_base & (y_logic != -100)
                        loss_compat_wrong = (
                            w_nli * masked_bce_with_logits(
                                compat_nli_wrong,
                                compat_target_wrong,
                                mask_nli_wrong,
                                pos_weight=compat_positive_weight,
                            )
                            + w_logic * masked_bce_with_logits(
                                compat_logic_wrong,
                                compat_target_wrong,
                                mask_logic_wrong,
                                pos_weight=compat_positive_weight,
                            )
                        )
                    loss_compat = loss_compat_v2 + loss_compat_wrong

                loss_kl = rep1.new_tensor(0.0)
                if enable_kl:
                    mn = y_nli != -100
                    ml = y_logic != -100
                    if mn.any():
                        loss_kl = loss_kl + kl_consistency_loss(logit1_nli[mn], logit2_nli[mn])
                    if ml.any():
                        loss_kl = loss_kl + kl_consistency_loss(logit1_logic[ml], logit2_logic[ml])

                loss_mrel = ce_mrel(logit2_mrel, v2_corrupt) if enable_mrel else rep1.new_tensor(0.0)
                loss_sup = rep1.new_tensor(0.0)
                if enable_supcon and beta_supcon > 0:
                    use_nli_queue = (optim_step >= queue_warmup_steps) and nli_bank.is_ready()
                    loss_sup_nli = supcon_loss_with_queue(
                        rep1,
                        y_nli,
                        supcon_T,
                        nli_bank.reps if use_nli_queue else None,
                        nli_bank.labels if use_nli_queue else None,
                    )

                    batch_size = rep1.size(0)
                    pos_override = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=rep1.device)
                    neg_w_in = torch.ones((batch_size, batch_size), dtype=torch.float, device=rep1.device)
                    logic_valid = y_logic != -100
                    good_logic = logic_valid & (v2_forbidden == 0)
                    for i in range(batch_size):
                        if not logic_valid[i]:
                            continue
                        for j in range(batch_size):
                            if i == j or not logic_valid[j]:
                                continue
                            if (y_logic[i].item() == y_logic[j].item()) and good_logic[i] and good_logic[j]:
                                pos_override[i, j] = True

                    forb_pair_in = torch.clamp(
                        v2_forbidden.unsqueeze(1).float() + v2_forbidden.unsqueeze(0).float(),
                        0,
                        1,
                    )
                    neg_w_in = neg_w_in + forb_pair_in * (forbidden_neg_weight - 1.0)

                    use_logic_queue = (optim_step >= queue_warmup_steps) and logic_bank.is_ready()
                    loss_sup_logic = supcon_loss_with_queue(
                        reps=rep1,
                        labels=y_logic,
                        temperature=supcon_T,
                        queue_reps=logic_bank.reps if use_logic_queue else None,
                        queue_labels=logic_bank.labels if use_logic_queue else None,
                        pos_mask_override=pos_override,
                        neg_weight_inbatch=neg_w_in,
                        anchor_forbidden=v2_forbidden,
                        queue_forbidden=logic_bank.forbidden if use_logic_queue else None,
                        forbidden_neg_weight=forbidden_neg_weight,
                    )
                    loss_sup = loss_sup_nli + loss_sup_logic

                loss_total = (
                    loss_cls
                    + with_wrong_margin_weight * loss_margin
                    + lambda_compat * loss_compat
                    + lambda_kl * loss_kl
                    + gamma_mrel * loss_mrel
                    + beta_supcon * loss_sup
                )
                loss = loss_total / grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if global_step % log_every == 0:
                writer.add_scalar("train/loss_cls", float(loss_cls.detach().cpu().item()), global_step)
                writer.add_scalar("train/loss_cls_v1", float(loss_cls_v1.detach().cpu().item()), global_step)
                writer.add_scalar("train/loss_nli_v1_ce", float(loss_nli_v1.detach().cpu().item()), global_step)
                writer.add_scalar("train/loss_logic_v1_ce", float(loss_logic_v1.detach().cpu().item()), global_step)
                writer.add_scalar("train/loss_kl", float(loss_kl.detach().cpu().item()), global_step)
                writer.add_scalar("train/loss_mrel", float(loss_mrel.detach().cpu().item()), global_step)
                writer.add_scalar("train/loss_supcon", float(loss_sup.detach().cpu().item()), global_step)
                writer.add_scalar("train/loss_with_wrong_margin", float(loss_margin.detach().cpu().item()), global_step)
                writer.add_scalar("train/loss_compat", float(loss_compat.detach().cpu().item()), global_step)
                writer.add_scalar("train/loss_total", float(loss_total.detach().cpu().item()), global_step)
                writer.add_scalar("train/lr", float(get_lr(optimizer)), global_step)
                writer.add_scalar("train/progress", float(progress), global_step)
                writer.add_scalar("train/queue_nli_size", float(nli_bank.reps.size(0)), global_step)
                writer.add_scalar("train/queue_logic_size", float(logic_bank.reps.size(0)), global_step)
                writer.add_scalar("train/use_nli_batch", float(1.0 if use_nli_batch else 0.0), global_step)
                if alpha_v2_ce > 0:
                    writer.add_scalar("train/loss_cls_v2", float(loss_cls_v2.detach().cpu().item()), global_step)
                    writer.add_scalar("train/loss_nli_v2_ce", float(loss_nli_v2.detach().cpu().item()), global_step)
                    writer.add_scalar("train/loss_logic_v2_ce", float(loss_logic_v2.detach().cpu().item()), global_step)
                if with_wrong_margin_weight > 0:
                    writer.add_scalar("train/loss_with_wrong_margin_nli", float(loss_margin_nli.detach().cpu().item()), global_step)
                    writer.add_scalar("train/loss_with_wrong_margin_logic", float(loss_margin_logic.detach().cpu().item()), global_step)
                if lambda_compat > 0:
                    writer.add_scalar("train/loss_compat_v2", float(loss_compat_v2.detach().cpu().item()), global_step)
                    writer.add_scalar("train/loss_compat_wrong", float(loss_compat_wrong.detach().cpu().item()), global_step)

            if global_step % grad_accum != 0:
                continue

            optim_step += 1
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pbar.update(1)

            with torch.no_grad():
                rep1_det = rep1.detach()
                mn = y_nli != -100
                if mn.any():
                    nli_bank.enqueue(rep1_det[mn], y_nli[mn])
                ml = y_logic != -100
                if ml.any():
                    logic_bank.enqueue(rep1_det[ml], y_logic[ml], forbidden=v2_forbidden[ml])

            if eval_every > 0 and (global_step % eval_every == 0):
                del rep1, rep2, logit1_nli, logit1_logic, logit2_nli, logit2_logic, logit2_mrel
                cuda_cleanup()
                model.eval()

                r_no_k = run_one_eval_and_free("no_nli_v1", dev_no_k_loader, "nli", "v1", 3, [ID2NLI[i] for i in range(3)], global_step)
                r_no_a = run_one_eval_and_free("no_logic_v1", dev_no_a_loader, "logic", "v1", len(logic_label2id), [id2logic[i] for i in range(len(logic_label2id))], global_step)
                r_w_k = run_one_eval_and_free("with_nli_v2", dev_with_k_loader, "nli", "v2", 3, [ID2NLI[i] for i in range(3)], global_step)
                r_w_a = run_one_eval_and_free("with_logic_v2", dev_with_a_loader, "logic", "v2", len(logic_label2id), [id2logic[i] for i in range(len(logic_label2id))], global_step)

                if enable_wrong:
                    r_r_k = run_one_eval_and_free("wrong_nli_v2", dev_wrong_k_loader, "nli", "v2", 3, [ID2NLI[i] for i in range(3)], global_step)
                    r_r_a = run_one_eval_and_free("wrong_logic_v2", dev_wrong_a_loader, "logic", "v2", len(logic_label2id), [id2logic[i] for i in range(len(logic_label2id))], global_step)
                else:
                    r_r_k = _zero_result()
                    r_r_a = _zero_result()

                score_no = _pair_metric(r_no_k, r_no_a, selection_base_metric)
                score_with = _pair_metric(r_w_k, r_w_a, selection_base_metric)
                score_wrong = _pair_metric(r_r_k, r_r_a, selection_base_metric) if enable_wrong else 0.0
                score_gap = (score_with - score_wrong) if enable_wrong else 0.0

                if selection_metric == "with_only":
                    final_score = score_with
                elif selection_metric == "f1_with_gap":
                    final_score = (
                        selection_weight_no * score_no
                        + selection_weight_with * score_with
                        + selection_weight_wrong * score_wrong
                        + selection_gap_weight * score_gap
                    )
                else:
                    final_score = (
                        selection_weight_no * score_no
                        + selection_weight_with * score_with
                        + selection_weight_wrong * score_wrong
                    )
                writer.add_scalar("dev/score_no", score_no, global_step)
                writer.add_scalar("dev/score_with", score_with, global_step)
                writer.add_scalar("dev/score_wrong", score_wrong, global_step)
                writer.add_scalar("dev/score_gap_with_minus_wrong", score_gap, global_step)
                writer.add_scalar("dev/final_score", final_score, global_step)

                save_transition_analysis(eval_dir, "nli", dev_k_rows, "nli", r_no_k, r_w_k, r_r_k if enable_wrong else None, ID2NLI)
                save_transition_analysis(eval_dir, "logic", dev_a_rows, "logic", r_no_a, r_w_a, r_r_a if enable_wrong else None, id2logic)

                if final_score > (best_score + min_delta):
                    best_score = final_score
                    no_improve_count = 0
                    save_best(
                        exp_dir,
                        model,
                        tokenizer,
                        logic_label2id,
                        best_score,
                        {
                            "cfg": cfg,
                            "exp": exp,
                            "run_dir": run_dir,
                            "used_yaml": used_yaml_path,
                            "used_script": train_script_path,
                        },
                    )
                elif optim_step >= early_stop_start_optim_step:
                    no_improve_count += 1
                    if patience > 0 and no_improve_count >= patience:
                        early_stopped = True

                cuda_cleanup()
                model.train()

            if early_stopped:
                break

        if early_stopped:
            break

    pbar.close()
    writer.close()

    return {
        "exp_name": exp_name,
        "best_score": best_score,
        "enable_kl": enable_kl,
        "enable_dropout": enable_dropout,
        "enable_corruption": enable_corruption,
        "enable_mrel": enable_mrel,
        "enable_supcon": enable_supcon,
        "alpha_v2_ce": alpha_v2_ce,
        "v2_ce_only_clean": v2_ce_only_clean,
        "selection_metric": selection_metric,
        "selection_base_metric": selection_base_metric,
        "selection_gap_weight": selection_gap_weight,
        "with_wrong_margin_weight": with_wrong_margin_weight,
        "with_wrong_margin_value": with_wrong_margin_value,
        "with_wrong_margin_only_clean": with_wrong_margin_only_clean,
        "logic_class_weight_mode": logic_class_weight_mode,
        "logic_class_weight_smoothing": logic_class_weight_smoothing,
        "logic_class_weight_overrides": logic_class_weight_overrides,
        "logic_focal_gamma": logic_focal_gamma,
        "lambda_compat": lambda_compat,
        "compat_positive_weight": compat_positive_weight,
        "compat_use_with_view": compat_use_with_view,
        "compat_use_wrong_view": compat_use_wrong_view,
        "use_marker_compatibility_head": bool(model_cfg.get("use_marker_compatibility_head", False)),
        "use_compatibility_for_delta": bool(model_cfg.get("use_compatibility_for_delta", False)),
        "train_sampling_mode": train_sampling_mode,
        "use_learned_marker_gate": bool(model_cfg.get("use_learned_marker_gate", False)),
        "use_marker_aware_head": bool(model_cfg.get("use_marker_aware_head", True)),
        "prompt_style": str(prompt_cfg.get("style", "legacy")),
        "marker_source": str(marker_cfg.get("source", "predicted")),
        "exp_dir": exp_dir,
        "tb_dir": tb_dir,
        "early_stopped": early_stopped,
        "no_improve_count": no_improve_count,
        "patience": patience,
        "min_delta": min_delta,
        "early_stop_start_optim_step": early_stop_start_optim_step,
        "mixed_precision": mp,
        "quant_type": str(model_cfg.get("quant_type", "8bit")),
        "queue_nli_size": int(nli_bank.reps.size(0)),
        "queue_logic_size": int(logic_bank.reps.size(0)),
        "ratio_nli": ratio_nli,
        "ratio_logic": ratio_logic,
        "w_nli": w_nli,
        "w_logic": w_logic,
        "selection_weight_no": selection_weight_no,
        "selection_weight_with": selection_weight_with,
        "selection_weight_wrong": selection_weight_wrong,
        "microsteps_per_epoch": microsteps_per_epoch,
        "optim_steps_per_epoch": optim_steps_per_epoch,
        "strong_wrong_nli": strong_wrong_nli,
        "wrong_nli_exclude_topk": wrong_nli_exclude_topk,
        "wrong_nli_exclude_same_category": wrong_nli_exclude_same_category,
        "confidence_gating_enabled": confidence_gating_enabled,
        "confidence_gating_apply_train": confidence_gating_apply_train,
        "confidence_gating_apply_eval": confidence_gating_apply_eval,
        "confidence_gating_temperature": confidence_gating_temperature,
        "confidence_gating_min_top1_prob": confidence_gating_min_top1_prob,
        "confidence_gating_min_top1_gap": confidence_gating_min_top1_gap,
    }
