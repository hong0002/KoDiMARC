import os
import json
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kodimarc.common.io import ensure_dir, write_csv, write_json
from kodimarc.common.markers import CAT2ID, ID2CAT
from kodimarc.step2.metrics import confusion_matrix, macro_prf
from kodimarc.step2.model import Step2EncoderClassifier

# ============================================================
# 6) Eval
# ============================================================
@torch.no_grad()
def eval_collect(
    model: Step2EncoderClassifier,
    loader: DataLoader,
    device: torch.device,
    task: str,
    view: str,
    num_classes: int,
    use_autocast: bool,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, Any]:
    model.eval()
    all_row_id, all_y, all_pred, all_cat, all_forb = [], [], [], [], []
    all_top1_marker, all_v2_marker, all_topk_markers, all_topk_scores = [], [], [], []

    for batch in tqdm(loader, desc=f"Collect({task},{view})", leave=False):
        row_ids_full = batch.row_ids.to(device)

        if view == "v1":
            ids = batch.v1_input_ids.to(device)
            mask = batch.v1_attention_mask.to(device)
            token_type_ids = batch.v1_token_type_ids.to(device)
            marker_category_distribution = batch.v1_marker_cat_dist.to(device)
            marker_start = None
            marker_end = None
        else:
            ids = batch.v2_input_ids.to(device)
            mask = batch.v2_attention_mask.to(device)
            token_type_ids = batch.v2_token_type_ids.to(device)
            marker_category_distribution = batch.v2_marker_cat_dist.to(device)
            marker_start = batch.v2_marker_start.to(device)
            marker_end = batch.v2_marker_end.to(device)

        if use_autocast and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, logits_nli, logits_logic, _ = model(
                    ids,
                    mask,
                    token_type_ids=token_type_ids,
                    marker_start=marker_start,
                    marker_end=marker_end,
                    marker_category_distribution=marker_category_distribution,
                )
        else:
            _, logits_nli, logits_logic, _ = model(
                ids,
                mask,
                token_type_ids=token_type_ids,
                marker_start=marker_start,
                marker_end=marker_end,
                marker_category_distribution=marker_category_distribution,
            )

        if task == "nli":
            y_full = batch.y_nli.to(device)
            m = (y_full != -100)
            if m.sum() == 0:
                del row_ids_full, ids, mask, logits_nli, logits_logic, y_full
                continue
            logits = logits_nli[m]
            y = y_full[m]
        else:
            y_full = batch.y_logic.to(device)
            m = (y_full != -100)
            if m.sum() == 0:
                del row_ids_full, ids, mask, logits_nli, logits_logic, y_full
                continue
            logits = logits_logic[m]
            y = y_full[m]

        pred = logits.argmax(dim=-1)
        row_ids = row_ids_full[m]

        cat = batch.v2_marker_cat_id.to(device)[m] if view == "v2" else torch.full_like(y, CAT2ID["NONE"])
        forb = batch.v2_marker_is_forbidden.to(device)[m] if view == "v2" else torch.zeros_like(y)

        all_row_id.append(row_ids.detach().cpu())
        all_y.append(y.detach().cpu())
        all_pred.append(pred.detach().cpu())
        all_cat.append(cat.detach().cpu())
        all_forb.append(forb.detach().cpu())

        mask_idx = torch.nonzero(m.detach().cpu()).squeeze(1).tolist()
        all_top1_marker.extend([batch.top1_marker_text[i] for i in mask_idx])
        all_v2_marker.extend([batch.v2_marker_text[i] for i in mask_idx])
        all_topk_markers.extend([batch.topk_markers_text[i] for i in mask_idx])
        all_topk_scores.extend([batch.topk_scores_text[i] for i in mask_idx])

        del row_ids_full, row_ids, ids, mask, logits_nli, logits_logic, logits, y_full, y, pred, cat, forb

    if not all_y:
        return {
            "acc": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
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

    row_id = torch.cat(all_row_id, dim=0).tolist()
    y = torch.cat(all_y, dim=0).tolist()
    pred = torch.cat(all_pred, dim=0).tolist()
    cat = torch.cat(all_cat, dim=0).tolist()
    forb = torch.cat(all_forb, dim=0).tolist()

    acc = sum(int(a == b) for a, b in zip(y, pred)) / max(len(y), 1)
    mp, mr, mf = macro_prf(num_classes, y, pred)

    return {
        "acc": float(acc),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "row_id": row_id,
        "y": y,
        "pred": pred,
        "cat": cat,
        "forb": forb,
        "top1_marker_text": all_top1_marker,
        "v2_marker_text": all_v2_marker,
        "topk_markers_text": all_topk_markers,
        "topk_scores_text": all_topk_scores,
    }

def save_confusion_and_report(
    out_dir: str,
    prefix: str,
    num_classes: int,
    class_names: List[str],
    collected: Dict[str, Any],
):
    ensure_dir(out_dir)

    y = collected["y"]
    pred = collected["pred"]
    acc = collected["acc"]
    mp = collected.get("macro_precision", 0.0)
    mr = collected.get("macro_recall", 0.0)
    mf = collected.get("macro_f1", 0.0)

    cm = confusion_matrix(num_classes, y, pred)

    cm_csv = []
    cm_csv.append(["GT\\PRED"] + class_names)
    for i, row in enumerate(cm):
        cm_csv.append([class_names[i]] + row)
    write_csv(os.path.join(out_dir, f"{prefix}_confusion.csv"), cm_csv)

    write_json(os.path.join(out_dir, f"{prefix}_confusion.json"), {
        "acc": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf,
        "class_names": class_names,
        "confusion": cm,
    })

    cat_list = collected.get("cat", [])
    if cat_list:
        by_cat = {}
        for yt, pr, c in zip(y, pred, cat_list):
            cat_name = ID2CAT.get(int(c), "UNK")
            if cat_name not in by_cat:
                by_cat[cat_name] = {"total": 0, "correct": 0}
            by_cat[cat_name]["total"] += 1
            by_cat[cat_name]["correct"] += int(yt == pr)

        rows = []
        for cat_name, d in sorted(by_cat.items(), key=lambda x: x[0]):
            total = d["total"]
            correct = d["correct"]
            acc_c = (correct / total) if total > 0 else 0.0
            rows.append([cat_name, total, correct, acc_c])

        write_csv(
            os.path.join(out_dir, f"{prefix}_marker_category_report.csv"),
            rows,
            header=["marker_category", "total", "correct", "acc"]
        )

def save_transition_analysis(
    out_dir: str,
    prefix: str,
    rows_ref: List[Dict[str, Any]],
    task: str,
    no_col: Dict[str, Any],
    with_col: Dict[str, Any],
    wrong_col: Optional[Dict[str, Any]],
    label_name_map: Dict[int, str],
):
    ensure_dir(out_dir)

    no_map = {
        rid: (y, p)
        for rid, y, p in zip(no_col["row_id"], no_col["y"], no_col["pred"])
    }

    with_map = {
        rid: (y, p, c, t1, vm, tk, ts)
        for rid, y, p, c, t1, vm, tk, ts in zip(
            with_col["row_id"],
            with_col["y"],
            with_col["pred"],
            with_col["cat"],
            with_col["top1_marker_text"],
            with_col["v2_marker_text"],
            with_col["topk_markers_text"],
            with_col["topk_scores_text"],
        )
    }

    wrong_map = {}
    if wrong_col is not None:
        wrong_map = {
            rid: (y, p, c, t1, vm, tk, ts)
            for rid, y, p, c, t1, vm, tk, ts in zip(
                wrong_col["row_id"],
                wrong_col["y"],
                wrong_col["pred"],
                wrong_col["cat"],
                wrong_col["top1_marker_text"],
                wrong_col["v2_marker_text"],
                wrong_col["topk_markers_text"],
                wrong_col["topk_scores_text"],
            )
        }

    detail_rows = []
    n_no_to_with = 0
    n_with_to_wrong_drop = 0
    n_wrong_better_than_with = 0
    n_all_same = 0

    common_ids = sorted(
        set(no_map.keys()) &
        set(with_map.keys()) &
        (set(wrong_map.keys()) if wrong_col is not None else set(with_map.keys()))
    )

    for rid in common_ids:
        base_row = rows_ref[rid]
        y_no, p_no = no_map[rid]
        y_with, p_with, cat_with, top1_with, v2m_with, topk_with, topk_scores_with = with_map[rid]

        if wrong_col is not None:
            y_wrong, p_wrong, cat_wrong, top1_wrong, v2m_wrong, topk_wrong, topk_scores_wrong = wrong_map[rid]
        else:
            y_wrong, p_wrong, cat_wrong = y_with, p_with, CAT2ID["NONE"]
            top1_wrong, v2m_wrong, topk_wrong, topk_scores_wrong = "", "", [], []

        c_no = int(y_no == p_no)
        c_with = int(y_with == p_with)
        c_wrong = int(y_wrong == p_wrong)

        if (c_no == 0) and (c_with == 1):
            n_no_to_with += 1
        if (c_with == 1) and (c_wrong == 0):
            n_with_to_wrong_drop += 1
        if c_wrong > c_with:
            n_wrong_better_than_with += 1
        if (p_no == p_with) and (p_with == p_wrong):
            n_all_same += 1

        detail_rows.append([
            rid,
            base_row.get("premise", ""),
            base_row.get("hypothesis", ""),
            label_name_map.get(y_no, str(y_no)),
            label_name_map.get(p_no, str(p_no)),
            label_name_map.get(p_with, str(p_with)),
            label_name_map.get(p_wrong, str(p_wrong)),
            c_no,
            c_with,
            c_wrong,
            ID2CAT.get(int(cat_with), "UNK"),
            ID2CAT.get(int(cat_wrong), "UNK"),
            top1_with,
            v2m_with,
            " | ".join(topk_with) if topk_with else "",
            json.dumps(topk_scores_with, ensure_ascii=False),
            top1_wrong,
            v2m_wrong,
            " | ".join(topk_wrong) if topk_wrong else "",
            json.dumps(topk_scores_wrong, ensure_ascii=False),
        ])

    write_csv(
        os.path.join(out_dir, f"{prefix}_transition_detail.csv"),
        detail_rows,
        header=[
            "row_id", "premise", "hypothesis", "gold",
            "pred_no", "pred_with", "pred_wrong",
            "correct_no", "correct_with", "correct_wrong",
            "with_marker_cat", "wrong_marker_cat",
            "with_top1_marker", "with_used_marker", "with_topk_markers", "with_topk_scores",
            "wrong_top1_marker", "wrong_used_marker", "wrong_topk_markers", "wrong_topk_scores",
        ]
    )

    summary = {
        "task": task,
        "num_common_samples": len(common_ids),
        "no_to_with_improve_count": n_no_to_with,
        "with_to_wrong_drop_count": n_with_to_wrong_drop,
        "wrong_better_than_with_count": n_wrong_better_than_with,
        "all_same_prediction_count": n_all_same,
        "no_to_with_improve_ratio": (n_no_to_with / len(common_ids)) if common_ids else 0.0,
        "with_to_wrong_drop_ratio": (n_with_to_wrong_drop / len(common_ids)) if common_ids else 0.0,
        "wrong_better_than_with_ratio": (n_wrong_better_than_with / len(common_ids)) if common_ids else 0.0,
        "all_same_prediction_ratio": (n_all_same / len(common_ids)) if common_ids else 0.0,
    }
    write_json(os.path.join(out_dir, f"{prefix}_transition_summary.json"), summary)
