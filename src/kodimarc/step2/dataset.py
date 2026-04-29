import math
import torch
import random
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import Dataset
from dataclasses import dataclass

from kodimarc.common.markers import (
    ALL_LEX_MARKERS,
    ANTI_CATEGORY_MAP,
    CANONICAL_MARKER_BY_CAT,
    CAT2ID,
    LEX_CAT2MARKERS,
    LEX_MARKER2CAT,
    LOGIC_KO_TO_CAT,
    NLI_LABEL2ID,
    NLI_TO_CAT,
    NLI_WRONG_CATEGORY_MAP,
    marker_to_relation_special_token,
    marker_category,
)
from kodimarc.common.utils import linear_schedule, softmax_sample
from kodimarc.step2.prompt import build_prompt

# ============================================================
# 3) Dataset
# ============================================================
@dataclass
class Step2Batch:
    row_ids: torch.Tensor
    v1_input_ids: torch.Tensor
    v1_attention_mask: torch.Tensor
    v1_token_type_ids: torch.Tensor
    v1_marker_cat_dist: torch.Tensor
    v2_input_ids: torch.Tensor
    v2_attention_mask: torch.Tensor
    v2_token_type_ids: torch.Tensor
    v2_marker_cat_dist: torch.Tensor
    v2_marker_start: torch.Tensor
    v2_marker_end: torch.Tensor
    v_wrong_input_ids: torch.Tensor
    v_wrong_attention_mask: torch.Tensor
    v_wrong_token_type_ids: torch.Tensor
    v_wrong_marker_cat_dist: torch.Tensor
    v_wrong_marker_start: torch.Tensor
    v_wrong_marker_end: torch.Tensor
    y_nli: torch.Tensor
    y_logic: torch.Tensor
    v2_marker_is_forbidden: torch.Tensor
    v2_is_corrupt: torch.Tensor
    v2_marker_cat_id: torch.Tensor
    v_wrong_available: torch.Tensor
    v_wrong_marker_is_forbidden: torch.Tensor
    v_wrong_marker_cat_id: torch.Tensor

    top1_marker_text: List[str]
    v2_marker_text: List[str]
    v_wrong_marker_text: List[str]
    topk_markers_text: List[List[str]]
    topk_scores_text: List[List[float]]

class Step2Dataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer,
        logic_label2id: Dict[str, int],
        max_len: int,
        is_train: bool,
        seed: int,

        marker_temperature: float,
        dropout_nli_start: float,
        dropout_nli_end: float,
        dropout_logic_start: float,
        dropout_logic_end: float,
        dropout_unk_boost: float,

        corrupt_prob_nli: float,
        corrupt_prob_logic: float,

        logic_forbidden_categories: Dict[str, List[str]],
        mode: str,  # train | eval_no_marker | eval_with_marker | eval_wrong_marker
        enable_corruption: bool,
        enable_dropout: bool,
        include_wrong_view: bool = False,
        corrupt_prob_nli_end: Optional[float] = None,
        corrupt_prob_logic_end: Optional[float] = None,
        prompt_style: str = "legacy",
        prompt_input_style: str = "prompt",
        pair_marker_placement: str = "hypothesis_prefix",
        pair_marker_prefix: str = "[M] ",
        pair_marker_representation: str = "raw_text",
        pair_category_special_template: str = "[REL_{cat}]",
        pair_distribution_temperature: Optional[float] = None,
        train_sampling_mode: str = "temperature",
        marker_source: str = "predicted",
        confidence_gating_enabled: bool = False,
        confidence_gating_apply_train: bool = True,
        confidence_gating_apply_eval: bool = True,
        confidence_gating_temperature: Optional[float] = None,
        confidence_gating_min_top1_prob: float = 0.0,
        confidence_gating_min_top1_gap: float = 0.0,

        strong_wrong_nli: bool = True,
        wrong_nli_exclude_topk: bool = True,
        wrong_nli_exclude_same_category: bool = True,
    ):
        self.rows = rows
        self.tok = tokenizer
        self.logic_label2id = logic_label2id
        self.max_len = max_len
        self.is_train = is_train
        self.rng = random.Random(seed)

        self.marker_temperature = marker_temperature
        self.dropout_nli_start = dropout_nli_start
        self.dropout_nli_end = dropout_nli_end
        self.dropout_logic_start = dropout_logic_start
        self.dropout_logic_end = dropout_logic_end
        self.dropout_unk_boost = dropout_unk_boost

        self.corrupt_prob_nli = corrupt_prob_nli
        self.corrupt_prob_logic = corrupt_prob_logic
        self.corrupt_prob_nli_end = corrupt_prob_nli if corrupt_prob_nli_end is None else corrupt_prob_nli_end
        self.corrupt_prob_logic_end = corrupt_prob_logic if corrupt_prob_logic_end is None else corrupt_prob_logic_end

        self.logic_forbidden_categories = logic_forbidden_categories or {}
        self.mode = mode
        self.enable_corruption = enable_corruption
        self.enable_dropout = enable_dropout
        self.include_wrong_view = include_wrong_view
        self.marker_tag_ids = self.tok("[M]", add_special_tokens=False)["input_ids"]
        self.prompt_style = str(prompt_style)
        self.prompt_input_style = str(prompt_input_style).strip().lower()
        self.pair_marker_placement = str(pair_marker_placement).strip().lower()
        self.pair_marker_prefix = str(pair_marker_prefix)
        self.pair_marker_representation = str(pair_marker_representation).strip().lower()
        self.pair_category_special_template = str(pair_category_special_template)
        self.pair_distribution_temperature = (
            float(pair_distribution_temperature)
            if pair_distribution_temperature is not None
            else 1.0
        )
        self.train_sampling_mode = str(train_sampling_mode).strip().lower()
        self.marker_source = str(marker_source).strip().lower()
        self.confidence_gating_enabled = bool(confidence_gating_enabled)
        self.confidence_gating_apply_train = bool(confidence_gating_apply_train)
        self.confidence_gating_apply_eval = bool(confidence_gating_apply_eval)
        self.confidence_gating_temperature = (
            float(confidence_gating_temperature)
            if confidence_gating_temperature is not None
            else float(marker_temperature)
        )
        self.confidence_gating_min_top1_prob = float(confidence_gating_min_top1_prob)
        self.confidence_gating_min_top1_gap = float(confidence_gating_min_top1_gap)

        self.strong_wrong_nli = strong_wrong_nli
        self.wrong_nli_exclude_topk = wrong_nli_exclude_topk
        self.wrong_nli_exclude_same_category = wrong_nli_exclude_same_category

        self._progress = 0.0

    def set_progress(self, progress_0_1: float):
        self._progress = float(progress_0_1)

    def __len__(self):
        return len(self.rows)

    def _encode(self, text: str, text_pair: Optional[str] = None) -> Tuple[List[int], List[int], List[int]]:
        enc = self.tok(
            text,
            text_pair=text_pair,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = [0] * len(enc["input_ids"])
        return enc["input_ids"], enc["attention_mask"], token_type_ids

    def _build_pair_texts(
        self,
        premise: str,
        hypothesis: str,
        marker: Optional[str],
    ) -> Tuple[str, str]:
        if not marker:
            return premise, hypothesis
        if self.pair_marker_representation == "category_distribution":
            return premise, hypothesis

        rendered_marker = self._render_pair_marker(marker)
        marker_text = f"{self.pair_marker_prefix}{rendered_marker}"
        if self.pair_marker_placement == "premise_prefix":
            premise = f"{marker_text}\n{premise}"
        elif self.pair_marker_placement == "premise_suffix":
            premise = f"{premise}\n{marker_text}"
        elif self.pair_marker_placement == "hypothesis_suffix":
            hypothesis = f"{hypothesis}\n{marker_text}"
        else:
            hypothesis = f"{marker_text}\n{hypothesis}"
        return premise, hypothesis

    def _build_input(
        self,
        premise: str,
        hypothesis: str,
        marker: Optional[str],
        task: str,
    ) -> Tuple[List[int], List[int], List[int]]:
        if self.prompt_input_style == "pair_raw":
            text_a, text_b = self._build_pair_texts(premise, hypothesis, marker)
            return self._encode(text_a, text_pair=text_b)

        text = build_prompt(premise, hypothesis, marker=marker, task=task, style=self.prompt_style)
        return self._encode(text)

    def _sample_marker_topk(self, topk_markers, topk_scores) -> str:
        if self.is_train:
            if self.train_sampling_mode == "top1":
                best_i = max(range(len(topk_scores)), key=lambda i: topk_scores[i])
                return topk_markers[best_i]
            return softmax_sample(topk_markers, topk_scores, self.marker_temperature, self.rng)
        best_i = max(range(len(topk_scores)), key=lambda i: topk_scores[i])
        return topk_markers[best_i]

    def _render_pair_marker(self, marker: str) -> str:
        if self.pair_marker_representation == "category_special":
            return marker_to_relation_special_token(
                marker,
                template=self.pair_category_special_template,
            )
        return marker

    @staticmethod
    def _zero_cat_dist() -> List[float]:
        return [0.0] * len(CAT2ID)

    @staticmethod
    def _one_hot_cat_dist(cat: str) -> List[float]:
        dist = [0.0] * len(CAT2ID)
        dist[CAT2ID.get(cat, CAT2ID["UNK"])] = 1.0
        return dist

    def _topk_category_distribution(self, topk_markers: List[str], topk_scores: List[float]) -> List[float]:
        if not topk_markers or not topk_scores or len(topk_markers) != len(topk_scores):
            return self._zero_cat_dist()

        t = max(float(self.pair_distribution_temperature), 1e-6)
        mx = max(float(s) for s in topk_scores)
        exps = [math.exp((float(s) - mx) / t) for s in topk_scores]
        z = sum(exps)
        if z <= 0.0:
            return self._zero_cat_dist()

        dist = [0.0] * len(CAT2ID)
        for marker, e in zip(topk_markers, exps):
            prob = e / z
            cat = marker_category(marker)
            dist[CAT2ID.get(cat, CAT2ID["UNK"])] += prob
        return dist

    def _confidence_gate_marker(self, topk_markers: List[str], topk_scores: List[float]) -> bool:
        if self.marker_source != "predicted" or not self.confidence_gating_enabled:
            return False
        if self.is_train and not self.confidence_gating_apply_train:
            return False
        if (not self.is_train) and not self.confidence_gating_apply_eval:
            return False
        if not topk_markers or not topk_scores or len(topk_markers) != len(topk_scores):
            return False

        t = max(float(self.confidence_gating_temperature), 1e-6)
        mx = max(topk_scores)
        exps = [math.exp((float(s) - mx) / t) for s in topk_scores]
        z = sum(exps)
        if z <= 0.0:
            return False
        probs = [x / z for x in exps]
        sorted_probs = sorted(probs, reverse=True)
        top1_prob = sorted_probs[0]
        top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        top1_gap = top1_prob - top2_prob

        if top1_prob < self.confidence_gating_min_top1_prob:
            return True
        if top1_gap < self.confidence_gating_min_top1_gap:
            return True
        return False

    def _gold_marker_from_labels(
        self,
        nli_label: Optional[str],
        logic_label: Optional[str],
    ) -> Optional[str]:
        if nli_label is not None:
            cat = NLI_TO_CAT.get(nli_label)
            if cat is not None:
                return CANONICAL_MARKER_BY_CAT.get(cat)
        if logic_label is not None:
            cat = LOGIC_KO_TO_CAT.get(logic_label)
            if cat is not None:
                return CANONICAL_MARKER_BY_CAT.get(cat)
        return None

    def _sample_corrupt_marker_logic(self, logic_label: str, avoid: Optional[str]) -> Optional[str]:
        forbidden = self.logic_forbidden_categories.get(logic_label, [])
        pool: List[str] = []
        for cat in forbidden:
            pool.extend(LEX_CAT2MARKERS.get(cat, []))
        pool = [m for m in pool if m and (avoid is None or m != avoid)]
        if not pool:
            return None
        return self.rng.choice(pool)

    def _sample_corrupt_marker_generic(self, avoid: Optional[str], exclude: Optional[List[str]] = None) -> Optional[str]:
        exclude_set = set(exclude or [])
        pool = [m for m in ALL_LEX_MARKERS if m and (avoid is None or m != avoid) and (m not in exclude_set)]
        if not pool:
            return None
        return self.rng.choice(pool)

    def _sample_corrupt_marker_nli_strong(
        self,
        nli_label: Optional[str],
        used_marker: Optional[str],
        topk_markers: Optional[List[str]],
    ) -> Optional[str]:
        exclude_markers = set()
        if used_marker is not None:
            exclude_markers.add(used_marker)
        if self.wrong_nli_exclude_topk and topk_markers:
            exclude_markers.update(topk_markers)

        pref_cats = NLI_WRONG_CATEGORY_MAP.get(nli_label or "", [])
        if not pref_cats:
            used_cat = marker_category(used_marker) if used_marker is not None else "NONE"
            pref_cats = ANTI_CATEGORY_MAP.get(used_cat, ["CONTRAST", "CAUSAL", "ADD"])
        pool = []
        for cat in pref_cats:
            pool.extend(LEX_CAT2MARKERS.get(cat, []))

        gold_cat = NLI_TO_CAT.get(nli_label or "", marker_category(used_marker) if used_marker is not None else "NONE")
        if self.wrong_nli_exclude_same_category and gold_cat in LEX_CAT2MARKERS:
            same_cat_markers = set(LEX_CAT2MARKERS.get(gold_cat, []))
        else:
            same_cat_markers = set()

        pool = [m for m in pool if m not in exclude_markers and m not in same_cat_markers]
        if pool:
            return self.rng.choice(pool)

        pool2 = []
        for m in ALL_LEX_MARKERS:
            if m in exclude_markers:
                continue
            if self.wrong_nli_exclude_same_category and marker_category(m) == gold_cat:
                continue
            pool2.append(m)
        if pool2:
            return self.rng.choice(pool2)

        return self._sample_corrupt_marker_generic(avoid=used_marker, exclude=list(exclude_markers))

    def _is_forbidden_marker_for_logic(self, logic_label: str, marker: str) -> bool:
        cats = set(self.logic_forbidden_categories.get(logic_label, []))
        return marker_category(marker) in cats

    def _sample_wrong_marker(
        self,
        is_logic: bool,
        nli_label: Optional[str],
        logic_label: Optional[str],
        used_marker: Optional[str],
        topk_markers: Optional[List[str]],
    ) -> Optional[str]:
        if is_logic and logic_label is not None:
            cm = self._sample_corrupt_marker_logic(logic_label, avoid=used_marker)
            if cm is None:
                cm = self._sample_corrupt_marker_generic(avoid=used_marker, exclude=topk_markers or [])
            return cm

        if self.strong_wrong_nli:
            return self._sample_corrupt_marker_nli_strong(nli_label=nli_label, used_marker=used_marker, topk_markers=topk_markers)
        return self._sample_corrupt_marker_generic(avoid=used_marker, exclude=topk_markers or [])

    @staticmethod
    def _find_subsequence(seq: List[int], pattern: List[int], start: int = 0) -> int:
        if not pattern:
            return -1
        plen = len(pattern)
        upper = len(seq) - plen + 1
        for i in range(max(start, 0), max(upper, 0)):
            if seq[i:i + plen] == pattern:
                return i
        return -1

    def _find_marker_span(self, input_ids: List[int], marker_text: Optional[str]) -> Tuple[int, int]:
        if not marker_text:
            return -1, -1
        if self.pair_marker_representation == "category_distribution":
            return -1, -1

        if self.prompt_input_style == "pair_raw":
            rendered_marker = self._render_pair_marker(marker_text)
            marker_ids = self.tok(rendered_marker, add_special_tokens=False)["input_ids"]
            if not marker_ids:
                return -1, -1

            tag_start = self._find_subsequence(input_ids, self.marker_tag_ids, start=0)
            if tag_start >= 0:
                search_start = tag_start + len(self.marker_tag_ids)
                marker_start = self._find_subsequence(input_ids, marker_ids, start=search_start)
                if marker_start >= 0:
                    marker_end = min(marker_start + len(marker_ids), len(input_ids))
                    return marker_start, marker_end

            marker_start = self._find_subsequence(input_ids, marker_ids, start=0)
            if marker_start < 0:
                return -1, -1
            marker_end = min(marker_start + len(marker_ids), len(input_ids))
            return marker_start, marker_end

        if not self.marker_tag_ids:
            return -1, -1

        tag_start = self._find_subsequence(input_ids, self.marker_tag_ids, start=0)
        if tag_start < 0:
            return -1, -1

        marker_ids = self.tok(marker_text, add_special_tokens=False)["input_ids"]
        if not marker_ids:
            return tag_start, min(tag_start + len(self.marker_tag_ids), len(input_ids))

        search_start = tag_start + len(self.marker_tag_ids)
        marker_start = self._find_subsequence(input_ids, marker_ids, start=search_start)
        if marker_start < 0:
            marker_start = search_start
            marker_end = min(search_start + len(marker_ids), len(input_ids))
        else:
            marker_end = min(marker_start + len(marker_ids), len(input_ids))
        return marker_start, marker_end

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        s1 = r["premise"]
        s2 = r["hypothesis"]

        nli_lab = r.get("nli_label", None)
        logic_lab = r.get("logic_label", None)

        y_nli = -100
        y_logic = -100
        is_nli = False
        is_logic = False
        if nli_lab is not None:
            is_nli = True
            y_nli = NLI_LABEL2ID.get(nli_lab, -100)
        elif logic_lab is not None:
            is_logic = True
            y_logic = self.logic_label2id[logic_lab]

        prompt_task = "nli" if is_nli else ("logic" if is_logic else "generic")
        v1_ids, v1_mask, v1_type_ids = self._build_input(s1, s2, marker=None, task=prompt_task)
        v1_cat_dist = self._zero_cat_dist()

        topk_markers = r.get("step1_topk_markers", None) or []
        topk_scores = r.get("step1_topk_scores", None) or []

        top1_marker = None
        if topk_markers and topk_scores:
            top1_marker = self._sample_marker_topk(topk_markers, topk_scores)

        gold_marker = self._gold_marker_from_labels(nli_lab, logic_lab)
        base_marker = gold_marker if self.marker_source == "gold_canonical" else top1_marker
        marker_gated = False
        if base_marker is not None and self._confidence_gate_marker(topk_markers, topk_scores):
            base_marker = None
            marker_gated = True

        if base_marker is None and not topk_markers and not topk_scores:
            return {
                "row_id": idx,
                "v1_ids": v1_ids, "v1_mask": v1_mask, "v1_type_ids": v1_type_ids,
                "v1_cat_dist": v1_cat_dist,
                "v2_ids": v1_ids, "v2_mask": v1_mask, "v2_type_ids": v1_type_ids,
                "v2_cat_dist": self._zero_cat_dist(),
                "v2_marker_start": -1, "v2_marker_end": -1,
                "v_wrong_ids": v1_ids, "v_wrong_mask": v1_mask, "v_wrong_type_ids": v1_type_ids,
                "v_wrong_cat_dist": self._zero_cat_dist(),
                "v_wrong_marker_start": -1, "v_wrong_marker_end": -1,
                "y_nli": y_nli, "y_logic": y_logic,
                "v2_forbidden": 0,
                "v2_corrupt": 0,
                "v2_cat_id": CAT2ID["NONE"],
                "v_wrong_available": 0,
                "v_wrong_forbidden": 0,
                "v_wrong_cat_id": CAT2ID["NONE"],
                "top1_marker_text": top1_marker or "",
                "v2_marker_text": "",
                "v_wrong_marker_text": "",
                "topk_markers_text": topk_markers,
                "topk_scores_text": topk_scores,
            }

        used_marker = base_marker
        v2_is_corrupt = 0

        if self.mode == "eval_no_marker":
            used_marker = None

        elif self.mode == "eval_with_marker":
            pass

        elif self.mode == "eval_wrong_marker":
            if used_marker is not None:
                cm = self._sample_wrong_marker(is_logic, nli_lab, logic_lab, used_marker, topk_markers)
                if cm is not None:
                    used_marker = cm
                    v2_is_corrupt = 1

        else:
            if self.enable_corruption and self.is_train and used_marker is not None:
                cn = linear_schedule(self.corrupt_prob_nli, self.corrupt_prob_nli_end, self._progress)
                cl = linear_schedule(self.corrupt_prob_logic, self.corrupt_prob_logic_end, self._progress)
                corrupt_prob = cn if is_nli else (cl if is_logic else 0.0)
                if self.rng.random() < corrupt_prob:
                    cm = self._sample_wrong_marker(is_logic, nli_lab, logic_lab, used_marker, topk_markers)
                    if cm is not None:
                        used_marker = cm
                        v2_is_corrupt = 1

        v2_forbidden = 0
        if is_logic and logic_lab is not None and used_marker is not None:
            v2_forbidden = 1 if self._is_forbidden_marker_for_logic(logic_lab, used_marker) else 0

        do_drop = False
        if self.mode == "train" and self.enable_dropout and self.is_train and used_marker is not None:
            dn = linear_schedule(self.dropout_nli_start, self.dropout_nli_end, self._progress)
            dl = linear_schedule(self.dropout_logic_start, self.dropout_logic_end, self._progress)
            base_drop = dn if is_nli else (dl if is_logic else 0.0)
            if marker_category(used_marker) == "UNK":
                base_drop = min(1.0, base_drop + self.dropout_unk_boost)
            do_drop = (self.rng.random() < base_drop)

        if used_marker is None or do_drop:
            v2_ids, v2_mask, v2_type_ids = v1_ids, v1_mask, v1_type_ids
            v2_cat_dist = self._zero_cat_dist()
            cat_id = CAT2ID["NONE"]
            final_v2_marker_text = ""
            v2_marker_start, v2_marker_end = -1, -1
        else:
            v2_ids, v2_mask, v2_type_ids = self._build_input(s1, s2, marker=used_marker, task=prompt_task)
            cat = marker_category(used_marker)
            if self.pair_marker_representation == "category_distribution":
                if v2_is_corrupt:
                    v2_cat_dist = self._one_hot_cat_dist(cat)
                else:
                    v2_cat_dist = self._topk_category_distribution(topk_markers, topk_scores)
            else:
                v2_cat_dist = self._zero_cat_dist()
            cat_id = CAT2ID.get(cat, CAT2ID["UNK"])
            final_v2_marker_text = used_marker
            v2_marker_start, v2_marker_end = self._find_marker_span(v2_ids, used_marker)

        wrong_marker = None
        if self.include_wrong_view and base_marker is not None and not marker_gated:
            wrong_marker = self._sample_wrong_marker(is_logic, nli_lab, logic_lab, base_marker, topk_markers)

        if wrong_marker is None:
            v_wrong_ids, v_wrong_mask, v_wrong_type_ids = v1_ids, v1_mask, v1_type_ids
            v_wrong_cat_dist = self._zero_cat_dist()
            v_wrong_marker_start, v_wrong_marker_end = -1, -1
            wrong_available = 0
            wrong_forbidden = 0
            wrong_cat_id = CAT2ID["NONE"]
            final_wrong_marker_text = ""
        else:
            v_wrong_ids, v_wrong_mask, v_wrong_type_ids = self._build_input(s1, s2, marker=wrong_marker, task=prompt_task)
            if self.pair_marker_representation == "category_distribution":
                v_wrong_cat_dist = self._one_hot_cat_dist(marker_category(wrong_marker))
            else:
                v_wrong_cat_dist = self._zero_cat_dist()
            v_wrong_marker_start, v_wrong_marker_end = self._find_marker_span(v_wrong_ids, wrong_marker)
            wrong_available = 1
            wrong_forbidden = 0
            if is_logic and logic_lab is not None:
                wrong_forbidden = 1 if self._is_forbidden_marker_for_logic(logic_lab, wrong_marker) else 0
            wrong_cat = marker_category(wrong_marker)
            wrong_cat_id = CAT2ID.get(wrong_cat, CAT2ID["UNK"])
            final_wrong_marker_text = wrong_marker

        return {
            "row_id": idx,
            "v1_ids": v1_ids, "v1_mask": v1_mask, "v1_type_ids": v1_type_ids,
            "v1_cat_dist": v1_cat_dist,
            "v2_ids": v2_ids, "v2_mask": v2_mask, "v2_type_ids": v2_type_ids,
            "v2_cat_dist": v2_cat_dist,
            "v2_marker_start": v2_marker_start,
            "v2_marker_end": v2_marker_end,
            "v_wrong_ids": v_wrong_ids, "v_wrong_mask": v_wrong_mask, "v_wrong_type_ids": v_wrong_type_ids,
            "v_wrong_cat_dist": v_wrong_cat_dist,
            "v_wrong_marker_start": v_wrong_marker_start,
            "v_wrong_marker_end": v_wrong_marker_end,
            "y_nli": y_nli, "y_logic": y_logic,
            "v2_forbidden": v2_forbidden,
            "v2_corrupt": v2_is_corrupt,
            "v2_cat_id": cat_id,
            "v_wrong_available": wrong_available,
            "v_wrong_forbidden": wrong_forbidden,
            "v_wrong_cat_id": wrong_cat_id,

            "top1_marker_text": top1_marker if top1_marker is not None else "",
            "v2_marker_text": final_v2_marker_text,
            "v_wrong_marker_text": final_wrong_marker_text,
            "topk_markers_text": list(topk_markers) if topk_markers is not None else [],
            "topk_scores_text": list(topk_scores) if topk_scores is not None else [],
        }

class PadCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Step2Batch:
        def pad(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
            maxlen = max(len(s) for s in seqs)
            if self.pad_to_multiple_of is not None:
                m = self.pad_to_multiple_of
                maxlen = ((maxlen + m - 1) // m) * m
            out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            return out

        def pad_mask(seqs: List[List[int]]) -> torch.Tensor:
            maxlen = max(len(s) for s in seqs)
            if self.pad_to_multiple_of is not None:
                m = self.pad_to_multiple_of
                maxlen = ((maxlen + m - 1) // m) * m
            out = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, :len(s)] = 1
            return out

        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id

        row_ids = torch.tensor([b["row_id"] for b in batch], dtype=torch.long)
        v1_ids = pad([b["v1_ids"] for b in batch], pad_id)
        v2_ids = pad([b["v2_ids"] for b in batch], pad_id)
        v_wrong_ids = pad([b["v_wrong_ids"] for b in batch], pad_id)
        v1_mask = pad_mask([b["v1_mask"] for b in batch])
        v2_mask = pad_mask([b["v2_mask"] for b in batch])
        v_wrong_mask = pad_mask([b["v_wrong_mask"] for b in batch])
        v1_type_ids = pad([b["v1_type_ids"] for b in batch], 0)
        v2_type_ids = pad([b["v2_type_ids"] for b in batch], 0)
        v_wrong_type_ids = pad([b["v_wrong_type_ids"] for b in batch], 0)
        v1_cat_dist = torch.tensor([b["v1_cat_dist"] for b in batch], dtype=torch.float32)
        v2_cat_dist = torch.tensor([b["v2_cat_dist"] for b in batch], dtype=torch.float32)
        v_wrong_cat_dist = torch.tensor([b["v_wrong_cat_dist"] for b in batch], dtype=torch.float32)

        y_nli = torch.tensor([b["y_nli"] for b in batch], dtype=torch.long)
        y_logic = torch.tensor([b["y_logic"] for b in batch], dtype=torch.long)
        v2_marker_start = torch.tensor([b["v2_marker_start"] for b in batch], dtype=torch.long)
        v2_marker_end = torch.tensor([b["v2_marker_end"] for b in batch], dtype=torch.long)
        v_wrong_marker_start = torch.tensor([b["v_wrong_marker_start"] for b in batch], dtype=torch.long)
        v_wrong_marker_end = torch.tensor([b["v_wrong_marker_end"] for b in batch], dtype=torch.long)
        v2_forb = torch.tensor([b["v2_forbidden"] for b in batch], dtype=torch.long)
        v2_corr = torch.tensor([b["v2_corrupt"] for b in batch], dtype=torch.long)
        v2_cat = torch.tensor([b["v2_cat_id"] for b in batch], dtype=torch.long)
        v_wrong_avail = torch.tensor([b["v_wrong_available"] for b in batch], dtype=torch.long)
        v_wrong_forb = torch.tensor([b["v_wrong_forbidden"] for b in batch], dtype=torch.long)
        v_wrong_cat = torch.tensor([b["v_wrong_cat_id"] for b in batch], dtype=torch.long)

        return Step2Batch(
            row_ids=row_ids,
            v1_input_ids=v1_ids, v1_attention_mask=v1_mask, v1_token_type_ids=v1_type_ids, v1_marker_cat_dist=v1_cat_dist,
            v2_input_ids=v2_ids, v2_attention_mask=v2_mask, v2_token_type_ids=v2_type_ids, v2_marker_cat_dist=v2_cat_dist,
            v2_marker_start=v2_marker_start, v2_marker_end=v2_marker_end,
            v_wrong_input_ids=v_wrong_ids, v_wrong_attention_mask=v_wrong_mask, v_wrong_token_type_ids=v_wrong_type_ids, v_wrong_marker_cat_dist=v_wrong_cat_dist,
            v_wrong_marker_start=v_wrong_marker_start, v_wrong_marker_end=v_wrong_marker_end,
            y_nli=y_nli, y_logic=y_logic,
            v2_marker_is_forbidden=v2_forb,
            v2_is_corrupt=v2_corr,
            v2_marker_cat_id=v2_cat,
            v_wrong_available=v_wrong_avail,
            v_wrong_marker_is_forbidden=v_wrong_forb,
            v_wrong_marker_cat_id=v_wrong_cat,

            top1_marker_text=[b["top1_marker_text"] for b in batch],
            v2_marker_text=[b["v2_marker_text"] for b in batch],
            v_wrong_marker_text=[b["v_wrong_marker_text"] for b in batch],
            topk_markers_text=[b["topk_markers_text"] for b in batch],
            topk_scores_text=[b["topk_scores_text"] for b in batch],
        )
