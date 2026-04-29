from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import ujson
from torch.utils.data import Dataset

from kodimarc.common.markers import LEXICON, normalize_marker, normalize_whitespace

LABEL2ID = {
    "ADD": 0,
    "CONTRAST": 1,
    "CAUSAL": 2,
    "EXPLAN": 3,
    "CONCESS": 4,
    "COND": 5,
    "EXAMPLE": 6,
}

ALL_MARKERS = []
for _, markers in LEXICON.items():
    for marker in markers:
        normalized = normalize_marker(marker)
        if normalized and normalized not in ALL_MARKERS:
            ALL_MARKERS.append(normalized)

MARKER2ID = {marker: idx for idx, marker in enumerate(ALL_MARKERS)}
OTHER_MARKER = "OTHER"
MARKER2ID[OTHER_MARKER] = len(MARKER2ID)


@dataclass
class ConnectiveSFTConfig:
    train_path: str
    max_length: int = 512
    max_samples: Optional[int] = None


class ConnectiveSFTDataset(Dataset):
    """Response-only SFT dataset for Step1 discourse marker generation."""

    def __init__(self, cfg: ConnectiveSFTConfig, tokenizer) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.examples = []

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        with open(cfg.train_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if cfg.max_samples is not None and idx >= cfg.max_samples:
                    break

                ex = ujson.loads(line)
                instruction = normalize_whitespace(ex.get("instruction", ""))
                input_text = ex.get("input", "").strip()
                output_text = ex.get("output", "").strip()
                label_str = ex.get("label")
                marker_str = ex.get("marker", "")

                if not instruction or not input_text or not output_text or label_str is None:
                    continue
                if label_str not in LABEL2ID:
                    continue

                label_id = LABEL2ID[label_str]
                marker_id = MARKER2ID.get(normalize_marker(marker_str), MARKER2ID[OTHER_MARKER])

                prompt_text = (
                    "<s>### Instruction:\n"
                    f"{instruction}\n\n"
                    "### Input:\n"
                    f"{input_text}\n\n"
                    "### Output:\n"
                )
                output_segment = f"{output_text}</s>"

                prompt_enc = tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=cfg.max_length,
                    padding=False,
                    add_special_tokens=False,
                )
                output_enc = tokenizer(
                    output_segment,
                    truncation=True,
                    max_length=cfg.max_length,
                    padding=False,
                    add_special_tokens=False,
                )

                prompt_ids = prompt_enc["input_ids"]
                output_ids = output_enc["input_ids"]
                if len(prompt_ids) >= cfg.max_length - 1:
                    continue

                max_output_len = cfg.max_length - len(prompt_ids)
                output_ids = output_ids[:max_output_len]
                input_ids = prompt_ids + output_ids
                attention_mask = [1] * len(input_ids)
                labels = [-100] * len(prompt_ids) + output_ids[:]

                if len(input_ids) < cfg.max_length:
                    pad_len = cfg.max_length - len(input_ids)
                    input_ids += [pad_id] * pad_len
                    attention_mask += [0] * pad_len
                    labels += [-100] * pad_len

                self.examples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "label_id": label_id,
                        "marker_id": marker_id,
                    }
                )

        print(f"[ConnectiveSFTDataset] loaded {len(self.examples):,} examples from {cfg.train_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        result = {}
        for key, value in item.items():
            if key in {"label_id", "marker_id"}:
                result[key] = torch.tensor(value, dtype=torch.long)
            else:
                result[key] = torch.tensor(value)
        return result
