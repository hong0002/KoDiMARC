import gc
import math
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_cfg: str) -> torch.device:
    if device_cfg == 'cpu':
        return torch.device('cpu')
    if device_cfg == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def split_train_rows_by_task(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nli_rows = []
    logic_rows = []
    for row in rows:
        if row.get('nli_label') is not None:
            nli_rows.append(row)
        elif row.get('logic_label') is not None:
            logic_rows.append(row)
    return nli_rows, logic_rows


def build_logic_label_vocab(train_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    labels = sorted({row['logic_label'] for row in train_rows if row.get('logic_label') is not None})
    return {label: idx for idx, label in enumerate(labels)}


def softmax_sample(items: List[str], scores: List[float], temperature: float, rng: random.Random) -> str:
    temp = max(float(temperature), 1e-6)
    max_score = max(scores)
    exps = [math.exp((score - max_score) / temp) for score in scores]
    total = sum(exps)
    threshold = rng.random() * total
    cumulative = 0.0
    for item, value in zip(items, exps):
        cumulative += value
        if cumulative >= threshold:
            return item
    return items[-1]


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]['lr']


def linear_schedule(start: float, end: float, progress_0_1: float) -> float:
    progress = min(max(progress_0_1, 0.0), 1.0)
    return start + (end - start) * progress


def sanitize_name(text: str) -> str:
    text = text.strip().replace('/', '__')
    text = re.sub(r'[^0-9A-Za-z_\-\.]+', '_', text)
    return text[:180]


def now_seoul_str() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
