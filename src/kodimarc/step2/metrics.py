from typing import List, Tuple

# ============================================================
# 1.5) Metrics
# ============================================================
def macro_prf(num_classes: int, y: List[int], pred: List[int]) -> Tuple[float, float, float]:
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for yt, pr in zip(y, pred):
        if not (0 <= yt < num_classes) or not (0 <= pr < num_classes):
            continue
        if yt == pr:
            tp[yt] += 1
        else:
            fp[pr] += 1
            fn[yt] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    mp = sum(precisions) / num_classes if num_classes > 0 else 0.0
    mr = sum(recalls) / num_classes if num_classes > 0 else 0.0
    mf = sum(f1s) / num_classes if num_classes > 0 else 0.0
    return mp, mr, mf

def confusion_matrix(num_classes: int, y: List[int], pred: List[int]) -> List[List[int]]:
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for yt, pr in zip(y, pred):
        if 0 <= yt < num_classes and 0 <= pr < num_classes:
            cm[yt][pr] += 1
    return cm