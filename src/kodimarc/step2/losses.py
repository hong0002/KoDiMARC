import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ============================================================
# 5) Losses
# ============================================================
def kl_consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    pa = F.log_softmax(logits_a, dim=-1)
    pb = F.log_softmax(logits_b, dim=-1)
    qa = F.softmax(logits_a, dim=-1)
    qb = F.softmax(logits_b, dim=-1)
    kl_ab = F.kl_div(pa, qb, reduction="batchmean")
    kl_ba = F.kl_div(pb, qa, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)

def masked_ce(ce_fn: nn.CrossEntropyLoss, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = (y != -100)
    if m.any():
        return ce_fn(logits[m], y[m])
    return logits.new_tensor(0.0)

def masked_ce_with_mask(
    ce_fn: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    y: torch.Tensor,
    extra_mask: torch.Tensor
) -> torch.Tensor:
    m = (y != -100) & extra_mask
    if m.any():
        return ce_fn(logits[m], y[m])
    return logits.new_tensor(0.0)

def masked_focal_ce(
    logits: torch.Tensor,
    y: torch.Tensor,
    gamma: float,
    weight: Optional[torch.Tensor] = None,
    extra_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    m = (y != -100)
    if extra_mask is not None:
        m = m & extra_mask
    if not m.any():
        return logits.new_tensor(0.0)

    logits_sel = logits[m]
    y_sel = y[m].long()
    log_probs = F.log_softmax(logits_sel, dim=-1)
    probs = log_probs.exp()
    log_pt = log_probs.gather(1, y_sel.unsqueeze(1)).squeeze(1)
    pt = probs.gather(1, y_sel.unsqueeze(1)).squeeze(1)
    focal_factor = torch.pow((1.0 - pt).clamp_min(0.0), float(gamma))
    loss = -focal_factor * log_pt
    if weight is not None:
        class_weight = weight.to(device=logits_sel.device, dtype=logits_sel.dtype)
        loss = loss * class_weight.gather(0, y_sel)
    return loss.mean()

def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    m = mask.bool()
    if not m.any():
        return logits.new_tensor(0.0)

    logits_sel = logits[m]
    target_sel = target[m].to(dtype=logits_sel.dtype)
    kwargs = {}
    if pos_weight is not None and float(pos_weight) > 0.0:
        kwargs["pos_weight"] = torch.tensor(
            float(pos_weight),
            device=logits_sel.device,
            dtype=logits_sel.dtype,
        )
    return F.binary_cross_entropy_with_logits(logits_sel, target_sel, **kwargs)

def masked_margin_between_views(
    logits_with: torch.Tensor,
    logits_wrong: torch.Tensor,
    y: torch.Tensor,
    margin: float,
    extra_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    m = (y != -100)
    if extra_mask is not None:
        m = m & extra_mask
    if not m.any():
        return logits_with.new_tensor(0.0)

    y_sel = y[m].long().unsqueeze(1)
    gold_with = logits_with[m].gather(1, y_sel).squeeze(1)
    gold_wrong = logits_wrong[m].gather(1, y_sel).squeeze(1)
    return F.relu(float(margin) - (gold_with - gold_wrong)).mean()

def supcon_loss_with_queue(
    reps: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    queue_reps: Optional[torch.Tensor] = None,
    queue_labels: Optional[torch.Tensor] = None,
    pos_mask_override: Optional[torch.Tensor] = None,
    neg_weight_inbatch: Optional[torch.Tensor] = None,
    anchor_forbidden: Optional[torch.Tensor] = None,
    queue_forbidden: Optional[torch.Tensor] = None,
    forbidden_neg_weight: float = 2.0,
) -> torch.Tensor:
    device = reps.device
    valid = labels != -100
    if valid.sum() < 2:
        return reps.new_tensor(0.0)

    reps_a = reps[valid]
    labs_a = labels[valid]
    N = reps_a.size(0)
    reps_a = F.normalize(reps_a, dim=-1)

    use_q = (queue_reps is not None) and (queue_reps.numel() > 0)
    if use_q:
        reps_q = F.normalize(queue_reps.to(device=device, dtype=reps_a.dtype), dim=-1)
        Q = reps_q.size(0)
        reps_all = torch.cat([reps_a, reps_q], dim=0)
    else:
        Q = 0
        reps_all = reps_a

    sim = (reps_a @ reps_all.T) / max(float(temperature), 1e-6)

    logits_mask = torch.ones((N, N + Q), device=device, dtype=torch.bool)
    logits_mask[:, :N] &= ~torch.eye(N, device=device, dtype=torch.bool)

    if pos_mask_override is None:
        pos = (labs_a.unsqueeze(0) == labs_a.unsqueeze(1)) & (~torch.eye(N, device=device, dtype=torch.bool))
    else:
        idx = torch.nonzero(valid).squeeze(1)
        pm = pos_mask_override[idx][:, idx].to(device)
        pos = pm & (~torch.eye(N, device=device, dtype=torch.bool))

    w = torch.ones((N, N + Q), device=device, dtype=reps_a.dtype)

    if neg_weight_inbatch is not None:
        idx = torch.nonzero(valid).squeeze(1)
        nb = neg_weight_inbatch[idx][:, idx].to(device=device, dtype=reps_a.dtype)
        w[:, :N] = w[:, :N] * nb

    if anchor_forbidden is not None:
        af = anchor_forbidden[valid].to(device=device).float()
    else:
        af = torch.zeros((N,), device=device)

    if use_q and (queue_forbidden is not None):
        qf = queue_forbidden.to(device=device).float()
    else:
        qf = torch.zeros((Q,), device=device)

    if forbidden_neg_weight != 1.0:
        forb_pair_in = torch.clamp(af.unsqueeze(1) + af.unsqueeze(0), 0, 1)
        w[:, :N] = w[:, :N] * (1.0 + forb_pair_in * (forbidden_neg_weight - 1.0))
        if Q > 0:
            forb_pair_q = torch.clamp(af.unsqueeze(1) + qf.unsqueeze(0), 0, 1)
            w[:, N:] = w[:, N:] * (1.0 + forb_pair_q * (forbidden_neg_weight - 1.0))

    exp_sim = torch.exp(sim) * logits_mask.float() * w
    denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = sim - torch.log(denom)

    pos_count = pos.sum(dim=1).clamp_min(1)
    loss = -(log_prob[:, :N] * pos.float()).sum(dim=1) / pos_count
    return loss.mean()
