import torch.nn as nn
import torch
from typing import Optional

# ============================================================
# 4) Model
# ============================================================
class Step2EncoderClassifier(nn.Module):
    def __init__(
        self,
        base_model,
        hidden_size: int,
        num_logic: int,
        dropout: float = 0.1,
        head_dtype: Optional[torch.dtype] = None,
        use_marker_aware_head: bool = False,
        use_base_delta_head: bool = False,
        pooling_strategy: str = "last_token",
        delta_scale_nli: float = 1.0,
        delta_scale_logic: float = 1.0,
        use_category_distribution_feature: bool = False,
        num_marker_categories: int = 0,
        use_learned_marker_gate: bool = False,
        marker_gate_hidden_size: int = 0,
        marker_gate_init_bias: float = 1.0,
        use_marker_compatibility_head: bool = False,
        use_compatibility_for_delta: bool = False,
        compatibility_hidden_size: int = 0,
        compatibility_init_bias: float = 0.0,
    ):
        super().__init__()
        self.lm = base_model
        self.drop = nn.Dropout(dropout)
        self.use_marker_aware_head = use_marker_aware_head
        self.use_base_delta_head = use_base_delta_head and use_marker_aware_head
        self.pooling_strategy = str(pooling_strategy).lower()
        self.delta_scale_nli = float(delta_scale_nli)
        self.delta_scale_logic = float(delta_scale_logic)
        self.use_category_distribution_feature = bool(use_category_distribution_feature) and int(num_marker_categories) > 0
        self.num_marker_categories = int(num_marker_categories)
        self.use_learned_marker_gate = bool(use_learned_marker_gate) and self.use_base_delta_head
        self.marker_gate_hidden_size = int(marker_gate_hidden_size) if int(marker_gate_hidden_size) > 0 else int(hidden_size)
        self.marker_gate_init_bias = float(marker_gate_init_bias)
        self.use_marker_compatibility_head = bool(use_marker_compatibility_head) and self.use_marker_aware_head
        self.use_compatibility_for_delta = bool(use_compatibility_for_delta) and self.use_marker_compatibility_head
        self.compatibility_hidden_size = int(compatibility_hidden_size) if int(compatibility_hidden_size) > 0 else int(hidden_size)
        self.compatibility_init_bias = float(compatibility_init_bias)

        if head_dtype is None:
            head_dtype = getattr(base_model, "dtype", torch.float16)

        head_in = hidden_size
        joint_in = hidden_size * 4
        if self.use_marker_aware_head:
            self.marker_proj = nn.Linear(hidden_size, hidden_size).to(dtype=head_dtype)
            if self.use_category_distribution_feature:
                self.cat_dist_proj = nn.Linear(self.num_marker_categories, hidden_size, bias=False).to(dtype=head_dtype)
            if self.use_marker_compatibility_head:
                self.compatibility_proj = nn.Sequential(
                    nn.Linear(joint_in, self.compatibility_hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ).to(dtype=head_dtype)
                self.compatibility_head = nn.Linear(self.compatibility_hidden_size, 2).to(dtype=head_dtype)
                nn.init.constant_(self.compatibility_head.bias, self.compatibility_init_bias)
            if self.use_base_delta_head:
                self.delta_proj = nn.Sequential(
                    nn.Linear(joint_in, hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ).to(dtype=head_dtype)
                if self.use_learned_marker_gate:
                    self.marker_gate = nn.Sequential(
                        nn.Linear(joint_in, self.marker_gate_hidden_size),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.marker_gate_hidden_size, 2),
                    ).to(dtype=head_dtype)
                    nn.init.constant_(self.marker_gate[-1].bias, self.marker_gate_init_bias)
                self.base_nli_head = nn.Linear(hidden_size, 3).to(dtype=head_dtype)
                self.base_logic_head = nn.Linear(hidden_size, num_logic).to(dtype=head_dtype)
                self.delta_nli_head = nn.Linear(hidden_size, 3).to(dtype=head_dtype)
                self.delta_logic_head = nn.Linear(hidden_size, num_logic).to(dtype=head_dtype)
                self.mrel_head = nn.Linear(hidden_size, 2).to(dtype=head_dtype)
            else:
                head_in = hidden_size * 2

        if not self.use_base_delta_head:
            self.nli_head = nn.Linear(head_in, 3).to(dtype=head_dtype)
            self.logic_head = nn.Linear(head_in, num_logic).to(dtype=head_dtype)
            self.mrel_head = nn.Linear(head_in, 2).to(dtype=head_dtype)

    def _pool_marker(
        self,
        hs: torch.Tensor,
        marker_start: Optional[torch.Tensor],
        marker_end: Optional[torch.Tensor],
    ) -> torch.Tensor:
        marker_repr = torch.zeros((hs.size(0), hs.size(-1)), device=hs.device, dtype=hs.dtype)
        if marker_start is None or marker_end is None:
            return marker_repr

        for i in range(hs.size(0)):
            s = int(marker_start[i].item())
            e = int(marker_end[i].item())
            if s < 0 or e <= s or s >= hs.size(1):
                continue
            e = min(e, hs.size(1))
            marker_repr[i] = hs[i, s:e, :].mean(dim=0)
        return marker_repr

    def _marker_valid_mask(
        self,
        hs: torch.Tensor,
        marker_start: Optional[torch.Tensor],
        marker_end: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mask = torch.zeros((hs.size(0), 1), device=hs.device, dtype=hs.dtype)
        if marker_start is None or marker_end is None:
            return mask
        valid = (
            (marker_start >= 0)
            & (marker_end > marker_start)
            & (marker_start < hs.size(1))
        )
        if valid.any():
            mask[valid] = 1.0
        return mask

    def _pool_sequence(self, hs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "cls":
            return hs[:, 0, :]
        if self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).to(dtype=hs.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (hs * mask).sum(dim=1) / denom
        lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hs.size(0), device=hs.device)
        return hs[batch_idx, lengths, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        marker_start: Optional[torch.Tensor] = None,
        marker_end: Optional[torch.Tensor] = None,
        marker_category_distribution: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "output_hidden_states": False,
            "use_cache": False,
        }
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids
        out = self.lm(**model_kwargs)
        hs = out.last_hidden_state
        pooled = self._pool_sequence(hs, attention_mask)
        pooled = self.drop(pooled)

        head_input = pooled
        aux_outputs = {}
        if self.use_marker_aware_head:
            marker_repr = self._pool_marker(hs, marker_start, marker_end)
            marker_repr = self.drop(self.marker_proj(marker_repr))
            marker_mask = self._marker_valid_mask(hs, marker_start, marker_end)
            if self.use_category_distribution_feature and marker_category_distribution is not None:
                cat_dist = marker_category_distribution.to(device=hs.device, dtype=hs.dtype)
                cat_repr = self.drop(self.cat_dist_proj(cat_dist))
                marker_repr = marker_repr + cat_repr
                cat_mask = (cat_dist.sum(dim=-1, keepdim=True) > 0).to(dtype=hs.dtype)
                marker_mask = torch.maximum(marker_mask, cat_mask)
            joint_input = torch.cat(
                [
                    pooled,
                    marker_repr,
                    pooled * marker_repr,
                    pooled - marker_repr,
                ],
                dim=-1,
            )
            compat_nli_gate = marker_mask
            compat_logic_gate = marker_mask
            if self.use_marker_compatibility_head:
                compat_hidden = self.compatibility_proj(joint_input)
                compat_hidden = compat_hidden * marker_mask
                compat_logits = self.compatibility_head(compat_hidden)
                compat_nli_logits = compat_logits[:, 0]
                compat_logic_logits = compat_logits[:, 1]
                compat_probs = torch.sigmoid(compat_logits) * marker_mask
                compat_nli_gate = compat_probs[:, :1]
                compat_logic_gate = compat_probs[:, 1:2]
                aux_outputs["compat_nli_logits"] = compat_nli_logits
                aux_outputs["compat_logic_logits"] = compat_logic_logits
                aux_outputs["compat_nli_probs"] = compat_nli_gate.squeeze(-1)
                aux_outputs["compat_logic_probs"] = compat_logic_gate.squeeze(-1)
            if self.use_base_delta_head:
                delta_hidden = self.delta_proj(joint_input)
                delta_hidden = delta_hidden * marker_mask
                gate_nli = marker_mask
                gate_logic = marker_mask
                if self.use_learned_marker_gate:
                    gate_values = torch.sigmoid(self.marker_gate(joint_input)) * marker_mask
                    gate_nli = gate_values[:, :1]
                    gate_logic = gate_values[:, 1:2]
                    aux_outputs["learned_gate_nli"] = gate_nli.squeeze(-1)
                    aux_outputs["learned_gate_logic"] = gate_logic.squeeze(-1)
                if self.use_compatibility_for_delta and self.use_marker_compatibility_head:
                    gate_nli = gate_nli * compat_nli_gate
                    gate_logic = gate_logic * compat_logic_gate
                delta_nli = self.delta_nli_head(delta_hidden)
                delta_logic = self.delta_logic_head(delta_hidden)
                logits_nli = self.base_nli_head(pooled) + (self.delta_scale_nli * gate_nli * delta_nli)
                logits_logic = self.base_logic_head(pooled) + (self.delta_scale_logic * gate_logic * delta_logic)
                logits_mrel = self.mrel_head(delta_hidden)
                if return_aux:
                    return pooled, logits_nli, logits_logic, logits_mrel, aux_outputs
                return pooled, logits_nli, logits_logic, logits_mrel
            head_input = torch.cat([pooled, marker_repr], dim=-1)

        logits_nli = self.nli_head(head_input)
        logits_logic = self.logic_head(head_input)
        logits_mrel = self.mrel_head(head_input)
        if return_aux:
            return pooled, logits_nli, logits_logic, logits_mrel, aux_outputs
        return pooled, logits_nli, logits_logic, logits_mrel
