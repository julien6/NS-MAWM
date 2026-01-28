"""Rule Violation Rate (RVR) metric."""

from __future__ import annotations

import torch


def compute_rvr(
    y_pred: torch.Tensor,
    omega_d: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    viol = (torch.abs(y_pred - omega_d) > eps) & mask
    denom = mask.sum().clamp(min=1)
    return viol.sum().float() / denom.float()
