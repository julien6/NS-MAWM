"""Agnostic NS-MAWM metrics."""

from __future__ import annotations

import torch

from ns_mawm.features import FeatureSchema, FeatureType


def rule_violation_rate(
    prediction: torch.Tensor,
    symbolic_values: torch.Tensor,
    symbolic_mask: torch.Tensor,
    schema: FeatureSchema,
) -> torch.Tensor:
    violations = torch.zeros_like(symbolic_mask, dtype=torch.bool)
    for spec in schema.specs:
        sl = schema.slice(spec.name)
        mask = symbolic_mask[..., sl]
        if not mask.any():
            continue
        pred = prediction[..., sl]
        sym = symbolic_values[..., sl]
        if spec.feature_type == FeatureType.CATEGORICAL:
            bad = (pred.argmax(dim=-1) != sym.argmax(dim=-1)).unsqueeze(-1).expand_as(mask)
        elif spec.feature_type == FeatureType.BINARY:
            bad = (pred > 0.5) != (sym > 0.5)
        else:
            bad = torch.abs(pred - sym) > spec.tolerance
        violations[..., sl] = bad & mask
    return violations.float().sum() / symbolic_mask.float().sum().clamp_min(1.0)


def projection_magnitude(raw_prediction: torch.Tensor, projected_prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return raw_prediction.new_tensor(0.0)
    return torch.abs(projected_prediction - raw_prediction)[mask].mean()


def residual_error(prediction: torch.Tensor, target: torch.Tensor, covered_mask: torch.Tensor) -> torch.Tensor:
    residual_mask = ~covered_mask
    if not residual_mask.any():
        return prediction.new_tensor(0.0)
    return torch.mean((prediction[residual_mask] - target[residual_mask]) ** 2)
