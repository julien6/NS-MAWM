"""Overcooked-AI specific symbolic rules (minimal)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from nsmawm.symbolic.rule import Rule, RuleContext, RuleResult


@dataclass
class StayPutRule(Rule):
    """If action=stay, position features remain unchanged."""

    stay_action_index: int
    position_indices: Optional[Tuple[int, int]] = None
    feature_indices: Optional[dict] = None

    def apply(self, context: RuleContext) -> RuleResult:
        obs = context.obs_t
        act = context.act_t
        stay = act[..., self.stay_action_index] > 0.5
        x_idx, y_idx = self._resolve_position_indices()

        mask = torch.zeros_like(obs, dtype=torch.bool)
        pos_mask = stay.unsqueeze(-1).expand(*stay.shape, 2)
        mask[..., (x_idx, y_idx)] = pos_mask

        values = torch.where(mask, obs, torch.zeros_like(obs))
        return RuleResult(values=values, mask=mask)

    def _resolve_position_indices(self) -> Tuple[int, int]:
        if self.position_indices is not None:
            return self.position_indices
        if self.feature_indices is None:
            return (0, 1)
        positions = self.feature_indices.get("positions", {})
        return (positions.get("x", 0), positions.get("y", 1))


@dataclass
class PositionBoundsRule(Rule):
    """Clamp position features to be within the grid bounds."""

    grid_shape: Tuple[int, int]
    position_indices: Optional[Tuple[int, int]] = None
    feature_indices: Optional[dict] = None

    def apply(self, context: RuleContext) -> RuleResult:
        obs = context.obs_t
        values = obs.clone()
        width, height = self.grid_shape
        x_idx, y_idx = self._resolve_position_indices()

        values[..., x_idx] = obs[..., x_idx].clamp(0.0, float(width - 1))
        values[..., y_idx] = obs[..., y_idx].clamp(0.0, float(height - 1))

        mask = torch.zeros_like(obs, dtype=torch.bool)
        mask[..., x_idx] = True
        mask[..., y_idx] = True

        return RuleResult(values=values, mask=mask)

    def _resolve_position_indices(self) -> Tuple[int, int]:
        if self.position_indices is not None:
            return self.position_indices
        if self.feature_indices is None:
            return (0, 1)
        positions = self.feature_indices.get("positions", {})
        return (positions.get("x", 0), positions.get("y", 1))


def make_overcooked_rules(
    *,
    stay_action_index: int,
    grid_shape: Tuple[int, int],
    feature_indices: Optional[dict] = None,
) -> list[Rule]:
    """Convenience helper to build minimal Overcooked rules."""
    return [
        StayPutRule(stay_action_index=stay_action_index, feature_indices=feature_indices),
        PositionBoundsRule(grid_shape=grid_shape, feature_indices=feature_indices),
    ]
