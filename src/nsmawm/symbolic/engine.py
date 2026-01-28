"""Symbolic rule engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

import torch

from nsmawm.symbolic.rule import Rule, RuleContext, RuleResult


@dataclass
class EngineOutput:
    omega_d: torch.Tensor
    mask: torch.Tensor


class RuleEngine:
    """Aggregate symbolic rules to produce partial predictions and masks."""

    def __init__(self, rules: Iterable[Rule], collision: Literal["last", "error"] = "last"):
        self.rules: List[Rule] = list(rules)
        self.collision = collision

    def apply(self, context: RuleContext) -> EngineOutput:
        if not self.rules:
            shape = context.obs_t.shape
            device = context.obs_t.device
            omega_d = torch.zeros(shape, device=device)
            mask = torch.zeros(shape, dtype=torch.bool, device=device)
            return EngineOutput(omega_d=omega_d, mask=mask)

        omega_d = torch.zeros_like(context.obs_t)
        mask = torch.zeros_like(context.obs_t, dtype=torch.bool)
        for rule in self.rules:
            result: RuleResult = rule.apply(context)
            if self.collision == "error" and torch.any(mask & result.mask):
                raise ValueError("Rule collision detected")
            if self.collision == "last":
                omega_d = torch.where(result.mask, result.values, omega_d)
            else:
                omega_d = torch.where(result.mask & ~mask, result.values, omega_d)
            mask = mask | result.mask
        return EngineOutput(omega_d=omega_d, mask=mask)

    def __len__(self) -> int:
        return len(self.rules)
