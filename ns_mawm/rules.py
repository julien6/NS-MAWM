"""Agnostic symbolic transition rules."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

import torch

from ns_mawm.features import FeatureSchema


@dataclass(frozen=True)
class RuleContext:
    obs: torch.Tensor
    action: torch.Tensor
    history: torch.Tensor | None = None
    info: dict[str, object] | None = None


@dataclass(frozen=True)
class RulePrediction:
    values: torch.Tensor
    mask: torch.Tensor
    rule_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleConflict:
    rule_id: str
    overlap_count: int


@dataclass(frozen=True)
class RuleAudit:
    coverage: float
    covered_features: tuple[str, ...]
    conflicts: tuple[RuleConflict, ...]
    family_coverage: dict[str, float]


class SymbolicRule(Protocol):
    rule_id: str
    covered_features: tuple[str, ...]

    def predict(self, context: RuleContext) -> RulePrediction: ...


class SymbolicWorldModel:
    def __init__(self, rules: list[SymbolicRule], conflict_policy: str = "last"):
        self.rules = rules
        self.conflict_policy = conflict_policy
        self.last_conflicts: tuple[RuleConflict, ...] = ()

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        rule_ids: list[str] = []
        conflicts: list[RuleConflict] = []
        for rule in self.rules:
            pred = rule.predict(context)
            overlap = mask & pred.mask
            if overlap.any():
                conflict = RuleConflict(rule.rule_id, int(overlap.sum().item()))
                conflicts.append(conflict)
                if self.conflict_policy == "error":
                    raise ValueError(f"Symbolic rule conflict: {rule.rule_id}")
            write = pred.mask if self.conflict_policy == "last" else pred.mask & ~mask
            values = torch.where(write, pred.values, values)
            mask = mask | pred.mask
            rule_ids.append(rule.rule_id)
        self.last_conflicts = tuple(conflicts)
        return RulePrediction(values, mask, tuple(rule_ids))

    @property
    def covered_features(self) -> tuple[str, ...]:
        names: list[str] = []
        for rule in self.rules:
            names.extend(rule.covered_features)
        return tuple(dict.fromkeys(names))

    def audit(self, schema: FeatureSchema, sample_obs: torch.Tensor | None = None) -> RuleAudit:
        covered = schema.mask(self.covered_features)
        if sample_obs is not None and sample_obs.ndim > 1:
            covered_eval = covered.expand_as(sample_obs)
            coverage = schema.coverage(covered_eval)
            family = schema.family_summary(covered_eval)
        else:
            coverage = schema.coverage(covered)
            family = schema.family_summary(covered)
        return RuleAudit(coverage, self.covered_features, self.last_conflicts, family)


def select_rules_for_coverage(
    schema: FeatureSchema,
    rules: list[SymbolicRule],
    target_coverage: float,
) -> list[SymbolicRule]:
    if target_coverage <= 0:
        return []
    selected: list[SymbolicRule] = []
    for rule in rules:
        selected.append(rule)
        if schema.coverage(schema.mask(_covered_names(selected))) >= target_coverage:
            break
    return selected


def dropout_rules(rules: list[SymbolicRule], rate: float, seed: int) -> list[SymbolicRule]:
    rng = random.Random(seed)
    return [rule for rule in rules if rng.random() >= rate]


class NoisyRule:
    def __init__(self, base_rule: SymbolicRule, std: float, seed: int = 0):
        self.base_rule = base_rule
        self.std = std
        self.seed = seed
        self.rule_id = f"noisy:{base_rule.rule_id}"
        self.covered_features = base_rule.covered_features

    def predict(self, context: RuleContext) -> RulePrediction:
        pred = self.base_rule.predict(context)
        generator = torch.Generator(device=pred.values.device).manual_seed(self.seed)
        noise = torch.randn(pred.values.shape, generator=generator, device=pred.values.device) * self.std
        return RulePrediction(torch.where(pred.mask, pred.values + noise, pred.values), pred.mask, (self.rule_id,))


def _covered_names(rules: list[SymbolicRule]) -> tuple[str, ...]:
    names: list[str] = []
    for rule in rules:
        names.extend(rule.covered_features)
    return tuple(dict.fromkeys(names))
