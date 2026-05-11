"""Environment-specific symbolic rule inventories.

This module may name environments. The agnostic `ns_mawm` package must not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ns_mawm.features import FeatureSchema
from ns_mawm.rules import RuleContext, RulePrediction, SymbolicRule, select_rules_for_coverage


@dataclass
class FeaturePersistenceRule(SymbolicRule):
    rule_id: str
    covered_features: tuple[str, ...]
    schema: FeatureSchema

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        feature_mask = self.schema.mask(self.covered_features, device=context.obs.device)
        values[..., feature_mask] = context.obs[..., feature_mask]
        mask[..., feature_mask] = True
        return RulePrediction(values, mask, (self.rule_id,))


@dataclass
class BoundedFeatureRule(SymbolicRule):
    rule_id: str
    covered_features: tuple[str, ...]
    schema: FeatureSchema
    low: float
    high: float

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        feature_mask = self.schema.mask(self.covered_features, device=context.obs.device)
        values[..., feature_mask] = context.obs[..., feature_mask].clamp(self.low, self.high)
        mask[..., feature_mask] = True
        return RulePrediction(values, mask, (self.rule_id,))


class PredatorPreyBoundaryRule:
    rule_id = "predator_prey_boundary_motion"

    def __init__(self, schema: FeatureSchema, n_agents: int, width: int, height: int):
        self.schema = schema
        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.covered_features = tuple(
            f"predator_{i}.{axis}" for i in range(n_agents) for axis in ("x", "y")
        )

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        actions = context.action.reshape(context.action.shape[0], self.n_agents, 5)
        for i in range(self.n_agents):
            x_sl = self.schema.slice(f"predator_{i}.x")
            y_sl = self.schema.slice(f"predator_{i}.y")
            x = context.obs[..., x_sl]
            y = context.obs[..., y_sl]
            idx = actions[:, i].argmax(dim=-1, keepdim=True)
            next_x = x + (idx == 4).float() - (idx == 3).float()
            next_y = y + (idx == 2).float() - (idx == 1).float()
            values[..., x_sl] = next_x.clamp(0, self.width - 1)
            values[..., y_sl] = next_y.clamp(0, self.height - 1)
            mask[..., x_sl] = True
            mask[..., y_sl] = True
        return RulePrediction(values, mask, (self.rule_id,))


def rule_inventory(environment_name: str, schema: FeatureSchema, **kwargs) -> list[SymbolicRule]:
    key = environment_name.lower().replace("-", "_")
    if key in {"predator_prey", "predatorprey"}:
        return [
            PredatorPreyBoundaryRule(
                schema,
                n_agents=int(kwargs.get("n_agents", 2)),
                width=int(kwargs.get("width", 7)),
                height=int(kwargs.get("height", 7)),
            ),
            FeaturePersistenceRule("predator_prey_prey_persistence", ("prey.x", "prey.y"), schema),
        ]
    if key in {"gridcraft", "grid_craft"}:
        position_like = tuple(spec.name for spec in schema.specs if spec.family in {"agent_state"} and spec.name.endswith(("0", "1")))
        fallback = tuple(spec.name for spec in schema.specs[: max(1, min(4, len(schema.specs)))])
        return [
            FeaturePersistenceRule("gridcraft_local_persistence", position_like or fallback, schema),
            BoundedFeatureRule("gridcraft_feature_bounds", fallback, schema, low=0, high=99),
        ]
    if key in {"overcooked", "overcooked_ai"}:
        first = tuple(spec.name for spec in schema.specs[: max(1, min(6, len(schema.specs)))])
        return [FeaturePersistenceRule("overcooked_position_orientation_persistence", first, schema)]
    if key in {"smac", "smacv2"}:
        first = tuple(spec.name for spec in schema.specs[: max(1, min(8, len(schema.specs)))])
        return [
            FeaturePersistenceRule("smac_position_action_mask_persistence", first, schema),
            BoundedFeatureRule("smac_health_bounds", first, schema, low=-1, high=1),
        ]
    return []


def rules_for_coverage(environment_name: str, schema: FeatureSchema, coverage: float, **kwargs) -> list[SymbolicRule]:
    return select_rules_for_coverage(schema, rule_inventory(environment_name, schema, **kwargs), coverage)
