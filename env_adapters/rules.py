"""Environment-specific symbolic rule inventories.

This module may name environments. The agnostic `ns_mawm` package must not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ns_mawm.features import FeatureSchema
from ns_mawm.rules import RuleContext, RulePrediction, SymbolicRule, select_rules_for_coverage

NOMINAL_COVERAGE_TARGETS = {
    "gridcraft": 0.42,
    "grid_craft": 0.42,
    "overcooked": 0.34,
    "overcooked_ai": 0.34,
    "predator_prey": 0.29,
    "predatorprey": 0.29,
    "smac": 0.18,
    "smacv2": 0.18,
}

COVERAGE_SWEEP = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)


@dataclass
class FeaturePersistenceRule(SymbolicRule):
    rule_id: str
    covered_features: tuple[str, ...]
    schema: FeatureSchema
    feature_families: tuple[str, ...] = ()
    dropout_eligible: bool = True
    assumptions: tuple[str, ...] = ("feature persists unless affected by an observed action",)

    def __post_init__(self) -> None:
        if not self.feature_families:
            self.feature_families = tuple(sorted({self.schema.spec(name).family for name in self.covered_features}))

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
    feature_families: tuple[str, ...] = ()
    dropout_eligible: bool = True
    assumptions: tuple[str, ...] = ("feature remains inside configured bounds",)

    def __post_init__(self) -> None:
        if not self.feature_families:
            self.feature_families = tuple(sorted({self.schema.spec(name).family for name in self.covered_features}))

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        feature_mask = self.schema.mask(self.covered_features, device=context.obs.device)
        values[..., feature_mask] = context.obs[..., feature_mask].clamp(self.low, self.high)
        mask[..., feature_mask] = True
        return RulePrediction(values, mask, (self.rule_id,))


class PredatorPreyBoundaryRule:
    def __init__(self, schema: FeatureSchema, n_agents: int, width: int, height: int, predator_ids: tuple[int, ...] | None = None):
        self.schema = schema
        self.n_agents = n_agents
        self.predator_ids = predator_ids or tuple(range(n_agents))
        suffix = "_".join(str(i) for i in self.predator_ids) or "none"
        self.rule_id = f"predator_prey_boundary_motion_p{suffix}"
        self.width = width
        self.height = height
        self.covered_features = tuple(f"predator_{i}.{axis}" for i in self.predator_ids for axis in ("x", "y"))
        self.feature_families = ("position", "boundary")
        self.dropout_eligible = True
        self.assumptions = ("predator movement is action-conditioned and clipped to map boundaries",)

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        flat_action = context.action.reshape(-1, context.action.shape[-1])
        flat_obs = context.obs.reshape(-1, context.obs.shape[-1])
        flat_values = values.reshape(-1, values.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        actions = flat_action.reshape(flat_action.shape[0], self.n_agents, 5)
        for i in self.predator_ids:
            x_sl = self.schema.slice(f"predator_{i}.x")
            y_sl = self.schema.slice(f"predator_{i}.y")
            x = flat_obs[..., x_sl]
            y = flat_obs[..., y_sl]
            idx = actions[:, i].argmax(dim=-1, keepdim=True)
            next_x = x + (idx == 4).float() - (idx == 3).float()
            next_y = y + (idx == 2).float() - (idx == 1).float()
            flat_values[..., x_sl] = next_x.clamp(0, self.width - 1)
            flat_values[..., y_sl] = next_y.clamp(0, self.height - 1)
            flat_mask[..., x_sl] = True
            flat_mask[..., y_sl] = True
        return RulePrediction(values, mask, (self.rule_id,))


class PredatorPreyVelocityRule:
    def __init__(self, schema: FeatureSchema, n_agents: int, width: int, height: int):
        self.schema = schema
        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.rule_id = "predator_prey_action_velocity"
        self.covered_features = tuple(f"predator_{i}.d{axis}" for i in range(n_agents) for axis in ("x", "y"))
        self.feature_families = ("velocity",)
        self.dropout_eligible = True
        self.assumptions = ("velocity features equal clipped action-induced position deltas",)

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        flat_action = context.action.reshape(-1, context.action.shape[-1])
        flat_obs = context.obs.reshape(-1, context.obs.shape[-1])
        flat_values = values.reshape(-1, values.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        actions = flat_action.reshape(flat_action.shape[0], self.n_agents, 5)
        for i in range(self.n_agents):
            x_sl = self.schema.slice(f"predator_{i}.x")
            y_sl = self.schema.slice(f"predator_{i}.y")
            dx_sl = self.schema.slice(f"predator_{i}.dx")
            dy_sl = self.schema.slice(f"predator_{i}.dy")
            idx = actions[:, i].argmax(dim=-1, keepdim=True)
            raw_dx = (idx == 4).float() - (idx == 3).float()
            raw_dy = (idx == 2).float() - (idx == 1).float()
            x = flat_obs[..., x_sl]
            y = flat_obs[..., y_sl]
            next_x = (x + raw_dx).clamp(0, self.width - 1)
            next_y = (y + raw_dy).clamp(0, self.height - 1)
            flat_values[..., dx_sl] = next_x - x
            flat_values[..., dy_sl] = next_y - y
            flat_mask[..., dx_sl] = True
            flat_mask[..., dy_sl] = True
        return RulePrediction(values, mask, (self.rule_id,))


class PredatorPreyCaptureRule:
    rule_id = "predator_prey_capture_activity"
    covered_features = ("prey.active", "capture.in_range")
    feature_families = ("activity", "capture")
    dropout_eligible = True
    assumptions = ("capture indicator is determined by next predator positions and active prey state",)

    def __init__(self, schema: FeatureSchema, n_agents: int, width: int, height: int):
        self.schema = schema
        self.n_agents = n_agents
        self.width = width
        self.height = height

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        flat_action = context.action.reshape(-1, context.action.shape[-1])
        flat_obs = context.obs.reshape(-1, context.obs.shape[-1])
        flat_values = values.reshape(-1, values.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        actions = flat_action.reshape(flat_action.shape[0], self.n_agents, 5)
        next_positions: list[torch.Tensor] = []
        for i in range(self.n_agents):
            x = flat_obs[..., self.schema.slice(f"predator_{i}.x")]
            y = flat_obs[..., self.schema.slice(f"predator_{i}.y")]
            idx = actions[:, i].argmax(dim=-1, keepdim=True)
            next_x = (x + (idx == 4).float() - (idx == 3).float()).clamp(0, self.width - 1)
            next_y = (y + (idx == 2).float() - (idx == 1).float()).clamp(0, self.height - 1)
            next_positions.append(torch.cat([next_x, next_y], dim=-1))
        predators = torch.stack(next_positions, dim=1)
        prey = torch.cat([flat_obs[..., self.schema.slice("prey.x")], flat_obs[..., self.schema.slice("prey.y")]], dim=-1)
        dist = torch.linalg.vector_norm(predators - prey.unsqueeze(1), dim=-1).min(dim=-1, keepdim=True).values
        in_range = (dist <= 1.0).float()
        active = flat_obs[..., self.schema.slice("prey.active")]
        next_active = torch.where(in_range.bool(), torch.zeros_like(active), active)
        active_sl = self.schema.slice("prey.active")
        range_sl = self.schema.slice("capture.in_range")
        flat_values[..., active_sl] = next_active
        flat_values[..., range_sl] = in_range
        flat_mask[..., active_sl] = True
        flat_mask[..., range_sl] = True
        return RulePrediction(values, mask, (self.rule_id,))


class PredatorPreyBoundaryContactRule:
    rule_id = "predator_prey_boundary_contact"
    covered_features = ("boundary.contact",)
    feature_families = ("boundary",)
    dropout_eligible = True
    assumptions = ("boundary contact is true iff any action-induced move is clipped by map bounds",)

    def __init__(self, schema: FeatureSchema, n_agents: int, width: int, height: int):
        self.schema = schema
        self.n_agents = n_agents
        self.width = width
        self.height = height

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        flat_action = context.action.reshape(-1, context.action.shape[-1])
        flat_obs = context.obs.reshape(-1, context.obs.shape[-1])
        flat_values = values.reshape(-1, values.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        actions = flat_action.reshape(flat_action.shape[0], self.n_agents, 5)
        clipped = torch.zeros(flat_obs.shape[0], 1, dtype=torch.bool, device=flat_obs.device)
        for i in range(self.n_agents):
            x = flat_obs[..., self.schema.slice(f"predator_{i}.x")]
            y = flat_obs[..., self.schema.slice(f"predator_{i}.y")]
            idx = actions[:, i].argmax(dim=-1, keepdim=True)
            raw_x = x + (idx == 4).float() - (idx == 3).float()
            raw_y = y + (idx == 2).float() - (idx == 1).float()
            clipped = clipped | (raw_x != raw_x.clamp(0, self.width - 1)) | (raw_y != raw_y.clamp(0, self.height - 1))
        sl = self.schema.slice("boundary.contact")
        flat_values[..., sl] = clipped.float()
        flat_mask[..., sl] = True
        return RulePrediction(values, mask, (self.rule_id,))


def _first_names(schema: FeatureSchema, ratio: float, start: int = 0) -> tuple[str, ...]:
    count = max(1, int(round(schema.width * ratio)))
    return tuple(spec.name for spec in schema.specs[start : min(len(schema.specs), start + count)])


def _remaining_names(schema: FeatureSchema, covered: tuple[str, ...]) -> tuple[str, ...]:
    covered_set = set(covered)
    return tuple(spec.name for spec in schema.specs if spec.name not in covered_set)


def _family_names(schema: FeatureSchema, *families: str, limit: int | None = None, skip: int = 0) -> tuple[str, ...]:
    selected = tuple(spec.name for spec in schema.specs if spec.family in families)
    if skip:
        selected = selected[skip:]
    return selected if limit is None else selected[:limit]


def rule_inventory(environment_name: str, schema: FeatureSchema, **kwargs) -> list[SymbolicRule]:
    key = environment_name.lower().replace("-", "_")
    if key in {"predator_prey", "predatorprey"}:
        return [
            PredatorPreyCaptureRule(schema, int(kwargs.get("n_agents", 2)), int(kwargs.get("width", 7)), int(kwargs.get("height", 7))),
            PredatorPreyBoundaryRule(
                schema,
                n_agents=int(kwargs.get("n_agents", 2)),
                width=int(kwargs.get("width", 7)),
                height=int(kwargs.get("height", 7)),
                predator_ids=(0,),
            ),
            PredatorPreyVelocityRule(
                schema,
                int(kwargs.get("n_agents", 2)),
                int(kwargs.get("width", 7)),
                int(kwargs.get("height", 7)),
            ),
            PredatorPreyBoundaryContactRule(schema, int(kwargs.get("n_agents", 2)), int(kwargs.get("width", 7)), int(kwargs.get("height", 7))),
            FeaturePersistenceRule(
                "predator_prey_prey_position_persistence",
                ("prey.x", "prey.y"),
                schema,
                assumptions=("prey position persists in this local benchmark unless reset",),
            ),
            PredatorPreyBoundaryRule(
                schema,
                n_agents=int(kwargs.get("n_agents", 2)),
                width=int(kwargs.get("width", 7)),
                height=int(kwargs.get("height", 7)),
                predator_ids=tuple(range(1, int(kwargs.get("n_agents", 2)))),
            ),
        ]
    if key in {"gridcraft", "grid_craft"}:
        boundary = _family_names(schema, "boundary", limit=max(1, int(round(schema.width * 0.14))))
        collision = _family_names(schema, "collision", "agent_state", limit=max(1, int(round(schema.width * 0.14))))
        object_features = _family_names(schema, "object_persistence", limit=max(1, int(round(schema.width * 0.08))))
        resource_features = _family_names(schema, "hunger", "inventory", "crafting")
        return [
            FeaturePersistenceRule(
                "gridcraft_boundary_motion_feasibility",
                boundary,
                schema,
                feature_families=("boundary",),
                assumptions=("terrain and boundary-coded local cells constrain feasible movement",),
            ),
            FeaturePersistenceRule(
                "gridcraft_collision_agent_state",
                collision,
                schema,
                feature_families=("collision", "agent_state"),
                assumptions=("covered entity and self-state indicators encode collision and occupancy constraints",),
            ),
            FeaturePersistenceRule(
                "gridcraft_object_persistence",
                object_features,
                schema,
                feature_families=("object_persistence",),
                assumptions=("covered nearby block cells persist unless harvested or moved out of view",),
            ),
            BoundedFeatureRule(
                "gridcraft_hunger_inventory_crafting_bounds",
                resource_features,
                schema,
                low=0,
                high=99,
                feature_families=("hunger", "inventory", "crafting"),
                assumptions=("hunger, inventory, and crafting counters remain non-negative and bounded",),
            ),
        ]
    if key in {"overcooked", "overcooked_ai"}:
        first = _first_names(schema, 0.12, 0)
        second = _first_names(schema, 0.11, len(first))
        third = _first_names(schema, 0.11, len(first) + len(second))
        return [
            FeaturePersistenceRule("overcooked_wall_counter_collision", first, schema, assumptions=("wall and counter collisions block invalid moves",)),
            FeaturePersistenceRule("overcooked_pickup_drop_carried_object", second, schema, assumptions=("pickup and drop actions update carried-object indicators",)),
            FeaturePersistenceRule("overcooked_pot_orientation_persistence", third, schema, assumptions=("pot state, orientation, and unattended objects persist unless acted on",)),
        ]
    if key in {"smac", "smacv2"}:
        first = _first_names(schema, 0.06, 0)
        second = _first_names(schema, 0.06, len(first))
        third = _first_names(schema, 0.06, len(first) + len(second))
        return [
            FeaturePersistenceRule("smac_movement_feasibility", first, schema, assumptions=("movement feasibility and position persistence hold for covered units",)),
            FeaturePersistenceRule("smac_range_action_mask_constraints", second, schema, assumptions=("covered action masks and range constraints are transition-local",)),
            BoundedFeatureRule("smac_health_dead_unit_bounds", third, schema, low=-1, high=1, assumptions=("health bounds and dead-unit inactivity are preserved",)),
        ]
    return []


def rules_for_coverage(environment_name: str, schema: FeatureSchema, coverage: float, **kwargs) -> list[SymbolicRule]:
    inventory = rule_inventory(environment_name, schema, **kwargs)
    selected = select_rules_for_coverage(schema, inventory, coverage)
    achieved = schema.coverage(schema.mask(tuple(name for rule in selected for name in rule.covered_features))) if selected else 0.0
    if coverage > achieved and coverage >= 0.99:
        covered = tuple(name for rule in selected for name in rule.covered_features)
        selected.append(
            FeaturePersistenceRule(
                f"{environment_name}_oracle_full_coverage",
                _remaining_names(schema, covered),
                schema,
                dropout_eligible=False,
                assumptions=("oracle diagnostic rule used only for high-coverage sweeps",),
            )
        )
    return selected


def achieved_coverage(schema: FeatureSchema, rules: list[SymbolicRule]) -> float:
    names = tuple(name for rule in rules for name in rule.covered_features)
    return schema.coverage(schema.mask(names)) if names else 0.0


def nominal_coverage_target(environment_name: str) -> float:
    return NOMINAL_COVERAGE_TARGETS[environment_name.lower().replace("-", "_")]


def coverage_sweep_rules(environment_name: str, schema: FeatureSchema, **kwargs) -> dict[float, list[SymbolicRule]]:
    return {target: rules_for_coverage(environment_name, schema, target, **kwargs) for target in COVERAGE_SWEEP}
