#!/usr/bin/env python
"""Compose one baseline from env_adapters + marl_lib + wm_lib + ns_mawm."""

from __future__ import annotations

import torch

from env_adapters import make_environment
from marl_lib import MAPPOPolicy, collect_transitions
from ns_mawm import FeatureSchema, FeatureSpec, FeatureType, NSMAWM
from ns_mawm.core import IntegrationStrategy
from ns_mawm.rules import RulePrediction, SymbolicRule, SymbolicWorldModel, RuleContext
from wm_lib import make_world_model


class FirstFeaturePersistenceRule(SymbolicRule):
    rule_id = "first_feature_persistence"
    covered_features = ("f0",)

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        values[..., :1] = context.obs[..., :1]
        mask[..., :1] = True
        return RulePrediction(values, mask, (self.rule_id,))


def main() -> None:
    env = make_environment("predator_prey")
    schema = FeatureSchema.from_specs([FeatureSpec(f"f{i}", FeatureType.CONTINUOUS) for i in range(env.obs_dim)])
    policy = MAPPOPolicy(env.obs_dim, env.n_agents, env.action_dim)
    batch = collect_transitions(env, policy, steps=8)
    wm = make_world_model("rssm", env.obs_dim, env.n_agents * env.action_dim)
    ns_model = NSMAWM(wm, SymbolicWorldModel([FirstFeaturePersistenceRule()]), IntegrationStrategy.PROJECTION)
    out = ns_model(batch.obs, batch.action, rollout=True)
    print({"obs_shape": tuple(out.prediction.shape), "coverage": schema.coverage(out.symbolic_mask)})


if __name__ == "__main__":
    main()
