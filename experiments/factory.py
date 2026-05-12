"""Factories that compose env_adapters, marl_lib, wm_lib, and ns_mawm."""

from __future__ import annotations

from env_adapters import make_environment
from env_adapters.rules import rules_for_coverage
from marl_lib import MAPPOPolicy, QMIXPolicy, RandomPolicy, SACPolicy, ScriptedPolicy
from ns_mawm import NSMAWM
from ns_mawm.core import IntegrationStrategy
from ns_mawm.rules import SymbolicWorldModel
from wm_lib import make_world_model

from experiments.config import ExperimentConfig


def make_policy(name: str, obs_dim: int, n_agents: int, action_dim: int):
    if name == "Random":
        return RandomPolicy(n_agents, action_dim)
    if name == "Scripted":
        return ScriptedPolicy(n_agents, action_dim)
    if name == "QMIX":
        return QMIXPolicy(obs_dim, n_agents, action_dim)
    if name == "SAC":
        return SACPolicy(obs_dim, n_agents, action_dim)
    return MAPPOPolicy(obs_dim, n_agents, action_dim)


def make_components(config: ExperimentConfig):
    env = make_environment(config.environment)
    policy = make_policy(config.policy, env.obs_dim, env.n_agents, env.action_dim)
    if config.world_model == "none":
        return env, policy, None, None
    output_dim = None
    rules = rules_for_coverage(
        config.environment,
        env.schema,
        config.coverage,
        n_agents=env.n_agents,
        width=getattr(env, "width", 7),
        height=getattr(env, "height", 7),
    )
    symbolic = SymbolicWorldModel(rules, conflict_policy="last")
    if config.strategy == "residual" and rules:
        covered = env.schema.mask(symbolic.covered_features)
        output_dim = int((~covered).sum().item())
    wm = make_world_model(
        config.world_model,
        env.obs_dim,
        env.n_agents * env.action_dim,
        output_dim=output_dim,
        kl_free_nats=config.kl_free_nats,
        kl_balance=config.kl_balance,
    )
    ns_model = NSMAWM(wm, symbolic, IntegrationStrategy(config.strategy), schema=env.schema)
    return env, policy, wm, ns_model
