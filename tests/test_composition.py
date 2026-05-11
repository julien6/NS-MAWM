import torch

from env_adapters import make_environment
from marl_lib import MAPPOPolicy, QMIXPolicy, RandomPolicy, SACPolicy, ScriptedPolicy, collect_transitions
from ns_mawm import FeatureSchema, FeatureSpec, FeatureType, NSMAWM
from ns_mawm.core import IntegrationStrategy
from ns_mawm.metrics import rule_violation_rate
from experiments.registry import BASELINES
from ns_mawm.rules import RuleContext, RulePrediction, SymbolicRule, SymbolicWorldModel
from wm_lib import make_world_model


class FirstFeatureRule(SymbolicRule):
    rule_id = "first_feature"
    covered_features = ("f0",)

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        values[..., :1] = context.obs[..., :1]
        mask[..., :1] = True
        return RulePrediction(values, mask, (self.rule_id,))


def _schema(width: int) -> FeatureSchema:
    return FeatureSchema.from_specs([FeatureSpec(f"f{i}", FeatureType.CONTINUOUS) for i in range(width)])


def _policy(name: str, obs_dim: int, n_agents: int, action_dim: int):
    if name == "Random":
        return RandomPolicy(n_agents, action_dim)
    if name == "Scripted":
        return ScriptedPolicy(n_agents, action_dim)
    if name == "QMIX":
        return QMIXPolicy(obs_dim, n_agents, action_dim)
    if name == "SAC":
        return SACPolicy(obs_dim, n_agents, action_dim)
    return MAPPOPolicy(obs_dim, n_agents, action_dim)


def test_ns_mawm_composes_with_custom_wm_and_marl_on_predator_prey():
    env = make_environment("predator_prey")
    policy = MAPPOPolicy(env.obs_dim, env.n_agents, env.action_dim)
    batch = collect_transitions(env, policy, steps=6)
    wm = make_world_model("deterministic", env.obs_dim, env.n_agents * env.action_dim)
    model = NSMAWM(wm, SymbolicWorldModel([FirstFeatureRule()]), IntegrationStrategy.PROJECTION)
    out = model(batch.obs, batch.action, rollout=True)
    assert out.prediction.shape == batch.next_obs.shape
    assert rule_violation_rate(out.prediction, out.symbolic_values, out.symbolic_mask, _schema(env.obs_dim)) >= 0


def test_all_45_baselines_can_be_instantiated_as_compositions():
    env_name_map = {"GridCraft": "predator_prey", "Overcooked": "predator_prey", "PredatorPrey": "predator_prey", "SMACv2": "predator_prey"}
    wm_name_map = {"RSSM": "rssm", "Deterministic": "deterministic", "Transformer": "transformer"}
    for spec in BASELINES.values():
        env = make_environment(env_name_map.get(spec.environment, "predator_prey"))
        policy = _policy(spec.policy, env.obs_dim, env.n_agents, env.action_dim)
        obs = env.reset(seed=0).reshape(1, -1)
        action = policy.act(obs)
        assert action.shape[-1] == env.n_agents * env.action_dim
        if spec.wm != "MF":
            wm = make_world_model(wm_name_map[spec.wm], env.obs_dim, env.n_agents * env.action_dim)
            strategy = spec.strategy if spec.strategy != "-" else "none"
            model = NSMAWM(wm, SymbolicWorldModel([FirstFeatureRule()] if spec.coverage > 0 else []), strategy)
            out = model(obs, action, rollout=True)
            assert out.prediction.shape[-1] == env.obs_dim


def test_gridcraft_adapter_instantiates_when_package_is_available():
    env = make_environment("gridcraft", n_agents=2, max_steps=3)
    obs = env.reset(seed=0)
    action = RandomPolicy(env.n_agents, env.action_dim).act(obs.reshape(1, -1)).reshape(-1)
    next_obs, reward, done, info = env.step(action)
    assert next_obs.numel() == env.obs_dim
    assert reward.shape == (1,)
