import torch
from torch import nn
import pytest

from env_adapters import make_environment
from env_adapters.rules import achieved_coverage, nominal_coverage_target, rules_for_coverage
from experiments.reproduction_check import validate_paired_comparison, validate_reproduction_logs
from experiments.baseline_configs import all_baseline_configs, materialize_baseline_configs
from experiments.config import ExperimentConfig
from experiments.factory import make_components
from experiments.launch import executable_stages, write_manifest
from experiments.logging import aggregate_mean_sem
from experiments.matrix_status import build_status_table
from experiments.paper_readiness import build_readiness_report
from marl_lib import MAPPOPolicy, collect_transitions
from marl_lib.rollout import ReplayBuffer
from ns_mawm import FeatureSchema, FeatureSpec, FeatureType, NSMAWM
from ns_mawm.metrics import rule_violation_rate
from ns_mawm.rules import RuleContext, RulePrediction, SymbolicWorldModel
from wm_lib.models import RSSMWorldModel
from wm_lib.models import WorldModelOutput


class ConstantWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(2.0))

    def forward(self, obs, action, state=None):
        pred = torch.zeros_like(obs) + self.bias
        return WorldModelOutput(pred, torch.zeros(obs.shape[0], 1), torch.zeros(obs.shape[0], 1), state, {})


class FirstFeatureZeroRule:
    rule_id = "first_zero"
    covered_features = ("x",)
    feature_families = ("position",)
    dropout_eligible = True

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        mask[..., 0] = True
        return RulePrediction(values, mask, (self.rule_id,))


def _schema():
    return FeatureSchema.from_specs(
        [
            FeatureSpec("x", FeatureType.CONTINUOUS, family="position"),
            FeatureSpec("y", FeatureType.CONTINUOUS, family="position"),
        ]
    )


def test_smoke_mode_limits_seeds_full_mode_uses_all_seeds():
    cfg = ExperimentConfig(mode="smoke", seeds=(0, 1, 2))
    assert cfg.active_seeds == (0,)
    assert cfg.active_horizons == cfg.smoke_horizons
    full = ExperimentConfig(mode="full", seeds=(0, 1, 2))
    assert full.active_seeds == (0, 1, 2)
    assert full.active_horizons == full.horizons


def test_adapter_variants_are_deterministic_and_split_named():
    env = make_environment("predator_prey")
    variant = env.make_variants("UV")[0]
    obs_a = env.reset(seed=5, variant=variant)
    obs_b = env.reset(seed=5, variant=variant)
    assert torch.equal(obs_a, obs_b)
    assert "uv" in env.variant_id


def test_projection_changes_rollout_regularization_does_not():
    obs = torch.ones(2, 2)
    action = torch.zeros(2, 1)
    symbolic = SymbolicWorldModel([FirstFeatureZeroRule()])
    reg = NSMAWM(ConstantWM(), symbolic, "regularization", schema=_schema())
    proj = NSMAWM(ConstantWM(), symbolic, "projection", schema=_schema())
    assert torch.equal(reg(obs, action, rollout=True).prediction[:, 0], torch.full((2,), 2.0))
    assert torch.equal(proj(obs, action, rollout=True).prediction[:, 0], torch.zeros(2))


def test_all_baseline_configs_smoke_instantiate_without_world_model_for_mf():
    configs = all_baseline_configs(ExperimentConfig(mode="smoke"))
    assert len(configs) == 45
    env, policy, wm, model = make_components(configs["B01"])
    assert wm is None and model is None
    batch = collect_transitions(env, policy, steps=2, seed=0)
    assert batch.obs.shape[0] == 2
    env, policy, wm, model = make_components(configs["B20"])
    assert wm is not None and model is not None
    assert policy.act(env.reset(seed=0).reshape(1, -1)).shape[-1] == env.n_agents * env.action_dim


def test_aggregation_reports_std_sem_ci(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("baseline_id,seed,rvr\nB20,0,0.2\nB20,1,0.4\n", encoding="utf-8")
    table = aggregate_mean_sem(str(path), ["baseline_id"], ["rvr"])
    assert {"rvr_mean", "rvr_std", "rvr_sem", "rvr_ci95"}.issubset(table.columns)


def test_materialized_configs_include_smoke_and_full_for_all_baselines(tmp_path):
    written = materialize_baseline_configs(tmp_path)
    assert len(written) == 90
    assert (tmp_path / "smoke" / "B20.yaml").exists()
    assert (tmp_path / "full" / "B20.yaml").exists()


def test_launcher_skips_wm_only_stages_for_model_free_baselines_and_writes_manifest(tmp_path):
    mf = ExperimentConfig(mode="smoke", baseline_id="B01", world_model="none", output_dir=str(tmp_path))
    wm = ExperimentConfig(mode="smoke", baseline_id="B20", world_model="rssm", output_dir=str(tmp_path))
    stages = ["train", "rollout", "planning", "rule_dropout"]
    assert executable_stages(mf, stages) == ["train"]
    assert executable_stages(wm, stages) == stages
    manifest = write_manifest(tmp_path, configs={"B01": mf, "B20": wm}, requested_stages=stages, dry_run=True)
    text = manifest.read_text(encoding="utf-8")
    assert '"B01"' in text and '"B20"' in text
    assert '"dry_run": true' in text


def test_replay_buffer_sequence_sampling_shape():
    env = make_environment("predator_prey")
    policy = MAPPOPolicy(env.obs_dim, env.n_agents, env.action_dim)
    batch = collect_transitions(env, policy, steps=5, seed=0)
    replay = ReplayBuffer(capacity=10, seed=0)
    replay.extend(batch)
    seq = replay.sample_sequence(batch_size=2, sequence_length=3)
    assert seq.obs.shape == (2, 3, env.obs_dim)
    assert seq.action.shape == (2, 3, env.n_agents * env.action_dim)


def test_nominal_predator_prey_rule_coverage_close_to_article_target():
    env = make_environment("predator_prey")
    rules = rules_for_coverage("predator_prey", env.schema, 0.3, n_agents=env.n_agents)
    assert achieved_coverage(env.schema, rules) == pytest.approx(nominal_coverage_target("predator_prey"), abs=0.05)


def test_gridcraft_schema_is_semantic_and_nominal_coverage_is_reachable():
    env = make_environment("gridcraft")
    assert "agent_0.grid.terrain.0.0" in env.schema.names
    assert "agent_0.self.hp" in env.schema.names
    assert "agent_0.self.inventory.wood" in env.schema.names
    assert {"boundary", "collision", "object_persistence", "hunger", "inventory", "crafting"}.issubset(
        {spec.family for spec in env.schema.specs}
    )
    rules = rules_for_coverage("gridcraft", env.schema, 0.5, n_agents=env.n_agents)
    assert achieved_coverage(env.schema, rules) == pytest.approx(nominal_coverage_target("gridcraft"), abs=0.03)


def test_overcooked_nominal_coverage_and_claim_components_instantiate():
    env = make_environment("overcooked")
    assert env.obs_dim > 0
    assert {"position", "orientation", "carried_object", "pot_state", "object_persistence", "collision"}.issubset(
        {spec.family for spec in env.schema.specs}
    )
    rules = rules_for_coverage("overcooked", env.schema, 0.34, n_agents=env.n_agents)
    assert achieved_coverage(env.schema, rules) == pytest.approx(nominal_coverage_target("overcooked"), abs=0.02)
    cfg = ExperimentConfig(environment="overcooked", baseline_id="B33", world_model="rssm", strategy="projection", coverage=0.3)
    env, policy, wm, model = make_components(cfg)
    assert policy.act(env.reset(seed=0).reshape(1, -1)).shape[-1] == env.n_agents * env.action_dim
    assert wm is not None and model is not None


def test_smac_nominal_coverage_with_semantic_schema():
    schema = FeatureSchema.from_specs(
        FeatureSpec(f"smac.{family}_{i}", FeatureType.CONTINUOUS, family=family)
        for i, family in enumerate(("position", "movement_feasibility", "range", "action_mask", "health", "unit_activity") * 20)
    )
    rules = rules_for_coverage("smacv2", schema, 0.18, n_agents=8)
    assert achieved_coverage(schema, rules) == pytest.approx(nominal_coverage_target("smacv2"), abs=0.02)


def test_predator_prey_schema_snapshot_is_rich_and_stable():
    env = make_environment("predator_prey")
    assert env.schema.names == (
        "predator_0.x",
        "predator_0.y",
        "predator_1.x",
        "predator_1.y",
        "predator_0.dx",
        "predator_0.dy",
        "predator_1.dx",
        "predator_1.dy",
        "prey.x",
        "prey.y",
        "prey.active",
        "capture.in_range",
        "boundary.contact",
    )
    assert {spec.family for spec in env.schema.specs} == {"position", "velocity", "activity", "capture", "boundary"}


def test_predator_prey_rules_match_real_transition_on_covered_features():
    env = make_environment("predator_prey")
    env.reset(seed=0)
    env.predators = torch.tensor([[0.0, 0.0], [4.0, 4.0]])
    env.predator_delta = torch.zeros(2, 2)
    env.prey = torch.tensor([6.0, 6.0])
    env.prey_active = torch.tensor(1.0)
    env.boundary_contact = torch.tensor(0.0)
    env._update_capture_features()
    obs = env._obs()
    action = env.encode_action([3, 0])
    next_obs, _reward, _done, _info = env.step(action)
    rules = rules_for_coverage("predator_prey", env.schema, 0.3, n_agents=env.n_agents, width=env.width, height=env.height)
    sym = SymbolicWorldModel(rules).predict(RuleContext(obs=obs.reshape(1, -1), action=action.reshape(1, -1)))
    assert torch.allclose(sym.values[sym.mask], next_obs.reshape(1, -1)[sym.mask])
    assert float(rule_violation_rate(next_obs.reshape(1, -1), sym.values, sym.mask, env.schema)) == pytest.approx(0.0)


def test_rssm_reports_free_nats_and_kl_balance_on_sequences():
    model = RSSMWorldModel(obs_dim=5, action_dim=3, kl_free_nats=0.2, kl_balance=0.75)
    out = model(torch.zeros(2, 4, 5), torch.zeros(2, 4, 3))
    assert out.prediction.shape == (2, 4, 5)
    assert float(out.metrics["kl"].detach()) >= 0.2
    assert float(out.metrics["kl_free_nats"]) == pytest.approx(0.2)
    assert float(out.metrics["kl_balance"]) == pytest.approx(0.75)


def test_reproduction_check_validates_predator_prey_smoke_subset(tmp_path):
    (tmp_path / "eval_rollout.csv").write_text(
        "baseline_id,seed,horizon,compounding_error_slope,rvr,covered_rvr,wm_total_loss,obs_loss,kl_loss,reward_loss,done_loss,projection_magnitude,residual_error\n"
        "B31,0,3,0.1,0.4,0.4,1.0,2.0,0.0,0.1,0.1,0.0,2.0\n"
        "B31,0,5,0.1,0.4,0.4,1.0,2.0,0.0,0.1,0.1,0.0,2.0\n"
        "B34,0,3,0.1,0.2,0.2,1.0,1.5,0.0,0.1,0.1,1.0,1.5\n"
        "B34,0,5,0.1,0.2,0.2,1.0,1.5,0.0,0.1,0.1,1.0,1.5\n",
        encoding="utf-8",
    )
    (tmp_path / "train_world_model.csv").write_text(
        "baseline_id,seed,reward_per_resource,training_real_reward,generalization_gap\n"
        "B31,0,1.0,1.0,0.0\n"
        "B34,0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    result = validate_reproduction_logs(
        tmp_path,
        baselines=("B31", "B34"),
        required_seeds=1,
        required_horizons=(3, 5),
    )
    assert result.ok


def test_reproduction_check_requires_metrics_per_baseline_not_globally(tmp_path):
    (tmp_path / "eval_rollout.csv").write_text(
        "baseline_id,seed,horizon,compounding_error_slope,rvr,covered_rvr,wm_total_loss,obs_loss,kl_loss,reward_loss,done_loss,projection_magnitude,residual_error\n"
        "B31,0,3,0.1,0.4,0.4,1.0,2.0,0.0,0.1,0.1,0.0,2.0\n"
        "B34,0,3,0.1,0.2,0.2,1.0,1.5,0.0,0.1,0.1,1.0,\n",
        encoding="utf-8",
    )
    (tmp_path / "train_world_model.csv").write_text(
        "baseline_id,seed,reward_per_resource,training_real_reward,generalization_gap\n"
        "B31,0,1.0,1.0,0.0\n"
        "B34,0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    result = validate_reproduction_logs(
        tmp_path,
        baselines=("B31", "B34"),
        required_seeds=1,
        required_horizons=(3,),
    )
    assert not result.ok
    assert "B34:residual_error" in result.missing_metrics


def test_matrix_status_reports_complete_and_incomplete_stage_rows(tmp_path):
    (tmp_path / "train_world_model.csv").write_text(
        "baseline_id,seed,reward_per_resource,training_real_reward,generalization_gap\n"
        "B01,0,1.0,1.0,0.0\n"
        "B31,0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_rollout.csv").write_text(
        "baseline_id,seed,horizon,compounding_error_slope,rvr,covered_rvr,wm_total_loss,obs_loss,kl_loss,reward_loss,done_loss,projection_magnitude,residual_error\n"
        "B31,0,3,0.1,0.4,0.4,1.0,2.0,0.0,0.1,0.1,0.0,2.0\n",
        encoding="utf-8",
    )
    table = build_status_table(
        tmp_path,
        mode="smoke",
        stages=["train", "rollout"],
        baselines=("B01", "B31"),
        required_seeds=1,
        required_horizons=(3, 5),
    )
    rows = {row["baseline_id"]: row for row in table.to_dict(orient="records")}
    assert rows["B01"]["complete"] is True
    assert rows["B01"]["rollout"] == "skipped"
    assert rows["B31"]["complete"] is False
    assert "missing horizons" in rows["B31"]["rollout"]


def test_paper_readiness_reports_incomplete_matrix_until_all_stages_exist(tmp_path):
    (tmp_path / "train_world_model.csv").write_text(
        "baseline_id,seed,reward_per_resource,training_real_reward,generalization_gap\n"
        "B31,0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_rollout.csv").write_text(
        "baseline_id,seed,horizon,compounding_error_slope,rvr,covered_rvr,wm_total_loss,obs_loss,kl_loss,reward_loss,done_loss,projection_magnitude,residual_error\n"
        "B31,0,3,0.1,0.4,0.4,1.0,2.0,0.0,0.1,0.1,0.0,2.0\n"
        "B31,0,5,0.1,0.4,0.4,1.0,2.0,0.0,0.1,0.1,0.0,2.0\n",
        encoding="utf-8",
    )
    report, status = build_readiness_report(
        tmp_path,
        mode="smoke",
        baselines=("B31",),
        required_seeds=1,
        required_horizons=(3, 5),
        required_stages=("train", "rollout", "planning"),
    )
    assert not report.ok
    assert report.incomplete_baselines == ("B31",)
    assert "missing log" in str(status.iloc[0]["planning"])


def test_predator_prey_b31_b34_arms_share_schema_and_action_space():
    base = ExperimentConfig(environment="predator_prey", world_model="rssm", policy="MAPPO")
    configs = [
        base.__class__(**{**base.__dict__, "baseline_id": "B31", "strategy": "none", "coverage": 0.0, "comparison_arm": "neural"}),
        base.__class__(**{**base.__dict__, "baseline_id": "B34", "strategy": "regularization", "coverage": 0.3, "comparison_arm": "regularization"}),
        base.__class__(**{**base.__dict__, "baseline_id": "B34", "strategy": "projection", "coverage": 0.3, "comparison_arm": "projection"}),
        base.__class__(**{**base.__dict__, "baseline_id": "B34", "strategy": "residual", "coverage": 0.3, "comparison_arm": "residual"}),
    ]
    components = [make_components(config) for config in configs]
    schema_names = [component[0].schema.names for component in components]
    action_widths = [component[0].n_agents * component[0].action_dim for component in components]
    assert len(set(schema_names)) == 1
    assert len(set(action_widths)) == 1


def test_residual_b34_decoder_width_equals_uncovered_features():
    cfg = ExperimentConfig(environment="predator_prey", world_model="rssm", baseline_id="B34", strategy="residual", coverage=0.3)
    env, _policy, wm, model = make_components(cfg)
    covered = env.schema.mask(model.symbolic_model.covered_features)
    assert wm.output_dim == int((~covered).sum().item())


def test_paired_validation_fails_when_directional_rvr_is_missing(tmp_path):
    (tmp_path / "eval_rollout.csv").write_text(
        "baseline_id,comparison_arm,seed,horizon,compounding_error_slope,rvr,covered_rvr,wm_total_loss,obs_loss,kl_loss,reward_loss,done_loss,projection_magnitude,residual_error\n"
        "B31,neural,0,3,0.1,0.4,0.4,1.0,2.0,0.0,0.1,0.1,0.0,2.0\n"
        "B34,projection,0,3,0.1,0.4,0.4,1.0,2.0,0.0,0.1,0.1,1.0,2.0\n",
        encoding="utf-8",
    )
    (tmp_path / "train_world_model.csv").write_text(
        "baseline_id,comparison_arm,seed,reward_per_resource,training_real_reward,generalization_gap\n"
        "B31,neural,0,1.0,1.0,0.0\n"
        "B34,projection,0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    result = validate_paired_comparison(
        tmp_path,
        paired_baseline="B31",
        comparison_baselines=("B34",),
        comparison_arms=("projection",),
        required_seeds=1,
        required_horizons=(3,),
        require_directional_rvr_improvement=("projection",),
    )
    assert not result.ok
    assert result.directional_failures


def test_paired_validation_passes_when_projection_reduces_covered_rvr(tmp_path):
    (tmp_path / "eval_rollout.csv").write_text(
        "baseline_id,comparison_arm,seed,horizon,compounding_error_slope,rvr,covered_rvr,wm_total_loss,obs_loss,kl_loss,reward_loss,done_loss,projection_magnitude,residual_error\n"
        "B31,neural,0,3,0.1,0.4,0.4,1.0,2.0,0.0,0.1,0.1,0.0,2.0\n"
        "B34,projection,0,3,0.1,0.1,0.1,1.0,1.5,0.0,0.1,0.1,1.0,1.5\n",
        encoding="utf-8",
    )
    (tmp_path / "train_world_model.csv").write_text(
        "baseline_id,comparison_arm,seed,reward_per_resource,training_real_reward,generalization_gap\n"
        "B31,neural,0,1.0,1.0,0.0\n"
        "B34,projection,0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    result = validate_paired_comparison(
        tmp_path,
        paired_baseline="B31",
        comparison_baselines=("B34",),
        comparison_arms=("projection",),
        required_seeds=1,
        required_horizons=(3,),
        require_directional_rvr_improvement=("projection",),
    )
    assert result.ok


def test_paired_validation_supports_distinct_strategy_baseline_ids(tmp_path):
    (tmp_path / "eval_rollout.csv").write_text(
        "baseline_id,comparison_arm,seed,horizon,compounding_error_slope,rvr,covered_rvr,wm_total_loss,obs_loss,kl_loss,reward_loss,done_loss,projection_magnitude,residual_error\n"
        "B12,neural,0,3,0.1,0.5,0.5,1.0,2.0,0.0,0.1,0.1,0.0,2.0\n"
        "B25,projection,0,3,0.1,0.0,0.0,1.0,1.5,0.0,0.1,0.1,1.0,1.5\n"
        "B26,residual,0,3,0.1,0.0,0.0,1.0,1.4,0.0,0.1,0.1,1.0,1.4\n",
        encoding="utf-8",
    )
    (tmp_path / "train_world_model.csv").write_text(
        "baseline_id,comparison_arm,seed,reward_per_resource,training_real_reward,generalization_gap\n"
        "B12,neural,0,1.0,1.0,0.0\n"
        "B25,projection,0,1.0,1.0,0.0\n"
        "B26,residual,0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    result = validate_paired_comparison(
        tmp_path,
        paired_baseline="B12",
        comparison_baselines=("B25", "B26"),
        comparison_arms=("projection", "residual"),
        required_seeds=1,
        required_horizons=(3,),
        require_directional_rvr_improvement=("projection", "residual"),
    )
    assert result.ok
