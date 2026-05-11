import torch
import pytest

from experiments.baseline_configs import all_baseline_configs
from experiments.registry import BASELINES
from ns_mawm import FeatureSchema, FeatureSpec, FeatureType, NSMAWM
from ns_mawm.rules import (
    NoisyRule,
    RuleContext,
    RulePrediction,
    SymbolicRule,
    SymbolicWorldModel,
    dropout_rules,
    select_rules_for_coverage,
)
from wm_lib.models import DeterministicWorldModel


class PersistenceRule(SymbolicRule):
    rule_id = "persist_x"
    covered_features = ("agent.x",)

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        values[..., 0] = context.obs[..., 0]
        mask[..., 0] = True
        return RulePrediction(values, mask, (self.rule_id,))


class PersistenceRuleY(SymbolicRule):
    rule_id = "persist_y"
    covered_features = ("agent.y",)

    def predict(self, context: RuleContext) -> RulePrediction:
        values = torch.zeros_like(context.obs)
        mask = torch.zeros_like(context.obs, dtype=torch.bool)
        values[..., 1] = context.obs[..., 1]
        mask[..., 1] = True
        return RulePrediction(values, mask, (self.rule_id,))


def _schema() -> FeatureSchema:
    return FeatureSchema.from_specs(
        [
            FeatureSpec("agent.x", FeatureType.CONTINUOUS, owner="agent", family="position", tolerance=0.1),
            FeatureSpec("agent.y", FeatureType.CONTINUOUS, owner="agent", family="position", tolerance=0.1),
            FeatureSpec("agent.has_item", FeatureType.BINARY, owner="agent", family="inventory"),
        ]
    )


def test_feature_schema_round_trip_and_family_summary():
    schema = _schema()
    mapping = {"agent.x": 1.0, "agent.y": 2.0, "agent.has_item": 1.0}
    tensor = schema.encode_mapping(mapping)
    assert tensor.tolist() == [1.0, 2.0, 1.0]
    assert schema.decode_tensor(tensor)["agent.y"] == 2.0
    mask = schema.mask(["agent.x", "agent.y"])
    assert schema.coverage(mask) == pytest.approx(2 / 3)
    assert schema.family_summary(mask) == {"position": 1.0, "inventory": 0.0}


def test_rule_coverage_dropout_noise_and_audit():
    schema = _schema()
    rules = [PersistenceRule(), PersistenceRuleY()]
    selected = select_rules_for_coverage(schema, rules, target_coverage=0.5)
    assert [rule.rule_id for rule in selected] == ["persist_x", "persist_y"]
    assert len(dropout_rules(rules, rate=1.0, seed=0)) == 0

    symbolic = SymbolicWorldModel([NoisyRule(PersistenceRule(), std=0.01, seed=7)])
    obs = torch.tensor([[3.0, 4.0, 0.0]])
    pred = symbolic.predict(RuleContext(obs=obs, action=torch.zeros(1, 2)))
    assert pred.mask.tolist() == [[True, False, False]]
    audit = symbolic.audit(schema)
    assert audit.coverage == pytest.approx(1 / 3)
    assert audit.family_coverage["position"] == 0.5


def test_residual_strategy_assembles_batch_with_uncovered_width():
    schema = _schema()
    symbolic = SymbolicWorldModel([PersistenceRule()])
    wm = DeterministicWorldModel(obs_dim=3, action_dim=2, output_dim=2)
    model = NSMAWM(wm, symbolic, "residual", schema=schema)
    obs = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
    action = torch.zeros(2, 2)
    out = model(obs, action, rollout=True)
    assert out.raw_prediction.shape == (2, 2)
    assert out.prediction.shape == (2, 3)
    assert torch.equal(out.prediction[:, 0], obs[:, 0])


def test_generated_baseline_configs_cover_b01_to_b45():
    configs = all_baseline_configs()
    assert set(configs) == set(BASELINES)
    assert len(configs) == 45
    assert configs["B01"].world_model == "none"
    assert configs["B24"].strategy == "regularization"
    assert configs["B25"].strategy == "projection"
    assert configs["B26"].strategy == "residual"
