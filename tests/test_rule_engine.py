import torch

from nsmawm.symbolic.engine import RuleEngine
from nsmawm.symbolic.rule import Rule, RuleContext, RuleResult


class RuleA(Rule):
    def apply(self, context: RuleContext) -> RuleResult:
        values = torch.zeros_like(context.obs_t)
        mask = torch.zeros_like(context.obs_t, dtype=torch.bool)
        mask[..., 0] = True
        values[..., 0] = 1.0
        return RuleResult(values=values, mask=mask)


class RuleB(Rule):
    def apply(self, context: RuleContext) -> RuleResult:
        values = torch.zeros_like(context.obs_t)
        mask = torch.zeros_like(context.obs_t, dtype=torch.bool)
        mask[..., 1] = True
        values[..., 1] = 2.0
        return RuleResult(values=values, mask=mask)


def test_rule_engine_combines_masks():
    obs = torch.zeros(2, 1, 2)
    act = torch.zeros(2, 1, 1)
    engine = RuleEngine([RuleA(), RuleB()])
    output = engine.apply(RuleContext(obs_t=obs, act_t=act))
    assert torch.all(output.mask[..., 0])
    assert torch.all(output.mask[..., 1])
    assert torch.allclose(output.omega_d[0, 0], torch.tensor([1.0, 2.0]))
