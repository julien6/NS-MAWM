import torch

from nsmawm.metrics.rvr import compute_rvr
from nsmawm.models.mawm_backbone import BackboneConfig
from nsmawm.models.nsmawm import NSMAWM
from nsmawm.symbolic.engine import RuleEngine
from nsmawm.symbolic.rule import Rule, RuleContext, RuleResult


class StayRule(Rule):
    def apply(self, context: RuleContext) -> RuleResult:
        obs = context.obs_t
        act = context.act_t
        stay = act[..., 0] > 0.5
        mask = stay.unsqueeze(-1).expand_as(obs)
        values = torch.where(mask, obs, torch.zeros_like(obs))
        return RuleResult(values=values, mask=mask)


def test_projection_matches_symbolic():
    obs = torch.randn(4, 1, 2)
    act = torch.zeros(4, 1, 2)
    act[..., 0] = 1.0

    cfg = BackboneConfig(
        n_agents=1,
        n_features=2,
        action_dim=2,
        latent_dim=8,
        hidden_dim=8,
        encoder_hidden=16,
        decoder_hidden=16,
    )
    model = NSMAWM.from_config(cfg, rule_engine=RuleEngine([StayRule()]), strategy="proj")
    output = model.forward(obs, act, apply_projection=True)
    assert torch.allclose(output.prediction, output.omega_d)


def test_rvr_zero_after_projection():
    obs = torch.randn(4, 1, 2)
    act = torch.zeros(4, 1, 2)
    act[..., 0] = 1.0

    cfg = BackboneConfig(
        n_agents=1,
        n_features=2,
        action_dim=2,
        latent_dim=8,
        hidden_dim=8,
        encoder_hidden=16,
        decoder_hidden=16,
    )
    model = NSMAWM.from_config(cfg, rule_engine=RuleEngine([StayRule()]), strategy="proj")
    output = model.forward(obs, act, apply_projection=True)
    rvr = compute_rvr(output.prediction, output.omega_d, output.mask)
    assert rvr.item() == 0.0
