import torch

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


def generate_synthetic(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.randn(n, 1, 2)
    act = torch.zeros(n, 1, 2)
    stay_mask = torch.rand(n, 1) > 0.5
    act[..., 0] = stay_mask.float().squeeze(-1)
    act[..., 1] = (~stay_mask).float().squeeze(-1)
    delta = torch.randn(n, 1, 2) * 0.1
    next_obs = torch.where(stay_mask.unsqueeze(-1), obs, obs + delta)
    return obs, act, next_obs


def test_training_loss_decreases():
    torch.manual_seed(7)
    obs, act, next_obs = generate_synthetic(128)

    cfg = BackboneConfig(
        n_agents=1,
        n_features=2,
        action_dim=2,
        latent_dim=16,
        hidden_dim=16,
        encoder_hidden=32,
        decoder_hidden=32,
    )
    model = NSMAWM.from_config(cfg, rule_engine=RuleEngine([StayRule()]), strategy="reg", lambda_symb=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    initial_loss, _ = model.compute_loss(obs, act, next_obs)
    for _ in range(200):
        optimizer.zero_grad()
        loss, _ = model.compute_loss(obs, act, next_obs)
        loss.backward()
        optimizer.step()

    final_loss, _ = model.compute_loss(obs, act, next_obs)
    assert final_loss.item() < initial_loss.item()
