"""Toy demo for NS-MAWM."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from nsmawm.data.datasets import TransitionsDataset
from nsmawm.metrics.rvr import compute_rvr
from nsmawm.models.mawm_backbone import BackboneConfig
from nsmawm.models.nsmawm import NSMAWM
from nsmawm.symbolic.engine import RuleEngine
from nsmawm.symbolic.rule import Rule, RuleContext, RuleResult
from nsmawm.training.trainer import fit


class StayRule(Rule):
    """If action=stay, positions remain unchanged."""

    def apply(self, context: RuleContext) -> RuleResult:
        obs = context.obs_t
        act = context.act_t
        stay = act[..., 0] > 0.5
        mask = stay.unsqueeze(-1).expand_as(obs)
        values = torch.where(mask, obs, torch.zeros_like(obs))
        return RuleResult(values=values, mask=mask)


def generate_synthetic(n: int, n_agents: int, n_features: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.randn(n, n_agents, n_features)
    act = torch.zeros(n, n_agents, 2)
    stay_mask = torch.rand(n, n_agents) > 0.5
    act[..., 0] = stay_mask.float()
    act[..., 1] = (~stay_mask).float()
    delta = torch.randn(n, n_agents, n_features) * 0.1
    next_obs = torch.where(stay_mask.unsqueeze(-1), obs, obs + delta)
    return obs, act, next_obs


def main() -> None:
    torch.manual_seed(7)
    n_agents = 2
    n_features = 2

    obs, act, next_obs = generate_synthetic(512, n_agents, n_features)
    dataset = TransitionsDataset(obs, act, next_obs)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    backbone_cfg = BackboneConfig(
        n_agents=n_agents,
        n_features=n_features,
        action_dim=2,
        latent_dim=32,
        hidden_dim=32,
        encoder_hidden=64,
        decoder_hidden=64,
    )

    rule_engine = RuleEngine([StayRule()])
    model = NSMAWM.from_config(backbone_cfg, rule_engine=rule_engine, strategy="reg", lambda_symb=1.0)

    fit(model, train_loader, max_epochs=3, learning_rate=1e-3)

    model.eval()
    with torch.no_grad():
        pred = model.predict(obs[:32], act[:32])
        output = model.forward(obs[:32], act[:32], apply_projection=False)
        rvr_reg = compute_rvr(pred, output.omega_d, output.mask)
        print(f"RVR (proj) = {rvr_reg.item():.4f}")

    model_proj = NSMAWM.from_config(backbone_cfg, rule_engine=rule_engine, strategy="proj")
    model_proj.eval()
    with torch.no_grad():
        output = model_proj.forward(obs[:32], act[:32], apply_projection=True)
        rvr_proj = compute_rvr(output.prediction, output.omega_d, output.mask)
        print(f"RVR (projection) = {rvr_proj.item():.4f}")


if __name__ == "__main__":
    main()
