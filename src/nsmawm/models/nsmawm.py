"""NS-MAWM wrapper with neuro-symbolic strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from nsmawm.metrics.rvr import compute_rvr
from nsmawm.models.mawm_backbone import MAWMBackbone, BackboneConfig
from nsmawm.symbolic.correction import Correction, IdentityCorrection
from nsmawm.symbolic.engine import RuleEngine
from nsmawm.symbolic.rule import RuleContext


@dataclass
class NSMAWMOutput:
    prediction: torch.Tensor
    hidden: Tuple[torch.Tensor, torch.Tensor]
    omega_d: torch.Tensor
    mask: torch.Tensor


class NSMAWM(nn.Module):
    """Neuro-symbolic MAWM model with integration strategies."""

    def __init__(
        self,
        backbone: MAWMBackbone,
        rule_engine: Optional[RuleEngine] = None,
        strategy: str = "reg",
        lambda_symb: float = 1.0,
        post_correction: Optional[Correction] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.rule_engine = rule_engine or RuleEngine([])
        self.strategy = strategy
        self.lambda_symb = lambda_symb
        self.post_correction = post_correction or IdentityCorrection()

    @classmethod
    def from_config(
        cls,
        config: BackboneConfig,
        rule_engine: Optional[RuleEngine] = None,
        strategy: str = "reg",
        lambda_symb: float = 1.0,
    ) -> "NSMAWM":
        backbone = MAWMBackbone(config)
        return cls(backbone=backbone, rule_engine=rule_engine, strategy=strategy, lambda_symb=lambda_symb)

    def forward(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        apply_projection: bool = False,
        apply_correction: bool = True,
    ) -> NSMAWMOutput:
        context = RuleContext(obs_t=obs_t, act_t=act_t, prev_hidden=hidden)
        engine_out = self.rule_engine.apply(context)
        omega_d = engine_out.omega_d
        mask = engine_out.mask

        pred, hidden = self.backbone(obs_t, act_t, hidden)

        if self.strategy == "residual":
            pred = torch.where(mask, omega_d, pred)

        if apply_projection or self.strategy == "proj":
            pred = torch.where(mask, omega_d, pred)

        if apply_correction:
            pred = self.post_correction(pred)

        return NSMAWMOutput(prediction=pred, hidden=hidden, omega_d=omega_d, mask=mask)

    def compute_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        next_obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.forward(obs_t, act_t, hidden, apply_projection=False, apply_correction=False)
        pred = output.prediction
        mask = output.mask
        omega_d = output.omega_d
        loss_pred = torch.mean((pred - next_obs) ** 2)
        loss_symb = torch.tensor(0.0, device=loss_pred.device)
        if self.strategy == "reg":
            if torch.any(mask):
                loss_symb = torch.mean(((pred - omega_d) ** 2)[mask])
            loss = loss_pred + self.lambda_symb * loss_symb
        else:
            loss = loss_pred
        return loss, loss_symb

    def predict(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        output = self.forward(obs_t, act_t, apply_projection=True)
        return output.prediction

    def rollout(self, obs_t: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """Rollout predictions over an action sequence [B, T, n_agents, action_dim]."""
        hidden = None
        preds = []
        current = obs_t
        for t in range(act_seq.shape[1]):
            output = self.forward(current, act_seq[:, t], hidden)
            current = output.prediction
            hidden = output.hidden
            preds.append(current)
        return torch.stack(preds, dim=1)

    def evaluate_rvr(self, obs_t: torch.Tensor, act_t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        output = self.forward(obs_t, act_t, apply_projection=True)
        return compute_rvr(output.prediction, output.omega_d, output.mask, eps=eps)
