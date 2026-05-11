"""Agnostic NS-MAWM integration layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
from torch import nn
import torch.nn.functional as F

from ns_mawm.features import FeatureSchema
from ns_mawm.rules import RuleContext, SymbolicWorldModel


class IntegrationStrategy(str, Enum):
    NONE = "none"
    REGULARIZATION = "regularization"
    PROJECTION = "projection"
    RESIDUAL = "residual"


class WorldModelProtocol(Protocol):
    def forward(self, obs: torch.Tensor, action: torch.Tensor, state: object | None = None): ...


@dataclass(frozen=True)
class NSMAWMOutput:
    prediction: torch.Tensor
    raw_prediction: torch.Tensor
    reward: torch.Tensor
    done_logits: torch.Tensor
    symbolic_values: torch.Tensor
    symbolic_mask: torch.Tensor
    projection_magnitude: torch.Tensor
    state: object | None = None
    metrics: dict[str, torch.Tensor] | None = None


class NSMAWM(nn.Module):
    """Composes any neural WM with any symbolic transition model.

    This class does not know about environments, policies, or concrete WM architectures.
    """

    def __init__(
        self,
        world_model: nn.Module,
        symbolic_model: SymbolicWorldModel,
        strategy: IntegrationStrategy | str,
        lambda_symbolic: float = 1.0,
        schema: FeatureSchema | None = None,
    ):
        super().__init__()
        self.world_model = world_model
        self.symbolic_model = symbolic_model
        self.strategy = IntegrationStrategy(strategy)
        self.lambda_symbolic = lambda_symbolic
        self.schema = schema

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        state: object | None = None,
        history: torch.Tensor | None = None,
        rollout: bool = False,
    ) -> NSMAWMOutput:
        wm_out = self.world_model(obs, action, state)
        sym = self.symbolic_model.predict(RuleContext(obs=obs, action=action, history=history))
        raw_pred = wm_out.prediction
        pred = self._assemble_residual_prediction(raw_pred, sym.values, sym.mask)
        projection = pred.new_tensor(0.0)
        if rollout and self.strategy in {IntegrationStrategy.PROJECTION, IntegrationStrategy.RESIDUAL}:
            assembled = torch.where(sym.mask, sym.values, pred)
            projection = torch.abs(assembled - pred)[sym.mask].mean() if sym.mask.any() else projection
            pred = assembled
        return NSMAWMOutput(
            prediction=pred,
            raw_prediction=raw_pred,
            reward=wm_out.reward,
            done_logits=wm_out.done_logits,
            symbolic_values=sym.values,
            symbolic_mask=sym.mask,
            projection_magnitude=projection,
            state=getattr(wm_out, "state", None),
            metrics=getattr(wm_out, "metrics", {}),
        )

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self.forward(obs, action, rollout=False)
        pred_target = next_obs
        pred = out.prediction
        if self.strategy == IntegrationStrategy.RESIDUAL:
            if self._is_residual_width(out.raw_prediction, out.symbolic_mask):
                uncovered = ~out.symbolic_mask.reshape(-1, out.symbolic_mask.shape[-1]).any(dim=0)
                pred_target = next_obs[..., uncovered]
                pred = out.raw_prediction
            else:
                pred_target = torch.where(out.symbolic_mask, pred.detach(), next_obs)
                pred = torch.where(out.symbolic_mask, next_obs.detach(), pred)
        obs_loss = F.mse_loss(pred, pred_target)
        reward_loss = F.mse_loss(out.reward, reward)
        done_loss = F.binary_cross_entropy_with_logits(out.done_logits, done)
        symbolic_loss = pred.new_tensor(0.0)
        if self.strategy == IntegrationStrategy.REGULARIZATION and out.symbolic_mask.any():
            symbolic_loss = F.mse_loss(pred[out.symbolic_mask], out.symbolic_values[out.symbolic_mask])
        kl = (out.metrics or {}).get("kl", pred.new_tensor(0.0))
        total = obs_loss + reward_loss + done_loss + kl + self.lambda_symbolic * symbolic_loss
        return total, {
            "obs_loss": obs_loss,
            "reward_loss": reward_loss,
            "done_loss": done_loss,
            "kl": kl,
            "symbolic_loss": symbolic_loss,
        }

    def _assemble_residual_prediction(
        self,
        prediction: torch.Tensor,
        symbolic_values: torch.Tensor,
        symbolic_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.strategy != IntegrationStrategy.RESIDUAL:
            return prediction
        if not self._is_residual_width(prediction, symbolic_mask):
            return prediction
        full = torch.zeros_like(symbolic_values)
        uncovered = ~symbolic_mask.reshape(-1, symbolic_mask.shape[-1]).any(dim=0)
        full[..., uncovered] = prediction
        return torch.where(symbolic_mask, symbolic_values, full)

    @staticmethod
    def _is_residual_width(prediction: torch.Tensor, symbolic_mask: torch.Tensor) -> bool:
        if prediction.shape[-1] == symbolic_mask.shape[-1]:
            return False
        uncovered_width = int((~symbolic_mask.reshape(-1, symbolic_mask.shape[-1]).any(dim=0)).sum().item())
        return prediction.shape[-1] == uncovered_width
