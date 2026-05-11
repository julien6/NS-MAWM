"""Evaluation protocols from the paper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ns_mawm.metrics import rule_violation_rate


@dataclass(frozen=True)
class RolloutMetric:
    horizon: int
    obs_loss: float
    reward_loss: float
    done_loss: float
    kl_loss: float
    wm_total_loss: float
    compounding_error_slope: float
    rvr: float
    projection_magnitude: float


@torch.no_grad()
def open_loop_rollout(model, batch, schema, horizons: tuple[int, ...]) -> list[RolloutMetric]:
    rows: list[RolloutMetric] = []
    for horizon in horizons:
        current = batch.obs[:1]
        max_steps = min(horizon, batch.action.shape[0])
        obs_losses: list[torch.Tensor] = []
        reward_losses: list[torch.Tensor] = []
        done_losses: list[torch.Tensor] = []
        kls: list[torch.Tensor] = []
        rvrs: list[torch.Tensor] = []
        projections: list[torch.Tensor] = []
        for t in range(max_steps):
            action = batch.action[t : t + 1]
            out = model(current, action, rollout=True)
            target_obs = batch.next_obs[t : t + 1]
            obs_losses.append(F.mse_loss(out.prediction, target_obs))
            reward_losses.append(F.mse_loss(out.reward, batch.reward[t : t + 1]))
            done_losses.append(F.binary_cross_entropy_with_logits(out.done_logits, batch.done[t : t + 1]))
            kls.append((out.metrics or {}).get("kl", out.prediction.new_tensor(0.0)))
            rvrs.append(rule_violation_rate(out.prediction, out.symbolic_values, out.symbolic_mask, schema))
            projections.append(out.projection_magnitude)
            current = out.prediction
        y = torch.stack(obs_losses)
        x = torch.arange(y.numel(), dtype=torch.float32, device=y.device)
        slope = torch.cov(torch.stack([x, y]))[0, 1] / torch.var(x).clamp_min(1e-6) if y.numel() > 1 else y.new_tensor(0.0)
        total = torch.stack(obs_losses).mean() + torch.stack(reward_losses).mean() + torch.stack(done_losses).mean() + torch.stack(kls).mean()
        rows.append(
            RolloutMetric(
                horizon,
                float(torch.stack(obs_losses).mean()),
                float(torch.stack(reward_losses).mean()),
                float(torch.stack(done_losses).mean()),
                float(torch.stack(kls).mean()),
                float(total),
                float(slope),
                float(torch.stack(rvrs).mean()),
                float(torch.stack(projections).mean()),
            )
        )
    return rows


@torch.no_grad()
def random_shooting_plan(model, obs: torch.Tensor, n_agents: int, action_dim: int, candidates: int = 16, horizon: int = 3) -> torch.Tensor:
    best_score = None
    best_action = None
    for _ in range(candidates):
        current = obs
        score = obs.new_tensor(0.0)
        first_action = None
        for _t in range(horizon):
            idx = torch.randint(action_dim, (obs.shape[0], n_agents), device=obs.device)
            action = F.one_hot(idx, action_dim).float().reshape(obs.shape[0], -1)
            if first_action is None:
                first_action = action
            out = model(current, action, rollout=True)
            score = score + out.reward.mean()
            current = out.prediction
        if best_score is None or score > best_score:
            best_score = score
            best_action = first_action
    assert best_action is not None
    return best_action
