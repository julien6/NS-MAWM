"""Evaluation protocols from the paper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ns_mawm.metrics import feature_family_rvr, masked_mse, residual_error, rule_violation_rate
from ns_mawm.rules import RuleContext


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
    rollout_fidelity: float
    residual_error: float
    covered_rvr: float
    covered_feature_mse: float
    uncovered_feature_mse: float
    residual_decoder_width: int
    rvr_by_family: dict[str, float]


@torch.no_grad()
def open_loop_rollout(model, batch, schema, horizons: tuple[int, ...], reference_symbolic_model=None) -> list[RolloutMetric]:
    if batch.obs.ndim == 3:
        obs = batch.obs[0]
        action = batch.action[0]
        next_obs = batch.next_obs[0]
        reward = batch.reward[0]
        done = batch.done[0]
    else:
        obs, action, next_obs, reward, done = batch.obs, batch.action, batch.next_obs, batch.reward, batch.done
    rows: list[RolloutMetric] = []
    residual_decoder_width = int(getattr(getattr(model, "world_model", model), "output_dim", 0))
    for horizon in horizons:
        current = obs[:1]
        max_steps = min(horizon, action.shape[0])
        obs_losses: list[torch.Tensor] = []
        reward_losses: list[torch.Tensor] = []
        done_losses: list[torch.Tensor] = []
        kls: list[torch.Tensor] = []
        rvrs: list[torch.Tensor] = []
        covered_mses: list[torch.Tensor] = []
        uncovered_mses: list[torch.Tensor] = []
        projections: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        family_rvrs: dict[str, list[torch.Tensor]] = {}
        for t in range(max_steps):
            action_t = action[t : t + 1]
            out = model(current, action_t, rollout=True)
            if reference_symbolic_model is not None:
                reference = reference_symbolic_model.predict(RuleContext(obs=current, action=action_t))
                symbolic_values, symbolic_mask = reference.values, reference.mask
            else:
                symbolic_values, symbolic_mask = out.symbolic_values, out.symbolic_mask
            target_obs = next_obs[t : t + 1]
            obs_losses.append(F.mse_loss(out.prediction, target_obs))
            reward_losses.append(F.mse_loss(out.reward, reward[t : t + 1]))
            done_losses.append(F.binary_cross_entropy_with_logits(out.done_logits, done[t : t + 1]))
            kls.append((out.metrics or {}).get("kl", out.prediction.new_tensor(0.0)))
            rvrs.append(rule_violation_rate(out.prediction, symbolic_values, symbolic_mask, schema))
            covered_mses.append(masked_mse(out.prediction, target_obs, symbolic_mask))
            uncovered_mses.append(residual_error(out.prediction, target_obs, symbolic_mask))
            for family, value in feature_family_rvr(out.prediction, symbolic_values, symbolic_mask, schema).items():
                family_rvrs.setdefault(family, []).append(value)
            projections.append(out.projection_magnitude)
            residuals.append(residual_error(out.prediction, target_obs, symbolic_mask))
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
                float(1.0 / (1.0 + torch.stack(obs_losses).mean())),
                float(torch.stack(residuals).mean()),
                float(torch.stack(rvrs).mean()),
                float(torch.stack(covered_mses).mean()),
                float(torch.stack(uncovered_mses).mean()),
                residual_decoder_width,
                {family: float(torch.stack(values).mean()) for family, values in family_rvrs.items()},
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


@torch.no_grad()
def cem_plan(
    model,
    obs: torch.Tensor,
    n_agents: int,
    action_dim: int,
    candidates: int = 64,
    horizon: int = 5,
    iterations: int = 3,
    elite_frac: float = 0.25,
) -> torch.Tensor:
    probs = torch.full((horizon, n_agents, action_dim), 1.0 / action_dim, device=obs.device)
    elite_count = max(1, int(candidates * elite_frac))
    first_actions = None
    for _ in range(iterations):
        scores: list[torch.Tensor] = []
        sampled: list[torch.Tensor] = []
        for _candidate in range(candidates):
            current = obs
            score = obs.new_tensor(0.0)
            seq: list[torch.Tensor] = []
            for t in range(horizon):
                dist = torch.distributions.Categorical(probs=probs[t])
                idx = dist.sample((obs.shape[0],))
                action = F.one_hot(idx, action_dim).float().reshape(obs.shape[0], -1)
                seq.append(action)
                out = model(current, action, rollout=True)
                score = score + out.reward.mean()
                current = out.prediction
            scores.append(score)
            sampled.append(torch.stack(seq, dim=1))
        order = torch.stack(scores).argsort(descending=True)[:elite_count]
        elites = torch.cat([sampled[int(i)] for i in order], dim=0)
        first_actions = elites[:, 0]
        elite_idx = elites.reshape(-1, horizon, n_agents, action_dim).argmax(dim=-1)
        probs = F.one_hot(elite_idx, action_dim).float().mean(dim=0).clamp_min(1e-3)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    assert first_actions is not None
    return first_actions[: obs.shape[0]]
