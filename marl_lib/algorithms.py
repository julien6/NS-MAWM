"""Policy/data-generator algorithms.

These classes provide clean PyTorch interfaces for the B01-B45 baseline registry. The
MAPPO/QMIX/SAC classes are intentionally compact policy modules suitable for smoke and
ablation tests; training-specific losses can be layered on top without changing their API.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MultiAgentPolicy(nn.Module):
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def update(self, batch) -> dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str | torch.device = "cpu") -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))


class RandomPolicy(MultiAgentPolicy):
    def __init__(self, n_agents: int, action_dim: int, seed: int = 0):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.generator = torch.Generator().manual_seed(seed)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        batch = obs.shape[0]
        idx = torch.randint(self.action_dim, (batch, self.n_agents), generator=self.generator, device=obs.device)
        return torch.nn.functional.one_hot(idx, self.action_dim).float().reshape(batch, -1)


class ScriptedPolicy(RandomPolicy):
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        batch = obs.shape[0]
        idx = torch.zeros(batch, self.n_agents, dtype=torch.long, device=obs.device)
        return torch.nn.functional.one_hot(idx, self.action_dim).float().reshape(batch, -1)


class ActorCriticPolicy(MultiAgentPolicy):
    def __init__(self, obs_dim: int, n_agents: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_agents * action_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs).reshape(obs.shape[0], self.n_agents, self.action_dim)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        logits = self.logits(obs)
        if deterministic:
            idx = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample()
        return torch.nn.functional.one_hot(idx, self.action_dim).float().reshape(obs.shape[0], -1)

    def update(self, batch, learning_rate: float = 1e-3) -> dict[str, float]:
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        logits = self.logits(batch.obs)
        target = batch.action.reshape(batch.action.shape[0], self.n_agents, self.action_dim).argmax(dim=-1)
        policy_loss = F.cross_entropy(logits.reshape(-1, self.action_dim), target.reshape(-1))
        value_loss = F.mse_loss(self.value(batch.obs), batch.reward)
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "policy_total_loss": float(loss.detach()),
        }


class MAPPOPolicy(ActorCriticPolicy):
    pass


class QMIXPolicy(ActorCriticPolicy):
    def __init__(self, obs_dim: int, n_agents: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(obs_dim, n_agents, action_dim, hidden_dim)
        self.mixer = nn.Sequential(nn.Linear(n_agents, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def update(self, batch, learning_rate: float = 1e-3) -> dict[str, float]:
        metrics = super().update(batch, learning_rate)
        metrics["qmix_mixer_norm"] = float(sum(p.detach().abs().mean() for p in self.mixer.parameters()))
        return metrics


class SACPolicy(ActorCriticPolicy):
    def __init__(self, obs_dim: int, n_agents: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(obs_dim, n_agents, action_dim, hidden_dim)
        self.log_alpha = nn.Parameter(torch.zeros(()))

    def update(self, batch, learning_rate: float = 1e-3) -> dict[str, float]:
        metrics = super().update(batch, learning_rate)
        metrics["sac_alpha"] = float(self.log_alpha.exp().detach())
        return metrics
