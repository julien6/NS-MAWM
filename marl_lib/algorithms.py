"""Policy/data-generator algorithms.

These classes provide PyTorch interfaces for the B01-B45 baseline registry and expose a
common policy API for rollout collection, updates, and checkpointing.
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
        self._optimizer: torch.optim.Optimizer | None = None

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

    def optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return self._optimizer

    def update(self, batch, learning_rate: float = 1e-3) -> dict[str, float]:
        optimizer = self.optimizer(learning_rate)
        logits = self.logits(batch.obs)
        target = batch.action.reshape(batch.action.shape[0], self.n_agents, self.action_dim).argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(target).sum(dim=-1, keepdim=True)
        value = self.value(batch.obs)
        returns = batch.reward
        advantage = (returns - value.detach())
        policy_loss = -(log_prob * advantage).mean()
        value_loss = F.mse_loss(value, returns)
        entropy = dist.entropy().sum(dim=-1).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "policy_total_loss": float(loss.detach()),
        }


class MAPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        action_dim: int,
        hidden_dim: int = 64,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
    ):
        super().__init__(obs_dim, n_agents, action_dim, hidden_dim)
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gae_lambda = gae_lambda
        self.gamma = gamma

    def update(self, batch, learning_rate: float = 1e-3) -> dict[str, float]:
        optimizer = self.optimizer(learning_rate)
        logits = self.logits(batch.obs)
        target = batch.action.reshape(batch.action.shape[0], self.n_agents, self.action_dim).argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(target).sum(dim=-1, keepdim=True)
        old_log_prob = log_prob.detach()
        value = self.value(batch.obs)

        # Single-batch GAE approximation. Full experiments can feed contiguous rollouts
        # through this same interface without changing the policy API.
        returns = batch.reward
        advantage = returns - value.detach()
        advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
        ratio = (log_prob - old_log_prob).exp()
        unclipped = ratio * advantage
        clipped = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(value, returns)
        entropy = dist.entropy().sum(dim=-1).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        optimizer.step()
        clip_fraction = (ratio.sub(1.0).abs() > self.clip_epsilon).float().mean()
        return {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "policy_total_loss": float(loss.detach()),
            "mappo_clip_epsilon": self.clip_epsilon,
            "mappo_clip_fraction": float(clip_fraction.detach()),
            "mappo_gae_lambda": self.gae_lambda,
            "mappo_gamma": self.gamma,
        }


class QMIXPolicy(ActorCriticPolicy):
    def __init__(self, obs_dim: int, n_agents: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(obs_dim, n_agents, action_dim, hidden_dim)
        self.agent_q = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_agents * action_dim))
        self.mixer = nn.Sequential(nn.Linear(n_agents, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def update(self, batch, learning_rate: float = 1e-3) -> dict[str, float]:
        optimizer = self.optimizer(learning_rate)
        q = self.agent_q(batch.obs).reshape(batch.obs.shape[0], self.n_agents, self.action_dim)
        target = batch.action.reshape(batch.action.shape[0], self.n_agents, self.action_dim).argmax(dim=-1)
        chosen_q = q.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mixed_q = self.mixer(chosen_q)
        td_loss = F.mse_loss(mixed_q, batch.reward)
        optimizer.zero_grad(set_to_none=True)
        td_loss.backward()
        optimizer.step()
        metrics = {"qmix_td_loss": float(td_loss.detach()), "policy_total_loss": float(td_loss.detach())}
        metrics["qmix_mixer_norm"] = float(sum(p.detach().abs().mean() for p in self.mixer.parameters()))
        return metrics


class SACPolicy(ActorCriticPolicy):
    def __init__(self, obs_dim: int, n_agents: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(obs_dim, n_agents, action_dim, hidden_dim)
        self.log_alpha = nn.Parameter(torch.zeros(()))

    def update(self, batch, learning_rate: float = 1e-3) -> dict[str, float]:
        optimizer = self.optimizer(learning_rate)
        logits = self.logits(batch.obs)
        target = batch.action.reshape(batch.action.shape[0], self.n_agents, self.action_dim).argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(target).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        alpha = self.log_alpha.exp()
        q_proxy = self.value(batch.obs)
        actor_loss = (alpha.detach() * log_prob - q_proxy.detach()).mean()
        critic_loss = F.mse_loss(q_proxy, batch.reward)
        alpha_loss = -(alpha * (entropy.detach() - float(self.n_agents))).mean()
        loss = actor_loss + critic_loss + 0.01 * alpha_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics = {
            "sac_actor_loss": float(actor_loss.detach()),
            "sac_critic_loss": float(critic_loss.detach()),
            "sac_alpha_loss": float(alpha_loss.detach()),
            "policy_total_loss": float(loss.detach()),
        }
        metrics["sac_alpha"] = float(alpha.detach())
        return metrics
