"""World-model architectures used by the reproduction tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class WorldModelOutput:
    prediction: torch.Tensor
    reward: torch.Tensor
    done_logits: torch.Tensor
    state: object | None = None
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


class WorldModelProtocol(Protocol):
    obs_dim: int
    action_dim: int
    output_dim: int

    def forward(self, obs: torch.Tensor, action: torch.Tensor, state: object | None = None) -> WorldModelOutput: ...


class StructuredDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeterministicWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = output_dim or obs_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.core = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.decoder = StructuredDecoder(hidden_dim, self.output_dim, hidden_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.done = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, state: object | None = None) -> WorldModelOutput:
        single = obs.ndim == 2
        if single:
            obs, action = obs.unsqueeze(1), action.unsqueeze(1)
        z = self.encoder(torch.cat([obs, action], dim=-1))
        h, next_state = self.core(z, state)
        pred, reward, done = self.decoder(h), self.reward(h), self.done(h)
        if single:
            pred, reward, done = pred[:, 0], reward[:, 0], done[:, 0]
        return WorldModelOutput(pred, reward, done, next_state)


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor


class RSSMWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        stoch_dim: int = 16,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = output_dim or obs_dim
        self.stoch_dim = stoch_dim
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.rnn = nn.GRUCell(stoch_dim + action_dim, hidden_dim)
        self.prior = nn.Linear(hidden_dim, 2 * stoch_dim)
        self.posterior = nn.Linear(2 * hidden_dim, 2 * stoch_dim)
        self.decoder = StructuredDecoder(hidden_dim + stoch_dim, self.output_dim, hidden_dim)
        self.reward = nn.Linear(hidden_dim + stoch_dim, 1)
        self.done = nn.Linear(hidden_dim + stoch_dim, 1)

    def initial(self, batch: int, device: torch.device) -> RSSMState:
        return RSSMState(torch.zeros(batch, self.rnn.hidden_size, device=device), torch.zeros(batch, self.stoch_dim, device=device))

    @staticmethod
    def _stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = x.chunk(2, dim=-1)
        return mean, F.softplus(raw_std) + 1e-4

    def forward(self, obs: torch.Tensor, action: torch.Tensor, state: object | None = None) -> WorldModelOutput:
        single = obs.ndim == 2
        if single:
            obs, action = obs.unsqueeze(1), action.unsqueeze(1)
        batch, time, _ = obs.shape
        s = state if isinstance(state, RSSMState) else self.initial(batch, obs.device)
        preds: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        kls: list[torch.Tensor] = []
        for t in range(time):
            deter = self.rnn(torch.cat([s.stoch, action[:, t]], dim=-1), s.deter)
            prior_mean, prior_std = self._stats(self.prior(deter))
            emb = self.obs_encoder(obs[:, t])
            post_mean, post_std = self._stats(self.posterior(torch.cat([deter, emb], dim=-1)))
            stoch = post_mean + torch.randn_like(post_std) * post_std
            feat = torch.cat([deter, stoch], dim=-1)
            preds.append(self.decoder(feat))
            rewards.append(self.reward(feat))
            dones.append(self.done(feat))
            kl = torch.log(prior_std / post_std) + (post_std.pow(2) + (post_mean - prior_mean).pow(2)) / (2 * prior_std.pow(2)) - 0.5
            kls.append(kl.mean(dim=-1, keepdim=True))
            s = RSSMState(deter, stoch)
        pred = torch.stack(preds, dim=1)
        reward = torch.stack(rewards, dim=1)
        done = torch.stack(dones, dim=1)
        kl_t = torch.stack(kls, dim=1).mean()
        if single:
            pred, reward, done = pred[:, 0], reward[:, 0], done[:, 0]
        return WorldModelOutput(pred, reward, done, s, {"kl": kl_t})


class TransformerWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
        heads: int = 2,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = output_dim or obs_dim
        self.input = nn.Linear(obs_dim + action_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, batch_first=True)
        self.core = nn.TransformerEncoder(layer, layers)
        self.decoder = StructuredDecoder(hidden_dim, self.output_dim, hidden_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.done = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, state: object | None = None) -> WorldModelOutput:
        single = obs.ndim == 2
        if single:
            obs, action = obs.unsqueeze(1), action.unsqueeze(1)
        x = self.input(torch.cat([obs, action], dim=-1))
        t = x.shape[1]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.core(x, mask=mask)
        pred, reward, done = self.decoder(h), self.reward(h), self.done(h)
        if single:
            pred, reward, done = pred[:, -1], reward[:, -1], done[:, -1]
        return WorldModelOutput(pred, reward, done, state)
