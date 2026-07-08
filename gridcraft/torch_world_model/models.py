from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


GRID_CELLS = 49
TERRAIN_CLASSES = 3
BLOCK_CLASSES = 4
ENTITY_CLASSES = 4
SELF_FEATURES = 11
OBS_SIZE = GRID_CELLS * (TERRAIN_CLASSES + BLOCK_CLASSES + ENTITY_CLASSES) + SELF_FEATURES
ACTION_SIZE = 15


class TorchGridcraftVAE(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, z_size: int = 64, hidden_size: int = 512, kl_tolerance: float = 0.5):
        super().__init__()
        self.obs_size = obs_size
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.kl_tolerance = kl_tolerance
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, z_size)
        self.logvar = nn.Linear(hidden_size, z_size)
        self.decoder = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_size),
        )

    def encode_mu_logvar(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs.float())
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def encode(self, obs: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mu, logvar = self.encode_mu_logvar(obs)
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z.float())

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_mu_logvar(obs)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar

    def loss(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        decoded, mu, logvar = self(obs)
        terrain = categorical_plane_loss(decoded, obs, 0, TERRAIN_CLASSES)
        block_offset = GRID_CELLS * TERRAIN_CLASSES
        blocks = categorical_plane_loss(decoded, obs, block_offset, BLOCK_CLASSES)
        entity_offset = GRID_CELLS * (TERRAIN_CLASSES + BLOCK_CLASSES)
        entities = categorical_plane_loss(decoded, obs, entity_offset, ENTITY_CLASSES)
        grid_loss = terrain + blocks + entities
        self_loss = F.mse_loss(decoded[:, -SELF_FEATURES:], obs[:, -SELF_FEATURES:])
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = torch.maximum(kl, torch.full_like(kl, self.kl_tolerance * self.z_size)).mean()
        loss = grid_loss + self_loss + kl
        return loss, {
            "vae_loss": float(loss.detach().cpu()),
            "vae_grid_loss": float(grid_loss.detach().cpu()),
            "vae_self_loss": float(self_loss.detach().cpu()),
            "vae_kl_loss": float(kl.detach().cpu()),
        }

    @torch.no_grad()
    def decode_tabular(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded = self.decode(z)
        cursor = 0
        terrain = decoded[:, cursor:cursor + GRID_CELLS * TERRAIN_CLASSES].reshape(-1, GRID_CELLS, TERRAIN_CLASSES).argmax(-1)
        cursor += GRID_CELLS * TERRAIN_CLASSES
        blocks = decoded[:, cursor:cursor + GRID_CELLS * BLOCK_CLASSES].reshape(-1, GRID_CELLS, BLOCK_CLASSES).argmax(-1)
        cursor += GRID_CELLS * BLOCK_CLASSES
        entities = decoded[:, cursor:cursor + GRID_CELLS * ENTITY_CLASSES].reshape(-1, GRID_CELLS, ENTITY_CLASSES).argmax(-1)
        self_vec = decoded[:, -SELF_FEATURES:]
        hp_hunger = torch.round(self_vec[:, :2] * 20.0).clamp(0, 20)
        inventory = torch.round(self_vec[:, 2:] * 10.0).clamp(0, 99)
        return {
            "grid": torch.stack([
                terrain.reshape(-1, 7, 7),
                blocks.reshape(-1, 7, 7),
                entities.reshape(-1, 7, 7),
            ], dim=1).long(),
            "self": torch.cat([hp_hunger, inventory], dim=1).long(),
        }


class TorchGridcraftRNN(nn.Module):
    def __init__(self, z_size: int = 64, action_size: int = ACTION_SIZE, rnn_size: int = 128, num_mixture: int = 5):
        super().__init__()
        self.z_size = z_size
        self.action_size = action_size
        self.rnn_size = rnn_size
        self.num_mixture = num_mixture
        self.rnn = nn.LSTM(input_size=z_size + action_size, hidden_size=rnn_size, batch_first=True)
        self.head = nn.Linear(rnn_size, z_size * num_mixture * 3 + 2)
        self.obs_head = nn.Linear(rnn_size, OBS_SIZE)

    def forward(self, z: torch.Tensor, actions: torch.Tensor, state=None):
        actions = actions.long().clamp(0, self.action_size - 1)
        action_oh = F.one_hot(actions, self.action_size).float()
        out, next_state = self.rnn(torch.cat([z, action_oh], dim=-1), state)
        raw = self.head(out)
        return self.split_output(raw), next_state

    def forward_with_observation(self, z: torch.Tensor, actions: torch.Tensor, state=None):
        actions = actions.long().clamp(0, self.action_size - 1)
        action_oh = F.one_hot(actions, self.action_size).float()
        out, next_state = self.rnn(torch.cat([z, action_oh], dim=-1), state)
        raw = self.head(out)
        obs = self.obs_head(out)
        return self.split_output(raw), obs, next_state

    def step(self, z: torch.Tensor, action: torch.Tensor, state=None, deterministic: bool = True):
        z_in = z[:, None, :]
        action_in = action[:, None]
        (logmix, mean, logstd, reward, done_logit), next_state = self.forward(z_in, action_in, state)
        logmix = logmix[:, 0]
        mean = mean[:, 0]
        logstd = logstd[:, 0]
        reward = reward[:, 0]
        done_logit = done_logit[:, 0]
        if deterministic:
            mix = logmix.exp()
            next_z = (mix * mean).sum(dim=-1)
        else:
            next_z = sample_mdn(logmix, mean, logstd)
        return next_z, reward, done_logit, next_state

    def step_with_observation(self, z: torch.Tensor, action: torch.Tensor, state=None, deterministic: bool = True):
        z_in = z[:, None, :]
        action_in = action[:, None]
        (logmix, mean, logstd, reward, done_logit), obs, next_state = self.forward_with_observation(z_in, action_in, state)
        logmix = logmix[:, 0]
        mean = mean[:, 0]
        logstd = logstd[:, 0]
        reward = reward[:, 0]
        done_logit = done_logit[:, 0]
        obs = obs[:, 0]
        if deterministic:
            mix = logmix.exp()
            next_z = (mix * mean).sum(dim=-1)
        else:
            next_z = sample_mdn(logmix, mean, logstd)
        return next_z, reward, done_logit, next_state, obs

    def split_output(self, raw: torch.Tensor):
        mdn = raw[..., : self.z_size * self.num_mixture * 3]
        reward = raw[..., -2]
        done_logit = raw[..., -1]
        mdn = mdn.reshape(*raw.shape[:-1], self.z_size, self.num_mixture * 3)
        logmix, mean, logstd = torch.split(mdn, self.num_mixture, dim=-1)
        return F.log_softmax(logmix, dim=-1), mean, logstd.clamp(-6.0, 2.0), reward, done_logit

    def loss(
        self,
        z_seq: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        mean_mse_weight: float = 10.0,
        reward_loss_weight: float = 1.0,
        done_loss_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target_z = z_seq[:, 1:]
        pred_input = z_seq[:, :-1]
        (logmix, mean, logstd, reward_pred, done_logit), _ = self.forward(pred_input, actions)
        target = target_z.unsqueeze(-1)
        log_prob = logmix - 0.5 * ((target - mean) / logstd.exp()).pow(2) - logstd - 0.5 * math.log(2.0 * math.pi)
        z_nll = -torch.logsumexp(log_prob, dim=-1).mean()
        expected_z = (logmix.exp() * mean).sum(dim=-1)
        mean_mse = F.mse_loss(expected_z, target_z)
        reward_loss = F.mse_loss(reward_pred, rewards)
        done_loss = F.binary_cross_entropy_with_logits(done_logit, dones.float())
        loss = (
            z_nll
            + float(mean_mse_weight) * mean_mse
            + float(reward_loss_weight) * reward_loss
            + float(done_loss_weight) * done_loss
        )
        return loss, {
            "training_wm_total_loss": float(loss.detach().cpu()),
            "training_obs_loss": float(z_nll.detach().cpu()),
            "training_mean_mse": float(mean_mse.detach().cpu()),
            "training_reward_loss": float(reward_loss.detach().cpu()),
            "training_done_loss": float(done_loss.detach().cpu()),
            "training_mean_mse_weight": float(mean_mse_weight),
            "training_reward_loss_weight": float(reward_loss_weight),
            "training_done_loss_weight": float(done_loss_weight),
        }


def categorical_plane_loss(decoded: torch.Tensor, target: torch.Tensor, offset: int, depth: int) -> torch.Tensor:
    logits = decoded[:, offset:offset + GRID_CELLS * depth].reshape(-1, GRID_CELLS, depth)
    labels = target[:, offset:offset + GRID_CELLS * depth].reshape(-1, GRID_CELLS, depth).argmax(-1)
    return F.cross_entropy(logits.reshape(-1, depth), labels.reshape(-1))


def sample_mdn(logmix: torch.Tensor, mean: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    batch, z_size, num_mix = logmix.shape
    component = torch.distributions.Categorical(logits=logmix.reshape(-1, num_mix)).sample().reshape(batch, z_size)
    gather = component.unsqueeze(-1)
    selected_mean = torch.gather(mean, -1, gather).squeeze(-1)
    selected_logstd = torch.gather(logstd, -1, gather).squeeze(-1)
    return selected_mean + selected_logstd.exp() * torch.randn_like(selected_mean)


def load_world_model_config(checkpoint_dir: str | Path) -> dict:
    path = Path(checkpoint_dir) / "world_model_config.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def make_vae_from_config(config: dict | None = None) -> TorchGridcraftVAE:
    vae_config = (config or {}).get("vae", {})
    return TorchGridcraftVAE(
        z_size=int(vae_config.get("z_size", 64)),
        hidden_size=int(vae_config.get("hidden_size", 512)),
        kl_tolerance=float(vae_config.get("kl_tolerance", 0.5)),
    )


def make_rnn_from_config(config: dict | None = None) -> TorchGridcraftRNN:
    vae_config = (config or {}).get("vae", {})
    rnn_config = (config or {}).get("rnn", {})
    return TorchGridcraftRNN(
        z_size=int(rnn_config.get("z_size", vae_config.get("z_size", 64))),
        action_size=int(rnn_config.get("action_size", ACTION_SIZE)),
        rnn_size=int(rnn_config.get("rnn_size", 128)),
        num_mixture=int(rnn_config.get("num_mixture", 5)),
    )
