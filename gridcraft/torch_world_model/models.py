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
STRUCTURED_EVENT_NAMES = (
    "move_success",
    "harvest_success",
    "pickup_success",
    "eat_success",
    "craft_plank",
    "craft_stick",
    "craft_wood_tool",
    "craft_stone_tool",
    "tool_equipped",
    "mob_hit",
    "mob_kill_armed",
    "agent_death",
    "milestone_level_1",
    "milestone_level_2",
    "milestone_level_3",
    "milestone_level_4",
    "milestone_level_5",
    "milestone_level_6",
    "milestone_level_7",
    "milestone_level_8",
)
STRUCTURED_EVENT_DIM = len(STRUCTURED_EVENT_NAMES)


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


def split_observation_vector(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cursor = 0
    terrain = obs[..., cursor : cursor + GRID_CELLS * TERRAIN_CLASSES].reshape(*obs.shape[:-1], GRID_CELLS, TERRAIN_CLASSES)
    cursor += GRID_CELLS * TERRAIN_CLASSES
    blocks = obs[..., cursor : cursor + GRID_CELLS * BLOCK_CLASSES].reshape(*obs.shape[:-1], GRID_CELLS, BLOCK_CLASSES)
    cursor += GRID_CELLS * BLOCK_CLASSES
    entities = obs[..., cursor : cursor + GRID_CELLS * ENTITY_CLASSES].reshape(*obs.shape[:-1], GRID_CELLS, ENTITY_CLASSES)
    self_vec = obs[..., -SELF_FEATURES:]
    return terrain, blocks, entities, self_vec


def observation_labels(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    terrain, blocks, entities, self_vec = split_observation_vector(obs)
    return terrain.argmax(-1), blocks.argmax(-1), entities.argmax(-1), self_vec.float()


def logits_to_observation_vector(
    terrain_logits: torch.Tensor,
    block_logits: torch.Tensor,
    entity_logits: torch.Tensor,
    self_pred: torch.Tensor,
) -> torch.Tensor:
    terrain = F.one_hot(terrain_logits.argmax(-1), TERRAIN_CLASSES).float().flatten(start_dim=-2)
    blocks = F.one_hot(block_logits.argmax(-1), BLOCK_CLASSES).float().flatten(start_dim=-2)
    entities = F.one_hot(entity_logits.argmax(-1), ENTITY_CLASSES).float().flatten(start_dim=-2)
    self_vec = self_pred.float().clamp(0.0, 1.0)
    return torch.cat([terrain, blocks, entities, self_vec], dim=-1)


class StructuredGridcraftWorldModel(nn.Module):
    """Factorized Gridcraft world model for discrete local observations."""

    def __init__(
        self,
        grid_embed_dim: int = 32,
        cnn_channels: int = 128,
        self_hidden_size: int = 128,
        agent_hidden_size: int = 256,
        attention_heads: int = 4,
        num_attention_layers: int = 1,
        transition_hidden_size: int = 256,
        action_size: int = ACTION_SIZE,
        event_dim: int = STRUCTURED_EVENT_DIM,
    ):
        super().__init__()
        self.grid_embed_dim = int(grid_embed_dim)
        self.cnn_channels = int(cnn_channels)
        self.self_hidden_size = int(self_hidden_size)
        self.agent_hidden_size = int(agent_hidden_size)
        self.attention_heads = int(attention_heads)
        self.num_attention_layers = int(num_attention_layers)
        self.transition_hidden_size = int(transition_hidden_size)
        self.action_size = int(action_size)
        self.event_dim = int(event_dim)

        self.terrain_embedding = nn.Embedding(TERRAIN_CLASSES, self.grid_embed_dim)
        self.block_embedding = nn.Embedding(BLOCK_CLASSES, self.grid_embed_dim)
        self.entity_embedding = nn.Embedding(ENTITY_CLASSES, self.grid_embed_dim)
        in_channels = self.grid_embed_dim * 3
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(in_channels, self.cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.cnn_channels, self.cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.cnn_channels * 7 * 7, self.agent_hidden_size),
            nn.ReLU(),
        )
        self.self_encoder = nn.Sequential(
            nn.Linear(SELF_FEATURES, self.self_hidden_size),
            nn.ReLU(),
            nn.Linear(self.self_hidden_size, self.self_hidden_size),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(self.action_size, self.self_hidden_size)
        fusion_dim = self.agent_hidden_size + self.self_hidden_size + self.self_hidden_size
        self.agent_fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.agent_hidden_size),
            nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.agent_hidden_size,
            nhead=max(1, self.attention_heads),
            dim_feedforward=max(self.agent_hidden_size * 2, 64),
            batch_first=True,
            dropout=0.0,
            activation="relu",
        )
        self.agent_attention = nn.TransformerEncoder(encoder_layer, num_layers=max(1, self.num_attention_layers))
        self.transition = nn.GRUCell(self.agent_hidden_size, self.transition_hidden_size)
        self.hidden_to_agent = nn.Sequential(
            nn.Linear(self.transition_hidden_size, self.agent_hidden_size),
            nn.ReLU(),
        )
        self.terrain_head = nn.Linear(self.agent_hidden_size, GRID_CELLS * TERRAIN_CLASSES)
        self.block_head = nn.Linear(self.agent_hidden_size, GRID_CELLS * BLOCK_CLASSES)
        self.entity_head = nn.Linear(self.agent_hidden_size, GRID_CELLS * ENTITY_CLASSES)
        self.self_head = nn.Linear(self.agent_hidden_size, SELF_FEATURES)
        self.reward_head = nn.Linear(self.agent_hidden_size, 1)
        self.done_head = nn.Linear(self.agent_hidden_size, 1)
        self.event_head = nn.Linear(self.agent_hidden_size, self.event_dim)

    def initial_hidden(self, batch: int, agents: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, agents, self.transition_hidden_size, device=device)

    def encode_step(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        terrain, blocks, entities, self_vec = observation_labels(obs)
        batch, agents = obs.shape[:2]
        terrain_emb = self.terrain_embedding(terrain.long())
        block_emb = self.block_embedding(blocks.long())
        entity_emb = self.entity_embedding(entities.long())
        grid = torch.cat([terrain_emb, block_emb, entity_emb], dim=-1)
        grid = grid.reshape(batch * agents, 7, 7, -1).permute(0, 3, 1, 2).contiguous()
        grid_repr = self.grid_encoder(grid).reshape(batch, agents, -1)
        self_repr = self.self_encoder(self_vec.float())
        action_repr = self.action_embedding(action.long().clamp(0, self.action_size - 1))
        fused = self.agent_fusion(torch.cat([grid_repr, self_repr, action_repr], dim=-1))
        return self.agent_attention(fused)

    def decode_hidden(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        agent = self.hidden_to_agent(hidden)
        return {
            "terrain_logits": self.terrain_head(agent).reshape(*agent.shape[:-1], GRID_CELLS, TERRAIN_CLASSES),
            "block_logits": self.block_head(agent).reshape(*agent.shape[:-1], GRID_CELLS, BLOCK_CLASSES),
            "entity_logits": self.entity_head(agent).reshape(*agent.shape[:-1], GRID_CELLS, ENTITY_CLASSES),
            "self_pred": torch.sigmoid(self.self_head(agent)),
            "reward_pred": self.reward_head(agent),
            "done_logit": self.done_head(agent),
            "event_logits": self.event_head(agent),
        }

    def step(self, obs: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if obs.ndim != 3:
            raise ValueError(f"expected obs [batch, agents, features], got {tuple(obs.shape)}")
        batch, agents = obs.shape[:2]
        if hidden is None:
            hidden = self.initial_hidden(batch, agents, device=obs.device)
        encoded = self.encode_step(obs.float(), action.long())
        next_hidden = self.transition(
            encoded.reshape(batch * agents, -1),
            hidden.reshape(batch * agents, -1),
        ).reshape(batch, agents, -1)
        return self.decode_hidden(next_hidden), next_hidden

    def decode_to_obs_vector(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return logits_to_observation_vector(
            outputs["terrain_logits"],
            outputs["block_logits"],
            outputs["entity_logits"],
            outputs["self_pred"],
        )

    def rollout_teacher_forced(self, obs_seq: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        preds: dict[str, list[torch.Tensor]] = {
            "terrain_logits": [],
            "block_logits": [],
            "entity_logits": [],
            "self_pred": [],
            "reward_pred": [],
            "done_logit": [],
            "event_logits": [],
        }
        hidden = None
        for t in range(actions.shape[1]):
            out, hidden = self.step(obs_seq[:, t], actions[:, t], hidden)
            for key, value in out.items():
                preds[key].append(value)
        return {key: torch.stack(values, dim=1) for key, values in preds.items()}

    def loss(
        self,
        obs_seq: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        events: torch.Tensor | None = None,
        reward_loss_weight: float = 10.0,
        done_loss_weight: float = 5.0,
        event_loss_weight: float = 5.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = self.rollout_teacher_forced(obs_seq[:, :-1], actions)
        target_obs = obs_seq[:, 1:]
        terrain_target, block_target, entity_target, self_target = observation_labels(target_obs)
        terrain_loss = F.cross_entropy(
            outputs["terrain_logits"].reshape(-1, TERRAIN_CLASSES),
            terrain_target.reshape(-1).long(),
        )
        block_loss = F.cross_entropy(
            outputs["block_logits"].reshape(-1, BLOCK_CLASSES),
            block_target.reshape(-1).long(),
        )
        entity_loss = F.cross_entropy(
            outputs["entity_logits"].reshape(-1, ENTITY_CLASSES),
            entity_target.reshape(-1).long(),
        )
        self_loss = F.smooth_l1_loss(outputs["self_pred"], self_target.float())
        reward_target = rewards.unsqueeze(-1).float()
        if dones.ndim == rewards.ndim - 1:
            dones = dones.unsqueeze(-1).expand_as(rewards)
        done_target = dones.unsqueeze(-1).float()
        reward_loss = F.mse_loss(outputs["reward_pred"], reward_target)
        done_loss = F.binary_cross_entropy_with_logits(outputs["done_logit"], done_target)
        if events is None:
            event_target = torch.zeros_like(outputs["event_logits"])
        else:
            event_target = events.float()
        event_loss = F.binary_cross_entropy_with_logits(outputs["event_logits"], event_target)
        loss = (
            terrain_loss
            + block_loss
            + entity_loss
            + self_loss
            + float(reward_loss_weight) * reward_loss
            + float(done_loss_weight) * done_loss
            + float(event_loss_weight) * event_loss
        )
        with torch.no_grad():
            terrain_acc = (outputs["terrain_logits"].argmax(-1) == terrain_target).float().mean()
            block_acc = (outputs["block_logits"].argmax(-1) == block_target).float().mean()
            entity_acc = (outputs["entity_logits"].argmax(-1) == entity_target).float().mean()
            done_pred = torch.sigmoid(outputs["done_logit"]) > 0.5
            done_acc = (done_pred == done_target.bool()).float().mean()
        return loss, {
            "training_structured_total_loss": float(loss.detach().cpu()),
            "training_structured_terrain_loss": float(terrain_loss.detach().cpu()),
            "training_structured_block_loss": float(block_loss.detach().cpu()),
            "training_structured_entity_loss": float(entity_loss.detach().cpu()),
            "training_structured_self_loss": float(self_loss.detach().cpu()),
            "training_structured_reward_loss": float(reward_loss.detach().cpu()),
            "training_structured_done_loss": float(done_loss.detach().cpu()),
            "training_structured_event_loss": float(event_loss.detach().cpu()),
            "training_structured_terrain_acc": float(terrain_acc.detach().cpu()),
            "training_structured_block_acc": float(block_acc.detach().cpu()),
            "training_structured_entity_acc": float(entity_acc.detach().cpu()),
            "training_structured_done_accuracy": float(done_acc.detach().cpu()),
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


def make_structured_from_config(config: dict | None = None) -> StructuredGridcraftWorldModel:
    structured = (config or {}).get("structured", {})
    return StructuredGridcraftWorldModel(
        grid_embed_dim=int(structured.get("grid_embed_dim", 32)),
        cnn_channels=int(structured.get("cnn_channels", 128)),
        self_hidden_size=int(structured.get("self_hidden_size", 128)),
        agent_hidden_size=int(structured.get("agent_hidden_size", 256)),
        attention_heads=int(structured.get("attention_heads", 4)),
        num_attention_layers=int(structured.get("num_attention_layers", 1)),
        transition_hidden_size=int(structured.get("transition_hidden_size", 256)),
        action_size=int(structured.get("action_size", ACTION_SIZE)),
        event_dim=int(structured.get("event_dim", STRUCTURED_EVENT_DIM)),
    )
