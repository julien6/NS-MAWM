from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import VGridcraftConfig
from .env import VectorizedGridcraftEnv


def dataset_key(config: VGridcraftConfig, episodes: int, max_steps: int, seed: int) -> str:
    payload = {
        "config": asdict(config),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "seed": int(seed),
        "version": 2,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def dataset_path(dataset_dir: str | os.PathLike[str], config: VGridcraftConfig, episodes: int, max_steps: int, seed: int) -> Path:
    return Path(dataset_dir) / f"gridcraft_{dataset_key(config, episodes, max_steps, seed)}.pt"


def collect_or_load_dataset(
    dataset_dir: str | os.PathLike[str],
    episodes: int,
    max_steps: int,
    num_envs: int,
    device: str,
    seed: int,
    config: VGridcraftConfig | None = None,
    reuse: bool = True,
    force_recollect: bool = False,
) -> tuple[dict[str, torch.Tensor], Path, bool]:
    config = config or VGridcraftConfig(max_steps=max_steps, seed=seed)
    path = dataset_path(dataset_dir, config, episodes, max_steps, seed)
    print(
        "[dataset] requested "
        f"path={path} episodes={episodes} max_steps={max_steps} "
        f"num_envs={num_envs} num_agents={config.num_agents} seed={seed} "
        f"reuse={int(bool(reuse))} force_recollect={int(bool(force_recollect))}",
        flush=True,
    )
    if reuse and path.exists() and not force_recollect:
        print(f"[dataset] reusing cached transitions from {path}", flush=True)
        return torch.load(path, map_location="cpu"), path, True
    if force_recollect and path.exists():
        print(f"[dataset] force recollect enabled; replacing cached transitions at {path}", flush=True)
    else:
        print(f"[dataset] cache miss; collecting transitions into {path}", flush=True)
    data = collect_dataset(
        episodes=episodes,
        max_steps=max_steps,
        num_envs=num_envs,
        device=device,
        seed=seed,
        config=config,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] assembled compact dataset size={dataset_nbytes(data) / (1024 ** 3):.2f} GiB", flush=True)
    torch.save(data, path)
    print(f"[dataset] saved transitions to {path}", flush=True)
    return data, path, False


def collect_dataset(
    episodes: int,
    max_steps: int,
    num_envs: int,
    device: str,
    seed: int,
    config: VGridcraftConfig | None = None,
) -> dict[str, torch.Tensor]:
    config = config or VGridcraftConfig(max_steps=max_steps, seed=seed)
    env = VectorizedGridcraftEnv(num_envs=num_envs, num_agents=config.num_agents, device=device, seed=seed, config=config)
    obs_grid_records = []
    obs_self_records = []
    action_records = []
    reward_records = []
    done_records = []
    completed = 0
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed + 991)
    print(
        "[dataset] collection started "
        f"episodes={episodes} max_steps={max_steps} num_envs={num_envs} "
        f"num_agents={config.num_agents} device={env.device}",
        flush=True,
    )
    while completed < episodes:
        obs = env.reset()
        episode_obs = []
        episode_self = []
        episode_actions = []
        episode_rewards = []
        episode_done = []
        active = torch.ones((num_envs,), dtype=torch.bool, device=env.device)
        for _ in range(max_steps):
            episode_obs.append(obs["grid"].detach().to(torch.int8).cpu())
            episode_self.append(obs["self"].detach().to(torch.int16).cpu())
            actions = torch.randint(0, config.action_size, (num_envs, config.num_agents), generator=generator, device=env.device)
            next_obs, rewards, done, truncated, _ = env.step(actions)
            terminal = done | truncated
            episode_actions.append(actions.detach().cpu())
            episode_rewards.append(rewards.detach().cpu())
            episode_done.append(terminal.detach().cpu())
            obs = next_obs
            active &= ~terminal
            if not bool(active.any()):
                break
        obs_grid_tensor = torch.stack(episode_obs, dim=1)
        obs_self_tensor = torch.stack(episode_self, dim=1)
        action_tensor = torch.stack(episode_actions, dim=1)
        reward_tensor = torch.stack(episode_rewards, dim=1)
        done_tensor = torch.stack(episode_done, dim=1)
        take = min(num_envs, episodes - completed)
        obs_grid_records.append(obs_grid_tensor[:take])
        obs_self_records.append(obs_self_tensor[:take])
        action_records.append(action_tensor[:take])
        reward_records.append(reward_tensor[:take])
        done_records.append(done_tensor[:take])
        completed += take
        if completed == episodes or completed % max(num_envs * 10, 1) == 0:
            print(f"[dataset] collected {completed}/{episodes} episodes", flush=True)
    return {
        "obs_grid": torch.cat(obs_grid_records, dim=0).to(torch.int8),
        "obs_self": torch.cat(obs_self_records, dim=0).to(torch.int16),
        "action": torch.cat(action_records, dim=0).long(),
        "reward": torch.cat(reward_records, dim=0).float(),
        "done": torch.cat(done_records, dim=0).bool(),
        "metadata": {
            "episodes": int(episodes),
            "max_steps": int(max_steps),
            "num_envs": int(num_envs),
            "seed": int(seed),
            "config": asdict(config),
            "storage": "tabular_compact_v2",
        },
    }


def has_compact_observations(data: dict[str, Any]) -> bool:
    return "obs_grid" in data and "obs_self" in data


def dataset_nbytes(data: dict[str, Any]) -> int:
    total = 0
    for value in data.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return int(total)


def observation_shape(data: dict[str, Any]) -> tuple[int, int, int]:
    if has_compact_observations(data):
        episodes, steps, agents = data["obs_grid"].shape[:3]
        return int(episodes), int(steps), int(agents)
    obs = data["obs"]
    if obs.ndim == 4:
        episodes, steps, agents = obs.shape[:3]
        return int(episodes), int(steps), int(agents)
    if obs.ndim == 3:
        episodes, steps = obs.shape[:2]
        return int(episodes), int(steps), 1
    raise ValueError(f"unsupported observation tensor shape: {tuple(obs.shape)}")


def vector_from_tabular(grid: torch.Tensor, self_vec: torch.Tensor, config: VGridcraftConfig | None = None) -> torch.Tensor:
    config = config or VGridcraftConfig()
    terrain = F.one_hot(grid[..., 0, :, :].long().clamp(0, config.terrain_classes - 1), config.terrain_classes).float().flatten(start_dim=-3)
    blocks = F.one_hot(grid[..., 1, :, :].long().clamp(0, config.block_classes - 1), config.block_classes).float().flatten(start_dim=-3)
    entities = F.one_hot(grid[..., 2, :, :].long().clamp(0, config.entity_classes - 1), config.entity_classes).float().flatten(start_dim=-3)
    numeric = torch.zeros_like(self_vec, dtype=torch.float32)
    numeric[..., :2] = self_vec[..., :2].float() / 20.0
    numeric[..., 2:] = self_vec[..., 2:].float().clamp(0, 10) / 10.0
    return torch.cat([terrain, blocks, entities, numeric], dim=-1)


def observation_vectors(data: dict[str, Any], episode_limit: int | None = None) -> torch.Tensor:
    if has_compact_observations(data):
        grid = data["obs_grid"]
        self_vec = data["obs_self"]
        if episode_limit is not None:
            grid = grid[:episode_limit]
            self_vec = self_vec[:episode_limit]
        return vector_from_tabular(grid, self_vec)
    obs = data["obs"]
    if episode_limit is not None:
        obs = obs[:episode_limit]
    if obs.ndim == 3:
        obs = obs.unsqueeze(2)
    return obs.float()


class RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict[str, Any]):
        self.compact = has_compact_observations(data)
        if self.compact:
            self.grid = data["obs_grid"]
            self.self_vec = data["obs_self"]
            self.episodes, self.steps, self.agents = self.grid.shape[:3]
        else:
            obs = data["obs"]
            if obs.ndim == 3:
                obs = obs.unsqueeze(2)
            self.obs = obs.reshape(-1, obs.shape[-1]).float()

    def __len__(self) -> int:
        if self.compact:
            return int(self.episodes * self.steps * self.agents)
        return int(self.obs.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.compact:
            agent = index % self.agents
            tmp = index // self.agents
            step = tmp % self.steps
            episode = tmp // self.steps
            return vector_from_tabular(self.grid[episode, step, agent], self.self_vec[episode, step, agent])
        return self.obs[index]

    def __getitems__(self, indices: list[int]) -> torch.Tensor:
        if not self.compact:
            return [item for item in self.obs[indices]]
        idx = torch.as_tensor(indices, dtype=torch.long)
        agent = idx % self.agents
        tmp = idx // self.agents
        step = tmp % self.steps
        episode = tmp // self.steps
        return [item for item in vector_from_tabular(self.grid[episode, step, agent], self.self_vec[episode, step, agent])]


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, z: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, seq_len: int):
        self.z = z.float()
        self.actions = actions.long()
        self.rewards = rewards.float()
        self.dones = dones.float()
        self.seq_len = int(seq_len)
        self.episodes = self.z.shape[0]
        self.steps = self.z.shape[1] - 1
        if self.steps < self.seq_len:
            raise ValueError("sequence length is longer than collected episodes")

    def __len__(self) -> int:
        return self.episodes * (self.steps - self.seq_len + 1)

    def __getitem__(self, index: int):
        episode = index // (self.steps - self.seq_len + 1)
        start = index % (self.steps - self.seq_len + 1)
        end = start + self.seq_len + 1
        return (
            self.z[episode, start:end],
            self.actions[episode, start:start + self.seq_len],
            self.rewards[episode, start:start + self.seq_len],
            self.dones[episode, start:start + self.seq_len],
        )
