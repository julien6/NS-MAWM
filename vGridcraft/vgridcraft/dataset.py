from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import VGridcraftConfig
from .env import VectorizedGridcraftEnv


def dataset_key(config: VGridcraftConfig, episodes: int, max_steps: int, seed: int) -> str:
    payload = {
        "config": asdict(config),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "seed": int(seed),
        "version": 1,
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
    if reuse and path.exists() and not force_recollect:
        return torch.load(path, map_location="cpu"), path, True
    data = collect_dataset(
        episodes=episodes,
        max_steps=max_steps,
        num_envs=num_envs,
        device=device,
        seed=seed,
        config=config,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
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
    obs_records = []
    action_records = []
    reward_records = []
    done_records = []
    completed = 0
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed + 991)
    while completed < episodes:
        obs = env.reset()
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_done = []
        active = torch.ones((num_envs,), dtype=torch.bool, device=env.device)
        for _ in range(max_steps):
            episode_obs.append(obs["vector"].detach().cpu())
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
        obs_tensor = torch.stack(episode_obs, dim=1)
        action_tensor = torch.stack(episode_actions, dim=1)
        reward_tensor = torch.stack(episode_rewards, dim=1)
        done_tensor = torch.stack(episode_done, dim=1)
        take = min(num_envs, episodes - completed)
        obs_records.append(obs_tensor[:take])
        action_records.append(action_tensor[:take])
        reward_records.append(reward_tensor[:take])
        done_records.append(done_tensor[:take])
        completed += take
    return {
        "obs": torch.cat(obs_records, dim=0).float(),
        "action": torch.cat(action_records, dim=0).long(),
        "reward": torch.cat(reward_records, dim=0).float(),
        "done": torch.cat(done_records, dim=0).bool(),
        "metadata": {
            "episodes": int(episodes),
            "max_steps": int(max_steps),
            "num_envs": int(num_envs),
            "seed": int(seed),
            "config": asdict(config),
        },
    }


class RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict[str, Any]):
        self.obs = data["obs"].reshape(-1, data["obs"].shape[-1]).float()

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.obs[index]


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
