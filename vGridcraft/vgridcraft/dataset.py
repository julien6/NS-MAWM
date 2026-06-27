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
            "version": 3,
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
    try:
        obs_grid_records = []
        obs_self_records = []
        action_records = []
        reward_records = []
        done_records = []
        valid_records = []
        episode_lengths = []
        completed = 0
        generator = torch.Generator(device=env.device)
        generator.manual_seed(seed + 991)
        print(
            "[dataset] collection started "
            f"episodes={episodes} max_steps={max_steps} num_envs={num_envs} "
            f"num_agents={config.num_agents} device={env.device}",
            flush=True,
        )
        obs = env.reset()
        per_env = [new_episode_buffers() for _ in range(num_envs)]
        while completed < episodes:
            actions = torch.randint(0, config.action_size, (num_envs, config.num_agents), generator=generator, device=env.device)
            next_obs, rewards, done, truncated, _ = env.step(actions)
            terminal = done | truncated
            full_length = torch.tensor(
                [len(buffers["actions"]) + 1 >= max_steps for buffers in per_env],
                dtype=torch.bool,
                device=env.device,
            )
            terminal = terminal | full_length

            for env_idx in range(num_envs):
                buffers = per_env[env_idx]
                buffers["obs_grid"].append(obs["grid"][env_idx].detach().to(torch.int8).cpu())
                buffers["obs_self"].append(obs["self"][env_idx].detach().to(torch.int16).cpu())
                buffers["actions"].append(actions[env_idx].detach().cpu())
                buffers["rewards"].append(rewards[env_idx].detach().cpu())
                buffers["done"].append(bool(terminal[env_idx].detach().cpu()))

            finished = torch.nonzero(terminal, as_tuple=False).flatten()
            if finished.numel() > 0:
                for env_idx in finished.detach().cpu().tolist():
                    if completed >= episodes:
                        break
                    finalized = finalize_episode(per_env[env_idx], max_steps=max_steps, config=config)
                    obs_grid_records.append(finalized["obs_grid"])
                    obs_self_records.append(finalized["obs_self"])
                    action_records.append(finalized["action"])
                    reward_records.append(finalized["reward"])
                    done_records.append(finalized["done"])
                    valid_records.append(finalized["transition_valid"])
                    episode_lengths.append(finalized["episode_length"])
                    per_env[env_idx] = new_episode_buffers()
                    completed += 1
                    if completed == episodes or completed % max(num_envs * 10, 1) == 0:
                        print(f"[dataset] collected {completed}/{episodes} episodes", flush=True)
                reset_ids = finished[: max(0, min(finished.numel(), num_envs))]
                if reset_ids.numel() > 0:
                    env.reset(env_ids=reset_ids)
                    next_obs = env.observation()
            obs = next_obs

        lengths = torch.as_tensor(episode_lengths, dtype=torch.long)
        skipped = int(episodes * max_steps - int(torch.stack(valid_records, dim=0).sum().item()))
        return {
            "obs_grid": torch.stack(obs_grid_records, dim=0).to(torch.int8),
            "obs_self": torch.stack(obs_self_records, dim=0).to(torch.int16),
            "action": torch.stack(action_records, dim=0).long(),
            "reward": torch.stack(reward_records, dim=0).float(),
            "done": torch.stack(done_records, dim=0).bool(),
            "transition_valid": torch.stack(valid_records, dim=0).bool(),
            "episode_length": lengths,
            "metadata": {
                "episodes": int(episodes),
                "max_steps": int(max_steps),
                "num_envs": int(num_envs),
                "seed": int(seed),
                "config": asdict(config),
                "storage": "tabular_compact_v3",
                "valid_transition_count": int(torch.stack(valid_records, dim=0).sum().item()),
                "invalid_padded_transition_count": skipped,
                "mean_episode_length": float(lengths.float().mean().item()) if lengths.numel() else 0.0,
                "truncation_or_done_rate": float((lengths.float() < float(max_steps)).float().mean().item()) if lengths.numel() else 0.0,
            },
        }
    finally:
        env.close()


def new_episode_buffers() -> dict[str, list[Any]]:
    return {"obs_grid": [], "obs_self": [], "actions": [], "rewards": [], "done": []}


def finalize_episode(buffers: dict[str, list[Any]], max_steps: int, config: VGridcraftConfig) -> dict[str, torch.Tensor | int]:
    length = len(buffers["actions"])
    if length <= 0:
        raise RuntimeError("cannot finalize an empty episode")
    pad = max_steps - length
    obs_grid = torch.stack(buffers["obs_grid"], dim=0)
    obs_self = torch.stack(buffers["obs_self"], dim=0)
    action = torch.stack(buffers["actions"], dim=0)
    reward = torch.stack(buffers["rewards"], dim=0)
    done = torch.as_tensor(buffers["done"], dtype=torch.bool)
    valid = torch.ones((length,), dtype=torch.bool)
    if pad > 0:
        obs_grid = torch.cat([obs_grid, obs_grid[-1:].expand(pad, *obs_grid.shape[1:])], dim=0)
        obs_self = torch.cat([obs_self, obs_self[-1:].expand(pad, *obs_self.shape[1:])], dim=0)
        action = torch.cat([action, torch.zeros((pad, config.num_agents), dtype=action.dtype)], dim=0)
        reward = torch.cat([reward, torch.zeros((pad, config.num_agents), dtype=reward.dtype)], dim=0)
        done = torch.cat([done, torch.ones((pad,), dtype=torch.bool)], dim=0)
        valid = torch.cat([valid, torch.zeros((pad,), dtype=torch.bool)], dim=0)
    return {
        "obs_grid": obs_grid[:max_steps],
        "obs_self": obs_self[:max_steps],
        "action": action[:max_steps],
        "reward": reward[:max_steps],
        "done": done[:max_steps],
        "transition_valid": valid[:max_steps],
        "episode_length": int(min(length, max_steps)),
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
            if "transition_valid" in data:
                valid = data["transition_valid"].bool()
                self.valid_indices = torch.nonzero(valid[:, :, None].expand(-1, -1, self.agents).reshape(-1), as_tuple=False).flatten()
            else:
                self.valid_indices = None
        else:
            obs = data["obs"]
            if obs.ndim == 3:
                obs = obs.unsqueeze(2)
            self.obs = obs.reshape(-1, obs.shape[-1]).float()

    def __len__(self) -> int:
        if self.compact:
            if self.valid_indices is not None:
                return int(self.valid_indices.numel())
            return int(self.episodes * self.steps * self.agents)
        return int(self.obs.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.compact:
            if self.valid_indices is not None:
                index = int(self.valid_indices[index])
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
        if self.valid_indices is not None:
            idx = self.valid_indices[idx]
        agent = idx % self.agents
        tmp = idx // self.agents
        step = tmp % self.steps
        episode = tmp // self.steps
        return [item for item in vector_from_tabular(self.grid[episode, step, agent], self.self_vec[episode, step, agent])]


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, z: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, seq_len: int, valid: torch.Tensor | None = None, obs: torch.Tensor | None = None):
        self.z = z.float()
        self.actions = actions.long()
        self.rewards = rewards.float()
        self.dones = dones.float()
        self.valid = valid.bool() if valid is not None else None
        self.obs = obs.float() if obs is not None else None
        self.seq_len = int(seq_len)
        self.episodes = self.z.shape[0]
        self.steps = self.z.shape[1] - 1
        if self.steps < self.seq_len:
            raise ValueError("sequence length is longer than collected episodes")
        self.indices = self._valid_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        episode, start = self.indices[index]
        end = start + self.seq_len + 1
        items = [
            self.z[episode, start:end],
            self.actions[episode, start:start + self.seq_len],
            self.rewards[episode, start:start + self.seq_len],
            self.dones[episode, start:start + self.seq_len],
        ]
        if self.obs is not None:
            items.append(self.obs[episode, start:end])
        return tuple(items)

    def _valid_indices(self) -> list[tuple[int, int]]:
        indices = []
        starts_per_episode = self.steps - self.seq_len + 1
        if self.valid is None:
            return [(episode, start) for episode in range(self.episodes) for start in range(starts_per_episode)]
        for episode in range(self.episodes):
            for start in range(starts_per_episode):
                end = start + self.seq_len
                if bool(self.valid[episode, start:end].all()):
                    done_window = self.dones[episode, start:end]
                    if done_window.numel() > 1 and bool(done_window[:-1].bool().any()):
                        continue
                    indices.append((episode, start))
        return indices
