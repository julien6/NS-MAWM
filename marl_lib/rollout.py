"""Rollout collection independent of concrete environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class FlatEnv(Protocol):
    n_agents: int
    action_dim: int
    obs_dim: int

    variant_id: str

    def reset(self, seed: int | None = None, variant: object | None = None) -> torch.Tensor: ...
    def step(self, action: torch.Tensor): ...


@dataclass(frozen=True)
class TransitionBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    next_obs: torch.Tensor
    variant_id: tuple[str, ...] = ()
    seed_id: torch.Tensor | None = None
    info: tuple[dict[str, object], ...] = ()

    def to(self, device: torch.device) -> "TransitionBatch":
        return TransitionBatch(
            self.obs.to(device),
            self.action.to(device),
            self.reward.to(device),
            self.done.to(device),
            self.next_obs.to(device),
            self.variant_id,
            None if self.seed_id is None else self.seed_id.to(device),
            self.info,
        )


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, seed: int = 0):
        self.capacity = capacity
        self.generator = torch.Generator().manual_seed(seed)
        self._obs: list[torch.Tensor] = []
        self._action: list[torch.Tensor] = []
        self._reward: list[torch.Tensor] = []
        self._done: list[torch.Tensor] = []
        self._next_obs: list[torch.Tensor] = []
        self._variant: list[str] = []
        self._seed: list[int] = []
        self._info: list[dict[str, object]] = []

    def __len__(self) -> int:
        return len(self._obs)

    def extend(self, batch: TransitionBatch, variant_id: str = "default", seed_id: int = 0) -> None:
        for i in range(batch.obs.shape[0]):
            self.push(
                batch.obs[i],
                batch.action[i],
                batch.reward[i],
                batch.done[i],
                batch.next_obs[i],
                batch.variant_id[i] if batch.variant_id else variant_id,
                int(batch.seed_id[i].item()) if batch.seed_id is not None else seed_id,
                batch.info[i] if batch.info else {},
            )

    def push(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
        variant_id: str,
        seed_id: int,
        info: dict[str, object] | None = None,
    ) -> None:
        if len(self) >= self.capacity:
            for store in (self._obs, self._action, self._reward, self._done, self._next_obs, self._variant, self._seed, self._info):
                store.pop(0)
        self._obs.append(obs.detach().cpu())
        self._action.append(action.detach().cpu())
        self._reward.append(reward.detach().cpu())
        self._done.append(done.detach().cpu())
        self._next_obs.append(next_obs.detach().cpu())
        self._variant.append(variant_id)
        self._seed.append(seed_id)
        self._info.append(info or {})

    def sample(self, batch_size: int) -> TransitionBatch:
        if len(self) == 0:
            raise ValueError("ReplayBuffer is empty")
        idx = torch.randint(len(self), (batch_size,), generator=self.generator)
        return TransitionBatch(
            torch.stack([self._obs[int(i)] for i in idx]),
            torch.stack([self._action[int(i)] for i in idx]),
            torch.stack([self._reward[int(i)] for i in idx]),
            torch.stack([self._done[int(i)] for i in idx]),
            torch.stack([self._next_obs[int(i)] for i in idx]),
            tuple(self._variant[int(i)] for i in idx),
            torch.tensor([self._seed[int(i)] for i in idx], dtype=torch.long),
            tuple(self._info[int(i)] for i in idx),
        )

    def sample_sequence(self, batch_size: int, sequence_length: int) -> TransitionBatch:
        if len(self) < sequence_length:
            raise ValueError("ReplayBuffer does not contain enough transitions for a sequence")
        max_start = len(self) - sequence_length + 1
        starts = torch.randint(max_start, (batch_size,), generator=self.generator)
        ranges = [[int(start) + offset for offset in range(sequence_length)] for start in starts]
        return TransitionBatch(
            torch.stack([torch.stack([self._obs[i] for i in row]) for row in ranges]),
            torch.stack([torch.stack([self._action[i] for i in row]) for row in ranges]),
            torch.stack([torch.stack([self._reward[i] for i in row]) for row in ranges]),
            torch.stack([torch.stack([self._done[i] for i in row]) for row in ranges]),
            torch.stack([torch.stack([self._next_obs[i] for i in row]) for row in ranges]),
            tuple(self._variant[row[0]] for row in ranges),
            torch.tensor([self._seed[row[0]] for row in ranges], dtype=torch.long),
            tuple(self._info[row[0]] for row in ranges),
        )


def collect_transitions(env: FlatEnv, policy, steps: int, seed: int = 0, variant: object | None = None) -> TransitionBatch:
    obs = env.reset(seed=seed, variant=variant).reshape(1, -1)
    obs_list: list[torch.Tensor] = []
    action_list: list[torch.Tensor] = []
    reward_list: list[torch.Tensor] = []
    done_list: list[torch.Tensor] = []
    next_obs_list: list[torch.Tensor] = []
    info_list: list[dict[str, object]] = []
    for t in range(steps):
        action = policy.act(obs)
        next_obs, reward, done, info = env.step(action.reshape(-1))
        next_obs = next_obs.reshape(1, -1)
        obs_list.append(obs.squeeze(0))
        action_list.append(action.squeeze(0))
        reward_list.append(torch.as_tensor(reward, dtype=torch.float32).reshape(1))
        done_list.append(torch.as_tensor(done, dtype=torch.float32).reshape(1))
        next_obs_list.append(next_obs.squeeze(0))
        info_list.append(dict(info or {}))
        obs = env.reset(seed + t + 1, variant=variant).reshape(1, -1) if bool(torch.as_tensor(done).item()) else next_obs
    return TransitionBatch(
        obs=torch.stack(obs_list),
        action=torch.stack(action_list),
        reward=torch.stack(reward_list),
        done=torch.stack(done_list),
        next_obs=torch.stack(next_obs_list),
        variant_id=tuple(getattr(env, "variant_id", "default") for _ in obs_list),
        seed_id=torch.full((len(obs_list),), seed, dtype=torch.long),
        info=tuple(info_list),
    )


def train_policy(policy, replay: ReplayBuffer, steps: int = 1, batch_size: int = 32, learning_rate: float = 1e-3) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    for _ in range(steps):
        history.append(policy.update(replay.sample(min(batch_size, len(replay))), learning_rate=learning_rate))
    return history
