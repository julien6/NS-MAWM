"""Replay buffer for offline transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Transition:
    obs: torch.Tensor
    act: torch.Tensor
    next_obs: torch.Tensor
    done: Optional[torch.Tensor] = None


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._data: List[Transition] = []

    def push(self, transition: Transition) -> None:
        if len(self._data) >= self.capacity:
            self._data.pop(0)
        self._data.append(transition)

    def __len__(self) -> int:
        return len(self._data)

    def sample(self) -> Transition:
        if not self._data:
            raise ValueError("ReplayBuffer is empty")
        return self._data[-1]

    def as_tensors(self) -> Transition:
        obs = torch.stack([t.obs for t in self._data])
        act = torch.stack([t.act for t in self._data])
        next_obs = torch.stack([t.next_obs for t in self._data])
        done = None
        if self._data[0].done is not None:
            done = torch.stack([t.done for t in self._data])
        return Transition(obs=obs, act=act, next_obs=next_obs, done=done)
