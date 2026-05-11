"""Flat tensor adapter protocol and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from ns_mawm.features import FeatureSchema, FeatureSpec, FeatureType


@dataclass(frozen=True)
class FlatStep:
    obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    info: dict[str, Any]


@dataclass(frozen=True)
class ActionSpec:
    n_agents: int
    action_dim: int
    names: tuple[str, ...] = ()

    @property
    def flat_dim(self) -> int:
        return self.n_agents * self.action_dim


class EnvironmentAdapter(Protocol):
    n_agents: int
    action_dim: int
    obs_dim: int
    variant_id: str

    @property
    def schema(self) -> FeatureSchema: ...
    @property
    def action_spec(self) -> ActionSpec: ...
    def reset(self, seed: int | None = None) -> torch.Tensor: ...
    def step(self, action: torch.Tensor): ...
    def decode(self, obs: torch.Tensor) -> dict[str, object]: ...
    def encode_action(self, actions: Any) -> torch.Tensor: ...


def flatten_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        chunks = [flatten_value(value[key]) for key in sorted(value)]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    return arr.reshape(-1)


def flatten_multiagent_obs(obs: dict[str, Any], agent_order: list[str]) -> torch.Tensor:
    chunks = [flatten_value(obs[agent]) for agent in agent_order if agent in obs]
    return torch.from_numpy(np.concatenate(chunks).astype(np.float32))


def generic_schema(width: int, prefix: str = "feature", owner: str | None = None, family: str = "observation") -> FeatureSchema:
    return FeatureSchema.from_specs(
        FeatureSpec(f"{prefix}_{i}", FeatureType.CONTINUOUS, owner=owner, family=family)
        for i in range(width)
    )
