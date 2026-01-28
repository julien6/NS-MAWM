"""Datasets for NS-MAWM."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset


class TransitionsDataset(Dataset):
    def __init__(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor):
        self.obs = obs
        self.act = act
        self.next_obs = next_obs

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "next_obs": self.next_obs[idx],
        }
