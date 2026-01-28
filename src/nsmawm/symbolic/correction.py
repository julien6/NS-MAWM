"""Symbolic post-correction operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class Correction(Protocol):
    def __call__(self, prediction: torch.Tensor) -> torch.Tensor:  # pragma: no cover - protocol
        ...


@dataclass
class IdentityCorrection:
    def __call__(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction


@dataclass
class DiscretizeCorrection:
    """Simple discretization correction for categorical/binary features."""

    threshold: float = 0.5

    def __call__(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.where(prediction >= self.threshold, torch.ones_like(prediction), torch.zeros_like(prediction))
