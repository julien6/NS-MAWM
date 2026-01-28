"""Feature schema and mask utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str  # continuous | categorical | binary
    dim: int = 1
    determinable: bool = False


class FeatureSchema:
    """Ordered feature schema with index mapping."""

    def __init__(self, specs: Iterable[FeatureSpec]):
        self._specs: List[FeatureSpec] = list(specs)
        self._indices = {spec.name: idx for idx, spec in enumerate(self._specs)}

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)

    def index(self, name: str) -> int:
        return self._indices[name]

    @property
    def determinable_mask(self) -> torch.Tensor:
        """Return a 1D mask for determinable features."""
        return torch.tensor([spec.determinable for spec in self._specs], dtype=torch.bool)

    def broadcast_mask(self, batch_shape: torch.Size) -> torch.Tensor:
        """Broadcast determinable mask to [*batch_shape, n_features]."""
        base = self.determinable_mask
        view_shape = (1,) * len(batch_shape) + (len(self._specs),)
        return base.view(view_shape).expand(*batch_shape, len(self._specs))
