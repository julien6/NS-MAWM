"""Environment-agnostic structured feature descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import torch


class FeatureType(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    INTEGER = "integer"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    feature_type: FeatureType
    width: int = 1
    owner: str | None = None
    family: str = "default"
    tolerance: float = 1e-3
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.feature_type == FeatureType.CATEGORICAL and self.categories:
            object.__setattr__(self, "width", len(self.categories))


@dataclass(frozen=True)
class FeatureSchema:
    specs: tuple[FeatureSpec, ...]

    @classmethod
    def from_specs(cls, specs: Iterable[FeatureSpec]) -> "FeatureSchema":
        specs_t = tuple(specs)
        names = [s.name for s in specs_t]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique")
        return cls(specs_t)

    @property
    def width(self) -> int:
        return sum(spec.width for spec in self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def owners(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(spec.owner for spec in self.specs if spec.owner is not None))

    def slice(self, name: str) -> slice:
        start = 0
        for spec in self.specs:
            end = start + spec.width
            if spec.name == name:
                return slice(start, end)
            start = end
        raise KeyError(name)

    def spec(self, name: str) -> FeatureSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def mask(self, names: Iterable[str], *, device: torch.device | None = None) -> torch.Tensor:
        out = torch.zeros(self.width, dtype=torch.bool, device=device)
        for name in names:
            out[self.slice(name)] = True
        return out

    def inverse_mask(self, names: Iterable[str], *, device: torch.device | None = None) -> torch.Tensor:
        return ~self.mask(names, device=device)

    def indices(self, names: Iterable[str], *, device: torch.device | None = None) -> torch.Tensor:
        mask = self.mask(names, device=device)
        return mask.nonzero(as_tuple=False).reshape(-1)

    def select(self, tensor: torch.Tensor, names: Iterable[str]) -> torch.Tensor:
        return tensor[..., self.indices(names, device=tensor.device)]

    def insert(
        self,
        base: torch.Tensor,
        values: torch.Tensor,
        names: Iterable[str],
    ) -> torch.Tensor:
        out = base.clone()
        out[..., self.indices(names, device=base.device)] = values
        return out

    def zeros(self, *batch_shape: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros((*batch_shape, self.width), dtype=torch.float32, device=device)

    def validate(self, tensor: torch.Tensor) -> None:
        if tensor.shape[-1] != self.width:
            raise ValueError(f"expected final dim {self.width}, got {tensor.shape[-1]}")

    def family_summary(self, mask: torch.Tensor) -> dict[str, float]:
        covered: dict[str, float] = {}
        totals: dict[str, float] = {}
        flat_mask = mask.reshape(-1, mask.shape[-1]).any(dim=0) if mask.ndim > 1 else mask
        for spec in self.specs:
            sl = self.slice(spec.name)
            covered.setdefault(spec.family, 0.0)
            totals.setdefault(spec.family, 0.0)
            covered[spec.family] += float(flat_mask[sl].float().sum().item())
            totals[spec.family] += float(spec.width)
        return {family: covered[family] / totals[family] for family in totals}

    def coverage(self, mask: torch.Tensor) -> float:
        if mask.ndim > 1:
            mask = mask.reshape(-1, mask.shape[-1]).any(dim=0)
        return float(mask.float().mean().item()) if mask.numel() else 0.0

    def encode_mapping(self, values: Mapping[str, object], *, device: torch.device | None = None) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for spec in self.specs:
            raw = values[spec.name]
            if spec.feature_type == FeatureType.CATEGORICAL:
                idx = spec.categories.index(str(raw)) if isinstance(raw, str) else int(raw)
                chunk = torch.zeros(spec.width, device=device)
                chunk[idx] = 1.0
            else:
                chunk = torch.as_tensor(raw, dtype=torch.float32, device=device).reshape(-1)
            if chunk.numel() != spec.width:
                raise ValueError(f"{spec.name} expected width {spec.width}, got {chunk.numel()}")
            chunks.append(chunk)
        return torch.cat(chunks, dim=-1)

    def decode_tensor(self, tensor: torch.Tensor) -> dict[str, object]:
        self.validate(tensor)
        decoded: dict[str, object] = {}
        for spec in self.specs:
            value = tensor[..., self.slice(spec.name)]
            if spec.feature_type == FeatureType.CATEGORICAL:
                idx = int(value.argmax(dim=-1).reshape(-1)[0].item())
                decoded[spec.name] = spec.categories[idx] if spec.categories else idx
            elif spec.feature_type == FeatureType.BINARY:
                decoded[spec.name] = (value > 0.5).float()
            elif spec.feature_type == FeatureType.INTEGER:
                decoded[spec.name] = value.round().long()
            else:
                decoded[spec.name] = value
        return decoded


@dataclass(frozen=True)
class FeatureSelector:
    schema: FeatureSchema
    names: tuple[str, ...]

    @property
    def mask(self) -> torch.Tensor:
        return self.schema.mask(self.names)

    @property
    def indices(self) -> torch.Tensor:
        return self.schema.indices(self.names)

    @property
    def width(self) -> int:
        return int(self.mask.sum().item())
