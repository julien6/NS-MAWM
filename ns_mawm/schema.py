from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal, Mapping
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class BlockSpec:
    name: str
    owner: int | Literal["shared"]
    kind: Literal["scalar", "categorical"]
    categories: tuple[str, ...] | None
    slice: tuple[int, int]
    unit: str | None = "native"
    scale: tuple[float, float] | None = (0.0, 1.0)
    eval_tol: float | None = 1e-5
    comp_tol: float | None = 0.0
    # Shared blocks occupy an explicitly declared column; they are never duplicated.
    column: int = 0

    @property
    def col(self) -> int:
        return self.owner if isinstance(self.owner, int) else self.column


class Schema:
    def __init__(self, blocks: list[BlockSpec], shape: tuple[int, int]):
        self.blocks, self.shape = tuple(blocks), tuple(shape)
        self.by_name = {b.name: b for b in blocks}
        if len(self.by_name) != len(blocks):
            raise ValueError("Duplicate block names")
        occupied = torch.zeros(shape, dtype=torch.int)
        for b in blocks:
            lo, hi = b.slice
            if not 0 <= lo < hi <= shape[0] or not 0 <= b.col < shape[1]:
                raise ValueError(f"Invalid location: {b.name}")
            if b.kind == "categorical":
                if not b.categories or len(set(b.categories)) != hi - lo or len(b.categories) != hi - lo:
                    raise ValueError(f"Invalid categories: {b.name}")
            elif b.kind != "scalar" or hi - lo != 1 or b.scale is None or not all(math.isfinite(v) for v in b.scale) or b.unit is None or b.scale[1] <= 0 or b.eval_tol is None or b.comp_tol is None or min(b.eval_tol, b.comp_tol) < 0:
                raise ValueError(f"Invalid scalar: {b.name}")
            occupied[lo:hi, b.col] += 1
        if not torch.all(occupied == 1):
            raise ValueError("Schema must cover every encoded entry exactly once")

    def get(self, x: Tensor, b: BlockSpec) -> Tensor:
        return x[..., b.slice[0]:b.slice[1], b.col]

    def validate_value(self, name: str, value: float | str) -> None:
        b = self.by_name[name]
        if b.kind == "categorical":
            if value not in b.categories:
                raise ValueError(f"Illegal category for {name}: {value}")
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Nonfinite/non-scalar assignment: {name}")

    def encode(self, values: Mapping[str, float | str], *, partial: bool = False) -> Tensor:
        if not partial and set(values) != set(self.by_name):
            raise ValueError("Observation is incomplete")
        out = torch.zeros(self.shape)
        rows, columns, entries = [], [], []
        for name, v in values.items():
            self.validate_value(name, v)
            b = self.by_name[name]
            rows.append(b.slice[0] + b.categories.index(v) if b.kind == "categorical" else b.slice[0])
            columns.append(b.col)
            entries.append(1.0 if b.kind == "categorical" else float(v))
        if rows:
            out[rows, columns] = torch.tensor(entries)

        return out

    def decode(self, x: Tensor) -> dict[str, float | str]:
        if tuple(x.shape) != self.shape or not torch.isfinite(x).all():
            raise ValueError("Invalid observation tensor")
        result = {}
        for b in self.blocks:
            v = self.get(x, b)
            if b.kind == "categorical":
                if v.sum() <= 0:
                    raise ValueError("Cannot decode categorical padding")
                result[b.name] = b.categories[int(v.argmax())]
            else:
                result[b.name] = float(v.item())
        return result

    def probabilities(self, logits: Tensor) -> Tensor:
        out = logits.clone()
        for b in self.blocks:
            if b.kind == "categorical":
                out[..., b.slice[0]:b.slice[1], b.col] = self.get(logits, b).softmax(-1)
        return out

    def losses(self, output: Tensor, target: Tensor, *, training: bool) -> Tensor:
        losses = []
        for b in self.blocks:
            p, y = self.get(output, b), self.get(target, b)
            if b.kind == "categorical":
                loss = -(y * p.log_softmax(-1)).sum(-1) if training else (p - y).square().sum(-1) / 2
            else:
                loss = ((p - y) / b.scale[1]).square().squeeze(-1)
            losses.append(loss)
        return torch.stack(losses, -1)

    def block_mask(self, mask: Tensor) -> Tensor:
        return torch.stack([self.get(mask, b).amin(-1) for b in self.blocks], -1)

    def fit(self, train_observations: Tensor) -> Schema:
        blocks = []
        for b in self.blocks:
            if b.kind == "scalar":
                v = self.get(train_observations, b).float()
                b = replace(b, scale=(float(v.mean()), (float(v.std(unbiased=False)) if float(v.std(unbiased=False)) > 1e-6 else 1.0)))
            blocks.append(b)
        return Schema(blocks, self.shape)

    def to_dict(self) -> dict:
        return {"shape": self.shape, "blocks": [asdict(b) for b in self.blocks]}

    @classmethod
    def from_dict(cls, obj: dict) -> Schema:
        blocks = []
        for record in obj["blocks"]:
            r = dict(record)
            for key in ("slice", "scale", "categories"):
                if r.get(key) is not None:
                    r[key] = tuple(r[key])
            blocks.append(BlockSpec(**r))
        return cls(blocks, tuple(obj["shape"]))
