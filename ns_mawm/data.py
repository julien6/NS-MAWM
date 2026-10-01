from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from .rules import digest


@dataclass
class Episode:
    id: str
    observations: torch.Tensor  # time+1, features, agents; native units
    actions: torch.Tensor      # time, agents
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    seed: int
    policy: str = "random"

    def validate(self):
        length = len(self.actions)
        if len(self.observations) != length + 1 or any(len(x) != length for x in (self.rewards, self.terminated, self.truncated)):
            raise ValueError("Episode transition lengths disagree")
        if not torch.isfinite(self.observations).all() or not torch.isfinite(self.rewards).all():
            raise ValueError("Dataset contains nonfinite values")
        if self.actions.ndim != 2 or self.observations.ndim != 3 or self.actions.shape[1] != self.observations.shape[2]:
            raise ValueError("Malformed joint observations/actions")
        if (self.terminated[:-1] | self.truncated[:-1]).any():
            raise ValueError("Episode continues after termination/truncation")


class Dataset:
    def __init__(self, episodes, manifest, frozen=None):
        self._episodes = {e.id: e for e in episodes}
        if len(self._episodes) != len(episodes):
            raise ValueError("Duplicate episode ids")
        ids = [eid for part in manifest["splits"].values() for eid in part]
        if len(ids) != len(set(ids)) or set(ids) != set(self._episodes):
            raise ValueError("Splits must partition complete episodes")
        for e in episodes:
            e.validate()
        claimed = manifest.get("hash")
        self.manifest = {k: v for k, v in manifest.items() if k != "hash"}
        self.hash = digest(self.manifest)
        if claimed is not None and claimed != self.hash:
            raise ValueError("Split manifest hash mismatch")
        self.frozen = frozen

    @classmethod
    def split(cls, episodes, seed=0, fractions=(.6, .15, .1, .15), metadata=None):
        if len(episodes) < 4 or len(fractions) != 4 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
            raise ValueError("Four nonempty episode splits are required")
        ids = [e.id for e in episodes]
        np.random.default_rng(seed).shuffle(ids)
        desired = np.array(fractions) * len(ids)
        counts = np.maximum(1, desired.astype(int))
        while counts.sum() < len(ids):
            counts[int(np.argmax(desired-counts))] += 1
        while counts.sum() > len(ids):
            eligible = np.where(counts > 1, counts-desired, -np.inf)
            counts[int(np.argmax(eligible))] -= 1
        parts, offset = {}, 0
        for name, count in zip(("train", "rev", "select", "test"), counts):
            parts[name] = ids[offset:offset + count]
            offset += count
        return cls(episodes, {"splits": parts, "seed": seed, "metadata": metadata or {}})

    def get(self, split, *, purpose="train"):
        allowed = {"train": {"train"}, "prompt": {"rev"}, "revision": {"rev"},
                   "selection": {"select"}, "evaluate": {"rev", "select", "test"}}
        if split not in allowed.get(purpose, set()):
            raise PermissionError(f"{purpose} cannot access {split}")
        if split == "test" and self.frozen is None:
            raise PermissionError("Freeze the library and all hyperparameters before test access")
        return [self._episodes[eid] for eid in self.manifest["splits"][split]]

    def freeze(self, library_hash, configuration):
        token = {"library_hash": library_hash, "configuration_hash": digest(configuration)}
        if self.frozen is not None and self.frozen != token:
            raise PermissionError("Test protocol is already frozen")
        self.frozen = token
        return token

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for episode in self._episodes.values():
            destination = path / (digest(episode.id) + ".pt")
            if not destination.exists():
                torch.save(vars(episode), destination)
            else:
                existing = torch.load(destination, weights_only=True)
                if any(not torch.equal(existing[k], v) if isinstance(v, torch.Tensor) else existing[k] != v for k,v in vars(episode).items()):
                    raise ValueError("Existing episode differs; choose a new dataset directory")
        manifest = {**self.manifest, "hash": self.hash}
        (path / "splits.json").write_text(json.dumps(manifest, indent=2))
        if self.frozen is not None:
            (path / "frozen.json").write_text(json.dumps(self.frozen, indent=2))

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        manifest = json.loads((path / "splits.json").read_text())
        episodes = [Episode(**torch.load(path / (digest(eid) + ".pt"), weights_only=True))
                    for ids in manifest["splits"].values() for eid in ids]
        frozen = json.loads((path / "frozen.json").read_text()) if (path / "frozen.json").exists() else None
        return cls(episodes, manifest, frozen)

    def summary(self):
        return {part: {"episodes": len(ids), "transitions": sum(len(self._episodes[i].actions) for i in ids),
                       "lengths": [len(self._episodes[i].actions) for i in ids],
                       "policies": {p: sum(self._episodes[i].policy == p for i in ids) / len(ids)
                                    for p in sorted({self._episodes[i].policy for i in ids})}}
                for part, ids in self.manifest["splits"].items()}
