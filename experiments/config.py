"""Experiment configuration and loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ExperimentConfig:
    baseline_id: str = "B20"
    environment: str = "predator_prey"
    world_model: str = "rssm"
    policy: str = "MAPPO"
    strategy: str = "regularization"
    coverage: float = 0.3
    train_regime: str = "MV"
    eval_regime: str = "UV"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    smoke_steps: int = 12
    train_updates: int = 3
    batch_size: int = 8
    horizons: tuple[int, ...] = (10, 25, 50)
    output_dir: str = "runs"


def load_config(path: str | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    data = OmegaConf.structured(ExperimentConfig)
    if path:
        data = OmegaConf.merge(data, OmegaConf.load(Path(path)))
    if overrides:
        data = OmegaConf.merge(data, OmegaConf.from_dotlist(overrides))
    obj = OmegaConf.to_object(data)
    assert isinstance(obj, ExperimentConfig)
    return obj
