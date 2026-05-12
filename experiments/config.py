"""Experiment configuration and loading."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "smoke"
    baseline_id: str = "B20"
    environment: str = "predator_prey"
    world_model: str = "rssm"
    policy: str = "MAPPO"
    strategy: str = "regularization"
    coverage: float = 0.3
    train_regime: str = "MV"
    eval_regime: str = "UV"
    seed: int | None = None
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    smoke_steps: int = 12
    full_steps: int = 2048
    update_budget: int | None = None
    train_updates: int = 3
    full_train_updates: int = 1000
    batch_size: int = 8
    sequence_length: int = 4
    wm_learning_rate: float = 1e-3
    policy_learning_rate: float = 1e-3
    kl_free_nats: float = 0.0
    kl_balance: float = 0.5
    horizons: tuple[int, ...] = (10, 25, 50)
    smoke_horizons: tuple[int, ...] = (3, 5)
    planner_horizon: int = 10
    planner_candidates: int = 256
    rule_dropout_rates: tuple[float, ...] = (0.25, 0.5, 0.75)
    log_resources: bool = True
    strict_reproduction: bool = False
    comparison_arm: str | None = None
    reference_coverage: float | None = None
    paired_baseline_id: str | None = None
    output_dir: str = "runs"

    @property
    def is_smoke(self) -> bool:
        return self.mode == "smoke"

    @property
    def active_seeds(self) -> tuple[int, ...]:
        if self.seed is not None:
            return (self.seed,)
        return self.seeds[:1] if self.is_smoke else self.seeds

    @property
    def active_steps(self) -> int:
        return self.smoke_steps if self.is_smoke else self.full_steps

    @property
    def active_train_updates(self) -> int:
        if self.update_budget is not None:
            return self.update_budget
        return self.train_updates if self.is_smoke else self.full_train_updates

    @property
    def active_horizons(self) -> tuple[int, ...]:
        return self.smoke_horizons if self.is_smoke else self.horizons

    def validate(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be 'smoke' or 'full'")
        if self.train_regime not in {"SV", "MV"}:
            raise ValueError("train_regime must be 'SV' or 'MV'")
        if self.eval_regime not in {"SV", "KV", "UV"}:
            raise ValueError("eval_regime must be 'SV', 'KV', or 'UV'")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        if self.reference_coverage is not None and not 0.0 <= self.reference_coverage <= 1.0:
            raise ValueError("reference_coverage must be in [0, 1]")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.wm_learning_rate <= 0 or self.policy_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.kl_free_nats < 0:
            raise ValueError("kl_free_nats must be non-negative")
        if not 0.0 <= self.kl_balance <= 1.0:
            raise ValueError("kl_balance must be in [0, 1]")
        if self.mode == "full" and len(self.active_seeds) < 5:
            raise ValueError("full mode requires at least 5 active seeds")


def load_config(path: str | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    data = OmegaConf.structured(ExperimentConfig)
    if path:
        data = OmegaConf.merge(data, OmegaConf.load(Path(path)))
    if overrides:
        data = OmegaConf.merge(data, OmegaConf.from_dotlist(overrides))
    obj = OmegaConf.to_object(data)
    assert isinstance(obj, ExperimentConfig)
    obj.validate()
    return obj


def smoke_config(config: ExperimentConfig) -> ExperimentConfig:
    return replace(config, mode="smoke")
