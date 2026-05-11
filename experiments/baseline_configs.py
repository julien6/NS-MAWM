"""Generated experiment configs for the B01-B45 reproduction matrix."""

from __future__ import annotations

from dataclasses import replace

from experiments.config import ExperimentConfig
from experiments.registry import BASELINES, BaselineSpec


ENVIRONMENT_NAMES = {
    "GridCraft": "gridcraft",
    "Overcooked": "overcooked",
    "PredatorPrey": "predator_prey",
    "SMACv2": "smacv2",
}

WORLD_MODEL_NAMES = {
    "MF": "none",
    "RSSM": "rssm",
    "Deterministic": "deterministic",
    "Transformer": "transformer",
}


def config_for_baseline(spec: BaselineSpec, base: ExperimentConfig | None = None) -> ExperimentConfig:
    cfg = base or ExperimentConfig()
    return replace(
        cfg,
        baseline_id=spec.baseline_id,
        environment=ENVIRONMENT_NAMES[spec.environment],
        world_model=WORLD_MODEL_NAMES[spec.wm],
        policy=spec.policy,
        strategy="none" if spec.strategy == "-" else spec.strategy,
        coverage=spec.coverage,
        train_regime=spec.train_regime,
        eval_regime=spec.eval_regime,
    )


def all_baseline_configs(base: ExperimentConfig | None = None) -> dict[str, ExperimentConfig]:
    return {baseline_id: config_for_baseline(spec, base) for baseline_id, spec in BASELINES.items()}
