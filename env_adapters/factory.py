"""Environment adapter factory."""

from __future__ import annotations

from env_adapters.gridcraft_adapter import GridCraftAdapter
from env_adapters.overcooked_adapter import OvercookedAdapter
from env_adapters.predator_prey_adapter import PredatorPreyAdapter
from env_adapters.smac_adapter import SMACAdapter


def make_environment(name: str, **kwargs):
    key = name.lower().replace("-", "_")
    if key in {"gridcraft", "grid_craft"}:
        return GridCraftAdapter(**kwargs)
    if key in {"overcooked", "overcooked_ai"}:
        return OvercookedAdapter(**kwargs)
    if key in {"predator_prey", "predatorprey"}:
        return PredatorPreyAdapter(**kwargs)
    if key in {"smac", "smacv2"}:
        return SMACAdapter(**kwargs)
    raise ValueError(f"Unknown environment: {name}")
