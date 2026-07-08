from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STAGE_ALIASES = {
    "quick": "screen",
    "standard": "promote",
    "serious": "final",
    "screen": "screen",
    "promote": "promote",
    "final": "final",
    "auto": "auto",
}


def normalize_stage(value: str | None, fallback: str = "screen") -> str:
    text = str(value or fallback).strip().lower()
    if text not in STAGE_ALIASES:
        raise ValueError(f"unsupported HPO stage {value!r}; expected one of {', '.join(sorted(STAGE_ALIASES))}")
    return STAGE_ALIASES[text]


def auto_stage(results_root: str | Path, family: str) -> str:
    family_dir = Path(results_root) / family
    if not (family_dir / "screen_results.json").exists():
        return "screen"
    if not (family_dir / "promoted_configs.json").exists():
        return "promote"
    return "final"


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open() as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def fixed_config_env(config: dict[str, Any]) -> str:
    return json.dumps(config.get("hyperparameters", config), sort_keys=True)
