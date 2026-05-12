"""Report B01-B45 run completeness from seed-level logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.baseline_configs import all_baseline_configs
from experiments.config import ExperimentConfig
from experiments.launch import executable_stages
from experiments.registry import BASELINES


STAGE_LOGS = {
    "train": "train_world_model.csv",
    "rollout": "eval_rollout.csv",
    "planning": "eval_planning.csv",
    "rule_dropout": "eval_rule_dropout.csv",
    "noisy_rules": "eval_noisy_rules.csv",
    "ood": "eval_rollout.csv",
}


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def stage_status(
    runs: Path,
    config: ExperimentConfig,
    stage: str,
    *,
    required_seeds: int,
    required_horizons: tuple[int, ...],
) -> tuple[bool, str]:
    frame = _load(runs / STAGE_LOGS[stage])
    if frame.empty:
        return False, "missing log"
    if "baseline_id" not in frame.columns:
        return False, "missing baseline_id column"
    rows = frame[frame["baseline_id"].astype(str) == config.baseline_id]
    if rows.empty:
        return False, "missing baseline rows"
    if "seed" not in rows.columns:
        return False, "missing seed column"
    seed_count = int(rows["seed"].nunique())
    if seed_count < required_seeds:
        return False, f"seeds {seed_count}/{required_seeds}"
    if stage in {"rollout", "ood"} and config.world_model != "none":
        if "horizon" not in rows.columns:
            return False, "missing horizon column"
        observed = {int(h) for h in rows["horizon"].dropna().unique()}
        missing = tuple(sorted(set(required_horizons) - observed))
        if missing:
            return False, f"missing horizons {missing}"
    return True, "ok"


def build_status_table(
    runs: str | Path,
    *,
    mode: str,
    stages: list[str],
    baselines: tuple[str, ...] | None = None,
    required_seeds: int | None = None,
    required_horizons: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    runs_path = Path(runs)
    base = ExperimentConfig(mode=mode, output_dir=str(runs_path))
    configs = all_baseline_configs(base)
    selected = baselines or tuple(configs)
    required_seed_count = required_seeds if required_seeds is not None else len(base.active_seeds)
    required_horizon_set = required_horizons or base.active_horizons
    rows: list[dict[str, object]] = []
    for baseline_id in selected:
        config = configs[baseline_id]
        needed_stages = executable_stages(config, stages)
        stage_results = {
            stage: stage_status(
                runs_path,
                config,
                stage,
                required_seeds=required_seed_count,
                required_horizons=required_horizon_set,
            )
            for stage in needed_stages
        }
        complete = all(ok for ok, _reason in stage_results.values())
        row: dict[str, object] = {
            "baseline_id": baseline_id,
            "family": BASELINES[baseline_id].family,
            "environment": BASELINES[baseline_id].environment,
            "wm": BASELINES[baseline_id].wm,
            "policy": BASELINES[baseline_id].policy,
            "complete": complete,
        }
        for stage in stages:
            if stage in stage_results:
                ok, reason = stage_results[stage]
                row[stage] = "ok" if ok else reason
            else:
                row[stage] = "skipped"
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--stages", nargs="+", default=["train", "rollout", "planning"])
    parser.add_argument("--baselines", default="")
    parser.add_argument("--required-seeds", type=int, default=None)
    parser.add_argument("--required-horizons", default="")
    parser.add_argument("--fail-incomplete", action="store_true")
    args = parser.parse_args()

    horizons = tuple(int(item) for item in args.required_horizons.split(",") if item.strip()) or None
    table = build_status_table(
        args.runs,
        mode=args.mode,
        stages=args.stages,
        baselines=_parse_tuple(args.baselines) or None,
        required_seeds=args.required_seeds,
        required_horizons=horizons,
    )
    print(table.to_string(index=False))
    if args.fail_incomplete and not bool(table["complete"].all()):
        raise SystemExit("matrix status incomplete")


if __name__ == "__main__":
    main()
