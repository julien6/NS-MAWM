"""Launch one baseline, one family, or the full B01-B45 smoke/full matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from experiments.baseline_configs import all_baseline_configs, configs_for_family
from experiments.config import ExperimentConfig
from experiments.registry import BASELINES
from experiments.reproduction_check import validate_reproduction_logs


STAGE_MODULES = {
    "train": "experiments.train_world_model",
    "rollout": "experiments.eval_rollout",
    "planning": "experiments.eval_planning",
    "rule_dropout": "experiments.eval_rule_dropout",
    "noisy_rules": "experiments.eval_noisy_rules",
    "ood": "experiments.eval_ood",
}

WM_ONLY_STAGES = {"rollout", "planning", "rule_dropout", "noisy_rules", "ood"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _stage_log_name(stage: str) -> str:
    return {
        "train": "train_world_model.csv",
        "rollout": "eval_rollout.csv",
        "planning": "eval_planning.csv",
        "rule_dropout": "eval_rule_dropout.csv",
        "noisy_rules": "eval_noisy_rules.csv",
        "ood": "eval_rollout.csv",
    }[stage]


def _stage_complete(config: ExperimentConfig, stage: str) -> bool:
    frame = _read_csv(Path(config.output_dir) / _stage_log_name(stage))
    if frame.empty or "baseline_id" not in frame.columns:
        return False
    rows = frame[frame["baseline_id"].astype(str) == config.baseline_id]
    if rows.empty or "seed" not in rows.columns:
        return False
    if rows["seed"].nunique() < len(config.active_seeds):
        return False
    if stage in {"rollout", "ood"} and config.world_model != "none":
        if "horizon" not in rows.columns:
            return False
        observed = {int(h) for h in rows["horizon"].dropna().unique()}
        return set(config.active_horizons).issubset(observed)
    return True


def executable_stages(config: ExperimentConfig, stages: list[str]) -> list[str]:
    """Return stages that make sense for this baseline.

    Model-free baselines skip WM-only stages because their metrics come from
    training/control logs rather than open-loop WM rollouts.
    """

    filtered: list[str] = []
    for stage in stages:
        if stage not in STAGE_MODULES:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(STAGE_MODULES)}")
        if config.world_model == "none" and stage in WM_ONLY_STAGES:
            continue
        filtered.append(stage)
    return filtered


def _write_temp_config(config: ExperimentConfig, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{config.baseline_id}_{config.mode}.yaml"
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    return path


def _run_module(module: str, config_path: Path) -> None:
    subprocess.run([sys.executable, "-m", module, "--config", str(config_path)], check=True)


def select_configs(args: argparse.Namespace) -> dict[str, ExperimentConfig]:
    base = ExperimentConfig(mode=args.mode, output_dir=args.output_dir)
    configs = all_baseline_configs(base)
    if args.baseline_id:
        if args.baseline_id not in configs:
            raise ValueError(f"Unknown baseline id {args.baseline_id!r}")
        return {args.baseline_id: configs[args.baseline_id]}
    if args.family:
        selected = configs_for_family(args.family, base)
        if not selected:
            raise ValueError(f"No baselines matched family {args.family!r}")
        return selected
    return configs


def write_manifest(
    output_dir: str | Path,
    *,
    configs: dict[str, ExperimentConfig],
    requested_stages: list[str],
    dry_run: bool,
) -> Path:
    path = Path(output_dir) / "launch_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": next(iter(configs.values())).mode if configs else None,
        "dry_run": dry_run,
        "requested_stages": requested_stages,
        "baselines": {
            baseline_id: {
                **asdict(config),
                "registry_family": BASELINES[baseline_id].family,
                "executable_stages": executable_stages(config, requested_stages),
            }
            for baseline_id, config in configs.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--baseline-id")
    target.add_argument("--family")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--stages", nargs="+", default=["train", "rollout"])
    parser.add_argument("--continue-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-after", action="store_true")
    parser.add_argument("--required-seeds", type=int, default=None)
    parser.add_argument("--required-horizons", default=None)
    args = parser.parse_args()

    selected = select_configs(args)
    manifest = write_manifest(args.output_dir, configs=selected, requested_stages=args.stages, dry_run=args.dry_run)
    print(f"wrote launch manifest to {manifest}")
    temp_root = Path(args.output_dir) / "_run_configs"
    for config in selected.values():
        config_path = _write_temp_config(config, temp_root)
        for stage in executable_stages(config, args.stages):
            if args.continue_existing and _stage_complete(config, stage):
                print(f"skip complete {config.baseline_id}:{stage}")
                continue
            print(f"run {config.baseline_id}:{stage}")
            if not args.dry_run:
                _run_module(STAGE_MODULES[stage], config_path)

    if args.validate_after and not args.dry_run:
        required_horizons = (
            tuple(int(item) for item in args.required_horizons.split(",") if item)
            if args.required_horizons
            else next(iter(selected.values())).active_horizons
        )
        required_seeds = args.required_seeds if args.required_seeds is not None else len(next(iter(selected.values())).active_seeds)
        result = validate_reproduction_logs(
            args.output_dir,
            baselines=tuple(selected),
            required_seeds=required_seeds,
            required_horizons=required_horizons,
        )
        if not result.ok:
            raise SystemExit(
                "Launch validation failed: "
                f"missing baselines={list(result.missing_baselines)}; "
                f"missing metrics={list(result.missing_metrics)}; "
                f"insufficient seeds={result.insufficient_seeds}; "
                f"missing horizons={result.missing_horizons}"
            )
        print(f"validated {len(selected)} baselines in {args.output_dir}")


if __name__ == "__main__":
    main()
