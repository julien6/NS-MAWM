"""Build baseline summary tables from seed-level logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.logging import aggregate_mean_sem
from experiments.registry import BASELINES
from experiments.reproduction_check import DEFAULT_TABLE_METRICS, validate_reproduction_logs


TABLE_METRICS = list(DEFAULT_TABLE_METRICS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--baselines", default="")
    parser.add_argument("--required-seeds", type=int, default=5)
    parser.add_argument("--required-horizons", default="10,25,50")
    parser.add_argument("--required-metrics", default=",".join(TABLE_METRICS))
    args = parser.parse_args()
    runs = Path(args.runs)
    selected_baselines = tuple(x.strip() for x in args.baselines.split(",") if x.strip()) or tuple(BASELINES)
    required_horizons = tuple(int(x) for x in args.required_horizons.split(",") if x)
    required_metrics = tuple(x.strip() for x in args.required_metrics.split(",") if x.strip())
    if args.strict:
        result = validate_reproduction_logs(
            runs,
            baselines=selected_baselines,
            required_seeds=args.required_seeds,
            required_horizons=required_horizons,
            required_metrics=required_metrics,
        )
        if not result.ok:
            raise SystemExit(
                f"Missing baselines={list(result.missing_baselines)}; "
                f"missing metrics={list(result.missing_metrics)}; "
                f"insufficient seeds={result.insufficient_seeds}; "
                f"missing horizons={result.missing_horizons}"
            )
    frames = []
    for name in ("eval_rollout.csv", "train_world_model.csv"):
        path = runs / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"No seed-level logs found in {runs}")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    missing_baselines = sorted(set(selected_baselines) - set(frame.get("baseline_id", [])))
    available_metrics = [metric for metric in TABLE_METRICS if metric in frame.columns]
    missing_seeds: dict[str, int] = {}
    if "seed" in frame.columns and "baseline_id" in frame.columns:
        counts = frame[frame["baseline_id"].astype(str).isin(selected_baselines)].groupby("baseline_id")["seed"].nunique()
        missing_seeds = {str(k): int(v) for k, v in counts.items() if int(v) < args.required_seeds}
    required_horizons = set(required_horizons)
    missing_horizons: dict[str, list[int]] = {}
    if "horizon" in frame.columns and "baseline_id" in frame.columns:
        for baseline_id, group in frame.dropna(subset=["horizon"]).groupby("baseline_id"):
            if str(baseline_id) not in selected_baselines:
                continue
            observed = {int(h) for h in group["horizon"].unique()}
            missing = sorted(required_horizons - observed)
            if missing:
                missing_horizons[str(baseline_id)] = missing
    if not available_metrics:
        raise SystemExit("Logs do not contain any Table 4 metrics")
    combined_path = runs / "_table4_combined.csv"
    frame.to_csv(combined_path, index=False)
    table = aggregate_mean_sem(str(combined_path), ["baseline_id"], available_metrics)
    print(table.to_string(index=False))
    if missing_baselines:
        print(f"# Missing baselines in current logs: {', '.join(missing_baselines[:10])}{'...' if len(missing_baselines) > 10 else ''}")
    if missing_seeds:
        print(f"# Baselines with fewer than {args.required_seeds} seeds: {missing_seeds}")
    if missing_horizons:
        print(f"# Missing required horizons: {missing_horizons}")


if __name__ == "__main__":
    main()
