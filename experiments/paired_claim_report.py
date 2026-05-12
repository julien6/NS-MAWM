"""Summarize paired baseline comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.reproduction_check import validate_paired_comparison


def _sem(series: pd.Series) -> float:
    return float(series.sem()) if len(series) > 1 else 0.0


def _parse_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_report(runs: str | Path, paired_baseline: str, comparison_baselines: tuple[str, ...], arms: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(Path(runs) / "eval_rollout.csv")
    base = frame[frame["baseline_id"].astype(str) == paired_baseline]
    comparison = frame[frame["baseline_id"].astype(str).isin(comparison_baselines)]
    rows: list[dict[str, object]] = []
    for arm in arms:
        arm_rows = comparison[comparison["comparison_arm"].fillna("").astype(str) == arm]
        arm_baselines = ",".join(sorted(set(arm_rows["baseline_id"].astype(str)))) if not arm_rows.empty else ""
        merged = base.merge(arm_rows, on=["seed", "horizon"], suffixes=("_base", "_arm"))
        for horizon, group in merged.groupby("horizon"):
            rvr_delta = group["covered_rvr_arm"] - group["covered_rvr_base"]
            mse_delta = group["obs_loss_arm"] - group["obs_loss_base"]
            rows.append(
                {
                    "arm": arm,
                    "baseline": arm_baselines,
                    "horizon": int(horizon),
                    "paired_n": int(len(group)),
                    "base_rvr_mean": float(group["covered_rvr_base"].mean()),
                    "arm_rvr_mean": float(group["covered_rvr_arm"].mean()),
                    "delta_rvr_mean": float(rvr_delta.mean()),
                    "delta_rvr_sem": _sem(rvr_delta),
                    "base_mse_mean": float(group["obs_loss_base"].mean()),
                    "arm_mse_mean": float(group["obs_loss_arm"].mean()),
                    "delta_mse_mean": float(mse_delta.mean()),
                    "delta_mse_sem": _sem(mse_delta),
                    "directional_rvr_pass": bool((rvr_delta < 0).any()) if arm in {"projection", "residual"} else True,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--paired-baseline", required=True)
    parser.add_argument("--comparison-baselines", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--required-seeds", type=int, default=1)
    parser.add_argument("--required-horizons", default="3,5")
    args = parser.parse_args()
    arms = _parse_tuple(args.arms)
    comparison_baselines = _parse_tuple(args.comparison_baselines)
    horizons = tuple(int(item) for item in args.required_horizons.split(",") if item)
    if args.strict:
        result = validate_paired_comparison(
            args.runs,
            paired_baseline=args.paired_baseline,
            comparison_baselines=comparison_baselines,
            comparison_arms=arms,
            required_seeds=args.required_seeds,
            required_horizons=horizons,
            require_directional_rvr_improvement=tuple(arm for arm in arms if arm in {"projection", "residual"}),
        )
        if not result.ok:
            raise SystemExit(f"paired comparison failed: {result}")
    print(build_report(args.runs, args.paired_baseline, comparison_baselines, arms).to_string(index=False))


if __name__ == "__main__":
    main()
