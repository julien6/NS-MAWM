"""Report paired Predator-Prey comparison results from logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.reproduction_check import validate_paired_comparison


def _sem(series: pd.Series) -> float:
    return float(series.sem()) if len(series) > 1 else 0.0


def build_report(runs: str | Path, arms: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(Path(runs) / "eval_rollout.csv")
    base = frame[frame["baseline_id"].astype(str) == "B31"]
    rows: list[dict[str, object]] = []
    for arm in arms:
        arm_rows = frame[(frame["baseline_id"].astype(str) == "B34") & (frame["comparison_arm"].fillna("").astype(str) == arm)]
        merged = base.merge(arm_rows, on=["seed", "horizon"], suffixes=("_b31", "_b34"))
        for horizon, group in merged.groupby("horizon"):
            rvr_delta = group["covered_rvr_b34"] - group["covered_rvr_b31"]
            mse_delta = group["obs_loss_b34"] - group["obs_loss_b31"]
            rows.append(
                {
                    "arm": arm,
                    "horizon": int(horizon),
                    "paired_n": int(len(group)),
                    "b31_rvr_mean": float(group["covered_rvr_b31"].mean()),
                    "b34_rvr_mean": float(group["covered_rvr_b34"].mean()),
                    "delta_rvr_mean": float(rvr_delta.mean()),
                    "delta_rvr_sem": _sem(rvr_delta),
                    "b31_mse_mean": float(group["obs_loss_b31"].mean()),
                    "b34_mse_mean": float(group["obs_loss_b34"].mean()),
                    "delta_mse_mean": float(mse_delta.mean()),
                    "delta_mse_sem": _sem(mse_delta),
                    "directional_rvr_pass": bool((rvr_delta < 0).any()) if arm in {"projection", "residual"} else True,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/predator_prey_claim_path")
    parser.add_argument("--arms", default="regularization,projection,residual")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--required-seeds", type=int, default=1)
    parser.add_argument("--required-horizons", default="3,5")
    args = parser.parse_args()
    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    horizons = tuple(int(item) for item in args.required_horizons.split(",") if item)
    if args.strict:
        result = validate_paired_comparison(
            args.runs,
            paired_baseline="B31",
            comparison_baselines=("B34",),
            comparison_arms=arms,
            required_seeds=args.required_seeds,
            required_horizons=horizons,
            require_directional_rvr_improvement=tuple(arm for arm in arms if arm in {"projection", "residual"}),
        )
        if not result.ok:
            raise SystemExit(f"paired comparison failed: {result}")
    report = build_report(args.runs, arms)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
