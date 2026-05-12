"""Strict validation for reproduction logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.registry import BASELINES


DEFAULT_TABLE_METRICS = (
    "compounding_error_slope",
    "rvr",
    "covered_rvr",
    "wm_total_loss",
    "obs_loss",
    "kl_loss",
    "reward_loss",
    "done_loss",
    "projection_magnitude",
    "residual_error",
    "reward_per_resource",
    "training_real_reward",
    "generalization_gap",
)

MODEL_FREE_METRICS = (
    "reward_per_resource",
    "training_real_reward",
    "generalization_gap",
)

WORLD_MODEL_METRICS = DEFAULT_TABLE_METRICS


@dataclass(frozen=True)
class ReproductionCheckResult:
    baselines: tuple[str, ...]
    missing_baselines: tuple[str, ...]
    missing_metrics: tuple[str, ...]
    insufficient_seeds: dict[str, int]
    missing_horizons: dict[str, tuple[int, ...]]
    directional_failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (
            self.missing_baselines
            or self.missing_metrics
            or self.insufficient_seeds
            or self.missing_horizons
            or self.directional_failures
        )


def _load_logs(runs: Path) -> pd.DataFrame:
    frames = []
    for name in (
        "eval_rollout.csv",
        "train_world_model.csv",
        "eval_planning.csv",
        "eval_ood.csv",
        "eval_rule_dropout.csv",
        "eval_noisy_rules.csv",
    ):
        path = runs / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No seed-level reproduction logs found in {runs}")
    return pd.concat(frames, ignore_index=True, sort=False)


def validate_reproduction_logs(
    runs: str | Path,
    *,
    baselines: tuple[str, ...] | None = None,
    required_seeds: int = 5,
    required_horizons: tuple[int, ...] = (10, 25, 50),
    required_metrics: tuple[str, ...] | None = DEFAULT_TABLE_METRICS,
) -> ReproductionCheckResult:
    runs_path = Path(runs)
    frame = _load_logs(runs_path)
    selected = baselines or tuple(BASELINES)
    unknown = tuple(sorted(set(selected) - set(BASELINES)))
    if unknown:
        raise ValueError(f"Unknown baseline ids: {', '.join(unknown)}")

    observed_baselines = set(frame.get("baseline_id", pd.Series(dtype=str)).dropna().astype(str))
    missing_baselines = tuple(sorted(set(selected) - observed_baselines))
    selected_frame = frame[frame["baseline_id"].astype(str).isin(selected)] if "baseline_id" in frame.columns else frame.iloc[0:0]

    missing_metric_items: list[str] = []
    for baseline_id in selected:
        spec = BASELINES[baseline_id]
        group = selected_frame[selected_frame["baseline_id"].astype(str) == baseline_id] if "baseline_id" in selected_frame else selected_frame.iloc[0:0]
        if group.empty:
            continue
        metrics_for_baseline = MODEL_FREE_METRICS if spec.wm == "MF" else (required_metrics or WORLD_MODEL_METRICS)
        for metric in metrics_for_baseline:
            if metric not in group.columns or not group[metric].notna().any():
                missing_metric_items.append(f"{baseline_id}:{metric}")
    missing_metrics = tuple(sorted(missing_metric_items))

    insufficient_seeds: dict[str, int] = {}
    if "seed" in selected_frame.columns and "baseline_id" in selected_frame.columns:
        for baseline_id in selected:
            observed = selected_frame[selected_frame["baseline_id"].astype(str) == baseline_id]["seed"].nunique()
            if int(observed) < required_seeds:
                insufficient_seeds[baseline_id] = int(observed)
    elif required_seeds > 0:
        insufficient_seeds = {baseline_id: 0 for baseline_id in selected}

    missing_horizons: dict[str, tuple[int, ...]] = {}
    if required_horizons:
        eval_path = runs_path / "eval_rollout.csv"
        eval_frame = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
        for baseline_id in selected:
            spec = BASELINES[baseline_id]
            if spec.wm == "MF":
                continue
            if "baseline_id" not in eval_frame.columns or "horizon" not in eval_frame.columns:
                missing_horizons[baseline_id] = required_horizons
                continue
            group = eval_frame[eval_frame["baseline_id"].astype(str) == baseline_id]
            observed = {int(h) for h in group.get("horizon", pd.Series(dtype=int)).dropna().unique()}
            missing = tuple(sorted(set(required_horizons) - observed))
            if missing:
                missing_horizons[baseline_id] = missing

    return ReproductionCheckResult(selected, missing_baselines, missing_metrics, insufficient_seeds, missing_horizons)


def validate_paired_comparison(
    runs: str | Path,
    *,
    paired_baseline: str,
    comparison_baselines: tuple[str, ...],
    comparison_arms: tuple[str, ...],
    required_seeds: int,
    required_horizons: tuple[int, ...],
    require_directional_rvr_improvement: tuple[str, ...] = (),
) -> ReproductionCheckResult:
    selected = (paired_baseline, *comparison_baselines)
    base_result = validate_reproduction_logs(
        runs,
        baselines=selected,
        required_seeds=required_seeds,
        required_horizons=required_horizons,
        required_metrics=DEFAULT_TABLE_METRICS,
    )
    runs_path = Path(runs)
    eval_path = runs_path / "eval_rollout.csv"
    if not eval_path.exists():
        return ReproductionCheckResult(
            base_result.baselines,
            base_result.missing_baselines,
            base_result.missing_metrics,
            base_result.insufficient_seeds,
            {**base_result.missing_horizons, paired_baseline: required_horizons},
            base_result.directional_failures,
        )
    frame = pd.read_csv(eval_path)
    failures: list[str] = list(base_result.directional_failures)
    required_columns = {"baseline_id", "seed", "horizon", "comparison_arm", "covered_rvr", "obs_loss"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        failures.append(f"missing paired comparison columns: {sorted(missing_columns)}")
        return ReproductionCheckResult(
            base_result.baselines,
            base_result.missing_baselines,
            base_result.missing_metrics,
            base_result.insufficient_seeds,
            base_result.missing_horizons,
            tuple(failures),
        )

    base = frame[frame["baseline_id"].astype(str) == paired_baseline]
    if base.empty:
        failures.append(f"missing paired baseline rows for {paired_baseline}")
    comparison_frame = frame[frame["baseline_id"].astype(str).isin(comparison_baselines)]
    for arm in comparison_arms:
        arm_rows = comparison_frame[comparison_frame["comparison_arm"].fillna("").astype(str) == arm]
        if arm_rows.empty:
            failures.append(f"missing comparison arm {arm} in baselines {comparison_baselines}")
            continue
        merged = base.merge(arm_rows, on=["seed", "horizon"], suffixes=("_base", "_arm"))
        expected_pairs = required_seeds * len(required_horizons)
        if len(merged) < expected_pairs:
            failures.append(f"insufficient paired rows for arm {arm}: {len(merged)}/{expected_pairs}")
            continue
        if arm in require_directional_rvr_improvement:
            delta = merged["covered_rvr_arm"] - merged["covered_rvr_base"]
            if not (delta < 0).any():
                baselines = sorted(set(arm_rows["baseline_id"].astype(str)))
                failures.append(
                    f"{arm} ({baselines}) did not reduce covered_rvr versus {paired_baseline}; "
                    f"mean_delta={float(delta.mean()):.6f}"
                )

    return ReproductionCheckResult(
        base_result.baselines,
        base_result.missing_baselines,
        base_result.missing_metrics,
        base_result.insufficient_seeds,
        base_result.missing_horizons,
        tuple(failures),
    )


def _parse_tuple(value: str, cast=str) -> tuple:
    if not value:
        return ()
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--baselines", default="")
    parser.add_argument("--required-seeds", type=int, default=5)
    parser.add_argument("--required-horizons", default="10,25,50")
    parser.add_argument("--required-metrics", default=",".join(DEFAULT_TABLE_METRICS))
    parser.add_argument("--paired-comparison", action="store_true")
    parser.add_argument("--paired-baseline", default="B31")
    parser.add_argument("--comparison-baselines", default="B34")
    parser.add_argument("--comparison-arms", default="regularization,projection,residual")
    parser.add_argument("--require-directional-rvr-improvement", default="")
    args = parser.parse_args()

    if args.paired_comparison:
        result = validate_paired_comparison(
            args.runs,
            paired_baseline=args.paired_baseline,
            comparison_baselines=_parse_tuple(args.comparison_baselines, str),
            comparison_arms=_parse_tuple(args.comparison_arms, str),
            required_seeds=args.required_seeds,
            required_horizons=_parse_tuple(args.required_horizons, int),
            require_directional_rvr_improvement=_parse_tuple(args.require_directional_rvr_improvement, str),
        )
    else:
        result = validate_reproduction_logs(
            args.runs,
            baselines=_parse_tuple(args.baselines, str) or None,
            required_seeds=args.required_seeds,
            required_horizons=_parse_tuple(args.required_horizons, int),
            required_metrics=_parse_tuple(args.required_metrics, str),
        )
    if not result.ok:
        raise SystemExit(
            "Reproduction check failed: "
            f"missing baselines={list(result.missing_baselines)}; "
            f"missing metrics={list(result.missing_metrics)}; "
            f"insufficient seeds={result.insufficient_seeds}; "
            f"missing horizons={result.missing_horizons}; "
            f"directional failures={list(result.directional_failures)}"
        )
    print(f"reproduction check ok for {len(result.baselines)} baselines in {args.runs}")


if __name__ == "__main__":
    main()
