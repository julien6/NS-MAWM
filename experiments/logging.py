"""CSV/JSON experiment logging and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd


class RunLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, object]] = []

    def log(self, row: Mapping[str, object]) -> None:
        self.rows.append(dict(row))

    def flush(self, stem: str = "metrics") -> None:
        frame = pd.DataFrame(self.rows)
        csv_path = self.output_dir / f"{stem}.csv"
        json_path = self.output_dir / f"{stem}.json"
        if csv_path.exists() and not frame.empty:
            frame = pd.concat([pd.read_csv(csv_path), frame], ignore_index=True, sort=False)
        frame.to_csv(csv_path, index=False)
        rows = frame.to_dict(orient="records")
        with (self.output_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, default=str)
        metric_cols = [
            col
            for col in frame.columns
            if col not in {
                "baseline_id",
                "seed",
                "environment",
                "variant_id",
                "train_regime",
                "eval_regime",
                "wm",
                "policy",
                "strategy",
                "mode",
                "comparison_arm",
                "reference_coverage",
                "paired_baseline_id",
                "commit_hash",
                "timestamp",
            }
            and pd.api.types.is_numeric_dtype(frame[col])
        ]
        if "baseline_id" in frame.columns and metric_cols:
            summary = pd.concat(
                [
                    frame.groupby("baseline_id", dropna=False)[metric_cols].mean().add_suffix("_mean"),
                    frame.groupby("baseline_id", dropna=False)[metric_cols].std().add_suffix("_std"),
                    frame.groupby("baseline_id", dropna=False)[metric_cols].sem().add_suffix("_sem"),
                ],
                axis=1,
            ).reset_index()
            summary.to_csv(self.output_dir / f"{stem}_summary.csv", index=False)


def aggregate_mean_sem(csv_path: str, group_by: list[str], metrics: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    grouped = frame.groupby(group_by, dropna=False)
    return pd.concat(
        [
            grouped[metrics].mean().add_suffix("_mean"),
            grouped[metrics].std().add_suffix("_std"),
            grouped[metrics].sem().add_suffix("_sem"),
            (1.96 * grouped[metrics].sem()).add_suffix("_ci95"),
        ],
        axis=1,
    ).reset_index()


def paired_seed_delta(frame: pd.DataFrame, baseline_a: str, baseline_b: str, metric: str) -> dict[str, float]:
    left = frame[frame["baseline_id"] == baseline_a][["seed", metric]].rename(columns={metric: "a"})
    right = frame[frame["baseline_id"] == baseline_b][["seed", metric]].rename(columns={metric: "b"})
    paired = left.merge(right, on="seed")
    if paired.empty:
        return {"paired_n": 0.0, "mean_delta": float("nan"), "sem_delta": float("nan")}
    delta = paired["b"] - paired["a"]
    return {"paired_n": float(len(delta)), "mean_delta": float(delta.mean()), "sem_delta": float(delta.sem())}
