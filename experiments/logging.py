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
        pd.DataFrame(self.rows).to_csv(self.output_dir / f"{stem}.csv", index=False)
        with (self.output_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2, default=str)


def aggregate_mean_sem(csv_path: str, group_by: list[str], metrics: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    grouped = frame.groupby(group_by, dropna=False)
    return pd.concat([grouped[metrics].mean().add_suffix("_mean"), grouped[metrics].sem().add_suffix("_sem")], axis=1).reset_index()
