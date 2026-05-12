"""Generate simple figures from logged CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    path = Path(args.runs) / "eval_rollout.csv"
    if not path.exists():
        if args.strict:
            raise SystemExit(f"{path} missing")
        print(f"{path} missing; run python -m experiments.eval_rollout first")
        return
    frame = pd.read_csv(path)
    required = {"baseline_id", "horizon", "obs_loss", "compounding_error_slope", "rvr"}
    missing = required - set(frame.columns)
    if args.strict and missing:
        raise SystemExit(f"missing required figure columns: {sorted(missing)}")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(frame["horizon"], frame["obs_loss"], marker="o")
    ax.set_xlabel("horizon")
    ax.set_ylabel("obs_loss")
    fig.tight_layout()
    out = Path(args.runs) / "compounding_error.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
