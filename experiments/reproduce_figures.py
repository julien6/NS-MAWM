"""Generate simple figures from logged CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    path = Path("runs/eval_rollout.csv")
    if not path.exists():
        print("runs/eval_rollout.csv missing; run python -m experiments.eval_rollout first")
        return
    frame = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(frame["horizon"], frame["obs_loss"], marker="o")
    ax.set_xlabel("horizon")
    ax.set_ylabel("obs_loss")
    fig.tight_layout()
    out = Path("runs/compounding_error.png")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
