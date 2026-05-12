"""Evaluate held-out variant behavior by forcing UV evaluation."""

from __future__ import annotations

import sys

from experiments.eval_rollout import main


if __name__ == "__main__":
    if not any(arg.startswith("eval_regime=") for arg in sys.argv[1:]):
        sys.argv.append("eval_regime=UV")
    main()
