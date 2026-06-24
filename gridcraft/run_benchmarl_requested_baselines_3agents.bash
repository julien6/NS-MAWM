#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Sequential 3-agent Gridcraft baseline campaign.
#
# Runs:
# - B00_model-free-control
# - B25_residual_k0.3
# - B25_projection_k0.3
# - B25_regularization_k0.3
# - B26_residual_k0.6
# - B26_projection_k0.6
# - B26_regularization_k0.6
#
# This delegates to run_benchmarl_requested_baselines.bash, which contains the
# serious default budgets and one-W&B-run-per-baseline orchestration.

export NUM_AGENTS=3

echo "Launching requested Gridcraft baselines with NUM_AGENTS=${NUM_AGENTS}"
exec ./run_benchmarl_requested_baselines.bash
