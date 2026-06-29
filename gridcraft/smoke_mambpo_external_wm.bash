#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
BASELINE_ID="${BASELINE_ID:-B25_projection_k0.3}"
NUM_AGENTS="${NUM_AGENTS:-3}"
SEED="${SEED:-1}"
WM_RUN_DIR="${WM_RUN_DIR:-runs_benchmarl/${BASELINE_ID}_a${NUM_AGENTS}_seed${SEED}}"
DEVICE="${DEVICE:-cpu}"
WANDB_FLAG="${WANDB_FLAG:-}"

echo "=== Smoke MAMBPO external Gridcraft world model ==="
echo "baseline=${BASELINE_ID}"
echo "wm_run_dir=${WM_RUN_DIR}"

"$PYTHON_BIN" run_benchmarl_marl_gridcraft.py \
  --algorithm mambpo \
  --baseline-id "$BASELINE_ID" \
  --wm-run-dir "$WM_RUN_DIR" \
  --num-agents "$NUM_AGENTS" \
  --seed "$SEED" \
  --num-envs 8 \
  --max-steps 32 \
  --max-iters 2 \
  --frames-per-batch 96 \
  --marl-train-batch-size 48 \
  --marl-optimizer-steps 1 \
  --marl-eval-every-iters 1 \
  --marl-eval-episodes 1 \
  --mb-world-model-train-epochs 0 \
  --mb-world-model-batch-size 16 \
  --mb-imagined-horizon 3 \
  --mb-imagined-branches 2 \
  --mb-lambda-imagined 0.5 \
  --device "$DEVICE" \
  --no-wandb-videos \
  $WANDB_FLAG
