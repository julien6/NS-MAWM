#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
NUM_ENVS="${NUM_ENVS:-256}"
EPISODES="${EPISODES:-5000}"
MAX_STEPS="${MAX_STEPS:-500}"
VAE_STEPS="${VAE_STEPS:-10000}"
RNN_STEPS="${RNN_STEPS:-10000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-1024}"
WM_NUM_WORKERS="${WM_NUM_WORKERS:-8}"
WANDB_FLAG="${WANDB_FLAG:---wandb}"

BASELINES=(
  B00_model-free-control
  B25_residual_k0.3
  B25_projection_k0.3
  B25_regularization_k0.3
  B26_residual_k0.6
  B26_projection_k0.6
  B26_regularization_k0.6
)

for baseline in "${BASELINES[@]}"; do
  "$PYTHON_BIN" run_benchmarl_gridcraft.py \
    --baseline-id "$baseline" \
    --phase all \
    --num-envs "$NUM_ENVS" \
    --episodes "$EPISODES" \
    --max-steps "$MAX_STEPS" \
    --vae-steps "$VAE_STEPS" \
    --rnn-steps "$RNN_STEPS" \
    --wm-batch-size "$WM_BATCH_SIZE" \
    --wm-num-workers "$WM_NUM_WORKERS" \
    --eval-every 1000 \
    --video-every 1000 \
    --device "$DEVICE" \
    $WANDB_FLAG
done
