#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
BASELINE_ID="${BASELINE_ID:-B25_projection_k0.3}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
WANDB_FLAG="${WANDB_FLAG:---wandb}"
RUN_NAME="${RUN_NAME:-${BASELINE_ID}_full_seed${SEED}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-${BASELINE_ID}}"

# World model + lightweight policy evaluation.
WM_NUM_ENVS="${WM_NUM_ENVS:-128}"
WM_EPISODES="${WM_EPISODES:-1000}"
WM_MAX_STEPS="${WM_MAX_STEPS:-200}"
VAE_STEPS="${VAE_STEPS:-2000}"
RNN_STEPS="${RNN_STEPS:-2000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-512}"
WM_NUM_WORKERS="${WM_NUM_WORKERS:-4}"
WM_EVAL_EVERY="${WM_EVAL_EVERY:-500}"
WM_HORIZONS="${WM_HORIZONS:-1 5 10 25 50}"

# Native BenchMARL MAPPO train/eval.
MARL_NUM_ENVS="${MARL_NUM_ENVS:-64}"
MARL_MAX_STEPS="${MARL_MAX_STEPS:-200}"
MARL_MAX_ITERS="${MARL_MAX_ITERS:-50}"
MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-2048}"

echo "=== Step 1/2: World Model train/eval + lightweight policy eval (${BASELINE_ID}) ==="
echo "W&B run id: ${WANDB_RUN_ID}"
"$PYTHON_BIN" run_benchmarl_gridcraft.py \
  --baseline-id "$BASELINE_ID" \
  --phase all \
  --seed "$SEED" \
  --num-envs "$WM_NUM_ENVS" \
  --episodes "$WM_EPISODES" \
  --max-steps "$WM_MAX_STEPS" \
  --vae-steps "$VAE_STEPS" \
  --rnn-steps "$RNN_STEPS" \
  --wm-batch-size "$WM_BATCH_SIZE" \
  --wm-num-workers "$WM_NUM_WORKERS" \
  --seq-len 32 \
  --eval-every "$WM_EVAL_EVERY" \
  --horizons $WM_HORIZONS \
  --device "$DEVICE" \
  --wandb-id "$WANDB_RUN_ID" \
  --wandb-name "$RUN_NAME" \
  --wandb-group "$WANDB_GROUP" \
  $WANDB_FLAG

echo "=== Step 2/2: Native BenchMARL MAPPO train/eval on vGridcraft ==="
"$PYTHON_BIN" run_benchmarl_mappo_gridcraft.py \
  --seed "$SEED" \
  --num-envs "$MARL_NUM_ENVS" \
  --max-steps "$MARL_MAX_STEPS" \
  --max-iters "$MARL_MAX_ITERS" \
  --frames-per-batch "$MARL_FRAMES_PER_BATCH" \
  --device "$DEVICE" \
  --wandb-id "$WANDB_RUN_ID" \
  --wandb-name "$RUN_NAME" \
  --wandb-group "$WANDB_GROUP" \
  $WANDB_FLAG

echo "=== Completed full baseline pipeline (${BASELINE_ID}) ==="
