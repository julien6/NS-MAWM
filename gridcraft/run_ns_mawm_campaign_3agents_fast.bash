#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Fast scientific campaign after invariant and micro-run validation.
# Videos are disabled by default to keep runtime and W&B storage under control.

export SEEDS="${SEEDS:-1 2 3}"
export NUM_AGENTS="${NUM_AGENTS:-3}"
export BASELINES="${BASELINES:-B00_model-free-control B10_neural_k0.0 B25_residual_k0.3 B25_projection_k0.3 B25_regularization_k0.3 B26_residual_k0.6 B26_projection_k0.6 B26_regularization_k0.6}"

export MODEL_FREE_DOWNSTREAM_ALGO="${MODEL_FREE_DOWNSTREAM_ALGO:-masac}"
export MODEL_BASED_DOWNSTREAM_ALGO="${MODEL_BASED_DOWNSTREAM_ALGO:-mambpo}"

export WM_EPISODES="${WM_EPISODES:-30000}"
export WM_MAX_STEPS="${WM_MAX_STEPS:-500}"
export VAE_STEPS="${VAE_STEPS:-50000}"
export RNN_STEPS="${RNN_STEPS:-50000}"
export WM_BATCH_SIZE="${WM_BATCH_SIZE:-4096}"
export WM_EVAL_EVERY="${WM_EVAL_EVERY:-2500}"
export WM_VIDEO_EVERY="${WM_VIDEO_EVERY:-0}"
export WM_HORIZONS="${WM_HORIZONS:-1 5 10 25 50 100}"

export MARL_MAX_ITERS="${MARL_MAX_ITERS:-3000}"
export MARL_NUM_ENVS="${MARL_NUM_ENVS:-512}"
export MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-8192}"
export MARL_TRAIN_BATCH_SIZE="${MARL_TRAIN_BATCH_SIZE:-1024}"
export MARL_OPTIMIZER_STEPS="${MARL_OPTIMIZER_STEPS:-4}"
export MARL_EVAL_EVERY_ITERS="${MARL_EVAL_EVERY_ITERS:-20}"
export MARL_EVAL_EPISODES="${MARL_EVAL_EPISODES:-4}"
export MARL_VIDEO_EVERY_ITERS="${MARL_VIDEO_EVERY_ITERS:-0}"
export MB_WORLD_MODEL_TRAIN_EPOCHS="${MB_WORLD_MODEL_TRAIN_EPOCHS:-0}"

export WANDB_FLAG="${WANDB_FLAG:---wandb}"

echo "=== Running NS-MAWM 3-agent fast campaign ==="
echo "Baselines: ${BASELINES}"
echo "Seeds: ${SEEDS}"
exec ./run_benchmarl_requested_baselines_3agents_fast_scientific.bash
