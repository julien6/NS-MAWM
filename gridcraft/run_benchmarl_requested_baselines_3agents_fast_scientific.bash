#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Faster scientific 3-agent campaign.
#
# This profile keeps the same seven baselines as the long 3-agent campaign but
# reduces wall-clock time by using a larger vectorized batch, fewer MAPPO
# optimization passes per iteration, less frequent evaluation, and shorter
# videos. Override any variable from the shell for runtime probes.

export NUM_AGENTS="${NUM_AGENTS:-3}"

# World model.
export WM_NUM_ENVS="${WM_NUM_ENVS:-1024}"
export WM_EPISODES="${WM_EPISODES:-30000}"
export WM_MAX_STEPS="${WM_MAX_STEPS:-500}"
export VAE_STEPS="${VAE_STEPS:-50000}"
export RNN_STEPS="${RNN_STEPS:-50000}"
export WM_BATCH_SIZE="${WM_BATCH_SIZE:-4096}"
export WM_NUM_WORKERS="${WM_NUM_WORKERS:-8}"
export WM_EVAL_EVERY="${WM_EVAL_EVERY:-5000}"
export WM_VIDEO_EVERY="${WM_VIDEO_EVERY:-10000}"
export WM_HORIZONS="${WM_HORIZONS:-1 5 10 25 50 100}"

# Native BenchMARL MAPPO.
export MARL_NUM_ENVS="${MARL_NUM_ENVS:-512}"
export MARL_MAX_STEPS="${MARL_MAX_STEPS:-500}"
export MARL_MAX_ITERS="${MARL_MAX_ITERS:-600}"
export MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-8192}"
export MAPPO_MINIBATCH_SIZE="${MAPPO_MINIBATCH_SIZE:-1024}"
export MAPPO_MINIBATCH_ITERS="${MAPPO_MINIBATCH_ITERS:-2}"
export MAPPO_EVAL_EVERY_ITERS="${MAPPO_EVAL_EVERY_ITERS:-25}"
export MAPPO_EVAL_EPISODES="${MAPPO_EVAL_EPISODES:-4}"
export MAPPO_HIDDEN_SIZE="${MAPPO_HIDDEN_SIZE:-256}"

# Evaluation media.
export VIDEO_MAX_STEPS="${VIDEO_MAX_STEPS:-75}"
export VIDEO_FPS="${VIDEO_FPS:-10}"

echo "Launching fast scientific Gridcraft baselines with NUM_AGENTS=${NUM_AGENTS}"
exec ./run_benchmarl_requested_baselines.bash
