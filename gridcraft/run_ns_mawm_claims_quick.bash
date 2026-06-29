#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Quick W&B experiment to check the main NS-MAWM claims before launching a
# serious campaign. It is intentionally short: use it to validate routing,
# RVR pre/post, symbolic losses, compounding metrics, videos, and early
# downstream MAMBPO signals, not as final scientific evidence.

export RUN_INVARIANTS="${RUN_INVARIANTS:-1}"
if [[ "$RUN_INVARIANTS" == "1" ]]; then
  echo "=== Step 0: validating NS-MAWM invariants ==="
  ./validate_ns_mawm_invariants.bash
fi

export SEEDS="${SEEDS:-1}"
export NUM_AGENTS="${NUM_AGENTS:-3}"
export BASELINES="${BASELINES:-B00_model-free-control B10_neural_k0.0 B25_regularization_k0.3 B25_projection_k0.3 B25_residual_k0.3}"

# Use a compact but representative reliable PSTR subset. Override with
# ENABLED_PSTR_RULES="" to use the full catalogue.
export ENABLED_PSTR_RULES="${ENABLED_PSTR_RULES:-PSTR_INDIV_STATIC_TERRAIN_SHIFT PSTR_INDIV_STATIC_BLOCK_SHIFT PSTR_INDIV_CENTER_AGENT PSTR_INDIV_HARVEST_TREE_WOOD PSTR_INDIV_EAT_APPLE PSTR_INDIV_CRAFT_PLANK}"

export MODEL_FREE_DOWNSTREAM_ALGO="${MODEL_FREE_DOWNSTREAM_ALGO:-masac}"
export MODEL_BASED_DOWNSTREAM_ALGO="${MODEL_BASED_DOWNSTREAM_ALGO:-mambpo}"

# World model: enough evaluations to see trends without doing a long run.
export WM_NUM_ENVS="${WM_NUM_ENVS:-128}"
export WM_EPISODES="${WM_EPISODES:-2048}"
export WM_MAX_STEPS="${WM_MAX_STEPS:-96}"
export VAE_STEPS="${VAE_STEPS:-2500}"
export RNN_STEPS="${RNN_STEPS:-2500}"
export WM_BATCH_SIZE="${WM_BATCH_SIZE:-512}"
export WM_NUM_WORKERS="${WM_NUM_WORKERS:-4}"
export WM_EVAL_EVERY="${WM_EVAL_EVERY:-500}"
export WM_VIDEO_EVERY="${WM_VIDEO_EVERY:-1000}"
export WM_HORIZONS="${WM_HORIZONS:-1 5 10 25}"
export JOINT_SYMBOLIC_TRAIN_EPISODES="${JOINT_SYMBOLIC_TRAIN_EPISODES:-32}"
export JOINT_SYMBOLIC_TRAIN_STEPS="${JOINT_SYMBOLIC_TRAIN_STEPS:-32}"

# Downstream MARL: short MAMBPO/MASAC phase to confirm that model-based
# baselines use imagined rollouts and log comparable metrics.
export MARL_NUM_ENVS="${MARL_NUM_ENVS:-64}"
export MARL_MAX_STEPS="${MARL_MAX_STEPS:-96}"
export MARL_MAX_ITERS="${MARL_MAX_ITERS:-80}"
export MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-512}"
export MARL_TRAIN_BATCH_SIZE="${MARL_TRAIN_BATCH_SIZE:-256}"
export MARL_OPTIMIZER_STEPS="${MARL_OPTIMIZER_STEPS:-1}"
export MARL_EVAL_EVERY_ITERS="${MARL_EVAL_EVERY_ITERS:-10}"
export MARL_EVAL_EPISODES="${MARL_EVAL_EPISODES:-2}"
export MARL_VIDEO_EVERY_ITERS="${MARL_VIDEO_EVERY_ITERS:-40}"
export MARL_HIDDEN_SIZE="${MARL_HIDDEN_SIZE:-256}"

# Keep MAMBPO focused on the external Gridcraft world-model signal.
export MB_WORLD_MODEL_TRAIN_EPOCHS="${MB_WORLD_MODEL_TRAIN_EPOCHS:-0}"
export MB_WORLD_MODEL_BATCH_SIZE="${MB_WORLD_MODEL_BATCH_SIZE:-128}"
export MB_WORLD_MODEL_HIDDEN_SIZE="${MB_WORLD_MODEL_HIDDEN_SIZE:-256}"
export MB_IMAGINED_HORIZON="${MB_IMAGINED_HORIZON:-5}"
export MB_IMAGINED_BRANCHES="${MB_IMAGINED_BRANCHES:-4}"
export MB_LAMBDA_IMAGINED="${MB_LAMBDA_IMAGINED:-0.5}"

# Media: keep videos enabled but sparse. They now show timestep/action/reward
# and real-vs-imagined columns.
export VIDEO_MAX_STEPS="${VIDEO_MAX_STEPS:-40}"
export VIDEO_FPS="${VIDEO_FPS:-8}"

export SHARED_MODEL_DIR="${SHARED_MODEL_DIR:-shared_models}"
export REUSE_VAE_CACHE="${REUSE_VAE_CACHE:-1}"
export REUSE_LATENT_CACHE="${REUSE_LATENT_CACHE:-1}"
export FORCE_VAE_RETRAIN="${FORCE_VAE_RETRAIN:-0}"
export FORCE_LATENT_REENCODE="${FORCE_LATENT_REENCODE:-0}"

export WANDB_FLAG="${WANDB_FLAG:---wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-ns-mawm-gridcraft}"

echo "=== Running quick NS-MAWM claims check ==="
echo "Baselines: ${BASELINES}"
echo "Seeds: ${SEEDS}"
echo "Agents: ${NUM_AGENTS}"
echo "PSTR rules: ${ENABLED_PSTR_RULES:-all}"
echo "W&B project: ${WANDB_PROJECT}"
echo "This is a quick diagnostic run, not the final scientific campaign."

exec ./run_benchmarl_requested_baselines_3agents_fast_scientific.bash
