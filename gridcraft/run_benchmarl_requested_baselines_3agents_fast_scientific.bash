#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Medium scientific 3-agent campaign.
#
# This profile keeps the same seven baselines as the long 3-agent campaign but
# targets roughly two hours per baseline on Spark. It is calibrated from an
# observed 3-agent B00 run where 600 policy iterations took
# around 10-12 minutes. The defaults below do about 10x that policy work while
# keeping vectorization, larger minibatches, periodic evaluation, and shorter
# videos. Override any variable from the shell for runtime probes.

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
if [[ "${AUTO_RESOURCE_PROFILE:-0}" == "1" && "${RESOURCE_PROFILE_APPLIED:-0}" != "1" ]]; then
  RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}"
  echo "[resource-profile] applying ${RESOURCE_PROFILE} to fast scientific baseline campaign"
  eval "$("$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target campaign --format shell)"
  "$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target campaign --format summary >&2 || true
  export RESOURCE_PROFILE_APPLIED=1
fi

export NUM_AGENTS="${NUM_AGENTS:-3}"

# World model.
export WM_NUM_ENVS="${WM_NUM_ENVS:-1024}"
export WM_EPISODES="${WM_EPISODES:-30000}"
export WM_MAX_STEPS="${WM_MAX_STEPS:-500}"
export VAE_STEPS="${VAE_STEPS:-50000}"
export RNN_STEPS="${RNN_STEPS:-50000}"
export WM_BATCH_SIZE="${WM_BATCH_SIZE:-4096}"
export WM_NUM_WORKERS="${WM_NUM_WORKERS:-8}"
export WM_SEQ_LEN="${WM_SEQ_LEN:-32}"
export VAE_Z_SIZE="${VAE_Z_SIZE:-64}"
export VAE_HIDDEN_SIZE="${VAE_HIDDEN_SIZE:-512}"
export VAE_KL_TOLERANCE="${VAE_KL_TOLERANCE:-0.5}"
export RNN_SIZE="${RNN_SIZE:-128}"
export RNN_NUM_MIXTURE="${RNN_NUM_MIXTURE:-5}"
export WM_MEAN_MSE_WEIGHT="${WM_MEAN_MSE_WEIGHT:-10.0}"
export WM_REWARD_LOSS_WEIGHT="${WM_REWARD_LOSS_WEIGHT:-1.0}"
export WM_DONE_LOSS_WEIGHT="${WM_DONE_LOSS_WEIGHT:-1.0}"
export WM_LEARNING_RATE="${WM_LEARNING_RATE:-0.001}"
export LAMBDA_SYM="${LAMBDA_SYM:-1.0}"
export LAMBDA_RESIDUAL="${LAMBDA_RESIDUAL:-0.25}"
export HPO_RESULTS_DIR="${HPO_RESULTS_DIR:-hpo_results/world_model}"
export REUSE_WM_HPO_CONFIG="${REUSE_WM_HPO_CONFIG:-1}"
export REQUIRE_WM_HPO="${REQUIRE_WM_HPO:-0}"
export REQUIRED_WM_HPO_STAGE="${REQUIRED_WM_HPO_STAGE:-}"
export MARL_HPO_RESULTS_DIR="${MARL_HPO_RESULTS_DIR:-hpo_results/marl}"
export REUSE_MARL_HPO_CONFIG="${REUSE_MARL_HPO_CONFIG:-1}"
export REQUIRE_MARL_HPO="${REQUIRE_MARL_HPO:-0}"
export REQUIRED_MARL_HPO_STAGE="${REQUIRED_MARL_HPO_STAGE:-}"
export WM_EVAL_EVERY="${WM_EVAL_EVERY:-2500}"
export WM_VIDEO_EVERY="${WM_VIDEO_EVERY:-5000}"
export WM_HORIZONS="${WM_HORIZONS:-1 5 10 25 50 100}"

# Residual variants compute an additional observation-space head/loss. Keeping
# the neural batch size there can OOM on GPUs that handle B10 comfortably.
export RESIDUAL_WM_BATCH_SIZE="${RESIDUAL_WM_BATCH_SIZE:-2048}"
export RESIDUAL_SYMBOLIC_TRAIN_SAMPLES="${RESIDUAL_SYMBOLIC_TRAIN_SAMPLES:-256}"

# Native BenchMARL MASAC/MAMBPO.
export MODEL_FREE_DOWNSTREAM_ALGO="${MODEL_FREE_DOWNSTREAM_ALGO:-masac}"
export MODEL_BASED_DOWNSTREAM_ALGO="${MODEL_BASED_DOWNSTREAM_ALGO:-mambpo}"
export MARL_NUM_ENVS="${MARL_NUM_ENVS:-512}"
export MARL_MAX_STEPS="${MARL_MAX_STEPS:-500}"
export MARL_MAX_ITERS="${MARL_MAX_ITERS:-3000}"
export MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-8192}"
export MARL_TRAIN_BATCH_SIZE="${MARL_TRAIN_BATCH_SIZE:-${MAPPO_MINIBATCH_SIZE:-1024}}"
export MARL_OPTIMIZER_STEPS="${MARL_OPTIMIZER_STEPS:-${MAPPO_MINIBATCH_ITERS:-4}}"
export MARL_EVAL_EVERY_ITERS="${MARL_EVAL_EVERY_ITERS:-${MAPPO_EVAL_EVERY_ITERS:-20}}"
export MARL_EVAL_EPISODES="${MARL_EVAL_EPISODES:-${MAPPO_EVAL_EPISODES:-4}}"
export MARL_VIDEO_EVERY_ITERS="${MARL_VIDEO_EVERY_ITERS:-${MAPPO_VIDEO_EVERY_ITERS:-250}}"
export MARL_HIDDEN_SIZE="${MARL_HIDDEN_SIZE:-${MAPPO_HIDDEN_SIZE:-256}}"

# Evaluation media.
export VIDEO_MAX_STEPS="${VIDEO_MAX_STEPS:-75}"
export VIDEO_FPS="${VIDEO_FPS:-10}"

echo "Launching fast scientific Gridcraft baselines with NUM_AGENTS=${NUM_AGENTS}"
echo "World Model defaults/HPO fallback: z=${VAE_Z_SIZE}, vae_hidden=${VAE_HIDDEN_SIZE}, rnn=${RNN_SIZE}, mixtures=${RNN_NUM_MIXTURE}, seq_len=${WM_SEQ_LEN}, reward_w=${WM_REWARD_LOSS_WEIGHT}, done_w=${WM_DONE_LOSS_WEIGHT}"
echo "World Model HPO registry: ${HPO_RESULTS_DIR} (reuse=${REUSE_WM_HPO_CONFIG}, require=${REQUIRE_WM_HPO})"
echo "MARL HPO registry: ${MARL_HPO_RESULTS_DIR} (reuse=${REUSE_MARL_HPO_CONFIG}, require=${REQUIRE_MARL_HPO})"
exec ./run_benchmarl_requested_baselines.bash
