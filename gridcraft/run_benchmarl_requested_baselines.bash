#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Runs exactly the requested Gridcraft baseline set:
# - B00: model-free MARL on the real vectorized Gridcraft environment.
# - B10: classic model-based VAE+MDN-RNN without NS-MAWM.
# - B25/B26: model-based baselines with NS-MAWM variants at k=0.3 and k=0.6.
#
# Each baseline delegates to run_full_benchmarl_baseline.bash, which produces one
# W&B run per baseline/seed and runs phases sequentially:
# B00 uses native BenchMARL MAPPO on real vGridcraft; model-based baselines train
# their world model, then use Dyna MAPPO inside that trained world model.

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
NUM_AGENTS="${NUM_AGENTS:-1}"
SEEDS="${SEEDS:-1 2 3}"
WANDB_FLAG="${WANDB_FLAG---wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ns-mawm-gridcraft}"
DRY_RUN="${DRY_RUN:-0}"

# World model defaults. These are intended for serious experimental runs, not
# smoke tests. They are deliberately expensive and should be launched on Spark
# or a comparable GPU/CPU node. Override from the shell for development.
WM_NUM_ENVS="${WM_NUM_ENVS:-512}"
WM_EPISODES="${WM_EPISODES:-50000}"
WM_MAX_STEPS="${WM_MAX_STEPS:-500}"
VAE_STEPS="${VAE_STEPS:-100000}"
RNN_STEPS="${RNN_STEPS:-100000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-2048}"
WM_NUM_WORKERS="${WM_NUM_WORKERS:-8}"
WM_EVAL_EVERY="${WM_EVAL_EVERY:-5000}"
WM_VIDEO_EVERY="${WM_VIDEO_EVERY:-10000}"
WM_HORIZONS="${WM_HORIZONS:-1 5 10 25 50 100}"
SHARED_MODEL_DIR="${SHARED_MODEL_DIR:-shared_models}"
REUSE_VAE_CACHE="${REUSE_VAE_CACHE:-1}"
FORCE_VAE_RETRAIN="${FORCE_VAE_RETRAIN:-0}"
REUSE_LATENT_CACHE="${REUSE_LATENT_CACHE:-1}"
FORCE_LATENT_REENCODE="${FORCE_LATENT_REENCODE:-0}"

# Downstream MARL defaults. B00 uses these values for native BenchMARL MAPPO on
# real vGridcraft; model-based baselines use them for Dyna MAPPO world-model
# rollouts where applicable.
MARL_NUM_ENVS="${MARL_NUM_ENVS:-256}"
MARL_MAX_STEPS="${MARL_MAX_STEPS:-500}"
MARL_MAX_ITERS="${MARL_MAX_ITERS:-2000}"
MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-8192}"
MAPPO_MINIBATCH_SIZE="${MAPPO_MINIBATCH_SIZE:-1024}"
MAPPO_MINIBATCH_ITERS="${MAPPO_MINIBATCH_ITERS:-2}"
MAPPO_EVAL_EVERY_ITERS="${MAPPO_EVAL_EVERY_ITERS:-25}"
MAPPO_EVAL_EPISODES="${MAPPO_EVAL_EPISODES:-4}"
MAPPO_VIDEO_EVERY_ITERS="${MAPPO_VIDEO_EVERY_ITERS:-250}"
MAPPO_HIDDEN_SIZE="${MAPPO_HIDDEN_SIZE:-256}"
DYNA_IMAGINED_HORIZON="${DYNA_IMAGINED_HORIZON:-32}"

# Evaluation media.
VIDEO_MAX_STEPS="${VIDEO_MAX_STEPS:-100}"
VIDEO_FPS="${VIDEO_FPS:-10}"

BASELINES="${BASELINES:-B00_model-free-control B10_neural_k0.0 B25_residual_k0.3 B25_projection_k0.3 B25_regularization_k0.3 B26_residual_k0.6 B26_projection_k0.6 B26_regularization_k0.6}"

echo "Running requested Gridcraft baselines:"
for baseline in $BASELINES; do
  printf '  - %s\n' "$baseline"
done
echo "Seeds: ${SEEDS}"
echo "W&B project: ${WANDB_PROJECT}"
echo "Device: ${DEVICE}"
echo "Agents: ${NUM_AGENTS}"
echo "World model: num_envs=${WM_NUM_ENVS}, episodes=${WM_EPISODES}, max_steps=${WM_MAX_STEPS}, vae_steps=${VAE_STEPS}, rnn_steps=${RNN_STEPS}, batch=${WM_BATCH_SIZE}, eval_every=${WM_EVAL_EVERY}"
echo "World model cache: shared_model_dir=${SHARED_MODEL_DIR}, reuse_vae=${REUSE_VAE_CACHE}, force_vae_retrain=${FORCE_VAE_RETRAIN}, reuse_latents=${REUSE_LATENT_CACHE}, force_latent_reencode=${FORCE_LATENT_REENCODE}"
echo "MARL: num_envs=${MARL_NUM_ENVS}, max_steps=${MARL_MAX_STEPS}, max_iters=${MARL_MAX_ITERS}, frames_per_batch=${MARL_FRAMES_PER_BATCH}, minibatch=${MAPPO_MINIBATCH_SIZE}, minibatch_iters=${MAPPO_MINIBATCH_ITERS}, eval_every_iters=${MAPPO_EVAL_EVERY_ITERS}, video_every_iters=${MAPPO_VIDEO_EVERY_ITERS}, dyna_horizon=${DYNA_IMAGINED_HORIZON}"
echo "MARL routing: B00/model-free uses real vGridcraft MAPPO; all model-based baselines use Dyna MAPPO inside the trained world model."
echo "Videos: wm_every=${WM_VIDEO_EVERY}, marl_every_iters=${MAPPO_VIDEO_EVERY_ITERS}, max_steps=${VIDEO_MAX_STEPS}, fps=${VIDEO_FPS}"

for seed in $SEEDS; do
  for baseline in $BASELINES; do
    echo
    echo "=== Baseline ${baseline}, seed ${seed} ==="
    PYTHON_BIN="$PYTHON_BIN" \
    BASELINE_ID="$baseline" \
    SEED="$seed" \
    DEVICE="$DEVICE" \
    NUM_AGENTS="$NUM_AGENTS" \
    WANDB_FLAG="$WANDB_FLAG" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    DRY_RUN="$DRY_RUN" \
    RUN_NAME="${baseline}_a${NUM_AGENTS}_full_seed${seed}" \
    WANDB_GROUP="$baseline" \
    WM_NUM_ENVS="$WM_NUM_ENVS" \
    WM_EPISODES="$WM_EPISODES" \
    WM_MAX_STEPS="$WM_MAX_STEPS" \
    VAE_STEPS="$VAE_STEPS" \
    RNN_STEPS="$RNN_STEPS" \
    WM_BATCH_SIZE="$WM_BATCH_SIZE" \
    WM_NUM_WORKERS="$WM_NUM_WORKERS" \
    WM_EVAL_EVERY="$WM_EVAL_EVERY" \
    WM_VIDEO_EVERY="$WM_VIDEO_EVERY" \
    WM_HORIZONS="$WM_HORIZONS" \
    SHARED_MODEL_DIR="$SHARED_MODEL_DIR" \
    REUSE_VAE_CACHE="$REUSE_VAE_CACHE" \
    FORCE_VAE_RETRAIN="$FORCE_VAE_RETRAIN" \
    REUSE_LATENT_CACHE="$REUSE_LATENT_CACHE" \
    FORCE_LATENT_REENCODE="$FORCE_LATENT_REENCODE" \
    MARL_NUM_ENVS="$MARL_NUM_ENVS" \
    MARL_MAX_STEPS="$MARL_MAX_STEPS" \
    MARL_MAX_ITERS="$MARL_MAX_ITERS" \
    MARL_FRAMES_PER_BATCH="$MARL_FRAMES_PER_BATCH" \
    MAPPO_MINIBATCH_SIZE="$MAPPO_MINIBATCH_SIZE" \
    MAPPO_MINIBATCH_ITERS="$MAPPO_MINIBATCH_ITERS" \
    MAPPO_EVAL_EVERY_ITERS="$MAPPO_EVAL_EVERY_ITERS" \
    MAPPO_EVAL_EPISODES="$MAPPO_EVAL_EPISODES" \
    MAPPO_VIDEO_EVERY_ITERS="$MAPPO_VIDEO_EVERY_ITERS" \
    MAPPO_HIDDEN_SIZE="$MAPPO_HIDDEN_SIZE" \
    DYNA_IMAGINED_HORIZON="$DYNA_IMAGINED_HORIZON" \
    VIDEO_MAX_STEPS="$VIDEO_MAX_STEPS" \
    VIDEO_FPS="$VIDEO_FPS" \
    ./run_full_benchmarl_baseline.bash
  done
done

echo
echo "=== Completed requested Gridcraft baseline set ==="
