#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
BASELINE_ID="${BASELINE_ID:-B25_projection_k0.3}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
NUM_AGENTS="${NUM_AGENTS:-1}"
WANDB_FLAG="${WANDB_FLAG---wandb}"
RUN_NAME="${RUN_NAME:-${BASELINE_ID}_a${NUM_AGENTS}_full_seed${SEED}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-${BASELINE_ID}}"
DRY_RUN="${DRY_RUN:-0}"

# World-model phase for model-based baselines; lightweight real policy smoke for B00.
WM_NUM_ENVS="${WM_NUM_ENVS:-128}"
WM_EPISODES="${WM_EPISODES:-1000}"
WM_MAX_STEPS="${WM_MAX_STEPS:-200}"
VAE_STEPS="${VAE_STEPS:-2000}"
RNN_STEPS="${RNN_STEPS:-2000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-512}"
WM_NUM_WORKERS="${WM_NUM_WORKERS:-4}"
WM_EVAL_EVERY="${WM_EVAL_EVERY:-500}"
WM_VIDEO_EVERY="${WM_VIDEO_EVERY:-$WM_EVAL_EVERY}"
WM_HORIZONS="${WM_HORIZONS:-1 5 10 25 50}"
JOINT_SYMBOLIC_TRAIN_EPISODES="${JOINT_SYMBOLIC_TRAIN_EPISODES:-8}"
JOINT_SYMBOLIC_TRAIN_STEPS="${JOINT_SYMBOLIC_TRAIN_STEPS:-8}"
VIDEO_MAX_STEPS="${VIDEO_MAX_STEPS:-100}"
VIDEO_FPS="${VIDEO_FPS:-10}"
SHARED_MODEL_DIR="${SHARED_MODEL_DIR:-shared_models}"
REUSE_VAE_CACHE="${REUSE_VAE_CACHE:-1}"
FORCE_VAE_RETRAIN="${FORCE_VAE_RETRAIN:-0}"
REUSE_LATENT_CACHE="${REUSE_LATENT_CACHE:-1}"
FORCE_LATENT_REENCODE="${FORCE_LATENT_REENCODE:-0}"

VAE_CACHE_ARGS=(--shared-model-dir "$SHARED_MODEL_DIR")
if [[ "$REUSE_VAE_CACHE" == "1" ]]; then
  VAE_CACHE_ARGS+=(--reuse-vae-cache)
else
  VAE_CACHE_ARGS+=(--no-reuse-vae-cache)
fi
if [[ "$FORCE_VAE_RETRAIN" == "1" ]]; then
  VAE_CACHE_ARGS+=(--force-vae-retrain)
fi
if [[ "$REUSE_LATENT_CACHE" == "1" ]]; then
  VAE_CACHE_ARGS+=(--reuse-latent-cache)
else
  VAE_CACHE_ARGS+=(--no-reuse-latent-cache)
fi
if [[ "$FORCE_LATENT_REENCODE" == "1" ]]; then
  VAE_CACHE_ARGS+=(--force-latent-reencode)
fi

# Downstream MARL train/eval. B00 uses real vGridcraft MASAC; every
# model-based baseline uses MAMBPO by default to expose the contribution of a
# learned world model during off-policy MARL.
MODEL_FREE_DOWNSTREAM_ALGO="${MODEL_FREE_DOWNSTREAM_ALGO:-masac}"
MODEL_BASED_DOWNSTREAM_ALGO="${MODEL_BASED_DOWNSTREAM_ALGO:-mambpo}"
case "$MODEL_BASED_DOWNSTREAM_ALGO" in
  mambpo|mb_mappo|mpc_cem|dyna_actor_critic|imagined_mappo)
    ;;
  *)
    echo "Unsupported MODEL_BASED_DOWNSTREAM_ALGO=${MODEL_BASED_DOWNSTREAM_ALGO}; expected mambpo, mb_mappo, mpc_cem, dyna_actor_critic, or imagined_mappo" >&2
    exit 2
    ;;
esac
case "$MODEL_FREE_DOWNSTREAM_ALGO" in
  masac|mappo)
    ;;
  *)
    echo "Unsupported MODEL_FREE_DOWNSTREAM_ALGO=${MODEL_FREE_DOWNSTREAM_ALGO}; expected masac or mappo" >&2
    exit 2
    ;;
esac
MARL_NUM_ENVS="${MARL_NUM_ENVS:-64}"
MARL_MAX_STEPS="${MARL_MAX_STEPS:-200}"
MARL_MAX_ITERS="${MARL_MAX_ITERS:-50}"
MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-2048}"
MARL_TRAIN_BATCH_SIZE="${MARL_TRAIN_BATCH_SIZE:-${MAPPO_MINIBATCH_SIZE:-1024}}"
MARL_OPTIMIZER_STEPS="${MARL_OPTIMIZER_STEPS:-${MAPPO_MINIBATCH_ITERS:-2}}"
MARL_EVAL_EVERY_ITERS="${MARL_EVAL_EVERY_ITERS:-${MAPPO_EVAL_EVERY_ITERS:-25}}"
MARL_EVAL_EPISODES="${MARL_EVAL_EPISODES:-${MAPPO_EVAL_EPISODES:-4}}"
MARL_VIDEO_EVERY_ITERS="${MARL_VIDEO_EVERY_ITERS:-${MAPPO_VIDEO_EVERY_ITERS:-250}}"
MARL_HIDDEN_SIZE="${MARL_HIDDEN_SIZE:-${MAPPO_HIDDEN_SIZE:-256}}"
for legacy_marl_env in MAPPO_MINIBATCH_SIZE MAPPO_MINIBATCH_ITERS MAPPO_EVAL_EVERY_ITERS MAPPO_EVAL_EPISODES MAPPO_VIDEO_EVERY_ITERS MAPPO_HIDDEN_SIZE; do
  if [[ -n "${!legacy_marl_env:-}" ]]; then
    echo "[naming] ${legacy_marl_env} is deprecated for generic MARL settings; prefer MARL_* variables." >&2
  fi
done
MB_WORLD_MODEL_TRAIN_EPOCHS="${MB_WORLD_MODEL_TRAIN_EPOCHS:-5}"
MB_WORLD_MODEL_BATCH_SIZE="${MB_WORLD_MODEL_BATCH_SIZE:-256}"
MB_WORLD_MODEL_HIDDEN_SIZE="${MB_WORLD_MODEL_HIDDEN_SIZE:-256}"
MB_IMAGINED_HORIZON="${MB_IMAGINED_HORIZON:-3}"
MB_IMAGINED_BRANCHES="${MB_IMAGINED_BRANCHES:-4}"
MB_LAMBDA_IMAGINED="${MB_LAMBDA_IMAGINED:-0.5}"
MPC_PLANNING_HORIZON="${MPC_PLANNING_HORIZON:-15}"
MPC_CEM_SAMPLES="${MPC_CEM_SAMPLES:-128}"
MPC_CEM_ITERS="${MPC_CEM_ITERS:-3}"
MPC_CEM_ELITE_FRAC="${MPC_CEM_ELITE_FRAC:-0.2}"
MARL_WANDB_STEP_OFFSET="${MARL_WANDB_STEP_OFFSET:-$((VAE_STEPS + RNN_STEPS + 1000))}"
DYNA_IMAGINED_HORIZON="${DYNA_IMAGINED_HORIZON:-32}"

case "$BASELINE_ID" in
  B00*|*model-free*)
    MODEL_BASED=0
    ;;
  *)
    MODEL_BASED=1
    ;;
esac

print_cmd() {
  printf '[dry-run]'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

if [[ "$MODEL_BASED" == "1" ]]; then
  echo "=== Step 1/2: World Model train/eval (${BASELINE_ID}) ==="
else
  echo "=== Step 1/2: Model-free lightweight real policy smoke (${BASELINE_ID}) ==="
fi
echo "W&B run id: ${WANDB_RUN_ID}"
WM_PHASE="policy"
if [[ "$MODEL_BASED" == "1" ]]; then
  WM_PHASE="world_model"
fi

WM_CMD=(
  "$PYTHON_BIN" run_benchmarl_gridcraft.py
  --baseline-id "$BASELINE_ID"
  --phase "$WM_PHASE"
  --seed "$SEED"
  --num-envs "$WM_NUM_ENVS"
  --num-agents "$NUM_AGENTS"
  --episodes "$WM_EPISODES"
  --max-steps "$WM_MAX_STEPS"
  --vae-steps "$VAE_STEPS"
  --rnn-steps "$RNN_STEPS"
  --wm-batch-size "$WM_BATCH_SIZE"
  --wm-num-workers "$WM_NUM_WORKERS"
  --seq-len 32
  --eval-every "$WM_EVAL_EVERY"
  --video-every "$WM_VIDEO_EVERY"
  --video-max-steps "$VIDEO_MAX_STEPS"
  --video-fps "$VIDEO_FPS"
  --horizons $WM_HORIZONS
  --joint-symbolic-train-episodes "$JOINT_SYMBOLIC_TRAIN_EPISODES"
  --joint-symbolic-train-steps "$JOINT_SYMBOLIC_TRAIN_STEPS"
  --device "$DEVICE"
  --wandb-id "$WANDB_RUN_ID"
  --wandb-name "$RUN_NAME"
  --wandb-group "$WANDB_GROUP"
  "${VAE_CACHE_ARGS[@]}"
)
if [[ -n "${WANDB_FLAG}" ]]; then
  WM_CMD+=($WANDB_FLAG)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  print_cmd "${WM_CMD[@]}"
else
  "${WM_CMD[@]}"
fi

if [[ "$MODEL_BASED" == "1" ]]; then
  if [[ "$MODEL_BASED_DOWNSTREAM_ALGO" == "mambpo" ]]; then
    echo "=== Step 2/2: MAMBPO train/eval on real vGridcraft with generated model rollouts ==="
    MARL_CMD=(
      "$PYTHON_BIN" run_benchmarl_marl_gridcraft.py
      --algorithm mambpo
      --baseline-id "$BASELINE_ID"
      --wm-run-dir "runs_benchmarl/${BASELINE_ID}_a${NUM_AGENTS}_seed${SEED}"
      --seed "$SEED"
      --num-envs "$MARL_NUM_ENVS"
      --num-agents "$NUM_AGENTS"
      --max-steps "$MARL_MAX_STEPS"
      --max-iters "$MARL_MAX_ITERS"
      --frames-per-batch "$MARL_FRAMES_PER_BATCH"
      --marl-train-batch-size "$MARL_TRAIN_BATCH_SIZE"
      --marl-optimizer-steps "$MARL_OPTIMIZER_STEPS"
      --marl-eval-every-iters "$MARL_EVAL_EVERY_ITERS"
      --marl-eval-episodes "$MARL_EVAL_EPISODES"
      --marl-video-every-iters "$MARL_VIDEO_EVERY_ITERS"
      --marl-hidden-size "$MARL_HIDDEN_SIZE"
      --mb-world-model-train-epochs "$MB_WORLD_MODEL_TRAIN_EPOCHS"
      --mb-world-model-batch-size "$MB_WORLD_MODEL_BATCH_SIZE"
      --mb-world-model-hidden-size "$MB_WORLD_MODEL_HIDDEN_SIZE"
      --mb-imagined-horizon "$MB_IMAGINED_HORIZON"
      --mb-imagined-branches "$MB_IMAGINED_BRANCHES"
      --mb-lambda-imagined "$MB_LAMBDA_IMAGINED"
      --device "$DEVICE"
      --wandb-id "$WANDB_RUN_ID"
      --wandb-name "$RUN_NAME"
      --wandb-group "$WANDB_GROUP"
      --wandb-step-offset "$MARL_WANDB_STEP_OFFSET"
      --video-max-steps "$VIDEO_MAX_STEPS"
      --video-fps "$VIDEO_FPS"
    )
  elif [[ "$MODEL_BASED_DOWNSTREAM_ALGO" == "mb_mappo" ]]; then
    echo "=== Step 2/2: Legacy MB-MAPPO train/eval on real vGridcraft with model-based critic targets ==="
    MARL_CMD=(
      "$PYTHON_BIN" run_benchmarl_marl_gridcraft.py
      --algorithm mb_mappo
      --baseline-id "$BASELINE_ID"
      --wm-run-dir "runs_benchmarl/${BASELINE_ID}_a${NUM_AGENTS}_seed${SEED}"
      --seed "$SEED"
      --num-envs "$MARL_NUM_ENVS"
      --num-agents "$NUM_AGENTS"
      --max-steps "$MARL_MAX_STEPS"
      --max-iters "$MARL_MAX_ITERS"
      --frames-per-batch "$MARL_FRAMES_PER_BATCH"
      --marl-train-batch-size "$MARL_TRAIN_BATCH_SIZE"
      --marl-optimizer-steps "$MARL_OPTIMIZER_STEPS"
      --marl-eval-every-iters "$MARL_EVAL_EVERY_ITERS"
      --marl-eval-episodes "$MARL_EVAL_EPISODES"
      --marl-video-every-iters "$MARL_VIDEO_EVERY_ITERS"
      --marl-hidden-size "$MARL_HIDDEN_SIZE"
      --mb-world-model-train-epochs "$MB_WORLD_MODEL_TRAIN_EPOCHS"
      --mb-world-model-batch-size "$MB_WORLD_MODEL_BATCH_SIZE"
      --mb-world-model-hidden-size "$MB_WORLD_MODEL_HIDDEN_SIZE"
      --mb-imagined-horizon "$MB_IMAGINED_HORIZON"
      --mb-imagined-branches "$MB_IMAGINED_BRANCHES"
      --mb-lambda-imagined "$MB_LAMBDA_IMAGINED"
      --device "$DEVICE"
      --wandb-id "$WANDB_RUN_ID"
      --wandb-name "$RUN_NAME"
      --wandb-group "$WANDB_GROUP"
      --wandb-step-offset "$MARL_WANDB_STEP_OFFSET"
      --video-max-steps "$VIDEO_MAX_STEPS"
      --video-fps "$VIDEO_FPS"
    )
  elif [[ "$MODEL_BASED_DOWNSTREAM_ALGO" == "mpc_cem" ]]; then
    echo "=== Step 2/2: MPC-CEM evaluation in real vGridcraft using the trained World Model ==="
    MARL_CMD=(
      "$PYTHON_BIN" run_benchmarl_mpc_cem_gridcraft.py
      --baseline-id "$BASELINE_ID"
      --wm-run-dir "runs_benchmarl/${BASELINE_ID}_a${NUM_AGENTS}_seed${SEED}"
      --seed "$SEED"
      --num-envs "$MARL_NUM_ENVS"
      --num-agents "$NUM_AGENTS"
      --max-steps "$MARL_MAX_STEPS"
      --planning-horizon "$MPC_PLANNING_HORIZON"
      --cem-samples "$MPC_CEM_SAMPLES"
      --cem-iters "$MPC_CEM_ITERS"
      --cem-elite-frac "$MPC_CEM_ELITE_FRAC"
      --device "$DEVICE"
      --wandb-id "$WANDB_RUN_ID"
      --wandb-name "$RUN_NAME"
      --wandb-group "$WANDB_GROUP"
      --wandb-step-offset "$MARL_WANDB_STEP_OFFSET"
      --video-max-steps "$VIDEO_MAX_STEPS"
      --video-fps "$VIDEO_FPS"
    )
  elif [[ "$MODEL_BASED_DOWNSTREAM_ALGO" == "imagined_mappo" ]]; then
    echo "=== Step 2/2: Legacy native BenchMARL MAPPO train/eval inside the trained World Model ==="
    MARL_CMD=(
      "$PYTHON_BIN" run_benchmarl_imagined_mappo_gridcraft.py
      --baseline-id "$BASELINE_ID"
      --wm-run-dir "runs_benchmarl/${BASELINE_ID}_a${NUM_AGENTS}_seed${SEED}"
      --seed "$SEED"
      --num-envs "$MARL_NUM_ENVS"
      --num-agents "$NUM_AGENTS"
      --max-steps "$MARL_MAX_STEPS"
      --max-iters "$MARL_MAX_ITERS"
      --frames-per-batch "$MARL_FRAMES_PER_BATCH"
      --marl-train-batch-size "$MARL_TRAIN_BATCH_SIZE"
      --marl-optimizer-steps "$MARL_OPTIMIZER_STEPS"
      --marl-eval-every-iters "$MARL_EVAL_EVERY_ITERS"
      --marl-eval-episodes "$MARL_EVAL_EPISODES"
      --marl-video-every-iters "$MARL_VIDEO_EVERY_ITERS"
      --marl-hidden-size "$MARL_HIDDEN_SIZE"
      --device "$DEVICE"
      --wandb-id "$WANDB_RUN_ID"
      --wandb-name "$RUN_NAME"
      --wandb-group "$WANDB_GROUP"
      --wandb-step-offset "$MARL_WANDB_STEP_OFFSET"
      --video-max-steps "$VIDEO_MAX_STEPS"
      --video-fps "$VIDEO_FPS"
    )
  else
    echo "=== Step 2/2: Dyna actor-critic train/eval inside the trained World Model only ==="
    MARL_CMD=(
      "$PYTHON_BIN" run_benchmarl_dyna_gridcraft.py
      --baseline-id "$BASELINE_ID"
      --wm-run-dir "runs_benchmarl/${BASELINE_ID}_a${NUM_AGENTS}_seed${SEED}"
      --seed "$SEED"
      --num-envs "$MARL_NUM_ENVS"
      --num-agents "$NUM_AGENTS"
      --max-steps "$MARL_MAX_STEPS"
      --max-iters "$MARL_MAX_ITERS"
      --imagined-horizon "$DYNA_IMAGINED_HORIZON"
      --marl-hidden-size "$MARL_HIDDEN_SIZE"
      --marl-eval-every-iters "$MARL_EVAL_EVERY_ITERS"
      --marl-eval-episodes "$MARL_EVAL_EPISODES"
      --marl-video-every-iters "$MARL_VIDEO_EVERY_ITERS"
      --device "$DEVICE"
      --wandb-id "$WANDB_RUN_ID"
      --wandb-name "$RUN_NAME"
      --wandb-group "$WANDB_GROUP"
      --wandb-step-offset "$MARL_WANDB_STEP_OFFSET"
      --video-max-steps "$VIDEO_MAX_STEPS"
      --video-fps "$VIDEO_FPS"
    )
  fi
  if [[ -n "${WANDB_FLAG}" ]]; then
    MARL_CMD+=($WANDB_FLAG)
  fi
else
  echo "=== Step 2/2: Native BenchMARL ${MODEL_FREE_DOWNSTREAM_ALGO^^} train/eval on real vGridcraft (model-free baseline only) ==="
  MARL_CMD=(
    "$PYTHON_BIN" run_benchmarl_marl_gridcraft.py
    --algorithm "$MODEL_FREE_DOWNSTREAM_ALGO"
    --seed "$SEED"
    --num-envs "$MARL_NUM_ENVS"
    --num-agents "$NUM_AGENTS"
    --max-steps "$MARL_MAX_STEPS"
    --max-iters "$MARL_MAX_ITERS"
    --frames-per-batch "$MARL_FRAMES_PER_BATCH"
    --marl-train-batch-size "$MARL_TRAIN_BATCH_SIZE"
    --marl-optimizer-steps "$MARL_OPTIMIZER_STEPS"
    --marl-eval-every-iters "$MARL_EVAL_EVERY_ITERS"
    --marl-eval-episodes "$MARL_EVAL_EPISODES"
    --marl-video-every-iters "$MARL_VIDEO_EVERY_ITERS"
    --marl-hidden-size "$MARL_HIDDEN_SIZE"
    --device "$DEVICE"
    --wandb-id "$WANDB_RUN_ID"
    --wandb-name "$RUN_NAME"
    --wandb-group "$WANDB_GROUP"
    --wandb-step-offset "$MARL_WANDB_STEP_OFFSET"
    --video-max-steps "$VIDEO_MAX_STEPS"
    --video-fps "$VIDEO_FPS"
  )
  if [[ -n "${WANDB_FLAG}" ]]; then
    MARL_CMD+=($WANDB_FLAG)
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  print_cmd "${MARL_CMD[@]}"
else
  "${MARL_CMD[@]}"
fi

echo "=== Completed full baseline pipeline (${BASELINE_ID}) ==="
