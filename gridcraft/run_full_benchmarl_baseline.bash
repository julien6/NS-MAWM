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
EXTERNAL_WM_RUN_DIR="${EXTERNAL_WM_RUN_DIR:-${WM_RUN_DIR:-}}"
WM_EXTERNAL_REUSED=0
WM_EXTERNAL_CHECKPOINT_CHECKSUM=""
WM_REWARD_LOSS_WEIGHT_USER_SET="${WM_REWARD_LOSS_WEIGHT+x}"
WM_DONE_LOSS_WEIGHT_USER_SET="${WM_DONE_LOSS_WEIGHT+x}"
WM_EVENT_LOSS_WEIGHT_USER_SET="${WM_EVENT_LOSS_WEIGHT+x}"

# World-model phase for model-based baselines; lightweight real policy smoke for B00.
WM_NUM_ENVS="${WM_NUM_ENVS:-128}"
WM_EPISODES="${WM_EPISODES:-1000}"
WM_MAX_STEPS="${WM_MAX_STEPS:-200}"
VAE_STEPS="${VAE_STEPS:-2000}"
RNN_STEPS="${RNN_STEPS:-2000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-512}"
WM_NUM_WORKERS="${WM_NUM_WORKERS:-4}"
WM_SEQ_LEN="${WM_SEQ_LEN:-32}"
WORLD_MODEL_ARCH="${WORLD_MODEL_ARCH:-vae_mdn_rnn}"
VAE_Z_SIZE="${VAE_Z_SIZE:-64}"
VAE_HIDDEN_SIZE="${VAE_HIDDEN_SIZE:-512}"
VAE_KL_TOLERANCE="${VAE_KL_TOLERANCE:-0.5}"
RNN_SIZE="${RNN_SIZE:-128}"
RNN_NUM_MIXTURE="${RNN_NUM_MIXTURE:-5}"
WM_MEAN_MSE_WEIGHT="${WM_MEAN_MSE_WEIGHT:-10.0}"
WM_REWARD_LOSS_WEIGHT="${WM_REWARD_LOSS_WEIGHT:-1.0}"
WM_DONE_LOSS_WEIGHT="${WM_DONE_LOSS_WEIGHT:-1.0}"
WM_EVENT_LOSS_WEIGHT="${WM_EVENT_LOSS_WEIGHT:-5.0}"
WM_GRID_EMBED_DIM="${WM_GRID_EMBED_DIM:-32}"
WM_CNN_CHANNELS="${WM_CNN_CHANNELS:-128}"
WM_SELF_HIDDEN_SIZE="${WM_SELF_HIDDEN_SIZE:-128}"
WM_AGENT_HIDDEN_SIZE="${WM_AGENT_HIDDEN_SIZE:-256}"
WM_ATTENTION_HEADS="${WM_ATTENTION_HEADS:-4}"
WM_NUM_ATTENTION_LAYERS="${WM_NUM_ATTENTION_LAYERS:-1}"
WM_TRANSITION_HIDDEN_SIZE="${WM_TRANSITION_HIDDEN_SIZE:-256}"
WM_LEARNING_RATE="${WM_LEARNING_RATE:-0.001}"
LAMBDA_SYM="${LAMBDA_SYM:-1.0}"
LAMBDA_RESIDUAL="${LAMBDA_RESIDUAL:-0.25}"
HPO_RESULTS_DIR="${HPO_RESULTS_DIR:-hpo_results/world_model}"
REUSE_WM_HPO_CONFIG="${REUSE_WM_HPO_CONFIG:-1}"
REQUIRE_WM_HPO="${REQUIRE_WM_HPO:-0}"
REQUIRED_WM_HPO_STAGE="${REQUIRED_WM_HPO_STAGE:-}"
WM_HPO_FAMILY="${WM_HPO_FAMILY:-}"
WM_HPO_CONFIG_REUSED="${WM_HPO_CONFIG_REUSED:-0}"
WM_HPO_CONFIG_PATH="${WM_HPO_CONFIG_PATH:-}"
WM_HPO_SCORE="${WM_HPO_SCORE:-}"
WM_HPO_BEST_RUN_URL="${WM_HPO_BEST_RUN_URL:-}"
WM_HPO_STAGE="${WM_HPO_STAGE:-}"
WM_HPO_DATASET_CHECKSUM="${WM_HPO_DATASET_CHECKSUM:-}"
WM_EVAL_EVERY="${WM_EVAL_EVERY:-500}"
WM_VIDEO_EVERY="${WM_VIDEO_EVERY:-$WM_EVAL_EVERY}"
WM_HORIZONS="${WM_HORIZONS:-1 5 10 25 50}"
ENABLED_PSTR_RULES="${ENABLED_PSTR_RULES:-}"
SYMBOLIC_TRAIN_SAMPLES="${SYMBOLIC_TRAIN_SAMPLES:-512}"
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
MARL_MODEL="${MARL_MODEL:-lstm}"
MARL_HIDDEN_SIZE="${MARL_HIDDEN_SIZE:-${MAPPO_HIDDEN_SIZE:-256}}"
MARL_LSTM_LAYERS="${MARL_LSTM_LAYERS:-1}"
MARL_LSTM_DROPOUT="${MARL_LSTM_DROPOUT:-0.0}"
MARL_LSTM_COMPILE="${MARL_LSTM_COMPILE:-0}"
MARL_LR="${MARL_LR:-0.00005}"
MARL_GAMMA="${MARL_GAMMA:-0.99}"
MARL_POLYAK_TAU="${MARL_POLYAK_TAU:-0.005}"
MARL_ALPHA_INIT="${MARL_ALPHA_INIT:-1.0}"
MARL_DISCRETE_TARGET_ENTROPY_WEIGHT="${MARL_DISCRETE_TARGET_ENTROPY_WEIGHT:-0.2}"
MARL_ENTROPY_PROFILE="${MARL_ENTROPY_PROFILE:-standard}"
MARL_TARGET_ENTROPY_WEIGHT_START="${MARL_TARGET_ENTROPY_WEIGHT_START:-}"
MARL_TARGET_ENTROPY_WEIGHT_END="${MARL_TARGET_ENTROPY_WEIGHT_END:-}"
MARL_ALPHA_INIT_START="${MARL_ALPHA_INIT_START:-}"
MARL_ALPHA_INIT_END="${MARL_ALPHA_INIT_END:-}"
MARL_FINETUNE_ITERS="${MARL_FINETUNE_ITERS:-0}"
MARL_MEMORY_SIZE="${MARL_MEMORY_SIZE:-1000000}"
MARL_HPO_RESULTS_DIR="${MARL_HPO_RESULTS_DIR:-hpo_results/marl}"
REUSE_MARL_HPO_CONFIG="${REUSE_MARL_HPO_CONFIG:-1}"
REQUIRE_MARL_HPO="${REQUIRE_MARL_HPO:-0}"
REQUIRED_MARL_HPO_STAGE="${REQUIRED_MARL_HPO_STAGE:-}"
MARL_HPO_CORE_REUSED="${MARL_HPO_CORE_REUSED:-0}"
MARL_HPO_CORE_SCORE="${MARL_HPO_CORE_SCORE:-}"
MARL_HPO_CORE_CONFIG_PATH="${MARL_HPO_CORE_CONFIG_PATH:-}"
MARL_HPO_IMAGINATION_REUSED="${MARL_HPO_IMAGINATION_REUSED:-0}"
MARL_HPO_IMAGINATION_SCORE="${MARL_HPO_IMAGINATION_SCORE:-}"
MARL_HPO_IMAGINATION_CONFIG_PATH="${MARL_HPO_IMAGINATION_CONFIG_PATH:-}"
MARL_HPO_CORE_STAGE="${MARL_HPO_CORE_STAGE:-}"
MARL_HPO_IMAGINATION_STAGE="${MARL_HPO_IMAGINATION_STAGE:-}"
MARL_HPO_IMAGINATION_CHECKPOINT_CHECKSUM="${MARL_HPO_IMAGINATION_CHECKPOINT_CHECKSUM:-}"
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
MAMBPO_IMAGINATION_MODE="${MAMBPO_IMAGINATION_MODE:-enabled}"
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
if [[ "$BASELINE_ID" == B11_* && "${WORLD_MODEL_ARCH:-vae_mdn_rnn}" == "vae_mdn_rnn" ]]; then
  WORLD_MODEL_ARCH="structured"
fi
if [[ "$BASELINE_ID" == B11_* ]]; then
  if [[ -z "$WM_REWARD_LOSS_WEIGHT_USER_SET" ]]; then
    WM_REWARD_LOSS_WEIGHT="10.0"
  fi
  if [[ -z "$WM_DONE_LOSS_WEIGHT_USER_SET" ]]; then
    WM_DONE_LOSS_WEIGHT="5.0"
  fi
  if [[ -z "$WM_EVENT_LOSS_WEIGHT_USER_SET" ]]; then
    WM_EVENT_LOSS_WEIGHT="5.0"
  fi
fi

if [[ "$MAMBPO_IMAGINATION_MODE" != "enabled" && "$MAMBPO_IMAGINATION_MODE" != "disabled" ]]; then
  echo "Unsupported MAMBPO_IMAGINATION_MODE=${MAMBPO_IMAGINATION_MODE}; expected enabled or disabled" >&2
  exit 2
fi
if [[ "$MAMBPO_IMAGINATION_MODE" == "disabled" ]]; then
  MB_LAMBDA_IMAGINED="0.0"
fi

checkpoint_checksum() {
  local checkpoint_dir="$1"
  "$PYTHON_BIN" - "$checkpoint_dir" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
if (root / "structured_wm.pt").is_file():
    paths = [root / "structured_wm.pt"]
elif (root / "vae.pt").is_file() and (root / "rnn.pt").is_file():
    paths = [root / "vae.pt", root / "rnn.pt"]
else:
    raise SystemExit(2)
payload = ":".join(f"{p.resolve()}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths)
print(hashlib.sha256(payload.encode()).hexdigest())
PY
}

if [[ "$MODEL_BASED" == "1" && -n "$EXTERNAL_WM_RUN_DIR" ]]; then
  EXTERNAL_WM_CHECKPOINT_DIR="${EXTERNAL_WM_RUN_DIR%/}/checkpoints"
  if [[ "$BASELINE_ID" == B11_* || "$WORLD_MODEL_ARCH" == "structured" ]]; then
    if [[ ! -f "${EXTERNAL_WM_CHECKPOINT_DIR}/structured_wm.pt" ]]; then
      echo "EXTERNAL_WM_RUN_DIR=${EXTERNAL_WM_RUN_DIR} does not contain checkpoints/structured_wm.pt" >&2
      exit 2
    fi
  else
    if [[ ! -f "${EXTERNAL_WM_CHECKPOINT_DIR}/vae.pt" || ! -f "${EXTERNAL_WM_CHECKPOINT_DIR}/rnn.pt" ]]; then
      echo "EXTERNAL_WM_RUN_DIR=${EXTERNAL_WM_RUN_DIR} must contain checkpoints/vae.pt and checkpoints/rnn.pt" >&2
      exit 2
    fi
  fi
  WM_EXTERNAL_REUSED=1
  WM_EXTERNAL_CHECKPOINT_CHECKSUM="$(checkpoint_checksum "$EXTERNAL_WM_CHECKPOINT_DIR")"
  echo "[wm-external] reusing checkpoint dir: ${EXTERNAL_WM_CHECKPOINT_DIR} checksum=${WM_EXTERNAL_CHECKPOINT_CHECKSUM}"
fi

if [[ "$MODEL_BASED" == "1" && "$REUSE_WM_HPO_CONFIG" == "1" ]]; then
  HPO_EXPORT_CMD=("$PYTHON_BIN" wm_hpo_registry.py export-env --baseline-id "$BASELINE_ID" --root "$HPO_RESULTS_DIR")
  if [[ "$REQUIRE_WM_HPO" == "1" ]]; then
    HPO_EXPORT_CMD+=(--require)
  fi
  if [[ -n "$REQUIRED_WM_HPO_STAGE" ]]; then
    HPO_EXPORT_CMD+=(--required-stage "$REQUIRED_WM_HPO_STAGE" --num-agents "$NUM_AGENTS")
  fi
  echo "[wm-hpo] checking best HPO config for ${BASELINE_ID} in ${HPO_RESULTS_DIR}"
  if HPO_EXPORT_OUTPUT="$("${HPO_EXPORT_CMD[@]}")"; then
    :
  else
    exit $?
  fi
  eval "$HPO_EXPORT_OUTPUT"
  if [[ "${WM_HPO_CONFIG_REUSED:-0}" == "1" ]]; then
    echo "[wm-hpo] using ${WM_HPO_FAMILY} config: ${WM_HPO_CONFIG_PATH} (score=${WM_HPO_SCORE})"
  fi
fi

if [[ "$REUSE_MARL_HPO_CONFIG" == "1" ]]; then
  MARL_HPO_EXPORT_CMD=("$PYTHON_BIN" marl_hpo_registry.py export-env --baseline-id "$BASELINE_ID" --downstream-algo "$MODEL_BASED_DOWNSTREAM_ALGO" --root "$MARL_HPO_RESULTS_DIR")
  if [[ "$REQUIRE_MARL_HPO" == "1" ]]; then
    MARL_HPO_EXPORT_CMD+=(--require)
  fi
  if [[ -n "$REQUIRED_MARL_HPO_STAGE" ]]; then
    MARL_HPO_EXPORT_CMD+=(--required-stage "$REQUIRED_MARL_HPO_STAGE" --num-agents "$NUM_AGENTS")
  fi
  MARL_HPO_EXPORT_CMD+=(--required-model-type "$MARL_MODEL")
  if [[ "$MAMBPO_IMAGINATION_MODE" == "disabled" ]]; then
    MARL_HPO_EXPORT_CMD+=(--allow-missing-imagination)
  fi
  echo "[marl-hpo] checking best MARL HPO config for ${BASELINE_ID} in ${MARL_HPO_RESULTS_DIR}"
  if MARL_HPO_EXPORT_OUTPUT="$("${MARL_HPO_EXPORT_CMD[@]}")"; then
    :
  else
    exit $?
  fi
  eval "$MARL_HPO_EXPORT_OUTPUT"
  if [[ "${MARL_HPO_CORE_REUSED:-0}" == "1" ]]; then
    echo "[marl-hpo] using masac_core config: ${MARL_HPO_CORE_CONFIG_PATH} (score=${MARL_HPO_CORE_SCORE})"
  fi
  if [[ "${MARL_HPO_IMAGINATION_REUSED:-0}" == "1" ]]; then
    echo "[marl-hpo] using mambpo_imagination config: ${MARL_HPO_IMAGINATION_CONFIG_PATH} (score=${MARL_HPO_IMAGINATION_SCORE})"
  fi
fi
echo "[marl-hpo] strict=${REQUIRE_MARL_HPO} core_reused=${MARL_HPO_CORE_REUSED:-0} imagination_reused=${MARL_HPO_IMAGINATION_REUSED:-0} imagination_mode=${MAMBPO_IMAGINATION_MODE}"

print_cmd() {
  printf '[dry-run]'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

if [[ "$MODEL_BASED" == "1" && "$WM_EXTERNAL_REUSED" == "1" ]]; then
  echo "=== Step 1/2: Reusing external World Model (${BASELINE_ID}) ==="
else
  if [[ "$MODEL_BASED" == "1" ]]; then
  echo "=== Step 1/2: World Model train/eval (${BASELINE_ID}) ==="
  else
  echo "=== Step 1/2: Model-free lightweight real policy smoke (${BASELINE_ID}) ==="
  fi
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
  --seq-len "$WM_SEQ_LEN"
  --world-model-arch "$WORLD_MODEL_ARCH"
  --vae-z-size "$VAE_Z_SIZE"
  --vae-hidden-size "$VAE_HIDDEN_SIZE"
  --vae-kl-tolerance "$VAE_KL_TOLERANCE"
  --rnn-size "$RNN_SIZE"
  --rnn-num-mixture "$RNN_NUM_MIXTURE"
  --learning-rate "$WM_LEARNING_RATE"
  --mean-mse-weight "$WM_MEAN_MSE_WEIGHT"
  --reward-loss-weight "$WM_REWARD_LOSS_WEIGHT"
  --done-loss-weight "$WM_DONE_LOSS_WEIGHT"
  --event-loss-weight "$WM_EVENT_LOSS_WEIGHT"
  --grid-embed-dim "$WM_GRID_EMBED_DIM"
  --cnn-channels "$WM_CNN_CHANNELS"
  --self-hidden-size "$WM_SELF_HIDDEN_SIZE"
  --agent-hidden-size "$WM_AGENT_HIDDEN_SIZE"
  --attention-heads "$WM_ATTENTION_HEADS"
  --num-attention-layers "$WM_NUM_ATTENTION_LAYERS"
  --transition-hidden-size "$WM_TRANSITION_HIDDEN_SIZE"
  --lambda-sym "$LAMBDA_SYM"
  --lambda-residual "$LAMBDA_RESIDUAL"
  --wm-hpo-family "${WM_HPO_FAMILY:-}"
  --wm-hpo-config-reused "${WM_HPO_CONFIG_REUSED:-0}"
  --wm-hpo-config-path "${WM_HPO_CONFIG_PATH:-}"
  --wm-hpo-stage "${WM_HPO_STAGE:-}"
  --wm-hpo-dataset-checksum "${WM_HPO_DATASET_CHECKSUM:-}"
  --eval-every "$WM_EVAL_EVERY"
  --video-every "$WM_VIDEO_EVERY"
  --video-max-steps "$VIDEO_MAX_STEPS"
  --video-fps "$VIDEO_FPS"
  --horizons $WM_HORIZONS
  --symbolic-train-samples "$SYMBOLIC_TRAIN_SAMPLES"
  --joint-symbolic-train-episodes "$JOINT_SYMBOLIC_TRAIN_EPISODES"
  --joint-symbolic-train-steps "$JOINT_SYMBOLIC_TRAIN_STEPS"
  --device "$DEVICE"
  --wandb-id "$WANDB_RUN_ID"
  --wandb-name "$RUN_NAME"
  --wandb-group "$WANDB_GROUP"
  "${VAE_CACHE_ARGS[@]}"
)
if [[ -n "${WM_HPO_SCORE:-}" ]]; then
  WM_CMD+=(--wm-hpo-score "$WM_HPO_SCORE")
fi
if [[ -n "${WM_HPO_BEST_RUN_URL:-}" ]]; then
  WM_CMD+=(--wm-hpo-best-run-url "$WM_HPO_BEST_RUN_URL")
fi
if [[ -n "$ENABLED_PSTR_RULES" ]]; then
  WM_CMD+=(--enabled-pstr-rules $ENABLED_PSTR_RULES)
fi
if [[ -n "${WANDB_FLAG}" ]]; then
  WM_CMD+=($WANDB_FLAG)
fi

if [[ "$WM_EXTERNAL_REUSED" == "1" ]]; then
  echo "[wm-external] skipping World Model training/evaluation because EXTERNAL_WM_RUN_DIR is set"
else
  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "${WM_CMD[@]}"
  else
    "${WM_CMD[@]}"
  fi
fi

if [[ "$MODEL_BASED" == "1" ]]; then
  MARL_WM_RUN_DIR="runs_benchmarl/${BASELINE_ID}_a${NUM_AGENTS}_seed${SEED}"
  if [[ "$WM_EXTERNAL_REUSED" == "1" ]]; then
    MARL_WM_RUN_DIR="$EXTERNAL_WM_RUN_DIR"
  fi
  if [[ "$MODEL_BASED_DOWNSTREAM_ALGO" == "mambpo" ]]; then
    if [[ "$MAMBPO_IMAGINATION_MODE" == "disabled" ]]; then
      echo "=== Step 2/2: MAMBPO train/eval on real vGridcraft with imagination disabled ==="
    else
      echo "=== Step 2/2: MAMBPO train/eval on real vGridcraft with generated model rollouts ==="
    fi
    MARL_CMD=(
      "$PYTHON_BIN" run_benchmarl_marl_gridcraft.py
      --algorithm mambpo
      --baseline-id "$BASELINE_ID"
      --wm-run-dir "$MARL_WM_RUN_DIR"
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
      --marl-model "$MARL_MODEL"
      --marl-hidden-size "$MARL_HIDDEN_SIZE"
      --marl-lstm-layers "$MARL_LSTM_LAYERS"
      --marl-lstm-dropout "$MARL_LSTM_DROPOUT"
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
      --wm-run-dir "$MARL_WM_RUN_DIR"
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
      --marl-model "$MARL_MODEL"
      --marl-hidden-size "$MARL_HIDDEN_SIZE"
      --marl-lstm-layers "$MARL_LSTM_LAYERS"
      --marl-lstm-dropout "$MARL_LSTM_DROPOUT"
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
      --wm-run-dir "$MARL_WM_RUN_DIR"
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
      --wm-run-dir "$MARL_WM_RUN_DIR"
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
      --wm-run-dir "$MARL_WM_RUN_DIR"
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
    --baseline-id "$BASELINE_ID"
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
    --marl-model "$MARL_MODEL"
    --marl-hidden-size "$MARL_HIDDEN_SIZE"
    --marl-lstm-layers "$MARL_LSTM_LAYERS"
    --marl-lstm-dropout "$MARL_LSTM_DROPOUT"
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

if [[ "${MARL_CMD[1]}" == "run_benchmarl_marl_gridcraft.py" ]]; then
  MARL_CMD+=(
    --marl-lr "$MARL_LR"
    --marl-gamma "$MARL_GAMMA"
    --marl-polyak-tau "$MARL_POLYAK_TAU"
    --marl-alpha-init "$MARL_ALPHA_INIT"
    --marl-discrete-target-entropy-weight "$MARL_DISCRETE_TARGET_ENTROPY_WEIGHT"
    --marl-entropy-profile "$MARL_ENTROPY_PROFILE"
    --marl-finetune-iters "$MARL_FINETUNE_ITERS"
    --marl-memory-size "$MARL_MEMORY_SIZE"
    --marl-hpo-core-reused "${MARL_HPO_CORE_REUSED:-0}"
    --marl-hpo-core-config-path "${MARL_HPO_CORE_CONFIG_PATH:-}"
    --marl-hpo-imagination-reused "${MARL_HPO_IMAGINATION_REUSED:-0}"
    --marl-hpo-imagination-config-path "${MARL_HPO_IMAGINATION_CONFIG_PATH:-}"
    --marl-hpo-core-stage "${MARL_HPO_CORE_STAGE:-}"
    --marl-hpo-imagination-stage "${MARL_HPO_IMAGINATION_STAGE:-}"
    --marl-hpo-imagination-checkpoint-checksum "${MARL_HPO_IMAGINATION_CHECKPOINT_CHECKSUM:-}"
    --marl-hpo-strict-mode "$REQUIRE_MARL_HPO"
    --wm-external-reused "$WM_EXTERNAL_REUSED"
    --wm-external-run-dir "${EXTERNAL_WM_RUN_DIR:-}"
    --wm-external-checkpoint-checksum "$WM_EXTERNAL_CHECKPOINT_CHECKSUM"
    --mambpo-imagination-mode "$MAMBPO_IMAGINATION_MODE"
  )
  if [[ "$MARL_LSTM_COMPILE" == "1" ]]; then
    MARL_CMD+=(--marl-lstm-compile)
  fi
  if [[ -n "${MARL_HPO_CORE_SCORE:-}" ]]; then
    MARL_CMD+=(--marl-hpo-core-score "$MARL_HPO_CORE_SCORE")
  fi
  if [[ -n "${MARL_HPO_IMAGINATION_SCORE:-}" ]]; then
    MARL_CMD+=(--marl-hpo-imagination-score "$MARL_HPO_IMAGINATION_SCORE")
  fi
  if [[ -n "$MARL_TARGET_ENTROPY_WEIGHT_START" ]]; then
    MARL_CMD+=(--marl-target-entropy-weight-start "$MARL_TARGET_ENTROPY_WEIGHT_START")
  fi
  if [[ -n "$MARL_TARGET_ENTROPY_WEIGHT_END" ]]; then
    MARL_CMD+=(--marl-target-entropy-weight-end "$MARL_TARGET_ENTROPY_WEIGHT_END")
  fi
  if [[ -n "$MARL_ALPHA_INIT_START" ]]; then
    MARL_CMD+=(--marl-alpha-init-start "$MARL_ALPHA_INIT_START")
  fi
  if [[ -n "$MARL_ALPHA_INIT_END" ]]; then
    MARL_CMD+=(--marl-alpha-init-end "$MARL_ALPHA_INIT_END")
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  print_cmd "${MARL_CMD[@]}"
else
  "${MARL_CMD[@]}"
fi

echo "=== Completed full baseline pipeline (${BASELINE_ID}) ==="
