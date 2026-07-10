#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
BASELINE_ID="${BASELINE_ID:-B00_model-free-control}"
SEEDS="${SEEDS:-1 2 3}"
NUM_AGENTS="${NUM_AGENTS:-3}"
DEVICE="${DEVICE:-cuda}"
WANDB_FLAG="${WANDB_FLAG:---wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ns-mawm-gridcraft}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-runs_benchmarl/native_marl}"
OUT_ROOT="${OUT_ROOT:-runs_benchmarl/entropy_consolidation}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-policy_hierarchy_eval_consolidated}"
DRY_RUN="${DRY_RUN:-0}"

MARL_MODEL="${MARL_MODEL:-lstm}"
MARL_HIDDEN_SIZE="${MARL_HIDDEN_SIZE:-256}"
MARL_LSTM_LAYERS="${MARL_LSTM_LAYERS:-1}"
MARL_LSTM_DROPOUT="${MARL_LSTM_DROPOUT:-0.0}"
MARL_LSTM_COMPILE="${MARL_LSTM_COMPILE:-0}"
MARL_MAX_ITERS="${MARL_MAX_ITERS:-200}"
MARL_NUM_ENVS="${MARL_NUM_ENVS:-256}"
MARL_MAX_STEPS="${MARL_MAX_STEPS:-500}"
MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-4096}"
MARL_TRAIN_BATCH_SIZE="${MARL_TRAIN_BATCH_SIZE:-512}"
MARL_OPTIMIZER_STEPS="${MARL_OPTIMIZER_STEPS:-4}"
MARL_EVAL_EVERY_ITERS="${MARL_EVAL_EVERY_ITERS:-20}"
MARL_EVAL_EPISODES="${MARL_EVAL_EPISODES:-4}"
MARL_VIDEO_EVERY_ITERS="${MARL_VIDEO_EVERY_ITERS:-0}"
MARL_MEMORY_SIZE="${MARL_MEMORY_SIZE:-200000}"
MARL_LR="${MARL_LR:-0.00005}"
MARL_GAMMA="${MARL_GAMMA:-0.99}"
MARL_POLYAK_TAU="${MARL_POLYAK_TAU:-0.005}"
MARL_ALPHA_INIT="${MARL_ALPHA_INIT:-0.03}"
MARL_DISCRETE_TARGET_ENTROPY_WEIGHT="${MARL_DISCRETE_TARGET_ENTROPY_WEIGHT:-0.01}"
MARL_ENTROPY_PROFILE="${MARL_ENTROPY_PROFILE:-low_entropy_finetune}"
MARL_FINETUNE_ITERS="${MARL_FINETUNE_ITERS:-$MARL_MAX_ITERS}"

EVAL_POLICY_MODES="${EVAL_POLICY_MODES:-deterministic,mode,temp_1.0,temp_0.5,temp_0.25,temp_0.1,sampled}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-500}"

print_cmd() {
  printf '[dry-run]'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "$@"
  else
    "$@"
  fi
}

find_checkpoint_for_seed() {
  local seed="$1"
  local explicit_var="CHECKPOINT_SEED_${seed}"
  local explicit="${!explicit_var:-}"
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return 0
  fi
  local checkpoint
  checkpoint="$(
    find "$CHECKPOINT_ROOT" -path '*/checkpoints/checkpoint_*.pt' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk -v seed="seed${seed}" '$0 ~ seed {sub(/^[^ ]+ /, ""); print; exit}'
  )"
  if [[ -n "$checkpoint" ]]; then
    printf '%s\n' "$checkpoint"
    return 0
  fi
  return 1
}

latest_checkpoint_under() {
  local root="$1"
  find "$root" -path '*/checkpoints/checkpoint_*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk '{sub(/^[^ ]+ /, ""); print; exit}'
}

echo "Gridcraft MASAC+LSTM low-entropy consolidation"
echo "  baseline:      ${BASELINE_ID}"
echo "  seeds:         ${SEEDS}"
echo "  checkpoint root: ${CHECKPOINT_ROOT}"
echo "  output root:   ${OUT_ROOT}"
echo "  alpha init:    ${MARL_ALPHA_INIT}"
echo "  entropy weight:${MARL_DISCRETE_TARGET_ENTROPY_WEIGHT}"
echo "  eval modes:    ${EVAL_POLICY_MODES}"

for seed in $SEEDS; do
  if ! input_checkpoint="$(find_checkpoint_for_seed "$seed")"; then
    echo "No input checkpoint found for seed ${seed} under ${CHECKPOINT_ROOT}." >&2
    echo "Set CHECKPOINT_SEED_${seed}=/path/to/checkpoint_*.pt explicitly." >&2
    exit 1
  fi

  run_name="${BASELINE_ID}_a${NUM_AGENTS}_seed${seed}_low_entropy_consolidation"
  run_id="${WANDB_RUN_ID:-${run_name}_$(date +%Y%m%d_%H%M%S)}"
  save_folder="${OUT_ROOT}/${BASELINE_ID}_a${NUM_AGENTS}_seed${seed}"
  echo "=== Consolidating seed=${seed} from ${input_checkpoint} ==="
  marl_cmd=(
    "$PYTHON_BIN" run_benchmarl_marl_gridcraft.py
    --algorithm masac \
    --baseline-id "$BASELINE_ID" \
    --restore-marl-checkpoint "$input_checkpoint" \
    --seed "$seed" \
    --num-envs "$MARL_NUM_ENVS" \
    --num-agents "$NUM_AGENTS" \
    --max-steps "$MARL_MAX_STEPS" \
    --max-iters "$MARL_MAX_ITERS" \
    --frames-per-batch "$MARL_FRAMES_PER_BATCH" \
    --marl-train-batch-size "$MARL_TRAIN_BATCH_SIZE" \
    --marl-optimizer-steps "$MARL_OPTIMIZER_STEPS" \
    --marl-eval-every-iters "$MARL_EVAL_EVERY_ITERS" \
    --marl-eval-episodes "$MARL_EVAL_EPISODES" \
    --marl-video-every-iters "$MARL_VIDEO_EVERY_ITERS" \
    --marl-model "$MARL_MODEL" \
    --marl-hidden-size "$MARL_HIDDEN_SIZE" \
    --marl-lstm-layers "$MARL_LSTM_LAYERS" \
    --marl-lstm-dropout "$MARL_LSTM_DROPOUT" \
    --marl-lr "$MARL_LR" \
    --marl-gamma "$MARL_GAMMA" \
    --marl-polyak-tau "$MARL_POLYAK_TAU" \
    --marl-alpha-init "$MARL_ALPHA_INIT" \
    --marl-discrete-target-entropy-weight "$MARL_DISCRETE_TARGET_ENTROPY_WEIGHT" \
    --marl-entropy-profile "$MARL_ENTROPY_PROFILE" \
    --marl-finetune-iters "$MARL_FINETUNE_ITERS" \
    --marl-memory-size "$MARL_MEMORY_SIZE" \
    --device "$DEVICE" \
    --save-folder "$save_folder" \
    --wandb-id "$run_id" \
    --wandb-name "$run_name" \
    --wandb-group "${BASELINE_ID}_low_entropy_consolidation" \
    --wandb-project "$WANDB_PROJECT" \
    --no-wandb-videos
  )
  if [[ "$MARL_LSTM_COMPILE" == "1" ]]; then
    marl_cmd+=(--marl-lstm-compile)
  fi
  if [[ -n "$WANDB_FLAG" ]]; then
    marl_cmd+=($WANDB_FLAG)
  fi
  run_cmd "${marl_cmd[@]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    consolidated_checkpoint="<dry-run-consolidated-checkpoint-seed-${seed}>"
  else
    consolidated_checkpoint="$(latest_checkpoint_under "$save_folder")"
    if [[ -z "$consolidated_checkpoint" ]]; then
      echo "No consolidated checkpoint found under ${save_folder}." >&2
      exit 2
    fi
  fi
  echo "=== Evaluating consolidated seed=${seed}: ${consolidated_checkpoint} ==="
  run_cmd env "CHECKPOINT_SEED_${seed}=$consolidated_checkpoint" \
    BASELINE_ID="$BASELINE_ID" \
    SEEDS="$seed" \
    NUM_AGENTS="$NUM_AGENTS" \
    DEVICE="$DEVICE" \
    EVAL_POLICY_MODES="$EVAL_POLICY_MODES" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_MAX_STEPS="$EVAL_MAX_STEPS" \
    OUT_DIR="$EVAL_OUT_DIR" \
    WANDB_FLAG="$WANDB_FLAG" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    ./evaluate_trained_policies_hierarchy.bash
done

echo "=== Completed MASAC+LSTM low-entropy consolidation ==="
