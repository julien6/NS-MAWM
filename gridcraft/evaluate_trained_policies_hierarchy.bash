#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
BASELINE_ID="${BASELINE_ID:-B00_model-free-control}"
SEEDS="${SEEDS:-1 2 3}"
NUM_AGENTS="${NUM_AGENTS:-3}"
DEVICE="${DEVICE:-cuda}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-500}"
EVAL_POLICY_MODES="${EVAL_POLICY_MODES:-deterministic,mode,sampled}"
WANDB_FLAG="${WANDB_FLAG---wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ns-mawm-gridcraft}"
OUT_DIR="${OUT_DIR:-policy_hierarchy_eval}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-runs_benchmarl/native_marl}"

find_latest_checkpoint() {
  local seed="$1"
  local pattern="*seed${seed}*"
  local checkpoint
  checkpoint="$(
    while IFS= read -r -d '' candidate; do
      run_dir="$(dirname "$(dirname "$candidate")")"
      if find "$run_dir" -maxdepth 1 -type f -name '*.json' -print0 \
        | xargs -0 -r grep -l "\"seed_${seed}\"" >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
      fi
    done < <(find "$CHECKPOINT_ROOT" -path "*/checkpoints/checkpoint_*.pt" -print0 2>/dev/null) \
      | xargs -r stat -c '%Y %n' \
      | sort -nr \
      | awk '{sub(/^[^ ]+ /, ""); print; exit}'
  )"
  if [[ -n "$checkpoint" ]]; then
    printf '%s\n' "$checkpoint"
    return 0
  fi
  checkpoint="$(
    find "$CHECKPOINT_ROOT" -path "*/checkpoints/checkpoint_*.pt" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk -v pattern="$pattern" '$0 ~ pattern {sub(/^[^ ]+ /, ""); print; exit}'
  )"
  if [[ -n "$checkpoint" ]]; then
    printf '%s\n' "$checkpoint"
    return 0
  fi
  checkpoint="$(
    find "$CHECKPOINT_ROOT" -path "*/checkpoints/checkpoint_*.pt" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk '{sub(/^[^ ]+ /, ""); print; exit}'
  )"
  if [[ -n "$checkpoint" ]]; then
    printf '%s\n' "$checkpoint"
    return 0
  fi
  return 1
}

echo "Gridcraft trained-policy hierarchy evaluation"
echo "  baseline:   $BASELINE_ID"
echo "  seeds:      $SEEDS"
echo "  agents:     $NUM_AGENTS"
echo "  device:     $DEVICE"
echo "  episodes:   $EVAL_EPISODES"
echo "  max_steps:  $EVAL_MAX_STEPS"
echo "  modes:      $EVAL_POLICY_MODES"
echo "  checkpoint root: $CHECKPOINT_ROOT"
echo

for seed in $SEEDS; do
  explicit_var="CHECKPOINT_SEED_${seed}"
  checkpoint="${!explicit_var:-}"
  if [[ -z "$checkpoint" ]]; then
    if ! checkpoint="$(find_latest_checkpoint "$seed")"; then
      echo "No checkpoint found for seed $seed under $CHECKPOINT_ROOT." >&2
      echo "Set CHECKPOINT_SEED_${seed}=/path/to/checkpoint_*.pt explicitly." >&2
      exit 1
    fi
  fi
  echo "=== Evaluating seed $seed ==="
  echo "checkpoint: $checkpoint"
  "$PYTHON_BIN" evaluate_trained_policies_hierarchy.py \
    --checkpoint "$checkpoint" \
    --baseline-id "$BASELINE_ID" \
    --seed "$seed" \
    --num-agents "$NUM_AGENTS" \
    --episodes "$EVAL_EPISODES" \
    --max-steps "$EVAL_MAX_STEPS" \
    --modes "$EVAL_POLICY_MODES" \
    --device "$DEVICE" \
    --out-dir "${OUT_DIR}/${BASELINE_ID}_a${NUM_AGENTS}_seed${seed}" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "${BASELINE_ID}_a${NUM_AGENTS}_seed${seed}_policy_hierarchy_eval" \
    --wandb-group "${BASELINE_ID}_policy_hierarchy_eval" \
    $WANDB_FLAG
done

echo "=== Completed trained-policy hierarchy evaluation ==="
