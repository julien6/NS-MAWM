#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
if [[ "${AUTO_RESOURCE_PROFILE:-0}" == "1" && "${RESOURCE_PROFILE_APPLIED:-0}" != "1" ]]; then
  RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}"
  echo "[resource-profile] applying ${RESOURCE_PROFILE} to B00/B11 lambda=0.2 final comparison"
  eval "$("$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target campaign --format shell)"
  "$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target campaign --format summary >&2 || true
  export RESOURCE_PROFILE_APPLIED=1
fi

NUM_AGENTS="${NUM_AGENTS:-3}"
SEEDS="${SEEDS:-1 2 3}"
B11_LAMBDA="${B11_LAMBDA:-0.2}"
B11_BASELINE_ID="${B11_BASELINE_ID:-B11_structured_neural_k0.0_lambda_0p2}"
RUN_B00="${RUN_B00:-0}"
RUN_B11="${RUN_B11:-1}"
RUN_POLICY_EVAL="${RUN_POLICY_EVAL:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
WANDB_FLAG="${WANDB_FLAG:---wandb}"
DRY_RUN="${DRY_RUN:-0}"
HPO_RESULTS_DIR="${HPO_RESULTS_DIR:-hpo_results/world_model}"
MARL_HPO_RESULTS_DIR="${MARL_HPO_RESULTS_DIR:-hpo_results/marl}"
COMPARISON_ID="${COMPARISON_ID:-b00_b11_lambda02_final_$(date -u +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-${COMPARISON_ID}}"

MARL_MAX_ITERS="${MARL_MAX_ITERS:-500}"
MARL_NUM_ENVS="${MARL_NUM_ENVS:-256}"
MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-4096}"
MARL_TRAIN_BATCH_SIZE="${MARL_TRAIN_BATCH_SIZE:-512}"
MARL_MEMORY_SIZE="${MARL_MEMORY_SIZE:-200000}"
MARL_EVAL_EVERY_ITERS="${MARL_EVAL_EVERY_ITERS:-20}"
MARL_EVAL_EPISODES="${MARL_EVAL_EPISODES:-4}"
MARL_VIDEO_EVERY_ITERS="${MARL_VIDEO_EVERY_ITERS:-0}"

EVAL_POLICY_MODES="${EVAL_POLICY_MODES:-deterministic,mode,temp_1.0,temp_0.5,temp_0.25,temp_0.1,sampled}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-500}"
ANALYSIS_OUT_DIR="${ANALYSIS_OUT_DIR:-analysis_b00_b11_lambda02_final_comparison}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_file() {
  local path="$1"
  local message="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ ! -f "$path" ]]; then
      echo "[dry-run] ${message}" >&2
    fi
    return 0
  fi
  if [[ ! -f "$path" ]]; then
    echo "$message" >&2
    exit 2
  fi
}

EXTERNAL_WM_RUN_DIR="${EXTERNAL_WM_RUN_DIR:-}"
if [[ -z "$EXTERNAL_WM_RUN_DIR" ]]; then
  if [[ "$DRY_RUN" == "1" && ! -f "${HPO_RESULTS_DIR}/structured_neural_k0.0/best_config.json" ]]; then
    EXTERNAL_WM_RUN_DIR="runs_benchmarl_hpo/structured_neural_k0.0/DRY_RUN_B11_structured_neural_k0.0_a${NUM_AGENTS}_seed1"
  else
    EXTERNAL_WM_RUN_DIR="$("$PYTHON_BIN" - "$HPO_RESULTS_DIR" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "structured_neural_k0.0" / "best_config.json"
if not path.is_file():
    raise SystemExit(f"Missing {path}")
checkpoint_dir = Path(json.loads(path.read_text())["checkpoint_dir"])
print(checkpoint_dir.parent)
PY
)"
  fi
fi

require_file "${EXTERNAL_WM_RUN_DIR%/}/checkpoints/structured_wm.pt" \
  "EXTERNAL_WM_RUN_DIR=${EXTERNAL_WM_RUN_DIR} does not contain checkpoints/structured_wm.pt"
require_file "${MARL_HPO_RESULTS_DIR}/masac_core/best_config.json" \
  "Missing ${MARL_HPO_RESULTS_DIR}/masac_core/best_config.json; run final MASAC+LSTM HPO first."

echo "B00 vs B11 StructuredWM+MAMBPO+LSTM final comparison"
echo "  seeds:        ${SEEDS}"
echo "  agents:       ${NUM_AGENTS}"
echo "  B11 lambda:   ${B11_LAMBDA}"
echo "  B11 baseline: ${B11_BASELINE_ID}"
echo "  comparison:   ${COMPARISON_ID}"
echo "  external WM:  ${EXTERNAL_WM_RUN_DIR}"
echo "  run B00:      ${RUN_B00}"
echo "  run B11:      ${RUN_B11}"
echo "  policy eval:  ${RUN_POLICY_EVAL}"
echo "  analysis:     ${RUN_ANALYSIS}"
echo "  dry-run:      ${DRY_RUN}"
echo

if [[ "$RUN_B00" == "1" ]]; then
  echo "=== Final B00 MASAC+LSTM runs ==="
  for seed in $SEEDS; do
    echo "--- B00 seed ${seed} ---"
    run_cmd env \
      AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" \
      RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      BASELINE_ID="B00_model-free-control" \
      SEED="$seed" \
      NUM_AGENTS="$NUM_AGENTS" \
      MODEL_FREE_DOWNSTREAM_ALGO=masac \
      MARL_MODEL=lstm \
      REUSE_MARL_HPO_CONFIG=1 \
      REQUIRE_MARL_HPO=1 \
      REQUIRED_MARL_HPO_STAGE=final \
      MARL_MAX_ITERS="$MARL_MAX_ITERS" \
      MARL_NUM_ENVS="$MARL_NUM_ENVS" \
      MARL_FRAMES_PER_BATCH="$MARL_FRAMES_PER_BATCH" \
      MARL_TRAIN_BATCH_SIZE="$MARL_TRAIN_BATCH_SIZE" \
      MARL_MEMORY_SIZE="$MARL_MEMORY_SIZE" \
      MARL_EVAL_EVERY_ITERS="$MARL_EVAL_EVERY_ITERS" \
      MARL_EVAL_EPISODES="$MARL_EVAL_EPISODES" \
      MARL_VIDEO_EVERY_ITERS="$MARL_VIDEO_EVERY_ITERS" \
      COMPARISON_ID="$COMPARISON_ID" \
      WANDB_GROUP="$WANDB_GROUP" \
      WANDB_FLAG="$WANDB_FLAG" \
      ./run_full_benchmarl_baseline.bash
  done
else
  echo "[comparison] skipping B00 training; existing B00 checkpoints/evals will be reused for analysis."
fi

if [[ "$RUN_B11" == "1" ]]; then
  echo "=== Final B11 StructuredWM+MAMBPO+LSTM lambda=${B11_LAMBDA} runs ==="
  for seed in $SEEDS; do
    echo "--- B11 seed ${seed} ---"
    run_cmd env \
      AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" \
      RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      BASELINE_ID="$B11_BASELINE_ID" \
      SEED="$seed" \
      NUM_AGENTS="$NUM_AGENTS" \
      MODEL_BASED_DOWNSTREAM_ALGO=mambpo \
      MARL_MODEL=lstm \
      EXTERNAL_WM_RUN_DIR="$EXTERNAL_WM_RUN_DIR" \
      MAMBPO_IMAGINATION_MODE=enabled \
      MB_LAMBDA_IMAGINED="$B11_LAMBDA" \
      REUSE_WM_HPO_CONFIG=1 \
      REUSE_MARL_HPO_CONFIG=1 \
      REUSE_MAMBPO_IMAGINATION_HPO=0 \
      HPO_RESULTS_DIR="$HPO_RESULTS_DIR" \
      MARL_HPO_RESULTS_DIR="$MARL_HPO_RESULTS_DIR" \
      REQUIRE_WM_HPO=1 \
      REQUIRE_MARL_HPO=1 \
      REQUIRED_WM_HPO_STAGE=final \
      REQUIRED_MARL_HPO_STAGE=final \
      MARL_MAX_ITERS="$MARL_MAX_ITERS" \
      MARL_NUM_ENVS="$MARL_NUM_ENVS" \
      MARL_FRAMES_PER_BATCH="$MARL_FRAMES_PER_BATCH" \
      MARL_TRAIN_BATCH_SIZE="$MARL_TRAIN_BATCH_SIZE" \
      MARL_MEMORY_SIZE="$MARL_MEMORY_SIZE" \
      MARL_EVAL_EVERY_ITERS="$MARL_EVAL_EVERY_ITERS" \
      MARL_EVAL_EPISODES="$MARL_EVAL_EPISODES" \
      MARL_VIDEO_EVERY_ITERS="$MARL_VIDEO_EVERY_ITERS" \
      COMPARISON_ID="$COMPARISON_ID" \
      WANDB_GROUP="$WANDB_GROUP" \
      WANDB_FLAG="$WANDB_FLAG" \
      ./run_full_benchmarl_baseline.bash
  done
else
  echo "[comparison] skipping B11 training; existing B11 lambda=0.2 checkpoints/evals will be reused for analysis."
fi

if [[ "$RUN_POLICY_EVAL" == "1" ]]; then
  echo "=== Posthoc B00 policy hierarchy evaluation ==="
  run_cmd env \
    BASELINE_ID="B00_model-free-control" \
    SEEDS="$SEEDS" \
    NUM_AGENTS="$NUM_AGENTS" \
    MARL_MODEL=lstm \
    EVAL_POLICY_MODES="$EVAL_POLICY_MODES" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_MAX_STEPS="$EVAL_MAX_STEPS" \
    COMPARISON_ID="$COMPARISON_ID" \
    WANDB_FLAG="$WANDB_FLAG" \
    ./evaluate_trained_policies_hierarchy.bash

  echo "=== Posthoc B11 lambda=0.2 policy hierarchy evaluation ==="
  run_cmd env \
    SEEDS="$SEEDS" \
    LAMBDA_VALUES="$B11_LAMBDA" \
    NUM_AGENTS="$NUM_AGENTS" \
    MARL_MODEL=lstm \
    EVAL_POLICY_MODES="$EVAL_POLICY_MODES" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_MAX_STEPS="$EVAL_MAX_STEPS" \
    COMPARISON_ID="$COMPARISON_ID" \
    WANDB_FLAG="$WANDB_FLAG" \
    ./evaluate_b11_mambpo_lambda_ablation.bash
else
  echo "[comparison] posthoc hierarchy evaluation disabled."
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  echo "=== Analysis B00 vs B11 lambda=0.2 ==="
  run_cmd "$PYTHON_BIN" analyze_b00_b10_lstm_comparison.py \
    --eval-root policy_hierarchy_eval \
    --baselines "B00_model-free-control,${B11_BASELINE_ID}" \
    --comparison-id "$COMPARISON_ID" \
    --latest-per-baseline-seed \
    --require-seeds 3 \
    --main-mode "temp_0.5" \
    --out-dir "$ANALYSIS_OUT_DIR"

  echo "Analysis written to ${ANALYSIS_OUT_DIR}"
else
  echo "[comparison] analysis disabled."
fi

echo "=== Completed B00 vs B11 lambda=0.2 final comparison ==="
