#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
if [[ "${AUTO_RESOURCE_PROFILE:-0}" == "1" && "${RESOURCE_PROFILE_APPLIED:-0}" != "1" ]]; then
  RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}"
  echo "[resource-profile] applying ${RESOURCE_PROFILE} to final B00/B11 scientific baselines"
  eval "$("$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target campaign --format shell)"
  "$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target campaign --format summary >&2 || true
  export RESOURCE_PROFILE_APPLIED=1
fi

NUM_AGENTS="${NUM_AGENTS:-3}"
SEEDS="${SEEDS:-1 2 3}"
COMPARISON_ID="${COMPARISON_ID:-b00_b11_final_$(date -u +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-${COMPARISON_ID}}"
WANDB_FLAG="${WANDB_FLAG:---wandb}"
DRY_RUN="${DRY_RUN:-0}"

RUN_HPO_IF_MISSING="${RUN_HPO_IF_MISSING:-0}"
RUN_B00="${RUN_B00:-1}"
RUN_B11="${RUN_B11:-1}"
RUN_POLICY_EVAL="${RUN_POLICY_EVAL:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

HPO_RESULTS_DIR="${HPO_RESULTS_DIR:-hpo_results/world_model}"
MARL_HPO_RESULTS_DIR="${MARL_HPO_RESULTS_DIR:-hpo_results/marl}"
REQUIRED_WM_HPO_STAGE="${REQUIRED_WM_HPO_STAGE:-final}"
REQUIRED_MARL_HPO_STAGE="${REQUIRED_MARL_HPO_STAGE:-final}"

B00_BASELINE_ID="${B00_BASELINE_ID:-B00_model-free-control}"
B11_LAMBDA="${B11_LAMBDA:-0.2}"
B11_LAMBDA_SUFFIX="${B11_LAMBDA_SUFFIX:-0p2}"
B11_BASELINE_ID="${B11_BASELINE_ID:-B11_structured_neural_k0.0_lambda_${B11_LAMBDA_SUFFIX}}"
B11_USE_MAMBPO_HPO="${B11_USE_MAMBPO_HPO:-0}"

MARL_MODEL="${MARL_MODEL:-lstm}"
MARL_MAX_ITERS="${MARL_MAX_ITERS:-500}"
MARL_NUM_ENVS="${MARL_NUM_ENVS:-256}"
MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-4096}"
MARL_TRAIN_BATCH_SIZE="${MARL_TRAIN_BATCH_SIZE:-512}"
MARL_MEMORY_SIZE="${MARL_MEMORY_SIZE:-200000}"
MARL_EVAL_EVERY_ITERS="${MARL_EVAL_EVERY_ITERS:-20}"
MARL_EVAL_EPISODES="${MARL_EVAL_EPISODES:-4}"
MARL_VIDEO_EVERY_ITERS="${MARL_VIDEO_EVERY_ITERS:-100}"

WM_VIDEO_EVERY="${WM_VIDEO_EVERY:-5000}"
VIDEO_MAX_STEPS="${VIDEO_MAX_STEPS:-100}"
VIDEO_FPS="${VIDEO_FPS:-10}"

EVAL_POLICY_MODES="${EVAL_POLICY_MODES:-deterministic,mode,temp_1.0,temp_0.5,temp_0.25,temp_0.1,sampled}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-500}"
ANALYSIS_OUT_DIR="${ANALYSIS_OUT_DIR:-analysis_final_b00_b11_scientific_baselines}"

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

validate_wm_hpo() {
  "$PYTHON_BIN" wm_hpo_registry.py validate \
    --hpo-family structured_neural_k0.0 \
    --root "$HPO_RESULTS_DIR" \
    --required-stage "$REQUIRED_WM_HPO_STAGE" \
    --num-agents "$NUM_AGENTS" \
    --require-checkpoints >/dev/null 2>&1
}

validate_marl_hpo() {
  local family="$1"
  shift
  "$PYTHON_BIN" marl_hpo_registry.py validate \
    --family "$family" \
    --root "$MARL_HPO_RESULTS_DIR" \
    --required-stage "$REQUIRED_MARL_HPO_STAGE" \
    --num-agents "$NUM_AGENTS" \
    --required-model-type "$MARL_MODEL" \
    "$@" >/dev/null 2>&1
}

run_missing_hpo_if_requested() {
  if validate_wm_hpo; then
    echo "[preflight] structured_neural_k0.0 WM HPO ${REQUIRED_WM_HPO_STAGE} config is valid."
  elif [[ "$RUN_HPO_IF_MISSING" == "1" ]]; then
    echo "[preflight] structured WM HPO is missing/invalid; running screen -> promote -> final."
    run_cmd env AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      HPO_STAGE=screen HPO_FAMILIES="structured_neural_k0.0" ./run_world_model_hpo_pipeline.bash
    run_cmd env AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      HPO_STAGE=promote HPO_FAMILIES="structured_neural_k0.0" ./run_world_model_hpo_pipeline.bash
    run_cmd env AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      HPO_STAGE=final HPO_FAMILIES="structured_neural_k0.0" HPO_SEEDS="$SEEDS" HPO_VIDEO_EVERY="$WM_VIDEO_EVERY" ./run_world_model_hpo_pipeline.bash
  elif [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] Missing or invalid final StructuredWM HPO config; continuing to print commands."
  else
    echo "Missing or invalid final StructuredWM HPO config. Run with RUN_HPO_IF_MISSING=1 or launch the WM HPO pipeline first." >&2
    exit 2
  fi

  if [[ -z "${EXTERNAL_WM_RUN_DIR:-}" && "$DRY_RUN" == "1" && ! -f "${HPO_RESULTS_DIR}/structured_neural_k0.0/best_config.json" ]]; then
    EXTERNAL_WM_RUN_DIR="runs_benchmarl_hpo/structured_neural_k0.0/DRY_RUN_B11_structured_neural_k0.0_a${NUM_AGENTS}_seed1"
  fi
  EXTERNAL_WM_RUN_DIR="${EXTERNAL_WM_RUN_DIR:-$("$PYTHON_BIN" - "$HPO_RESULTS_DIR" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1]) / "structured_neural_k0.0" / "best_config.json"
payload = json.loads(path.read_text())
print(Path(payload["checkpoint_dir"]).parent)
PY
)}"
  export EXTERNAL_WM_RUN_DIR
  require_file "${EXTERNAL_WM_RUN_DIR%/}/checkpoints/structured_wm.pt" \
    "EXTERNAL_WM_RUN_DIR=${EXTERNAL_WM_RUN_DIR} does not contain checkpoints/structured_wm.pt"

  if validate_marl_hpo masac_core; then
    echo "[preflight] masac_core MARL HPO ${REQUIRED_MARL_HPO_STAGE} ${MARL_MODEL} config is valid."
  elif [[ "$RUN_HPO_IF_MISSING" == "1" ]]; then
    echo "[preflight] MASAC-core HPO is missing/invalid; running screen -> promote -> final."
    run_cmd env AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      MARL_HPO_STAGE=screen MARL_HPO_FAMILIES="masac_core" MARL_HPO_MODEL="$MARL_MODEL" ./run_marl_hpo_pipeline.bash
    run_cmd env AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      MARL_HPO_STAGE=promote MARL_HPO_FAMILIES="masac_core" MARL_HPO_MODEL="$MARL_MODEL" ./run_marl_hpo_pipeline.bash
    run_cmd env AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      MARL_HPO_STAGE=final MARL_HPO_FAMILIES="masac_core" MARL_HPO_MODEL="$MARL_MODEL" MARL_HPO_SEEDS="$SEEDS" ./run_marl_hpo_pipeline.bash
  elif [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] Missing or invalid final masac_core MARL HPO config for ${MARL_MODEL}; continuing to print commands."
  else
    echo "Missing or invalid final masac_core MARL HPO config for ${MARL_MODEL}. Run with RUN_HPO_IF_MISSING=1 or launch MARL HPO first." >&2
    exit 2
  fi

  if [[ "$B11_USE_MAMBPO_HPO" == "1" ]]; then
    if validate_marl_hpo mambpo_imagination --external-checkpoint-dir "${EXTERNAL_WM_RUN_DIR%/}/checkpoints"; then
      echo "[preflight] mambpo_imagination MARL HPO ${REQUIRED_MARL_HPO_STAGE} config is valid for the selected WM."
    elif [[ "$RUN_HPO_IF_MISSING" == "1" ]]; then
      echo "[preflight] MAMBPO-imagination HPO is missing/invalid; running screen -> promote -> final."
      run_cmd env AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
        MARL_HPO_STAGE=screen MARL_HPO_FAMILIES="mambpo_imagination" MARL_HPO_BASELINE_ID="B11_structured_neural_k0.0" \
        MARL_HPO_WM_RUN_DIR="$EXTERNAL_WM_RUN_DIR" MARL_HPO_MODEL="$MARL_MODEL" ./run_marl_hpo_pipeline.bash
      run_cmd env AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
        MARL_HPO_STAGE=promote MARL_HPO_FAMILIES="mambpo_imagination" MARL_HPO_BASELINE_ID="B11_structured_neural_k0.0" \
        MARL_HPO_WM_RUN_DIR="$EXTERNAL_WM_RUN_DIR" MARL_HPO_MODEL="$MARL_MODEL" ./run_marl_hpo_pipeline.bash
      run_cmd env AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
        MARL_HPO_STAGE=final MARL_HPO_FAMILIES="mambpo_imagination" MARL_HPO_BASELINE_ID="B11_structured_neural_k0.0" \
        MARL_HPO_WM_RUN_DIR="$EXTERNAL_WM_RUN_DIR" MARL_HPO_MODEL="$MARL_MODEL" MARL_HPO_SEEDS="$SEEDS" ./run_marl_hpo_pipeline.bash
    elif [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] Missing or invalid final mambpo_imagination HPO config for this WM; continuing to print commands."
    else
      echo "Missing or invalid final mambpo_imagination HPO config for this WM. Set B11_USE_MAMBPO_HPO=0 to use fixed B11_LAMBDA=${B11_LAMBDA}." >&2
      exit 2
    fi
  else
    echo "[preflight] B11 will use fixed MB_LAMBDA_IMAGINED=${B11_LAMBDA} from the lambda ablation."
  fi
}

run_missing_hpo_if_requested

echo "Final scientific Gridcraft baselines"
echo "  comparison:   ${COMPARISON_ID}"
echo "  seeds:        ${SEEDS}"
echo "  agents:       ${NUM_AGENTS}"
echo "  B00:          ${B00_BASELINE_ID} MASAC+${MARL_MODEL}"
echo "  B11:          ${B11_BASELINE_ID} StructuredWM+MAMBPO+${MARL_MODEL}"
echo "  B11 lambda:   ${B11_LAMBDA} (use HPO=${B11_USE_MAMBPO_HPO})"
echo "  external WM:  ${EXTERNAL_WM_RUN_DIR}"
echo "  MARL budget:  iters=${MARL_MAX_ITERS}, envs=${MARL_NUM_ENVS}, frames/batch=${MARL_FRAMES_PER_BATCH}, train_batch=${MARL_TRAIN_BATCH_SIZE}"
echo "  videos:       WM every ${WM_VIDEO_EVERY}, MARL every ${MARL_VIDEO_EVERY_ITERS}"
echo "  dry-run:      ${DRY_RUN}"
echo

if [[ "$RUN_B00" == "1" ]]; then
  echo "=== Baseline 1/2: B00 MASAC+LSTM model-free ==="
  for seed in $SEEDS; do
    run_cmd env \
      AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" \
      RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      BASELINE_ID="$B00_BASELINE_ID" \
      SEED="$seed" \
      NUM_AGENTS="$NUM_AGENTS" \
      MODEL_FREE_DOWNSTREAM_ALGO=masac \
      MARL_MODEL="$MARL_MODEL" \
      REUSE_MARL_HPO_CONFIG=1 \
      REQUIRE_MARL_HPO=1 \
      REQUIRED_MARL_HPO_STAGE="$REQUIRED_MARL_HPO_STAGE" \
      MARL_MAX_ITERS="$MARL_MAX_ITERS" \
      MARL_NUM_ENVS="$MARL_NUM_ENVS" \
      MARL_FRAMES_PER_BATCH="$MARL_FRAMES_PER_BATCH" \
      MARL_TRAIN_BATCH_SIZE="$MARL_TRAIN_BATCH_SIZE" \
      MARL_MEMORY_SIZE="$MARL_MEMORY_SIZE" \
      MARL_EVAL_EVERY_ITERS="$MARL_EVAL_EVERY_ITERS" \
      MARL_EVAL_EPISODES="$MARL_EVAL_EPISODES" \
      MARL_VIDEO_EVERY_ITERS="$MARL_VIDEO_EVERY_ITERS" \
      VIDEO_MAX_STEPS="$VIDEO_MAX_STEPS" \
      VIDEO_FPS="$VIDEO_FPS" \
      COMPARISON_ID="$COMPARISON_ID" \
      WANDB_GROUP="$WANDB_GROUP" \
      WANDB_FLAG="$WANDB_FLAG" \
      ./run_full_benchmarl_baseline.bash
  done
else
  echo "[campaign] skipping B00 training."
fi

if [[ "$RUN_B11" == "1" ]]; then
  echo "=== Baseline 2/2: B11 StructuredWM+MAMBPO+LSTM model-based ==="
  for seed in $SEEDS; do
    run_cmd env \
      AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" \
      RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
      BASELINE_ID="$B11_BASELINE_ID" \
      SEED="$seed" \
      NUM_AGENTS="$NUM_AGENTS" \
      MODEL_BASED_DOWNSTREAM_ALGO=mambpo \
      MARL_MODEL="$MARL_MODEL" \
      EXTERNAL_WM_RUN_DIR="$EXTERNAL_WM_RUN_DIR" \
      MAMBPO_IMAGINATION_MODE=enabled \
      MB_LAMBDA_IMAGINED="$B11_LAMBDA" \
      REUSE_WM_HPO_CONFIG=1 \
      REUSE_MARL_HPO_CONFIG=1 \
      REUSE_MAMBPO_IMAGINATION_HPO="$B11_USE_MAMBPO_HPO" \
      REQUIRE_WM_HPO=1 \
      REQUIRE_MARL_HPO=1 \
      REQUIRED_WM_HPO_STAGE="$REQUIRED_WM_HPO_STAGE" \
      REQUIRED_MARL_HPO_STAGE="$REQUIRED_MARL_HPO_STAGE" \
      WM_VIDEO_EVERY="$WM_VIDEO_EVERY" \
      MARL_MAX_ITERS="$MARL_MAX_ITERS" \
      MARL_NUM_ENVS="$MARL_NUM_ENVS" \
      MARL_FRAMES_PER_BATCH="$MARL_FRAMES_PER_BATCH" \
      MARL_TRAIN_BATCH_SIZE="$MARL_TRAIN_BATCH_SIZE" \
      MARL_MEMORY_SIZE="$MARL_MEMORY_SIZE" \
      MARL_EVAL_EVERY_ITERS="$MARL_EVAL_EVERY_ITERS" \
      MARL_EVAL_EPISODES="$MARL_EVAL_EPISODES" \
      MARL_VIDEO_EVERY_ITERS="$MARL_VIDEO_EVERY_ITERS" \
      VIDEO_MAX_STEPS="$VIDEO_MAX_STEPS" \
      VIDEO_FPS="$VIDEO_FPS" \
      COMPARISON_ID="$COMPARISON_ID" \
      WANDB_GROUP="$WANDB_GROUP" \
      WANDB_FLAG="$WANDB_FLAG" \
      ./run_full_benchmarl_baseline.bash
  done
else
  echo "[campaign] skipping B11 training."
fi

if [[ "$RUN_POLICY_EVAL" == "1" ]]; then
  echo "=== Posthoc policy hierarchy evaluation: B00 ==="
  run_cmd env \
    BASELINE_ID="$B00_BASELINE_ID" \
    SEEDS="$SEEDS" \
    NUM_AGENTS="$NUM_AGENTS" \
    MARL_MODEL="$MARL_MODEL" \
    EVAL_POLICY_MODES="$EVAL_POLICY_MODES" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_MAX_STEPS="$EVAL_MAX_STEPS" \
    COMPARISON_ID="$COMPARISON_ID" \
    WANDB_GROUP="$WANDB_GROUP" \
    WANDB_FLAG="$WANDB_FLAG" \
    ./evaluate_trained_policies_hierarchy.bash

  echo "=== Posthoc policy hierarchy evaluation: B11 ==="
  run_cmd env \
    SEEDS="$SEEDS" \
    LAMBDA_VALUES="$B11_LAMBDA" \
    NUM_AGENTS="$NUM_AGENTS" \
    MARL_MODEL="$MARL_MODEL" \
    EVAL_POLICY_MODES="$EVAL_POLICY_MODES" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_MAX_STEPS="$EVAL_MAX_STEPS" \
    COMPARISON_ID="$COMPARISON_ID" \
    WANDB_GROUP="$WANDB_GROUP" \
    WANDB_FLAG="$WANDB_FLAG" \
    ./evaluate_b11_mambpo_lambda_ablation.bash
else
  echo "[campaign] skipping posthoc policy evaluation."
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  echo "=== Final analysis ==="
  run_cmd "$PYTHON_BIN" analyze_b00_b10_lstm_comparison.py \
    --eval-root policy_hierarchy_eval \
    --baselines "${B00_BASELINE_ID},${B11_BASELINE_ID}" \
    --comparison-id "$COMPARISON_ID" \
    --latest-per-baseline-seed \
    --require-seeds 3 \
    --main-mode "temp_0.5" \
    --out-dir "$ANALYSIS_OUT_DIR"
  echo "Analysis written to ${ANALYSIS_OUT_DIR}"
else
  echo "[campaign] skipping analysis."
fi

echo "=== Completed final scientific B00/B11 baselines (${COMPARISON_ID}) ==="
