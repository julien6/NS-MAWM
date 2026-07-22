#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
if [[ "${AUTO_RESOURCE_PROFILE:-0}" == "1" && "${RESOURCE_PROFILE_APPLIED:-0}" != "1" ]]; then
  RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}"
  echo "[resource-profile] applying ${RESOURCE_PROFILE} to B11 MAMBPO lambda ablation"
  eval "$("$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target campaign --format shell)"
  "$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target campaign --format summary >&2 || true
  export RESOURCE_PROFILE_APPLIED=1
fi

NUM_AGENTS="${NUM_AGENTS:-3}"
SEEDS="${SEEDS:-1 2 3}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.0 0.05 0.1 0.2 0.3 0.5}"
WANDB_FLAG="${WANDB_FLAG---wandb}"
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_STRICT_HPO="${DRY_RUN_STRICT_HPO:-0}"
HPO_RESULTS_DIR="${HPO_RESULTS_DIR:-hpo_results/world_model}"
MARL_HPO_RESULTS_DIR="${MARL_HPO_RESULTS_DIR:-hpo_results/marl}"
COMPARISON_ID="${COMPARISON_ID:-}"
REQUIRE_HPO_FOR_RUN=1
if [[ "$DRY_RUN" == "1" && "$DRY_RUN_STRICT_HPO" != "1" ]]; then
  REQUIRE_HPO_FOR_RUN=0
fi

EXTERNAL_WM_RUN_DIR="${EXTERNAL_WM_RUN_DIR:-}"
if [[ -z "$EXTERNAL_WM_RUN_DIR" ]]; then
  if [[ ! -f "${HPO_RESULTS_DIR}/structured_neural_k0.0/best_config.json" && "$REQUIRE_HPO_FOR_RUN" != "1" ]]; then
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

if [[ ! -f "${EXTERNAL_WM_RUN_DIR%/}/checkpoints/structured_wm.pt" && "$REQUIRE_HPO_FOR_RUN" == "1" ]]; then
  echo "EXTERNAL_WM_RUN_DIR=${EXTERNAL_WM_RUN_DIR} does not contain checkpoints/structured_wm.pt" >&2
  exit 2
fi
if [[ ! -f "${MARL_HPO_RESULTS_DIR}/masac_core/best_config.json" && "$REQUIRE_HPO_FOR_RUN" == "1" ]]; then
  echo "Missing ${MARL_HPO_RESULTS_DIR}/masac_core/best_config.json; run final MASAC+LSTM HPO first." >&2
  exit 2
fi

lambda_suffix() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
value = float(sys.argv[1])
text = f"{value:g}".replace(".", "p")
print(text)
PY
}

echo "B11 MAMBPO lambda ablation"
echo "  external WM: ${EXTERNAL_WM_RUN_DIR}"
echo "  lambdas:     ${LAMBDA_VALUES}"
echo "  seeds:       ${SEEDS}"
echo "  agents:      ${NUM_AGENTS}"
echo "  dry-run:     ${DRY_RUN}"
echo "  strict HPO:  ${REQUIRE_HPO_FOR_RUN}"
echo

for lambda_value in $LAMBDA_VALUES; do
  suffix="$(lambda_suffix "$lambda_value")"
  baseline_id="B11_structured_neural_k0.0_lambda_${suffix}"
  imagination_mode="enabled"
  if [[ "$lambda_value" == "0" || "$lambda_value" == "0.0" || "$lambda_value" == "0.00" ]]; then
    imagination_mode="disabled"
  fi
  echo "=== Lambda ${lambda_value} (${baseline_id}, imagination=${imagination_mode}) ==="
  for seed in $SEEDS; do
    echo "--- seed ${seed} ---"
    AUTO_RESOURCE_PROFILE="${AUTO_RESOURCE_PROFILE:-0}" \
    RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}" \
    BASELINE_ID="$baseline_id" \
    SEED="$seed" \
    NUM_AGENTS="$NUM_AGENTS" \
    MODEL_BASED_DOWNSTREAM_ALGO=mambpo \
    MARL_MODEL="${MARL_MODEL:-lstm}" \
    EXTERNAL_WM_RUN_DIR="$EXTERNAL_WM_RUN_DIR" \
    MAMBPO_IMAGINATION_MODE="$imagination_mode" \
    MB_LAMBDA_IMAGINED="$lambda_value" \
    REUSE_WM_HPO_CONFIG=1 \
    REUSE_MARL_HPO_CONFIG=1 \
    REUSE_MAMBPO_IMAGINATION_HPO=0 \
    HPO_RESULTS_DIR="$HPO_RESULTS_DIR" \
    MARL_HPO_RESULTS_DIR="$MARL_HPO_RESULTS_DIR" \
    COMPARISON_ID="$COMPARISON_ID" \
    REQUIRE_WM_HPO="$REQUIRE_HPO_FOR_RUN" \
    REQUIRE_MARL_HPO="$REQUIRE_HPO_FOR_RUN" \
    REQUIRED_WM_HPO_STAGE=final \
    REQUIRED_MARL_HPO_STAGE=final \
    MARL_MAX_ITERS="${MARL_MAX_ITERS:-500}" \
    MARL_NUM_ENVS="${MARL_NUM_ENVS:-256}" \
    MARL_FRAMES_PER_BATCH="${MARL_FRAMES_PER_BATCH:-4096}" \
    MARL_TRAIN_BATCH_SIZE="${MARL_TRAIN_BATCH_SIZE:-512}" \
    MARL_MEMORY_SIZE="${MARL_MEMORY_SIZE:-200000}" \
    MARL_EVAL_EVERY_ITERS="${MARL_EVAL_EVERY_ITERS:-20}" \
    MARL_EVAL_EPISODES="${MARL_EVAL_EPISODES:-4}" \
    MARL_VIDEO_EVERY_ITERS="${MARL_VIDEO_EVERY_ITERS:-0}" \
    WANDB_GROUP="${WANDB_GROUP:-B11_mambpo_lambda_ablation}" \
    WANDB_FLAG="$WANDB_FLAG" \
    DRY_RUN="$DRY_RUN" \
    ./run_full_benchmarl_baseline.bash
  done
done

echo "=== Completed B11 MAMBPO lambda ablation ==="
