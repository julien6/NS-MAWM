#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
NUM_AGENTS="${NUM_AGENTS:-3}"
SEEDS="${SEEDS:-1 2 3}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.0 0.05 0.1 0.2 0.3 0.5}"
EVAL_EPISODES="${EVAL_EPISODES:-8}"
EVAL_POLICY_MODES="${EVAL_POLICY_MODES:-deterministic,mode,temp_1.0,temp_0.5,temp_0.25,temp_0.1,sampled}"
WANDB_FLAG="${WANDB_FLAG---wandb}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-runs_benchmarl/native_marl}"
OUT_DIR="${OUT_DIR:-policy_hierarchy_eval}"

lambda_suffix() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(f"{value:g}".replace(".", "p"))
PY
}

find_checkpoint() {
  local baseline_id="$1"
  local seed="$2"
  "$PYTHON_BIN" - "$CHECKPOINT_ROOT" "$baseline_id" "$seed" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_id = sys.argv[2]
seed = int(sys.argv[3])
candidates = []
for checkpoint in root.rglob("checkpoints/checkpoint_*.pt"):
    run_dir = checkpoint.parent.parent
    matched = False
    for summary in run_dir.glob("*.json"):
        try:
            payload = json.loads(summary.read_text())
        except Exception:
            continue
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        if config.get("baseline_id") == baseline_id and int(config.get("seed", -1)) == seed:
            matched = True
            break
    if matched:
        candidates.append((checkpoint.stat().st_mtime, checkpoint))
if not candidates:
    raise SystemExit(1)
print(max(candidates, key=lambda row: row[0])[1])
PY
}

echo "B11 MAMBPO lambda posthoc evaluation"
echo "  lambdas: ${LAMBDA_VALUES}"
echo "  seeds:   ${SEEDS}"
echo "  modes:   ${EVAL_POLICY_MODES}"
echo

for lambda_value in $LAMBDA_VALUES; do
  suffix="$(lambda_suffix "$lambda_value")"
  baseline_id="B11_structured_neural_k0.0_lambda_${suffix}"
  echo "=== Evaluating ${baseline_id} ==="
  checkpoint_args=()
  for seed in $SEEDS; do
    checkpoint="$(find_checkpoint "$baseline_id" "$seed")"
    echo "seed ${seed}: ${checkpoint}"
    checkpoint_args+=("CHECKPOINT_SEED_${seed}=${checkpoint}")
  done
  env \
    BASELINE_ID="$baseline_id" \
    SEEDS="$SEEDS" \
    NUM_AGENTS="$NUM_AGENTS" \
    MARL_MODEL="${MARL_MODEL:-lstm}" \
    EVAL_POLICY_MODES="$EVAL_POLICY_MODES" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    OUT_DIR="$OUT_DIR" \
    WANDB_FLAG="$WANDB_FLAG" \
    "${checkpoint_args[@]}" \
    ./evaluate_trained_policies_hierarchy.bash
done

echo "=== Completed B11 MAMBPO lambda posthoc evaluation ==="
