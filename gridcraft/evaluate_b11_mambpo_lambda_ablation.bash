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

def payload_matches(payload: object, path: Path) -> bool:
    if not isinstance(payload, dict):
        return False
    config = payload.get("config", {})
    if not isinstance(config, dict):
        config = {}
    found_baseline = config.get("baseline_id") or payload.get("baseline_id")
    found_seed = config.get("seed") or payload.get("seed")
    if found_seed is None:
        marker = f"_seed{seed}"
        found_seed = seed if marker in path.name or marker in str(path.parent) else None
    try:
        found_seed_int = int(found_seed)
    except Exception:
        found_seed_int = -1
    return found_baseline == baseline_id and found_seed_int == seed

summary_candidates: list[Path] = []
checkpoint_candidates: list[tuple[float, Path]] = []
for checkpoint in root.rglob("checkpoints/checkpoint_*.pt"):
    run_dir = checkpoint.parent.parent
    matched = False
    for summary in run_dir.glob("*.json"):
        try:
            payload = json.loads(summary.read_text())
        except Exception:
            continue
        if payload_matches(payload, summary):
            matched = True
            break
    if matched:
        checkpoint_candidates.append((checkpoint.stat().st_mtime, checkpoint))
if checkpoint_candidates:
    print(max(checkpoint_candidates, key=lambda row: row[0])[1])
    raise SystemExit(0)

for summary in root.rglob("*_marl_summary.json"):
    try:
        payload = json.loads(summary.read_text())
    except Exception:
        continue
    if not payload_matches(payload, summary):
        continue
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    checkpoint_path = config.get("checkpoint_path") if isinstance(config, dict) else None
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            print(checkpoint)
            raise SystemExit(0)
        rel_checkpoint = root / checkpoint_path
        if rel_checkpoint.exists():
            print(rel_checkpoint)
            raise SystemExit(0)
    summary_candidates.append(summary)

if summary_candidates:
    summary = max(summary_candidates, key=lambda path: path.stat().st_mtime)
    summary_time = summary.stat().st_mtime
    nearby: list[tuple[float, float, Path]] = []
    for checkpoint in root.rglob("checkpoints/checkpoint_*.pt"):
        delta = abs(checkpoint.stat().st_mtime - summary_time)
        if delta <= 1800:
            nearby.append((delta, checkpoint.stat().st_mtime, checkpoint))
    if nearby:
        nearby.sort(key=lambda row: (row[0], -row[1]))
        print(nearby[0][2])
        raise SystemExit(0)

print(
    f"No checkpoint found for baseline={baseline_id} seed={seed} under {root}. "
    f"matching_summaries={len(summary_candidates)}",
    file=sys.stderr,
)
raise SystemExit(1)
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
