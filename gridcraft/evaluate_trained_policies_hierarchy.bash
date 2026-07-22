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
EVAL_POLICY_MODES="${EVAL_POLICY_MODES:-deterministic,mode,temp_1.0,temp_0.5,temp_0.25,temp_0.1,sampled}"
WANDB_FLAG="${WANDB_FLAG---wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ns-mawm-gridcraft}"
OUT_DIR="${OUT_DIR:-policy_hierarchy_eval}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-runs_benchmarl/native_marl}"
ALLOW_CHECKPOINT_FALLBACK="${ALLOW_CHECKPOINT_FALLBACK:-0}"
COMPARISON_ID="${COMPARISON_ID:-}"

find_latest_checkpoint() {
  local seed="$1"
  "$PYTHON_BIN" - "$CHECKPOINT_ROOT" "$BASELINE_ID" "$seed" "$ALLOW_CHECKPOINT_FALLBACK" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_id = sys.argv[2]
seed = int(sys.argv[3])
allow_fallback = sys.argv[4] == "1"

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
        found_seed = int(found_seed)
    except Exception:
        found_seed = -1
    return found_baseline == baseline_id and found_seed == seed

def text_file_matches(path: Path) -> bool:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return False
    if baseline_id not in text:
        return False
    seed_markers = (
        f"seed{seed}",
        f"seed_{seed}",
        f"seed: {seed}",
        f"seed: '{seed}'",
        f"seed: \"{seed}\"",
        f'"seed": {seed}',
        f"'seed': {seed}",
    )
    return any(marker in text for marker in seed_markers)

def run_dir_text_matches(run_dir: Path) -> bool:
    candidate_patterns = (
        "*.json",
        "*.yaml",
        "*.yml",
        "wandb/**/*.json",
        "wandb/**/*.yaml",
        "wandb/**/*.yml",
        "wandb/**/*.txt",
    )
    for pattern in candidate_patterns:
        for path in run_dir.glob(pattern):
            if path.is_file() and text_file_matches(path):
                return True
    return False

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
    if not matched and run_dir_text_matches(run_dir):
        matched = True
    if matched:
        checkpoint_candidates.append((checkpoint.stat().st_mtime, checkpoint))
if checkpoint_candidates:
    print(max(checkpoint_candidates, key=lambda row: row[0])[1])
    raise SystemExit(0)

summary_candidates: list[Path] = []
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

if allow_fallback:
    fallback = sorted(
        root.rglob("checkpoints/checkpoint_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if fallback:
        print(fallback[-1])
        raise SystemExit(0)

print(
    f"No checkpoint found for baseline={baseline_id} seed={seed} under {root}.",
    file=sys.stderr,
)
raise SystemExit(1)
PY
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
    --comparison-id "$COMPARISON_ID" \
    $WANDB_FLAG
done

echo "=== Completed trained-policy hierarchy evaluation ==="
