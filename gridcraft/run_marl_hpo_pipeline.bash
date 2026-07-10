#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
USER_SET_MARL_HPO_NUM_ENVS="${MARL_HPO_NUM_ENVS+x}"
USER_SET_MARL_HPO_FRAMES_PER_BATCH="${MARL_HPO_FRAMES_PER_BATCH+x}"
USER_SET_MARL_HPO_TRAIN_BATCH_SIZE="${MARL_HPO_TRAIN_BATCH_SIZE+x}"
USER_SET_MARL_HPO_MEMORY_SIZE="${MARL_HPO_MEMORY_SIZE+x}"
if [[ "${AUTO_RESOURCE_PROFILE:-0}" == "1" && "${RESOURCE_PROFILE_APPLIED:-0}" != "1" ]]; then
  RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}"
  echo "[resource-profile] applying ${RESOURCE_PROFILE} to MARL HPO"
  eval "$("$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target marl_hpo --format shell)"
  "$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target marl_hpo --format summary >&2 || true
  export RESOURCE_PROFILE_APPLIED=1
fi

PROJECT="${WANDB_PROJECT:-ns-mawm-gridcraft}"
ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ENTITY_ARGS=(--entity "$WANDB_ENTITY")
fi
RUN_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  RUN_ENTITY_ARGS=(--wandb-entity "$WANDB_ENTITY")
fi

MARL_HPO_RESULTS_DIR="${MARL_HPO_RESULTS_DIR:-hpo_results/marl}"
MARL_HPO_TRIALS_DIR="${MARL_HPO_TRIALS_DIR:-runs_benchmarl_marl_hpo}"
MARL_HPO_MODE="${MARL_HPO_MODE:-standard}"
MARL_HPO_STAGE="${MARL_HPO_STAGE:-$MARL_HPO_MODE}"
case "$MARL_HPO_STAGE" in
  quick) MARL_HPO_STAGE="screen" ;;
  standard) MARL_HPO_STAGE="promote" ;;
  serious) MARL_HPO_STAGE="final" ;;
esac
MARL_HPO_TOP_K="${MARL_HPO_TOP_K:-3}"
MARL_HPO_SEEDS="${MARL_HPO_SEEDS:-${MARL_HPO_SEED:-${SEED:-1}}}"
HPO_BENCHMARK_FIRST="${HPO_BENCHMARK_FIRST:-0}"
MARL_HPO_FAMILIES="${MARL_HPO_FAMILIES:-masac_core mambpo_imagination}"
FORCE_MARL_HPO="${FORCE_MARL_HPO:-0}"
if [[ "$MARL_HPO_STAGE" == "auto" ]]; then
  MARL_HPO_STAGE="final"
  for family in $MARL_HPO_FAMILIES; do
    if [[ ! -f "${MARL_HPO_RESULTS_DIR}/${family}/screen_results.json" ]]; then
      MARL_HPO_STAGE="screen"
      break
    fi
    if [[ ! -f "${MARL_HPO_RESULTS_DIR}/${family}/promoted_configs.json" ]]; then
      MARL_HPO_STAGE="promote"
    fi
  done
fi

case "$MARL_HPO_STAGE" in
  screen)
    MARL_HPO_COUNT="${MARL_HPO_COUNT:-3}"
    if [[ -z "$USER_SET_MARL_HPO_NUM_ENVS" ]]; then
      export MARL_HPO_NUM_ENVS="${MARL_HPO_SCREEN_NUM_ENVS:-256}"
    else
      export MARL_HPO_NUM_ENVS
    fi
    if [[ -z "$USER_SET_MARL_HPO_FRAMES_PER_BATCH" ]]; then
      export MARL_HPO_FRAMES_PER_BATCH="${MARL_HPO_SCREEN_FRAMES_PER_BATCH:-4096}"
    fi
    if [[ -z "$USER_SET_MARL_HPO_TRAIN_BATCH_SIZE" ]]; then
      export MARL_HPO_TRAIN_BATCH_SIZE="${MARL_HPO_SCREEN_TRAIN_BATCH_SIZE:-512}"
    fi
    if [[ -z "$USER_SET_MARL_HPO_MEMORY_SIZE" ]]; then
      export MARL_HPO_MEMORY_SIZE="${MARL_HPO_SCREEN_MEMORY_SIZE:-100000}"
    fi
    export MARL_HPO_MAX_STEPS="${MARL_HPO_MAX_STEPS:-64}"
    export MARL_HPO_MAX_ITERS="${MARL_HPO_MAX_ITERS:-2}"
    export MARL_HPO_EVAL_EVERY_ITERS="${MARL_HPO_EVAL_EVERY_ITERS:-1}"
    export MARL_HPO_EVAL_EPISODES="${MARL_HPO_EVAL_EPISODES:-1}"
    ;;
  promote)
    MARL_HPO_COUNT="${MARL_HPO_COUNT:-${MARL_HPO_TOP_K}}"
    export MARL_HPO_NUM_ENVS="${MARL_HPO_NUM_ENVS:-128}"
    export MARL_HPO_MAX_STEPS="${MARL_HPO_MAX_STEPS:-500}"
    export MARL_HPO_MAX_ITERS="${MARL_HPO_MAX_ITERS:-200}"
    export MARL_HPO_EVAL_EVERY_ITERS="${MARL_HPO_EVAL_EVERY_ITERS:-10}"
    export MARL_HPO_EVAL_EPISODES="${MARL_HPO_EVAL_EPISODES:-4}"
    ;;
  final)
    MARL_HPO_COUNT="${MARL_HPO_COUNT:-${MARL_HPO_TOP_K}}"
    export MARL_HPO_NUM_ENVS="${MARL_HPO_NUM_ENVS:-512}"
    export MARL_HPO_MAX_STEPS="${MARL_HPO_MAX_STEPS:-500}"
    export MARL_HPO_MAX_ITERS="${MARL_HPO_MAX_ITERS:-1000}"
    export MARL_HPO_EVAL_EVERY_ITERS="${MARL_HPO_EVAL_EVERY_ITERS:-20}"
    export MARL_HPO_EVAL_EPISODES="${MARL_HPO_EVAL_EPISODES:-4}"
    ;;
  *)
    echo "Unsupported MARL_HPO_STAGE=${MARL_HPO_STAGE}; expected screen, promote, final, auto, quick, standard, or serious." >&2
    exit 2
    ;;
esac

export MARL_HPO_DEVICE="${MARL_HPO_DEVICE:-${DEVICE:-cuda}}"
export MARL_HPO_NUM_AGENTS="${MARL_HPO_NUM_AGENTS:-${NUM_AGENTS:-3}}"
export MARL_HPO_SEED="${MARL_HPO_SEED:-${SEED:-1}}"
export MARL_HPO_VIDEO_EVERY_ITERS="${MARL_HPO_VIDEO_EVERY_ITERS:-0}"
export MARL_HPO_TRIALS_DIR
export MARL_HPO_CURRENT_STAGE="$MARL_HPO_STAGE"

if [[ -z "${MARL_HPO_WM_RUN_DIR:-}" ]]; then
  export MARL_HPO_WM_RUN_DIR="runs_benchmarl/B10_neural_k0.0_a${MARL_HPO_NUM_AGENTS}_seed${MARL_HPO_SEED}"
fi

config_for_family() {
  case "$1" in
    masac_core) echo "sweeps/marl_masac_core_hpo.yaml" ;;
    mambpo_imagination) echo "sweeps/marl_mambpo_imagination_hpo.yaml" ;;
    *)
      echo "Unsupported MARL HPO family: $1" >&2
      return 2
      ;;
  esac
}

echo "Gridcraft MARL HPO pipeline"
echo "  project:       ${PROJECT}"
echo "  mode/stage:    ${MARL_HPO_MODE}/${MARL_HPO_STAGE}"
echo "  count/family:  ${MARL_HPO_COUNT}"
echo "  top-k:         ${MARL_HPO_TOP_K}"
echo "  seeds:         ${MARL_HPO_SEEDS}"
echo "  families:      ${MARL_HPO_FAMILIES}"
echo "  results dir:   ${MARL_HPO_RESULTS_DIR}"
echo "  trials dir:    ${MARL_HPO_TRIALS_DIR}"
echo "  wm run dir:    ${MARL_HPO_WM_RUN_DIR}"
echo "  budget:        envs=${MARL_HPO_NUM_ENVS}, max_steps=${MARL_HPO_MAX_STEPS}, max_iters=${MARL_HPO_MAX_ITERS}"

if [[ "$HPO_BENCHMARK_FIRST" == "1" ]]; then
  echo "[marl-hpo] running throughput benchmark before HPO"
  "$PYTHON_BIN" benchmark_hpo_throughput.py --profile "${RESOURCE_PROFILE:-spark_max}" --target marl --out "${MARL_HPO_RESULTS_DIR}/throughput_report.json" || true
fi

for family in $MARL_HPO_FAMILIES; do
  mkdir -p "${MARL_HPO_TRIALS_DIR}/${family}"
  best_config="${MARL_HPO_RESULTS_DIR}/${family}/best_config.json"
  if [[ "$FORCE_MARL_HPO" != "1" && -f "$best_config" ]]; then
    reselection_stage="$("$PYTHON_BIN" - "$best_config" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
provenance = payload.get("provenance", {})
current = (
    provenance.get("environment_dynamics_version") == "gridcraft_dynamics_v2_armed_combat"
    and provenance.get("reward_schema_version") == "gridcraft_reward_v2_team_milestones"
)
print(payload.get("stage", "") if current and payload.get("selection_method") != "mean_across_seeds_v1" else "")
PY
)"
    if [[ -n "$reselection_stage" ]]; then
      reselection_budget="$("$PYTHON_BIN" - "$best_config" <<'PY'
import json
import sys
from pathlib import Path
print(json.dumps(json.loads(Path(sys.argv[1]).read_text()).get("budget", {}), separators=(",", ":")))
PY
)"
      echo "[marl-hpo] ${family}: recalculating stage=${reselection_stage} selection as a mean across seeds."
      "$PYTHON_BIN" marl_hpo_registry.py select-best \
        --family "$family" \
        --trials-root "${MARL_HPO_TRIALS_DIR}/${family}" \
        --results-root "$MARL_HPO_RESULTS_DIR" \
        --stage "$reselection_stage" \
        --top-k "$MARL_HPO_TOP_K" \
        --budget-json "$reselection_budget" >/dev/null
    fi
  fi
  if [[ "$FORCE_MARL_HPO" != "1" && -f "$best_config" ]]; then
    VALIDATE_ARGS=(
      "$PYTHON_BIN" marl_hpo_registry.py validate
      --family "$family"
      --root "$MARL_HPO_RESULTS_DIR"
      --required-stage "$MARL_HPO_STAGE"
      --num-agents "$MARL_HPO_NUM_AGENTS"
      --minimum-budget-json "{\"seeds\":\"${MARL_HPO_SEEDS}\",\"num_envs\":${MARL_HPO_NUM_ENVS},\"max_steps\":${MARL_HPO_MAX_STEPS},\"max_iters\":${MARL_HPO_MAX_ITERS}}"
    )
    if [[ "$family" == "mambpo_imagination" ]]; then
      VALIDATE_ARGS+=(--external-checkpoint-dir "${MARL_HPO_WM_RUN_DIR}/checkpoints")
    fi
    if "${VALIDATE_ARGS[@]}" >/dev/null 2>&1; then
      echo "[marl-hpo] ${family}: compatible ${MARL_HPO_STAGE} config found at ${best_config}; skipping."
      continue
    fi
    echo "[marl-hpo] ${family}: existing config is not valid for stage=${MARL_HPO_STAGE}, agents=${MARL_HPO_NUM_AGENTS}; archiving pre-fix results."
    archive_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    archive_root="${MARL_HPO_RESULTS_DIR}/pre_reward_hierarchy_fix/${archive_stamp}/${family}"
    mkdir -p "$archive_root"
    mv "$best_config" "${archive_root}/best_config.json"
    if [[ -d "${MARL_HPO_TRIALS_DIR}/${family}" ]]; then
      mv "${MARL_HPO_TRIALS_DIR}/${family}" "${archive_root}/trials"
    fi
    mkdir -p "${MARL_HPO_TRIALS_DIR}/${family}"
  fi
  if [[ "$family" == "mambpo_imagination" ]]; then
    checkpoint_dir="${MARL_HPO_WM_RUN_DIR}/checkpoints"
    if [[ ! -f "${checkpoint_dir}/vae.pt" || ! -f "${checkpoint_dir}/rnn.pt" ]]; then
      echo "[marl-hpo] mambpo_imagination requires trained WM checkpoints at ${checkpoint_dir}" >&2
      echo "Set MARL_HPO_WM_RUN_DIR or run B10_neural_k0.0 world-model first." >&2
      exit 2
    fi
  fi

  trial_glob_count="$(find "${MARL_HPO_TRIALS_DIR}/${family}" -name marl_hpo_trial_summary.json 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$MARL_HPO_STAGE" == "screen" || "$trial_glob_count" == "0" ]]; then
    sweep_config="$(config_for_family "$family")"
    if [[ "$MARL_HPO_STAGE" != "screen" && "$trial_glob_count" == "0" ]]; then
      echo "[marl-hpo] ${family}: no previous screen trials found; bootstrapping ${MARL_HPO_STAGE} with a sweep from ${sweep_config}"
    else
      echo "[marl-hpo] ${family}: creating screen sweep from ${sweep_config}"
    fi
    sweep_output_file="$(mktemp)"
    set +e
    ../.venv/bin/wandb sweep --project "$PROJECT" "${ENTITY_ARGS[@]}" "$sweep_config" 2>&1 | tee "$sweep_output_file"
    sweep_status="${PIPESTATUS[0]}"
    set -e
    if [[ "$sweep_status" != "0" ]]; then
      echo "[marl-hpo] wandb sweep failed with status ${sweep_status}." >&2
      echo "[marl-hpo] output was saved to ${sweep_output_file}" >&2
      exit "$sweep_status"
    fi
    SWEEP_ID="$(awk '
      /Creating sweep with ID:/ {print $NF}
      /wandb agent/ {print $NF}
    ' "$sweep_output_file" | tail -n 1)"
    if [[ -z "$SWEEP_ID" ]]; then
      echo "Could not parse sweep id from wandb sweep output for ${family}." >&2
      echo "[marl-hpo] output was saved to ${sweep_output_file}" >&2
      exit 1
    fi
    echo "[marl-hpo] ${family}: launching ${MARL_HPO_COUNT} ${MARL_HPO_STAGE} trials (${SWEEP_ID})"
    "$PYTHON_BIN" -m wandb agent --count "$MARL_HPO_COUNT" "$SWEEP_ID"
  else
    source_stage="screen"
    if [[ "$MARL_HPO_STAGE" == "final" ]]; then
      source_stage="promote"
    fi
    "$PYTHON_BIN" marl_hpo_registry.py write-stage-results \
      --family "$family" \
      --trials-root "${MARL_HPO_TRIALS_DIR}/${family}" \
      --results-root "$MARL_HPO_RESULTS_DIR" \
      --stage promote \
      --source-stage "$source_stage" \
      --top-k "$MARL_HPO_TOP_K" >/dev/null
    promoted_path="${MARL_HPO_RESULTS_DIR}/${family}/promoted_configs.json"
    echo "[marl-hpo] ${family}: replaying top configs from ${promoted_path}"
    config_count="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
data=json.loads(Path("${promoted_path}").read_text())
print(len(data.get("configs", [])))
PY
)"
    if [[ "$config_count" == "0" ]]; then
      echo "[marl-hpo] ${family}: no promoted configs available in ${promoted_path}" >&2
      exit 2
    fi
    for idx in $(seq 0 $((config_count - 1))); do
      fixed_config="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
data=json.loads(Path("${promoted_path}").read_text())
print(json.dumps(data["configs"][${idx}], sort_keys=True))
PY
)"
      for seed in $MARL_HPO_SEEDS; do
        echo "[marl-hpo] ${family}: ${MARL_HPO_STAGE} replay config=$((idx + 1))/${config_count} seed=${seed}"
        MARL_HPO_FIXED_CONFIG_JSON="$fixed_config" MARL_HPO_SEED="$seed" "$PYTHON_BIN" sweep_benchmarl_marl_hpo.py --hpo-family "$family" --wandb-project "$PROJECT" "${RUN_ENTITY_ARGS[@]}"
      done
    done
  fi

  echo "[marl-hpo] ${family}: selecting best local trial"
  "$PYTHON_BIN" marl_hpo_registry.py select-best \
    --family "$family" \
    --trials-root "${MARL_HPO_TRIALS_DIR}/${family}" \
    --results-root "$MARL_HPO_RESULTS_DIR" \
    --stage "$MARL_HPO_STAGE" \
    --top-k "$MARL_HPO_TOP_K" \
    --budget-json "{\"mode\":\"${MARL_HPO_MODE}\",\"stage\":\"${MARL_HPO_STAGE}\",\"count\":${MARL_HPO_COUNT},\"top_k\":${MARL_HPO_TOP_K},\"seeds\":\"${MARL_HPO_SEEDS}\",\"num_envs\":${MARL_HPO_NUM_ENVS},\"max_steps\":${MARL_HPO_MAX_STEPS},\"max_iters\":${MARL_HPO_MAX_ITERS}}"
done

"$PYTHON_BIN" marl_hpo_registry.py write-summary --results-root "$MARL_HPO_RESULTS_DIR" >/dev/null
echo "[marl-hpo] summary written to ${MARL_HPO_RESULTS_DIR}/summary.json"
