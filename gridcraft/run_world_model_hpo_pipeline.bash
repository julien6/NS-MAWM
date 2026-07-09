#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
if [[ "${AUTO_RESOURCE_PROFILE:-0}" == "1" && "${RESOURCE_PROFILE_APPLIED:-0}" != "1" ]]; then
  RESOURCE_PROFILE="${RESOURCE_PROFILE:-spark_max}"
  echo "[resource-profile] applying ${RESOURCE_PROFILE} to world-model HPO"
  eval "$("$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target wm_hpo --format shell)"
  "$PYTHON_BIN" resource_profile.py --profile "$RESOURCE_PROFILE" --target wm_hpo --format summary >&2 || true
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

HPO_RESULTS_DIR="${HPO_RESULTS_DIR:-hpo_results/world_model}"
HPO_TRIALS_DIR="${HPO_TRIALS_DIR:-runs_benchmarl_hpo}"
HPO_MODE="${HPO_MODE:-standard}"
HPO_STAGE="${HPO_STAGE:-$HPO_MODE}"
case "$HPO_STAGE" in
  quick) HPO_STAGE="screen" ;;
  standard) HPO_STAGE="promote" ;;
  serious) HPO_STAGE="final" ;;
esac
HPO_TOP_K="${HPO_TOP_K:-3}"
HPO_SEEDS="${HPO_SEEDS:-${HPO_SEED:-${SEED:-1}}}"
HPO_BENCHMARK_FIRST="${HPO_BENCHMARK_FIRST:-0}"
FORCE_WM_HPO="${FORCE_WM_HPO:-0}"
HPO_FAMILIES="${HPO_FAMILIES:-neural_k0.0 regularization_k0.3 regularization_k0.6 residual_k0.3 residual_k0.6}"
if [[ "$HPO_STAGE" == "auto" ]]; then
  HPO_STAGE="final"
  for family in $HPO_FAMILIES; do
    if [[ ! -f "${HPO_RESULTS_DIR}/${family}/screen_results.json" ]]; then
      HPO_STAGE="screen"
      break
    fi
    if [[ ! -f "${HPO_RESULTS_DIR}/${family}/promoted_configs.json" ]]; then
      HPO_STAGE="promote"
    fi
  done
fi

case "$HPO_STAGE" in
  screen)
    HPO_COUNT="${HPO_COUNT:-3}"
    export HPO_EPISODES="${HPO_EPISODES:-512}"
    export HPO_MAX_STEPS="${HPO_MAX_STEPS:-64}"
    export HPO_NUM_ENVS="${HPO_NUM_ENVS:-128}"
    export HPO_VAE_STEPS="${HPO_VAE_STEPS:-1000}"
    export HPO_RNN_STEPS="${HPO_RNN_STEPS:-1000}"
    export HPO_EVAL_EVERY="${HPO_EVAL_EVERY:-500}"
    export HPO_HORIZONS="${HPO_HORIZONS:-1 5 10}"
    ;;
  promote)
    HPO_COUNT="${HPO_COUNT:-${HPO_TOP_K}}"
    export HPO_EPISODES="${HPO_EPISODES:-4096}"
    export HPO_MAX_STEPS="${HPO_MAX_STEPS:-256}"
    export HPO_NUM_ENVS="${HPO_NUM_ENVS:-256}"
    export HPO_VAE_STEPS="${HPO_VAE_STEPS:-10000}"
    export HPO_RNN_STEPS="${HPO_RNN_STEPS:-10000}"
    export HPO_EVAL_EVERY="${HPO_EVAL_EVERY:-2500}"
    export HPO_HORIZONS="${HPO_HORIZONS:-1 5 10 25}"
    ;;
  final)
    HPO_COUNT="${HPO_COUNT:-${HPO_TOP_K}}"
    export HPO_EPISODES="${HPO_EPISODES:-30000}"
    export HPO_MAX_STEPS="${HPO_MAX_STEPS:-500}"
    export HPO_NUM_ENVS="${HPO_NUM_ENVS:-1024}"
    export HPO_VAE_STEPS="${HPO_VAE_STEPS:-50000}"
    export HPO_RNN_STEPS="${HPO_RNN_STEPS:-50000}"
    export HPO_EVAL_EVERY="${HPO_EVAL_EVERY:-5000}"
    export HPO_HORIZONS="${HPO_HORIZONS:-1 5 10 25 50 100}"
    ;;
  *)
    echo "Unsupported HPO_STAGE=${HPO_STAGE}; expected screen, promote, final, auto, quick, standard, or serious." >&2
    exit 2
    ;;
esac

export HPO_DEVICE="${HPO_DEVICE:-${DEVICE:-cuda}}"
export HPO_TRIALS_DIR
export HPO_RESULTS_DIR
export HPO_SHARED_MODEL_DIR="${HPO_SHARED_MODEL_DIR:-shared_models_hpo}"
export HPO_NUM_AGENTS="${HPO_NUM_AGENTS:-${NUM_AGENTS:-3}}"
export HPO_SEED="${HPO_SEED:-${SEED:-1}}"
export HPO_WM_NUM_WORKERS="${HPO_WM_NUM_WORKERS:-${WM_NUM_WORKERS:-4}}"
export HPO_VIDEO_EVERY="${HPO_VIDEO_EVERY:-0}"
export HPO_CURRENT_STAGE="$HPO_STAGE"

config_for_family() {
  case "$1" in
    neural_k0.0) echo "sweeps/world_model_neural_hpo.yaml" ;;
    regularization_k0.3) echo "sweeps/world_model_regularization_k03_hpo.yaml" ;;
    regularization_k0.6) echo "sweeps/world_model_regularization_k06_hpo.yaml" ;;
    residual_k0.3) echo "sweeps/world_model_residual_k03_hpo.yaml" ;;
    residual_k0.6) echo "sweeps/world_model_residual_k06_hpo.yaml" ;;
    *)
      echo "Unsupported HPO family: $1" >&2
      return 2
      ;;
  esac
}

echo "Gridcraft World Model HPO pipeline"
echo "  project:       ${PROJECT}"
echo "  mode/stage:    ${HPO_MODE}/${HPO_STAGE}"
echo "  count/family:  ${HPO_COUNT}"
echo "  top-k:         ${HPO_TOP_K}"
echo "  seeds:         ${HPO_SEEDS}"
echo "  families:      ${HPO_FAMILIES}"
echo "  results dir:   ${HPO_RESULTS_DIR}"
echo "  trials dir:    ${HPO_TRIALS_DIR}"
echo "  budget:        episodes=${HPO_EPISODES}, max_steps=${HPO_MAX_STEPS}, envs=${HPO_NUM_ENVS}, vae_steps=${HPO_VAE_STEPS}, rnn_steps=${HPO_RNN_STEPS}"

if [[ "$HPO_BENCHMARK_FIRST" == "1" ]]; then
  echo "[wm-hpo] running throughput benchmark before HPO"
  "$PYTHON_BIN" benchmark_hpo_throughput.py --profile "${RESOURCE_PROFILE:-spark_max}" --target wm --out "${HPO_RESULTS_DIR}/throughput_report.json" || true
fi

for family in $HPO_FAMILIES; do
  best_config="${HPO_RESULTS_DIR}/${family}/best_config.json"
  if [[ "$FORCE_WM_HPO" != "1" && -f "$best_config" ]]; then
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
      echo "[wm-hpo] ${family}: recalculating stage=${reselection_stage} selection as a mean across seeds."
      "$PYTHON_BIN" wm_hpo_registry.py select-best \
        --hpo-family "$family" \
        --trials-root "${HPO_TRIALS_DIR}/${family}" \
        --results-root "$HPO_RESULTS_DIR" \
        --stage "$reselection_stage" \
        --top-k "$HPO_TOP_K" \
        --budget-json "$reselection_budget" >/dev/null
    fi
  fi
  if [[ "$FORCE_WM_HPO" != "1" && -f "$best_config" ]]; then
    if "$PYTHON_BIN" wm_hpo_registry.py validate \
      --hpo-family "$family" \
      --root "$HPO_RESULTS_DIR" \
      --required-stage "$HPO_STAGE" \
      --num-agents "$HPO_NUM_AGENTS" \
      --minimum-budget-json "{\"seeds\":\"${HPO_SEEDS}\",\"episodes\":${HPO_EPISODES},\"max_steps\":${HPO_MAX_STEPS},\"vae_steps\":${HPO_VAE_STEPS},\"rnn_steps\":${HPO_RNN_STEPS}}" >/dev/null 2>&1; then
      echo "[wm-hpo] ${family}: compatible ${HPO_STAGE} config found at ${best_config}; skipping."
      continue
    fi
    echo "[wm-hpo] ${family}: existing config is not valid for stage=${HPO_STAGE}, agents=${HPO_NUM_AGENTS}; archiving pre-fix results."
    archive_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    archive_root="${HPO_RESULTS_DIR}/pre_reward_hierarchy_fix/${archive_stamp}/${family}"
    mkdir -p "$archive_root"
    mv "$best_config" "${archive_root}/best_config.json"
    if [[ -d "${HPO_TRIALS_DIR}/${family}" ]]; then
      mv "${HPO_TRIALS_DIR}/${family}" "${archive_root}/trials"
    fi
  fi

  trial_glob_count="$(find "${HPO_TRIALS_DIR}/${family}" -name hpo_trial_summary.json 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$HPO_STAGE" == "screen" || "$trial_glob_count" == "0" ]]; then
    sweep_config="$(config_for_family "$family")"
    if [[ "$HPO_STAGE" != "screen" && "$trial_glob_count" == "0" ]]; then
      echo "[wm-hpo] ${family}: no previous screen trials found; bootstrapping ${HPO_STAGE} with a sweep from ${sweep_config}"
    else
      echo "[wm-hpo] ${family}: creating screen sweep from ${sweep_config}"
    fi
    SWEEP_OUTPUT="$(../.venv/bin/wandb sweep --project "$PROJECT" "${ENTITY_ARGS[@]}" "$sweep_config" 2>&1)"
    printf '%s\n' "$SWEEP_OUTPUT"
    SWEEP_ID="$(printf '%s\n' "$SWEEP_OUTPUT" | awk '
      /Creating sweep with ID:/ {print $NF}
      /wandb agent/ {print $NF}
    ' | tail -n 1)"
    if [[ -z "$SWEEP_ID" ]]; then
      echo "Could not parse sweep id from wandb sweep output for ${family}." >&2
      exit 1
    fi
    echo "[wm-hpo] ${family}: launching ${HPO_COUNT} ${HPO_STAGE} trials (${SWEEP_ID})"
    "$PYTHON_BIN" -m wandb agent --count "$HPO_COUNT" "$SWEEP_ID"
  else
    source_stage="screen"
    if [[ "$HPO_STAGE" == "final" ]]; then
      source_stage="promote"
    fi
    "$PYTHON_BIN" wm_hpo_registry.py write-stage-results \
      --hpo-family "$family" \
      --trials-root "${HPO_TRIALS_DIR}/${family}" \
      --results-root "$HPO_RESULTS_DIR" \
      --stage promote \
      --source-stage "$source_stage" \
      --top-k "$HPO_TOP_K" >/dev/null
    promoted_path="${HPO_RESULTS_DIR}/${family}/promoted_configs.json"
    echo "[wm-hpo] ${family}: replaying top configs from ${promoted_path}"
    config_count="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
data=json.loads(Path("${promoted_path}").read_text())
print(len(data.get("configs", [])))
PY
)"
    if [[ "$config_count" == "0" ]]; then
      echo "[wm-hpo] ${family}: no promoted configs available in ${promoted_path}" >&2
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
      for seed in $HPO_SEEDS; do
        echo "[wm-hpo] ${family}: ${HPO_STAGE} replay config=$((idx + 1))/${config_count} seed=${seed}"
        HPO_FIXED_CONFIG_JSON="$fixed_config" HPO_SEED="$seed" "$PYTHON_BIN" sweep_benchmarl_wm_hpo.py --hpo-family "$family" --wandb-project "$PROJECT" "${RUN_ENTITY_ARGS[@]}"
      done
    done
  fi

  echo "[wm-hpo] ${family}: selecting best local trial"
  "$PYTHON_BIN" wm_hpo_registry.py select-best \
    --hpo-family "$family" \
    --trials-root "${HPO_TRIALS_DIR}/${family}" \
    --results-root "$HPO_RESULTS_DIR" \
    --stage "$HPO_STAGE" \
    --top-k "$HPO_TOP_K" \
    --budget-json "{\"mode\":\"${HPO_MODE}\",\"stage\":\"${HPO_STAGE}\",\"count\":${HPO_COUNT},\"top_k\":${HPO_TOP_K},\"seeds\":\"${HPO_SEEDS}\",\"episodes\":${HPO_EPISODES},\"max_steps\":${HPO_MAX_STEPS},\"num_envs\":${HPO_NUM_ENVS},\"vae_steps\":${HPO_VAE_STEPS},\"rnn_steps\":${HPO_RNN_STEPS}}"
done

"$PYTHON_BIN" wm_hpo_registry.py write-summary --results-root "$HPO_RESULTS_DIR" >/dev/null
echo "[wm-hpo] summary written to ${HPO_RESULTS_DIR}/summary.json"
