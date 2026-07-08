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

HPO_RESULTS_DIR="${HPO_RESULTS_DIR:-hpo_results/world_model}"
HPO_TRIALS_DIR="${HPO_TRIALS_DIR:-runs_benchmarl_hpo}"
HPO_MODE="${HPO_MODE:-standard}"
FORCE_WM_HPO="${FORCE_WM_HPO:-0}"
HPO_FAMILIES="${HPO_FAMILIES:-neural_k0.0 regularization_k0.3 regularization_k0.6 residual_k0.3 residual_k0.6}"

case "$HPO_MODE" in
  quick)
    HPO_COUNT="${HPO_COUNT:-3}"
    export HPO_EPISODES="${HPO_EPISODES:-512}"
    export HPO_MAX_STEPS="${HPO_MAX_STEPS:-64}"
    export HPO_NUM_ENVS="${HPO_NUM_ENVS:-128}"
    export HPO_VAE_STEPS="${HPO_VAE_STEPS:-1000}"
    export HPO_RNN_STEPS="${HPO_RNN_STEPS:-1000}"
    export HPO_EVAL_EVERY="${HPO_EVAL_EVERY:-500}"
    export HPO_HORIZONS="${HPO_HORIZONS:-1 5 10}"
    ;;
  standard)
    HPO_COUNT="${HPO_COUNT:-20}"
    export HPO_EPISODES="${HPO_EPISODES:-4096}"
    export HPO_MAX_STEPS="${HPO_MAX_STEPS:-256}"
    export HPO_NUM_ENVS="${HPO_NUM_ENVS:-256}"
    export HPO_VAE_STEPS="${HPO_VAE_STEPS:-10000}"
    export HPO_RNN_STEPS="${HPO_RNN_STEPS:-10000}"
    export HPO_EVAL_EVERY="${HPO_EVAL_EVERY:-2500}"
    export HPO_HORIZONS="${HPO_HORIZONS:-1 5 10 25}"
    ;;
  serious)
    HPO_COUNT="${HPO_COUNT:-50}"
    export HPO_EPISODES="${HPO_EPISODES:-30000}"
    export HPO_MAX_STEPS="${HPO_MAX_STEPS:-500}"
    export HPO_NUM_ENVS="${HPO_NUM_ENVS:-1024}"
    export HPO_VAE_STEPS="${HPO_VAE_STEPS:-50000}"
    export HPO_RNN_STEPS="${HPO_RNN_STEPS:-50000}"
    export HPO_EVAL_EVERY="${HPO_EVAL_EVERY:-5000}"
    export HPO_HORIZONS="${HPO_HORIZONS:-1 5 10 25 50 100}"
    ;;
  *)
    echo "Unsupported HPO_MODE=${HPO_MODE}; expected quick, standard, or serious." >&2
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
echo "  mode:          ${HPO_MODE}"
echo "  count/family:  ${HPO_COUNT}"
echo "  families:      ${HPO_FAMILIES}"
echo "  results dir:   ${HPO_RESULTS_DIR}"
echo "  trials dir:    ${HPO_TRIALS_DIR}"
echo "  budget:        episodes=${HPO_EPISODES}, max_steps=${HPO_MAX_STEPS}, envs=${HPO_NUM_ENVS}, vae_steps=${HPO_VAE_STEPS}, rnn_steps=${HPO_RNN_STEPS}"

for family in $HPO_FAMILIES; do
  best_config="${HPO_RESULTS_DIR}/${family}/best_config.json"
  if [[ "$FORCE_WM_HPO" != "1" && -f "$best_config" ]]; then
    echo "[wm-hpo] ${family}: existing best config found at ${best_config}; skipping."
    continue
  fi

  sweep_config="$(config_for_family "$family")"
  echo "[wm-hpo] ${family}: creating sweep from ${sweep_config}"
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
  echo "[wm-hpo] ${family}: launching ${HPO_COUNT} trials (${SWEEP_ID})"
  "$PYTHON_BIN" -m wandb agent --count "$HPO_COUNT" "$SWEEP_ID"

  echo "[wm-hpo] ${family}: selecting best local trial"
  "$PYTHON_BIN" wm_hpo_registry.py select-best \
    --hpo-family "$family" \
    --trials-root "${HPO_TRIALS_DIR}/${family}" \
    --results-root "$HPO_RESULTS_DIR" \
    --budget-json "{\"mode\":\"${HPO_MODE}\",\"count\":${HPO_COUNT},\"episodes\":${HPO_EPISODES},\"max_steps\":${HPO_MAX_STEPS},\"num_envs\":${HPO_NUM_ENVS},\"vae_steps\":${HPO_VAE_STEPS},\"rnn_steps\":${HPO_RNN_STEPS}}"
done

"$PYTHON_BIN" wm_hpo_registry.py write-summary --results-root "$HPO_RESULTS_DIR" >/dev/null
echo "[wm-hpo] summary written to ${HPO_RESULTS_DIR}/summary.json"
