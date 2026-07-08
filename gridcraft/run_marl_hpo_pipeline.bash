#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
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

MARL_HPO_RESULTS_DIR="${MARL_HPO_RESULTS_DIR:-hpo_results/marl}"
MARL_HPO_TRIALS_DIR="${MARL_HPO_TRIALS_DIR:-runs_benchmarl_marl_hpo}"
MARL_HPO_MODE="${MARL_HPO_MODE:-standard}"
MARL_HPO_FAMILIES="${MARL_HPO_FAMILIES:-masac_core mambpo_imagination}"
FORCE_MARL_HPO="${FORCE_MARL_HPO:-0}"

case "$MARL_HPO_MODE" in
  quick)
    MARL_HPO_COUNT="${MARL_HPO_COUNT:-3}"
    export MARL_HPO_NUM_ENVS="${MARL_HPO_NUM_ENVS:-16}"
    export MARL_HPO_MAX_STEPS="${MARL_HPO_MAX_STEPS:-64}"
    export MARL_HPO_MAX_ITERS="${MARL_HPO_MAX_ITERS:-2}"
    export MARL_HPO_EVAL_EVERY_ITERS="${MARL_HPO_EVAL_EVERY_ITERS:-1}"
    export MARL_HPO_EVAL_EPISODES="${MARL_HPO_EVAL_EPISODES:-1}"
    ;;
  standard)
    MARL_HPO_COUNT="${MARL_HPO_COUNT:-20}"
    export MARL_HPO_NUM_ENVS="${MARL_HPO_NUM_ENVS:-128}"
    export MARL_HPO_MAX_STEPS="${MARL_HPO_MAX_STEPS:-500}"
    export MARL_HPO_MAX_ITERS="${MARL_HPO_MAX_ITERS:-200}"
    export MARL_HPO_EVAL_EVERY_ITERS="${MARL_HPO_EVAL_EVERY_ITERS:-10}"
    export MARL_HPO_EVAL_EPISODES="${MARL_HPO_EVAL_EPISODES:-4}"
    ;;
  serious)
    MARL_HPO_COUNT="${MARL_HPO_COUNT:-50}"
    export MARL_HPO_NUM_ENVS="${MARL_HPO_NUM_ENVS:-512}"
    export MARL_HPO_MAX_STEPS="${MARL_HPO_MAX_STEPS:-500}"
    export MARL_HPO_MAX_ITERS="${MARL_HPO_MAX_ITERS:-1000}"
    export MARL_HPO_EVAL_EVERY_ITERS="${MARL_HPO_EVAL_EVERY_ITERS:-20}"
    export MARL_HPO_EVAL_EPISODES="${MARL_HPO_EVAL_EPISODES:-4}"
    ;;
  *)
    echo "Unsupported MARL_HPO_MODE=${MARL_HPO_MODE}; expected quick, standard, or serious." >&2
    exit 2
    ;;
esac

export MARL_HPO_DEVICE="${MARL_HPO_DEVICE:-${DEVICE:-cuda}}"
export MARL_HPO_NUM_AGENTS="${MARL_HPO_NUM_AGENTS:-${NUM_AGENTS:-3}}"
export MARL_HPO_SEED="${MARL_HPO_SEED:-${SEED:-1}}"
export MARL_HPO_VIDEO_EVERY_ITERS="${MARL_HPO_VIDEO_EVERY_ITERS:-0}"
export MARL_HPO_TRIALS_DIR

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
echo "  mode:          ${MARL_HPO_MODE}"
echo "  count/family:  ${MARL_HPO_COUNT}"
echo "  families:      ${MARL_HPO_FAMILIES}"
echo "  results dir:   ${MARL_HPO_RESULTS_DIR}"
echo "  trials dir:    ${MARL_HPO_TRIALS_DIR}"
echo "  wm run dir:    ${MARL_HPO_WM_RUN_DIR}"
echo "  budget:        envs=${MARL_HPO_NUM_ENVS}, max_steps=${MARL_HPO_MAX_STEPS}, max_iters=${MARL_HPO_MAX_ITERS}"

for family in $MARL_HPO_FAMILIES; do
  best_config="${MARL_HPO_RESULTS_DIR}/${family}/best_config.json"
  if [[ "$FORCE_MARL_HPO" != "1" && -f "$best_config" ]]; then
    echo "[marl-hpo] ${family}: existing best config found at ${best_config}; skipping."
    continue
  fi
  if [[ "$family" == "mambpo_imagination" ]]; then
    checkpoint_dir="${MARL_HPO_WM_RUN_DIR}/checkpoints"
    if [[ ! -f "${checkpoint_dir}/vae.pt" || ! -f "${checkpoint_dir}/rnn.pt" ]]; then
      echo "[marl-hpo] mambpo_imagination requires trained WM checkpoints at ${checkpoint_dir}" >&2
      echo "Set MARL_HPO_WM_RUN_DIR or run B10_neural_k0.0 world-model first." >&2
      exit 2
    fi
  fi

  sweep_config="$(config_for_family "$family")"
  echo "[marl-hpo] ${family}: creating sweep from ${sweep_config}"
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
  echo "[marl-hpo] ${family}: launching ${MARL_HPO_COUNT} trials (${SWEEP_ID})"
  "$PYTHON_BIN" -m wandb agent --count "$MARL_HPO_COUNT" "$SWEEP_ID"

  echo "[marl-hpo] ${family}: selecting best local trial"
  "$PYTHON_BIN" marl_hpo_registry.py select-best \
    --family "$family" \
    --trials-root "${MARL_HPO_TRIALS_DIR}/${family}" \
    --results-root "$MARL_HPO_RESULTS_DIR" \
    --budget-json "{\"mode\":\"${MARL_HPO_MODE}\",\"count\":${MARL_HPO_COUNT},\"num_envs\":${MARL_HPO_NUM_ENVS},\"max_steps\":${MARL_HPO_MAX_STEPS},\"max_iters\":${MARL_HPO_MAX_ITERS}}"
done

"$PYTHON_BIN" marl_hpo_registry.py write-summary --results-root "$MARL_HPO_RESULTS_DIR" >/dev/null
echo "[marl-hpo] summary written to ${MARL_HPO_RESULTS_DIR}/summary.json"
