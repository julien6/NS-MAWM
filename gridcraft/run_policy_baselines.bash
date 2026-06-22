#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-../.venv/bin/python}
SEEDS=${SEEDS:-"1"}
POLICY_UPDATES=${POLICY_UPDATES:-100}
EPISODES_PER_UPDATE=${EPISODES_PER_UPDATE:-8}
POLICY_EVAL_EVERY=${POLICY_EVAL_EVERY:-10}
POLICY_EVAL_EPISODES=${POLICY_EVAL_EPISODES:-10}
MAX_STEPS=${MAX_STEPS:-500}
PLANNING_HORIZON=${PLANNING_HORIZON:-15}
CEM_SAMPLES=${CEM_SAMPLES:-64}
WANDB_ARGS=()

if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-ns-mawm-gridcraft}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "${WANDB_MODE:-}" ]]; then
    WANDB_ARGS+=(--wandb-mode "$WANDB_MODE")
  fi
fi

for seed in $SEEDS; do
  "$PYTHON" run_baseline.py \
    --baseline-id B00 \
    --phase policy \
    --policy-baseline real_mappo \
    --python "$PYTHON" \
    --policy-updates "$POLICY_UPDATES" \
    --episodes-per-update "$EPISODES_PER_UPDATE" \
    --policy-eval-every "$POLICY_EVAL_EVERY" \
    --policy-eval-episodes "$POLICY_EVAL_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --seed "$seed" \
    "${WANDB_ARGS[@]}"

  for baseline in ${MODEL_BASELINES:-"B24 B25 B26 B27 B28 B29"}; do
    for policy in ${MODEL_POLICIES:-"imagined_mappo mpc_cem"}; do
      "$PYTHON" run_baseline.py \
        --baseline-id "$baseline" \
        --phase policy \
        --policy-baseline "$policy" \
        --python "$PYTHON" \
        --policy-updates "$POLICY_UPDATES" \
        --episodes-per-update "$EPISODES_PER_UPDATE" \
        --policy-eval-every "$POLICY_EVAL_EVERY" \
        --policy-eval-episodes "$POLICY_EVAL_EPISODES" \
        --max-steps "$MAX_STEPS" \
        --planning-horizon "$PLANNING_HORIZON" \
        --cem-samples "$CEM_SAMPLES" \
        --seed "$seed" \
        "${WANDB_ARGS[@]}"
    done
  done
done
