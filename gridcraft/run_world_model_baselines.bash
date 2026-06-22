#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-../.venv/bin/python}
BASELINES=${BASELINES:-"B10 B24 B25 B26 B27 B28 B29"}
SEEDS=${SEEDS:-"1"}
RNN_STEPS=${RNN_STEPS:-10000}
RNN_BATCH_SIZE=${RNN_BATCH_SIZE:-64}
RNN_SEQ_LEN=${RNN_SEQ_LEN:-32}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-500}
EVAL_HORIZON=${EVAL_HORIZON:-50}
EVAL_HORIZONS=${EVAL_HORIZONS:-"1 5 10 25 50"}
SERIES_LIMIT=${SERIES_LIMIT:-}
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
  for baseline in $BASELINES; do
    EXTRA_ARGS=()
    if [[ -n "$SERIES_LIMIT" ]]; then
      EXTRA_ARGS+=(--series-limit "$SERIES_LIMIT")
    fi
    if [[ -n "$EVAL_HORIZONS" ]]; then
      EXTRA_ARGS+=(--horizons)
      for horizon in $EVAL_HORIZONS; do
        EXTRA_ARGS+=("$horizon")
      done
    fi
    "$PYTHON" run_baseline.py \
      --baseline-id "$baseline" \
      --phase world_model \
      --python "$PYTHON" \
      --steps "$RNN_STEPS" \
      --batch-size "$RNN_BATCH_SIZE" \
      --seq-len "$RNN_SEQ_LEN" \
      --eval-every "$EVAL_EVERY" \
      --episodes "$EVAL_EPISODES" \
      --max-steps "$EVAL_MAX_STEPS" \
      --horizon-steps "$EVAL_HORIZON" \
      --seed "$seed" \
      "${EXTRA_ARGS[@]}" \
      "${WANDB_ARGS[@]}"
  done
done
