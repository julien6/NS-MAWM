#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-../.venv/bin/python}
LOG_DIR=${LOG_DIR:-trainlog}

RNN_STEPS=${RNN_STEPS:-10000}
RNN_BATCH_SIZE=${RNN_BATCH_SIZE:-64}
RNN_SEQ_LEN=${RNN_SEQ_LEN:-32}
RNN_SEED=${RNN_SEED:-1}
LAMBDA_SYM=${LAMBDA_SYM:-1.0}
SYMBOLIC_COVERAGE=${SYMBOLIC_COVERAGE:-1.0}

EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-500}
EVAL_HORIZON=${EVAL_HORIZON:-50}
WANDB_ARGS=()

if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_ARGS+=(--wandb)
  WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT:-ns-mawm-gridcraft}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "${WANDB_MODE:-}" ]]; then
    WANDB_ARGS+=(--wandb-mode "$WANDB_MODE")
  fi
fi

mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished ${name}"
}

run_step 10_ns_series "$PYTHON" series.py

run_step 11_ns_rnn_neural \
  "$PYTHON" rnn_train.py \
    --ns-variant neural \
    --steps "$RNN_STEPS" \
    --batch-size "$RNN_BATCH_SIZE" \
    --seq-len "$RNN_SEQ_LEN" \
    --seed "$RNN_SEED" \
    "${WANDB_ARGS[@]}"

run_step 12_ns_rnn_regularization \
  "$PYTHON" rnn_train.py \
    --ns-variant regularization \
    --lambda-sym "$LAMBDA_SYM" \
    --symbolic-coverage "$SYMBOLIC_COVERAGE" \
    --steps "$RNN_STEPS" \
    --batch-size "$RNN_BATCH_SIZE" \
    --seq-len "$RNN_SEQ_LEN" \
    --seed "$RNN_SEED" \
    "${WANDB_ARGS[@]}"

run_step 13_ns_rnn_residual \
  "$PYTHON" rnn_train.py \
    --ns-variant residual \
    --lambda-sym "$LAMBDA_SYM" \
    --symbolic-coverage "$SYMBOLIC_COVERAGE" \
    --steps "$RNN_STEPS" \
    --batch-size "$RNN_BATCH_SIZE" \
    --seq-len "$RNN_SEQ_LEN" \
    --seed "$RNN_SEED" \
    "${WANDB_ARGS[@]}"

run_step 14_ns_compare \
  "$PYTHON" compare_ns_variants.py \
    --episodes "$EVAL_EPISODES" \
    --max-steps "$EVAL_MAX_STEPS" \
    --horizon-steps "$EVAL_HORIZON" \
    --symbolic-coverage "$SYMBOLIC_COVERAGE" \
    "${WANDB_ARGS[@]}"

echo "NS-MAWM world-model training complete"
echo "summary: ${LOG_DIR}/ns_mawm_summary.json"
