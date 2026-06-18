#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-../.venv/bin/python}
LOG_DIR=${LOG_DIR:-trainlog}

EXTRACT_EPISODES=${EXTRACT_EPISODES:-5000}
EXTRACT_MAX_STEPS=${EXTRACT_MAX_STEPS:-500}
EXTRACT_SEED=${EXTRACT_SEED:-1}

VAE_STEPS=${VAE_STEPS:-10000}
VAE_BATCH_SIZE=${VAE_BATCH_SIZE:-256}
VAE_SEED=${VAE_SEED:-1}

RNN_STEPS=${RNN_STEPS:-10000}
RNN_BATCH_SIZE=${RNN_BATCH_SIZE:-64}
RNN_SEQ_LEN=${RNN_SEQ_LEN:-32}
RNN_SEED=${RNN_SEED:-1}

ES_EPISODES=${ES_EPISODES:-16}
ES_WORKERS=${ES_WORKERS:-64}
ES_GENERATIONS=${ES_GENERATIONS:-100}
ES_MAX_LEN=${ES_MAX_LEN:-500}
ES_OPTIMIZER=${ES_OPTIMIZER:-cma}
ES_SEED=${ES_SEED:-1}

mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished ${name}"
}

run_step 01_extract \
  "$PYTHON" extract.py \
    --episodes "$EXTRACT_EPISODES" \
    --max-steps "$EXTRACT_MAX_STEPS" \
    --seed "$EXTRACT_SEED"

run_step 02_vae \
  "$PYTHON" vae_train.py \
    --steps "$VAE_STEPS" \
    --batch-size "$VAE_BATCH_SIZE" \
    --seed "$VAE_SEED"

run_step 03_series \
  "$PYTHON" series.py

run_step 04_rnn \
  "$PYTHON" rnn_train.py \
    --steps "$RNN_STEPS" \
    --batch-size "$RNN_BATCH_SIZE" \
    --seq-len "$RNN_SEQ_LEN" \
    --seed "$RNN_SEED"

run_step 05_controller \
  "$PYTHON" train.py \
    -n "$ES_EPISODES" \
    -t "$ES_WORKERS" \
    -o "$ES_OPTIMIZER" \
    --generations "$ES_GENERATIONS" \
    --max_len "$ES_MAX_LEN" \
    --seed_start "$ES_SEED"

echo "training complete"
echo "best controller: log/gridcraftrnn.${ES_OPTIMIZER}.${ES_EPISODES}.${ES_WORKERS}.best.json"
