#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Requested protocol:
# - B00: model-free MARL in the real Gridcraft environment.
# - B24/B25/B26: NS-MAWM regularization/projection/residual with coverage 0.3.
# - B27/B28/B29: NS-MAWM regularization/projection/residual with coverage 0.6.
#
# Defaults are intentionally heavier than smoke tests. Override any variable
# from the shell when running on a larger or smaller machine.

export WANDB=${WANDB:-1}
export WANDB_PROJECT=${WANDB_PROJECT:-ns-mawm-gridcraft}
export PYTHON=${PYTHON:-../.venv/bin/python}
export SEEDS=${SEEDS:-"1 2 3"}

export BASELINES=${BASELINES:-"B24 B25 B26 B27 B28 B29"}
export RUN_B00=${RUN_B00:-1}

export EXTRACT_EPISODES=${EXTRACT_EPISODES:-20000}
export EXTRACT_MAX_STEPS=${EXTRACT_MAX_STEPS:-500}
export TRAIN_VAE=${TRAIN_VAE:-1}
export VAE_STEPS=${VAE_STEPS:-50000}
export VAE_BATCH_SIZE=${VAE_BATCH_SIZE:-256}

export RNN_STEPS=${RNN_STEPS:-100000}
export RNN_BATCH_SIZE=${RNN_BATCH_SIZE:-64}
export RNN_SEQ_LEN=${RNN_SEQ_LEN:-64}
export EVAL_EVERY=${EVAL_EVERY:-5000}
export EVAL_EPISODES=${EVAL_EPISODES:-200}
export EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-500}
export EVAL_HORIZON=${EVAL_HORIZON:-100}
export EVAL_HORIZONS=${EVAL_HORIZONS:-"1 5 10 25 50 100"}

export POLICY_UPDATES=${POLICY_UPDATES:-2000}
export EPISODES_PER_UPDATE=${EPISODES_PER_UPDATE:-16}
export POLICY_EVAL_EVERY=${POLICY_EVAL_EVERY:-25}
export POLICY_EVAL_EPISODES=${POLICY_EVAL_EPISODES:-50}
export MAX_STEPS=${MAX_STEPS:-500}
export PLANNING_HORIZON=${PLANNING_HORIZON:-25}
export CEM_SAMPLES=${CEM_SAMPLES:-256}
export BATCHED_CEM=${BATCHED_CEM:-1}
export BATCHED_CEM_SAMPLE_Z=${BATCHED_CEM_SAMPLE_Z:-0}
export BATCHED_CEM_SYMBOLIC_MODE=${BATCHED_CEM_SYMBOLIC_MODE:-cpu_projection}

export VIDEO_EPISODES=${VIDEO_EPISODES:-1}
export VIDEO_MAX_STEPS=${VIDEO_MAX_STEPS:-100}
export VIDEO_FPS=${VIDEO_FPS:-10}
export NO_WANDB_VIDEOS=${NO_WANDB_VIDEOS:-0}

WANDB_ARGS=()
COMMON_ARGS=()
HORIZON_ARGS=()

if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "${WANDB_MODE:-}" ]]; then
    WANDB_ARGS+=(--wandb-mode "$WANDB_MODE")
  fi
fi

if [[ "$TRAIN_VAE" == "1" ]]; then
  COMMON_ARGS+=(--train-vae)
fi
if [[ "$NO_WANDB_VIDEOS" == "1" ]]; then
  COMMON_ARGS+=(--no-wandb-videos)
fi
if [[ "$BATCHED_CEM" == "1" ]]; then
  COMMON_ARGS+=(--batched-cem)
fi
if [[ "$BATCHED_CEM_SAMPLE_Z" == "1" ]]; then
  COMMON_ARGS+=(--batched-cem-sample-z)
fi
COMMON_ARGS+=(--batched-cem-symbolic-mode "$BATCHED_CEM_SYMBOLIC_MODE")

if [[ -n "$EVAL_HORIZONS" ]]; then
  HORIZON_ARGS+=(--horizons)
  for horizon in $EVAL_HORIZONS; do
    HORIZON_ARGS+=("$horizon")
  done
fi

for seed in $SEEDS; do
  if [[ "$RUN_B00" == "1" ]]; then
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
      --video-episodes "$VIDEO_EPISODES" \
      --video-max-steps "$VIDEO_MAX_STEPS" \
      --video-fps "$VIDEO_FPS" \
      --seed "$seed" \
      "${COMMON_ARGS[@]}" \
      "${WANDB_ARGS[@]}"
  fi

  for baseline in $BASELINES; do
    "$PYTHON" run_baseline.py \
      --baseline-id "$baseline" \
      --phase all \
      --policy-baseline all \
      --python "$PYTHON" \
      --steps "$RNN_STEPS" \
      --batch-size "$RNN_BATCH_SIZE" \
      --seq-len "$RNN_SEQ_LEN" \
      --extract-episodes "$EXTRACT_EPISODES" \
      --extract-max-steps "$EXTRACT_MAX_STEPS" \
      --vae-steps "$VAE_STEPS" \
      --vae-batch-size "$VAE_BATCH_SIZE" \
      --eval-every "$EVAL_EVERY" \
      --episodes "$EVAL_EPISODES" \
      --max-steps "$MAX_STEPS" \
      --horizon-steps "$EVAL_HORIZON" \
      --policy-updates "$POLICY_UPDATES" \
      --episodes-per-update "$EPISODES_PER_UPDATE" \
      --policy-eval-every "$POLICY_EVAL_EVERY" \
      --policy-eval-episodes "$POLICY_EVAL_EPISODES" \
      --planning-horizon "$PLANNING_HORIZON" \
      --cem-samples "$CEM_SAMPLES" \
      --video-episodes "$VIDEO_EPISODES" \
      --video-max-steps "$VIDEO_MAX_STEPS" \
      --video-fps "$VIDEO_FPS" \
      --seed "$seed" \
      "${COMMON_ARGS[@]}" \
      "${HORIZON_ARGS[@]}" \
      "${WANDB_ARGS[@]}"
  done
done
