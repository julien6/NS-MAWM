#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-../.venv/bin/python}
SEEDS=${SEEDS:-"1"}
MODEL_BASELINES=${MODEL_BASELINES:-"B24 B25 B26 B27 B28 B29"}
MODEL_POLICIES=${MODEL_POLICIES:-"imagined_mappo mpc_cem"}
RUN_B00=${RUN_B00:-1}
POLICY_UPDATES=${POLICY_UPDATES:-100}
EPISODES_PER_UPDATE=${EPISODES_PER_UPDATE:-8}
POLICY_EVAL_EVERY=${POLICY_EVAL_EVERY:-10}
POLICY_EVAL_EPISODES=${POLICY_EVAL_EPISODES:-10}
MAX_STEPS=${MAX_STEPS:-500}
PLANNING_HORIZON=${PLANNING_HORIZON:-15}
CEM_SAMPLES=${CEM_SAMPLES:-64}
BATCHED_CEM=${BATCHED_CEM:-0}
BATCHED_CEM_SAMPLE_Z=${BATCHED_CEM_SAMPLE_Z:-0}
BATCHED_CEM_SYMBOLIC_MODE=${BATCHED_CEM_SYMBOLIC_MODE:-cpu_projection}
VIDEO_EPISODES=${VIDEO_EPISODES:-1}
VIDEO_MAX_STEPS=${VIDEO_MAX_STEPS:-100}
VIDEO_FPS=${VIDEO_FPS:-10}
NO_WANDB_VIDEOS=${NO_WANDB_VIDEOS:-0}
WANDB_ARGS=()
EXTRA_COMMON_ARGS=()
MODEL_EXTRA_ARGS=()

if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-ns-mawm-gridcraft}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "${WANDB_MODE:-}" ]]; then
    WANDB_ARGS+=(--wandb-mode "$WANDB_MODE")
  fi
fi

if [[ "$NO_WANDB_VIDEOS" == "1" ]]; then
  EXTRA_COMMON_ARGS+=(--no-wandb-videos)
fi
if [[ "$BATCHED_CEM" == "1" ]]; then
  MODEL_EXTRA_ARGS+=(--batched-cem)
fi
if [[ "$BATCHED_CEM_SAMPLE_Z" == "1" ]]; then
  MODEL_EXTRA_ARGS+=(--batched-cem-sample-z)
fi
MODEL_EXTRA_ARGS+=(--batched-cem-symbolic-mode "$BATCHED_CEM_SYMBOLIC_MODE")

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
      "${EXTRA_COMMON_ARGS[@]}" \
      "${WANDB_ARGS[@]}"
  fi

  for baseline in $MODEL_BASELINES; do
    for policy in $MODEL_POLICIES; do
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
        --video-episodes "$VIDEO_EPISODES" \
        --video-max-steps "$VIDEO_MAX_STEPS" \
        --video-fps "$VIDEO_FPS" \
        --seed "$seed" \
        "${EXTRA_COMMON_ARGS[@]}" \
        "${MODEL_EXTRA_ARGS[@]}" \
        "${WANDB_ARGS[@]}"
    done
  done
done
