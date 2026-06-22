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
export MODEL_BASELINES=${MODEL_BASELINES:-"B24 B25 B26 B27 B28 B29"}
export MODEL_POLICIES=${MODEL_POLICIES:-"imagined_mappo mpc_cem"}

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

export VIDEO_EPISODES=${VIDEO_EPISODES:-1}
export VIDEO_MAX_STEPS=${VIDEO_MAX_STEPS:-100}
export VIDEO_FPS=${VIDEO_FPS:-10}

./run_world_model_baselines.bash
./run_policy_baselines.bash
