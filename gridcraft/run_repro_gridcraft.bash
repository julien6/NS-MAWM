#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

./run_world_model_baselines.bash
./run_policy_baselines.bash
