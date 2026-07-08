#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Backward-compatible wrapper for the original neural-only HPO entry point.
export HPO_FAMILIES="${HPO_FAMILIES:-neural_k0.0}"
export HPO_COUNT="${HPO_COUNT:-${COUNT:-20}}"

exec ./run_world_model_hpo_pipeline.bash
