#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG=${1:-sweeps/smoke.yaml}
PROJECT=${WANDB_PROJECT:-ns-mawm-gridcraft}
ARGS=(sweep --project "$PROJECT")

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ARGS+=(--entity "$WANDB_ENTITY")
fi

ARGS+=("$CONFIG")

../.venv/bin/wandb "${ARGS[@]}"
