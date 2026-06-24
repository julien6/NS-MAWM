#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements-benchmarl-gridcraft.txt"
"$PYTHON_BIN" -m pip install -e "$ROOT_DIR/vGridcraft"
if [[ -d "$ROOT_DIR/BenchMARL" ]]; then
  "$PYTHON_BIN" -m pip install -e "$ROOT_DIR/BenchMARL"
fi

"$PYTHON_BIN" - <<'PY'
import torch
import torchrl
import tensordict
import vgridcraft
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("torchrl", torchrl.__version__)
print("vgridcraft", vgridcraft.VectorizedGridcraftEnv)
PY
