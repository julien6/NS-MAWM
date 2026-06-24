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

BENCHMARL_DIR="${BENCHMARL_DIR:-$ROOT_DIR/BenchMARL}"
if [[ -f "$BENCHMARL_DIR/setup.py" || -f "$BENCHMARL_DIR/pyproject.toml" ]]; then
  "$PYTHON_BIN" -m pip install -e "$BENCHMARL_DIR"
else
  cat >&2 <<EOF
BenchMARL was not installed because '$BENCHMARL_DIR' is not a Python project.
This is fine for run_benchmarl_gridcraft.py, which uses the standalone vectorized
runner. For native BenchMARL MAPPO, clone BenchMARL and rerun with:

  BENCHMARL_DIR=/path/to/BenchMARL ./scripts/setup_benchmarl_gridcraft_env.sh

EOF
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
