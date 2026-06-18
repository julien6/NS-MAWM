#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}. Create the virtualenv first, for example: python3.10 -m venv .venv" >&2
  exit 1
fi

echo "Using $(${PYTHON} --version) at ${PYTHON}"

missing=()
for cmd in swig mpicc xvfb-run; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    missing+=("${cmd}")
  fi
done

if ((${#missing[@]} > 0)); then
  echo "Some optional/native build tools are not on PATH: ${missing[*]}"
  echo "On Ubuntu/Debian install: sudo apt-get install -y build-essential swig openmpi-bin libopenmpi-dev xvfb"
  echo "Note: swig is required to build box2d-py when no prebuilt Box2D wheel is available."
fi

"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install -r "${ROOT_DIR}/requirements-worldmodels.txt"

"${PYTHON}" - <<'PY'
import importlib

modules = [
    "tensorflow",
    "gymnasium",
    "Box2D",
    "vizdoom",
    "cma",
    "mpi4py",
    "PIL",
    "pygame",
]

for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")
PY

echo "World Models environment is ready."
