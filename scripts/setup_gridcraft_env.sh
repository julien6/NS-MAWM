#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
GRIDCRAFT_DIR="${GRIDCRAFT_DIR:-${ROOT_DIR}/Gridcraft}"
GRIDCRAFT_REPO="${GRIDCRAFT_REPO:-https://github.com/julien6/Gridcraft.git}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}. Create .venv first." >&2
  exit 1
fi

if [[ ! -f "${GRIDCRAFT_DIR}/pyproject.toml" && ! -f "${GRIDCRAFT_DIR}/setup.py" ]]; then
  if [[ -e "${GRIDCRAFT_DIR}" ]]; then
    echo "${GRIDCRAFT_DIR} exists but is not a Python project with pyproject.toml or setup.py." >&2
    echo "Move it away or set GRIDCRAFT_DIR=/path/to/Gridcraft before rerunning." >&2
    exit 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "Missing git and ${GRIDCRAFT_DIR} is not present." >&2
    echo "Install git or clone manually: git clone ${GRIDCRAFT_REPO} ${GRIDCRAFT_DIR}" >&2
    exit 1
  fi
  echo "Cloning Gridcraft from ${GRIDCRAFT_REPO} into ${GRIDCRAFT_DIR}"
  git clone "${GRIDCRAFT_REPO}" "${GRIDCRAFT_DIR}"
fi

"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install -r "${ROOT_DIR}/requirements-worldmodels.txt"
"${PYTHON}" -m pip install -e "${GRIDCRAFT_DIR}"

"${PYTHON}" - <<PY
import sys
sys.path.insert(0, "${GRIDCRAFT_DIR}")
from gridcraft import GridcraftConfig, GridcraftEnv
import pettingzoo

env = GridcraftEnv(GridcraftConfig(width=12, height=12, num_agents=1, max_steps=5))
obs, infos = env.reset(seed=1)
print("gridcraft import ok", GridcraftConfig.__name__, GridcraftEnv.__name__)
print("pettingzoo", getattr(pettingzoo, "__version__", "unknown"))
print("agents", env.agents, "obs keys", list(obs))
env.close()
PY

echo "Gridcraft environment is ready."
