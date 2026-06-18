#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}. Create .venv first." >&2
  exit 1
fi

"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install -r "${ROOT_DIR}/requirements-worldmodels.txt"
"${PYTHON}" -m pip install -e "${ROOT_DIR}/Gridcraft"

"${PYTHON}" - <<PY
import sys
sys.path.insert(0, "${ROOT_DIR}/Gridcraft")
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
