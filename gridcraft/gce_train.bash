#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-../.venv/bin/python}

"$PYTHON" train.py -n 1 -t 1 -o cma "$@"
