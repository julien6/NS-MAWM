#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-../.venv/bin/python}

"$PYTHON" extract.py "$@"
