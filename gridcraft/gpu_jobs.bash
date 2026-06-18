#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-../.venv/bin/python}

"$PYTHON" vae_train.py "$@"
"$PYTHON" series.py
"$PYTHON" rnn_train.py
