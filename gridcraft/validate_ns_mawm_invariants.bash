#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

echo "=== NS-MAWM invariant validation: compile ==="
"$PYTHON_BIN" -m py_compile \
  BenchMARL/benchmarl/algorithms/mambpo.py \
  gridcraft/run_benchmarl_mappo_gridcraft.py \
  gridcraft/run_benchmarl_marl_gridcraft.py \
  gridcraft/run_benchmarl_gridcraft.py \
  gridcraft/ns_symbolic.py \
  gridcraft/diagnose_ns_mawm.py

echo "=== NS-MAWM invariant validation: diagnostics ==="
(
  cd gridcraft
  ../.venv/bin/python diagnose_ns_mawm.py --strict --json
)

echo "=== NS-MAWM invariant validation: tests ==="
(
  cd gridcraft
  ../.venv/bin/python -m pytest test_ns_symbolic.py test_pstr_rvr_table.py -q
)

echo "=== NS-MAWM invariant validation passed ==="
