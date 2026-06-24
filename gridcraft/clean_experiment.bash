#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" != "--yes" ]]; then
  cat <<'EOF'
This will remove generated Gridcraft World Models training artifacts:

  record/
  series/
  log/
  trainlog/
  tf_models/
  tf_vae/
  tf_rnn/
  tf_initial_z/
  vae/vae.json
  rnn/rnn.json
  initial_z/initial_z.json
  __pycache__/ and nested __pycache__ directories

Source files are kept.
Vectorized dataset caches under datasets/gridcraft/ are kept by default so
model-based baselines can reuse collected transitions across runs.

Run again with:

  ./clean_experiment.bash --yes

To also remove reusable vectorized datasets, run:

  ./clean_experiment.bash --yes --datasets

EOF
  exit 1
fi

rm -rf \
  record \
  series \
  log \
  trainlog \
  tf_models \
  tf_vae \
  tf_rnn \
  tf_initial_z

rm -f \
  vae/vae.json \
  rnn/rnn.json \
  initial_z/initial_z.json

find . -type d -name __pycache__ -prune -exec rm -rf {} +

if [[ "${2:-}" == "--datasets" ]]; then
  rm -rf datasets/gridcraft
  echo "Reusable vectorized dataset cache removed."
fi

mkdir -p \
  record \
  series \
  log \
  trainlog \
  vae \
  rnn \
  initial_z

echo "Gridcraft experiment artifacts removed."
echo "Reusable vectorized datasets are kept unless --datasets is passed."
echo "You can start a fresh run with:"
echo "  ./train_world_model.bash"
