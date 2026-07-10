#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BASE_GENERATED_DIRS=(
  record
  series
  log
  trainlog
  tf_models
  tf_vae
  tf_rnn
  tf_initial_z
)

COMPLETE_GENERATED_PATHS=(
  "${BASE_GENERATED_DIRS[@]}"
  .pytest_cache
  campaign_logs
  datasets/gridcraft
  hpo_results
  initial_z
  pstr_viz
  reward_hierarchy_diagnosis
  reward_hierarchy_diagnosis_v2
  runs
  runs_benchmarl
  runs_benchmarl_hpo
  runs_benchmarl_marl_hpo
  shared_models
  vae
  rnn
  wandb
)

print_sizes() {
  local path
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      du -sh "$path" 2>/dev/null || true
    fi
  done
}

remove_paths() {
  local path
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      rm -rf "$path"
    fi
  done
}

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

To inspect large BenchMARL/HPO checkpoints without deleting them, run:

  ./clean_experiment.bash --yes --benchmarl-checkpoints-dry-run

To remove only BenchMARL/HPO checkpoint files while keeping logs, W&B
metadata, summaries, and configs, run:

  ./clean_experiment.bash --yes --benchmarl-checkpoints

To remove all BenchMARL/HPO run directories, run:

  ./clean_experiment.bash --yes --benchmarl-runs

To inspect a full cleanup of generated training/evaluation artifacts, run:

  ./clean_experiment.bash --yes --complete-dry-run

To remove all generated training/evaluation artifacts, including local W&B
files, BenchMARL runs, HPO registries, reusable Gridcraft datasets, cached
latents, shared model caches, generated PSTR visualizations, and diagnostic
outputs, run:

  ./clean_experiment.bash --yes --complete

This keeps source files and scripts, but removes local experiment evidence.
Only use --complete after W&B runs are synced or when local logs are no longer
needed.

EOF
  exit 1
fi

DRY_RUN_ONLY=0
if [[ "$#" -gt 1 ]]; then
  DRY_RUN_ONLY=1
  for arg in "${@:2}"; do
    case "$arg" in
      --benchmarl-checkpoints-dry-run|--complete-dry-run)
        ;;
      *)
        DRY_RUN_ONLY=0
        ;;
    esac
  done
fi

if [[ "$DRY_RUN_ONLY" == "0" ]]; then
  remove_paths "${BASE_GENERATED_DIRS[@]}"

  rm -f \
    vae/vae.json \
    rnn/rnn.json \
    initial_z/initial_z.json

  find . -type d -name __pycache__ -prune -exec rm -rf {} +
fi

for arg in "${@:2}"; do
  case "$arg" in
    --datasets)
      rm -rf datasets/gridcraft
      echo "Reusable vectorized dataset cache removed."
      ;;
    --benchmarl-checkpoints-dry-run)
      echo "BenchMARL/HPO checkpoint files that would be removed:"
      find runs_benchmarl runs_benchmarl_marl_hpo runs_benchmarl_hpo \
        \( -path '*/checkpoints/checkpoint_*.pt' -o -name '*.pt.tmp' -o -name '*.tmp' \) \
        -type f -printf '%s %p\n' 2>/dev/null \
        | sort -nr \
        | awk '{size=$1; $1=""; printf "%.2f GiB%s\n", size/1024/1024/1024, $0}'
      ;;
    --benchmarl-checkpoints)
      echo "Removing BenchMARL/HPO checkpoint files and temporary checkpoint files."
      find runs_benchmarl runs_benchmarl_marl_hpo runs_benchmarl_hpo \
        \( -path '*/checkpoints/checkpoint_*.pt' -o -name '*.pt.tmp' -o -name '*.tmp' \) \
        -type f -delete 2>/dev/null || true
      ;;
    --benchmarl-runs)
      echo "Removing BenchMARL/HPO run directories."
      rm -rf runs_benchmarl/native_marl runs_benchmarl/native_mappo runs_benchmarl_marl_hpo runs_benchmarl_hpo
      ;;
    --complete-dry-run)
      echo "Full cleanup would remove these generated paths:"
      print_sizes "${COMPLETE_GENERATED_PATHS[@]}"
      echo
      echo "It would also remove nested __pycache__ directories."
      ;;
    --complete)
      echo "Removing all generated Gridcraft training/evaluation artifacts."
      remove_paths "${COMPLETE_GENERATED_PATHS[@]}"
      find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
      ;;
    *)
      echo "Unknown cleanup option: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "$DRY_RUN_ONLY" == "0" ]]; then
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
fi
