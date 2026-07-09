#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
NUM_AGENTS="${NUM_AGENTS:-3}"
SEEDS="${SEEDS:-1 2 3}"
HPO_SEEDS="${HPO_SEEDS:-1 2 3}"
MARL_HPO_SEEDS="${MARL_HPO_SEEDS:-1 2 3}"
HPO_RESULTS_DIR="${HPO_RESULTS_DIR:-hpo_results/world_model}"
MARL_HPO_RESULTS_DIR="${MARL_HPO_RESULTS_DIR:-hpo_results/marl}"
FORCE_WM_HPO="${FORCE_WM_HPO:-0}"
FORCE_MARL_HPO="${FORCE_MARL_HPO:-0}"
DRY_RUN="${DRY_RUN:-0}"

export NUM_AGENTS HPO_SEEDS MARL_HPO_SEEDS HPO_RESULTS_DIR MARL_HPO_RESULTS_DIR
export MODEL_FREE_DOWNSTREAM_ALGO=masac
export MODEL_BASED_DOWNSTREAM_ALGO=mambpo
export REUSE_WM_HPO_CONFIG=1
export REUSE_MARL_HPO_CONFIG=1
export REQUIRE_WM_HPO=1
export REQUIRE_MARL_HPO=1
export REQUIRED_WM_HPO_STAGE=final
export REQUIRED_MARL_HPO_STAGE=final

print_command() {
  printf '[dry-run]'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_command() {
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

run_wm_hpo_stages() {
  local stage
  for stage in screen promote final; do
    echo "=== World Model HPO: neural_k0.0 stage=${stage} ==="
    run_command env \
      HPO_STAGE="$stage" \
      HPO_MODE=serious \
      HPO_FAMILIES=neural_k0.0 \
      HPO_SEEDS="$HPO_SEEDS" \
      HPO_NUM_AGENTS="$NUM_AGENTS" \
      FORCE_WM_HPO="$FORCE_WM_HPO" \
      ./run_world_model_hpo_pipeline.bash
  done
}

run_marl_hpo_stages() {
  local family="$1"
  local wm_run_dir="${2:-}"
  local stage
  for stage in screen promote final; do
    echo "=== MARL HPO: ${family} stage=${stage} ==="
    command=(
      env
      MARL_HPO_STAGE="$stage"
      MARL_HPO_MODE=serious
      MARL_HPO_FAMILIES="$family"
      MARL_HPO_SEEDS="$MARL_HPO_SEEDS"
      MARL_HPO_NUM_AGENTS="$NUM_AGENTS"
      FORCE_MARL_HPO="$FORCE_MARL_HPO"
    )
    if [[ -n "$wm_run_dir" ]]; then
      command+=(MARL_HPO_WM_RUN_DIR="$wm_run_dir" REQUIRE_WM_HPO=1)
    fi
    command+=(./run_marl_hpo_pipeline.bash)
    run_command "${command[@]}"
  done
}

run_final_baseline() {
  local baseline="$1"
  echo "=== Final baseline: ${baseline}, seeds=${SEEDS} ==="
  run_command env \
    BASELINES="$baseline" \
    SEEDS="$SEEDS" \
    NUM_AGENTS="$NUM_AGENTS" \
    RESUME_COMPLETED=0 \
    CONTINUE_ON_ERROR=0 \
    REQUIRE_WM_HPO=1 \
    REQUIRED_WM_HPO_STAGE=final \
    REQUIRE_MARL_HPO=1 \
    REQUIRED_MARL_HPO_STAGE=final \
    MODEL_FREE_DOWNSTREAM_ALGO=masac \
    MODEL_BASED_DOWNSTREAM_ALGO=mambpo \
    ./run_benchmarl_requested_baselines_3agents_fast_scientific.bash
}

echo "Gridcraft neural comparison with mandatory HPO"
echo "  agents:             ${NUM_AGENTS}"
echo "  final seeds:        ${SEEDS}"
echo "  WM HPO seeds:       ${HPO_SEEDS}"
echo "  MARL HPO seeds:     ${MARL_HPO_SEEDS}"
echo "  force WM HPO:       ${FORCE_WM_HPO}"
echo "  force MARL HPO:     ${FORCE_MARL_HPO}"
echo "  WM HPO registry:    ${HPO_RESULTS_DIR}"
echo "  MARL HPO registry:  ${MARL_HPO_RESULTS_DIR}"

echo "=== Phase 1/6: mandatory MASAC core HPO ==="
run_marl_hpo_stages masac_core

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON_BIN" marl_hpo_registry.py validate \
    --family masac_core \
    --root "$MARL_HPO_RESULTS_DIR" \
    --required-stage final \
    --num-agents "$NUM_AGENTS" \
    --minimum-budget-json "{\"seeds\":\"${MARL_HPO_SEEDS}\",\"max_steps\":500,\"max_iters\":1000}"
fi

echo "=== Phase 2/6: final model-free MASAC baseline ==="
run_final_baseline B00_model-free-control

echo "=== Phase 3/6: mandatory neural World Model HPO ==="
run_wm_hpo_stages

echo "=== Phase 4/6: validate selected neural World Model checkpoint ==="
if [[ "$DRY_RUN" == "1" ]]; then
  print_command "$PYTHON_BIN" wm_hpo_registry.py validate \
    --hpo-family neural_k0.0 \
    --root "$HPO_RESULTS_DIR" \
    --required-stage final \
    --num-agents "$NUM_AGENTS" \
    --require-checkpoints \
    --minimum-budget-json "{\"seeds\":\"${HPO_SEEDS}\",\"episodes\":30000,\"max_steps\":500,\"vae_steps\":50000,\"rnn_steps\":50000}" \
    --print-checkpoint-dir
  WM_HPO_RUN_DIR="<best-neural-hpo-run-dir>"
else
  WM_CHECKPOINT_DIR="$("$PYTHON_BIN" wm_hpo_registry.py validate \
    --hpo-family neural_k0.0 \
    --root "$HPO_RESULTS_DIR" \
    --required-stage final \
    --num-agents "$NUM_AGENTS" \
    --require-checkpoints \
    --minimum-budget-json "{\"seeds\":\"${HPO_SEEDS}\",\"episodes\":30000,\"max_steps\":500,\"vae_steps\":50000,\"rnn_steps\":50000}" \
    --print-checkpoint-dir)"
  WM_HPO_RUN_DIR="$(dirname "$WM_CHECKPOINT_DIR")"
  echo "[orchestrator] MAMBPO HPO will use external World Model at ${WM_CHECKPOINT_DIR}"
fi

echo "=== Phase 5/6: mandatory MAMBPO imagination HPO ==="
run_marl_hpo_stages mambpo_imagination "$WM_HPO_RUN_DIR"

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON_BIN" marl_hpo_registry.py validate \
    --family mambpo_imagination \
    --root "$MARL_HPO_RESULTS_DIR" \
    --required-stage final \
    --num-agents "$NUM_AGENTS" \
    --minimum-budget-json "{\"seeds\":\"${MARL_HPO_SEEDS}\",\"max_steps\":500,\"max_iters\":1000}" \
    --external-checkpoint-dir "${WM_HPO_RUN_DIR}/checkpoints"
fi

echo "=== Phase 6/6: final neural World Model + MAMBPO baseline ==="
run_final_baseline B10_neural_k0.0

echo "=== Completed B00/B10 mandatory-HPO campaign ==="
