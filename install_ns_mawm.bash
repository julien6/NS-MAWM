#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_WORLDMODELS="${INSTALL_WORLDMODELS:-0}"
INSTALL_LEGACY="${INSTALL_LEGACY:-1}"
LEGACY_ISOLATED="${LEGACY_ISOLATED:-1}"
INSTALL_MAMBPO_LEGACY_REQUIREMENTS="${INSTALL_MAMBPO_LEGACY_REQUIREMENTS:-0}"
RUN_VERIFY="${RUN_VERIFY:-1}"

usage() {
  cat <<'EOF'
Usage: ./install_ns_mawm.bash [options]

Installs the NS-MAWM workspace.

Default behavior:
  - Creates/uses .venv for Gridcraft, vGridcraft, BenchMARL, W&B and current NS-MAWM code.
  - Installs SMAC, Overcooked_AI and MAMBPO in isolated venvs to avoid dependency conflicts.

Options:
  --venv PATH                    Main virtualenv path (default: .venv)
  --python BIN                   Python executable for the main venv (default: python3)
  --with-worldmodels             Also install legacy World Models requirements in the main venv.
  --skip-legacy                  Skip SMAC/Overcooked_AI/MAMBPO installation.
  --legacy-in-main               Install legacy modules into the main venv. Not recommended.
  --mambpo-legacy-requirements   Install MAMBPO/requirements.txt in its isolated venv.
                                 This is off by default because it pins old Python/Torch deps.
  --no-verify                    Skip import checks.
  -h, --help                     Show this help.

Environment variables mirror these flags:
  VENV_DIR, PYTHON_BIN, INSTALL_WORLDMODELS, INSTALL_LEGACY,
  LEGACY_ISOLATED, INSTALL_MAMBPO_LEGACY_REQUIREMENTS, RUN_VERIFY.
EOF
}

while (($#)); do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --with-worldmodels)
      INSTALL_WORLDMODELS=1
      shift
      ;;
    --skip-legacy)
      INSTALL_LEGACY=0
      shift
      ;;
    --legacy-in-main)
      LEGACY_ISOLATED=0
      shift
      ;;
    --mambpo-legacy-requirements)
      INSTALL_MAMBPO_LEGACY_REQUIREMENTS=1
      shift
      ;;
    --no-verify)
      RUN_VERIFY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

main_python() {
  echo "${VENV_DIR}/bin/python"
}

create_venv() {
  local venv_dir="$1"
  local python_bin="$2"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    echo "[install] creating virtualenv: ${venv_dir}"
    "${python_bin}" -m venv "${venv_dir}"
  fi
}

pip_install() {
  local python="$1"
  shift
  "${python}" -m pip install "$@"
}

install_editable_if_present() {
  local python="$1"
  local path="$2"
  local label="$3"
  if [[ -f "${path}/setup.py" || -f "${path}/pyproject.toml" ]]; then
    echo "[install] installing ${label} editable from ${path}"
    pip_install "${python}" -e "${path}"
  else
    echo "[install] skipping ${label}: no setup.py or pyproject.toml at ${path}" >&2
  fi
}

warn_native_tools() {
  local missing=()
  for cmd in swig mpicc; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      missing+=("${cmd}")
    fi
  done
  if ((${#missing[@]} > 0)); then
    cat >&2 <<EOF
[install] optional native build tools missing: ${missing[*]}
[install] On Ubuntu/Debian, consider:
          sudo apt-get install -y build-essential swig openmpi-bin libopenmpi-dev
EOF
  fi
}

verify_main() {
  local python="$1"
  "${python}" - <<'PY'
import importlib

modules = [
    "torch",
    "torchrl",
    "tensordict",
    "wandb",
    "benchmarl",
    "gridcraft",
    "vgridcraft",
    "numpy",
    "pygame",
    "PIL",
]

for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")

import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
PY
}

verify_legacy() {
  local label="$1"
  local python="$2"
  local module="$3"
  if [[ ! -x "${python}" ]]; then
    echo "[verify] ${label}: missing python at ${python}" >&2
    return 0
  fi
  "${python}" - <<PY
import importlib
module = importlib.import_module("${module}")
print("${label}: import ${module} ok", getattr(module, "__version__", "unknown"))
PY
}

install_main_stack() {
  local python
  python="$(main_python)"
  create_venv "${VENV_DIR}" "${PYTHON_BIN}"

  echo "[install] using main python: $("${python}" --version) at ${python}"
  warn_native_tools
  pip_install "${python}" --upgrade pip wheel "setuptools<82"

  echo "[install] installing core BenchMARL/Gridcraft requirements"
  pip_install "${python}" -r "${ROOT_DIR}/requirements-benchmarl-gridcraft.txt"
  pip_install "${python}" "protobuf>=6.31.1,<7"

  install_editable_if_present "${python}" "${ROOT_DIR}/Gridcraft" "Gridcraft"
  install_editable_if_present "${python}" "${ROOT_DIR}/vGridcraft" "vGridcraft"
  install_editable_if_present "${python}" "${ROOT_DIR}/BenchMARL" "BenchMARL"

  if [[ "${INSTALL_WORLDMODELS}" == "1" ]]; then
    echo "[install] installing legacy World Models requirements in main venv"
    pip_install "${python}" -r "${ROOT_DIR}/requirements-worldmodels.txt"
    pip_install "${python}" "protobuf>=6.31.1,<7"
  fi
}

install_legacy_isolated() {
  echo "[install] installing legacy modules in isolated virtualenvs"

  if [[ -d "${ROOT_DIR}/SMAC" ]]; then
    local smac_venv="${ROOT_DIR}/SMAC/.venv_ns_mawm"
    create_venv "${smac_venv}" "${PYTHON_BIN}"
    pip_install "${smac_venv}/bin/python" --upgrade pip wheel "setuptools<82"
    install_editable_if_present "${smac_venv}/bin/python" "${ROOT_DIR}/SMAC" "SMAC"
  fi

  if [[ -d "${ROOT_DIR}/Overcooked_AI" ]]; then
    local overcooked_venv="${ROOT_DIR}/Overcooked_AI/.venv_ns_mawm"
    create_venv "${overcooked_venv}" "${PYTHON_BIN}"
    pip_install "${overcooked_venv}/bin/python" --upgrade pip wheel "setuptools<82"
    install_editable_if_present "${overcooked_venv}/bin/python" "${ROOT_DIR}/Overcooked_AI" "Overcooked_AI"
  fi

  if [[ -d "${ROOT_DIR}/MAMBPO" ]]; then
    local mambpo_venv="${ROOT_DIR}/MAMBPO/.venv_ns_mawm"
    create_venv "${mambpo_venv}" "${PYTHON_BIN}"
    pip_install "${mambpo_venv}/bin/python" --upgrade pip wheel "setuptools<82"
    if [[ "${INSTALL_MAMBPO_LEGACY_REQUIREMENTS}" == "1" ]]; then
      echo "[install] installing MAMBPO legacy requirements; this may fail on modern Python/CUDA stacks"
      pip_install "${mambpo_venv}/bin/python" -r "${ROOT_DIR}/MAMBPO/requirements.txt"
    else
      echo "[install] skipping MAMBPO/requirements.txt by default because it pins old torch/tensorflow/gym versions"
    fi
    install_editable_if_present "${mambpo_venv}/bin/python" "${ROOT_DIR}/MAMBPO" "MAMBPO"
  fi
}

install_legacy_in_main() {
  local python
  python="$(main_python)"
  cat >&2 <<'EOF'
[install] WARNING: installing legacy modules in the main venv can downgrade protobuf
          and break modern W&B. Prefer the default isolated install.
EOF
  install_editable_if_present "${python}" "${ROOT_DIR}/SMAC" "SMAC"
  install_editable_if_present "${python}" "${ROOT_DIR}/Overcooked_AI" "Overcooked_AI"
  install_editable_if_present "${python}" "${ROOT_DIR}/MAMBPO" "MAMBPO"
  pip_install "${python}" "protobuf>=6.31.1,<7"
}

install_main_stack

if [[ "${INSTALL_LEGACY}" == "1" ]]; then
  if [[ "${LEGACY_ISOLATED}" == "1" ]]; then
    install_legacy_isolated
  else
    install_legacy_in_main
  fi
fi

if [[ "${RUN_VERIFY}" == "1" ]]; then
  echo "[verify] main environment"
  verify_main "$(main_python)"

  if [[ "${INSTALL_LEGACY}" == "1" && "${LEGACY_ISOLATED}" == "1" ]]; then
    verify_legacy "SMAC" "${ROOT_DIR}/SMAC/.venv_ns_mawm/bin/python" "smac" || true
    verify_legacy "Overcooked_AI" "${ROOT_DIR}/Overcooked_AI/.venv_ns_mawm/bin/python" "overcooked_ai_py" || true
    verify_legacy "MAMBPO" "${ROOT_DIR}/MAMBPO/.venv_ns_mawm/bin/python" "decentralizedlearning" || true
  fi
fi

cat <<EOF
[install] NS-MAWM installation completed.

Main environment:
  source ${VENV_DIR}/bin/activate

Legacy isolated environments:
  SMAC:          ${ROOT_DIR}/SMAC/.venv_ns_mawm
  Overcooked_AI: ${ROOT_DIR}/Overcooked_AI/.venv_ns_mawm
  MAMBPO:        ${ROOT_DIR}/MAMBPO/.venv_ns_mawm
EOF
