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
ENSURE_LEGACY_FORKS="${ENSURE_LEGACY_FORKS:-1}"
GRIDCRAFT_DIR="${GRIDCRAFT_DIR:-${ROOT_DIR}/Gridcraft}"
VGRIDCRAFT_DIR="${VGRIDCRAFT_DIR:-${ROOT_DIR}/vGridcraft}"
BENCHMARL_DIR="${BENCHMARL_DIR:-${ROOT_DIR}/BenchMARL}"
SMAC_DIR="${SMAC_DIR:-${ROOT_DIR}/SMAC}"
OVERCOOKED_AI_DIR="${OVERCOOKED_AI_DIR:-${ROOT_DIR}/Overcooked_AI}"
MAMBPO_DIR="${MAMBPO_DIR:-${ROOT_DIR}/MAMBPO}"
SMAC_REPO="${SMAC_REPO:-git@github.com:julien6/smac.git}"
OVERCOOKED_AI_REPO="${OVERCOOKED_AI_REPO:-git@github.com:julien6/overcooked_ai.git}"
LEGACY_REPO_BRANCH="${LEGACY_REPO_BRANCH:-ns-mawm}"

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
  --no-legacy-fork-check         Do not verify/clone SMAC and Overcooked_AI forks.
  --no-verify                    Skip import checks.
  -h, --help                     Show this help.

Environment variables mirror these flags:
  VENV_DIR, PYTHON_BIN, INSTALL_WORLDMODELS, INSTALL_LEGACY,
  LEGACY_ISOLATED, INSTALL_MAMBPO_LEGACY_REQUIREMENTS, RUN_VERIFY.
  GRIDCRAFT_DIR, VGRIDCRAFT_DIR, BENCHMARL_DIR, SMAC_DIR,
  OVERCOOKED_AI_DIR, MAMBPO_DIR can point to non-default local checkouts.
  SMAC_REPO, OVERCOOKED_AI_REPO and LEGACY_REPO_BRANCH configure fork checks.
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
    --no-legacy-fork-check)
      ENSURE_LEGACY_FORKS=0
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

is_python_project() {
  local path="$1"
  [[ -f "${path}/setup.py" || -f "${path}/pyproject.toml" ]]
}

install_editable_or_warn() {
  local python="$1"
  local path="$2"
  local label="$3"
  if [[ -d "${path}" ]]; then
    install_editable_if_present "${python}" "${path}" "${label}"
  else
    echo "[install] skipping ${label}: directory not found at ${path}" >&2
  fi
}

warn_uninitialized_project() {
  local path="$1"
  local label="$2"
  if [[ -d "${path}" && ! -f "${path}/setup.py" && ! -f "${path}/pyproject.toml" ]]; then
    cat >&2 <<EOF
[install] ${label} exists at ${path}, but no setup.py/pyproject.toml was found.
[install] If this is a submodule or nested clone, initialize it or set ${label}_DIR explicitly.
EOF
  fi
}

is_git_repo() {
  local path="$1"
  git -C "${path}" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

is_git_dirty() {
  local path="$1"
  [[ -n "$(git -C "${path}" status --porcelain 2>/dev/null)" ]]
}

https_fallback_url() {
  local repo_url="$1"
  if [[ "${repo_url}" =~ ^git@github.com:(.*)\.git$ ]]; then
    echo "https://github.com/${BASH_REMATCH[1]}.git"
  else
    echo "${repo_url}"
  fi
}

fetch_branch_with_fallback() {
  local path="$1"
  local remote_name="$2"
  local repo_url="$3"
  local branch="$4"
  local fallback_remote="${remote_name}_https"
  local fallback_url

  if git -C "${path}" fetch "${remote_name}" "${branch}" >&2; then
    echo "${remote_name}"
    return 0
  fi

  fallback_url="$(https_fallback_url "${repo_url}")"
  if [[ "${fallback_url}" == "${repo_url}" ]]; then
    echo "[repo-check] fetch failed for ${repo_url}; no HTTPS fallback available." >&2
    return 1
  fi

  echo "[repo-check] SSH fetch failed; trying HTTPS fallback ${fallback_url}" >&2
  if git -C "${path}" remote get-url "${fallback_remote}" >/dev/null 2>&1; then
    git -C "${path}" remote set-url "${fallback_remote}" "${fallback_url}"
  else
    git -C "${path}" remote add "${fallback_remote}" "${fallback_url}"
  fi
  git -C "${path}" fetch "${fallback_remote}" "${branch}" >&2
  echo "${fallback_remote}"
}

ensure_fork_checkout() {
  local path="$1"
  local label="$2"
  local repo_url="$3"
  local branch="$4"
  local remote_name="ns_mawm_expected"

  if ! command -v git >/dev/null 2>&1; then
    echo "[repo-check] git is not available; cannot verify ${label} fork." >&2
    return 0
  fi

  if [[ ! -e "${path}" ]]; then
    echo "[repo-check] cloning ${label} from ${repo_url} (${branch}) into ${path}"
    if ! git clone --branch "${branch}" "${repo_url}" "${path}"; then
      local fallback_url
      fallback_url="$(https_fallback_url "${repo_url}")"
      if [[ "${fallback_url}" == "${repo_url}" ]]; then
        echo "[repo-check] clone failed for ${repo_url}; no HTTPS fallback available." >&2
        return 1
      fi
      echo "[repo-check] SSH clone failed; trying HTTPS fallback ${fallback_url}" >&2
      git clone --branch "${branch}" "${fallback_url}" "${path}"
    fi
    return 0
  fi

  if [[ ! -d "${path}" ]]; then
    echo "[repo-check] ${label} path exists but is not a directory: ${path}" >&2
    return 0
  fi

  if ! is_git_repo "${path}"; then
    echo "[repo-check] ${label} at ${path} is not a git repository; leaving it unchanged." >&2
    return 0
  fi

  local origin_url current_branch expected_ref
  origin_url="$(git -C "${path}" remote get-url origin 2>/dev/null || true)"
  current_branch="$(git -C "${path}" branch --show-current 2>/dev/null || true)"

  echo "[repo-check] ${label}: path=${path}"
  echo "[repo-check] ${label}: origin=${origin_url:-none}, branch=${current_branch:-detached}"

  if [[ "${origin_url}" == "${repo_url}" ]]; then
    remote_name="origin"
  else
    echo "[repo-check] ${label}: origin does not match expected fork ${repo_url}; adding/updating remote ${remote_name}."
    if git -C "${path}" remote get-url "${remote_name}" >/dev/null 2>&1; then
      git -C "${path}" remote set-url "${remote_name}" "${repo_url}"
    else
      git -C "${path}" remote add "${remote_name}" "${repo_url}"
    fi
  fi
  expected_ref="refs/remotes/${remote_name}/${branch}"

  echo "[repo-check] ${label}: fetching ${remote_name}/${branch}"
  remote_name="$(fetch_branch_with_fallback "${path}" "${remote_name}" "${repo_url}" "${branch}")"
  expected_ref="refs/remotes/${remote_name}/${branch}"

  if [[ "${current_branch}" == "${branch}" ]]; then
    echo "[repo-check] ${label}: already on ${branch}."
    return 0
  fi

  if is_git_dirty "${path}"; then
    cat >&2 <<EOF
[repo-check] ${label}: worktree has local changes; not switching branch automatically.
[repo-check] ${label}: expected branch is ${branch}. Resolve local changes, then run:
              git -C ${path} checkout ${branch}
EOF
    return 0
  fi

  if git -C "${path}" show-ref --verify --quiet "refs/heads/${branch}"; then
    echo "[repo-check] ${label}: checking out existing local branch ${branch}"
    git -C "${path}" checkout "${branch}"
  else
    echo "[repo-check] ${label}: creating local branch ${branch} from ${remote_name}/${branch}"
    git -C "${path}" checkout -b "${branch}" --track "${expected_ref}"
  fi
}

ensure_legacy_forks() {
  if [[ "${ENSURE_LEGACY_FORKS}" != "1" || "${INSTALL_LEGACY}" != "1" ]]; then
    return 0
  fi
  ensure_fork_checkout "${SMAC_DIR}" "SMAC" "${SMAC_REPO}" "${LEGACY_REPO_BRANCH}"
  ensure_fork_checkout "${OVERCOOKED_AI_DIR}" "Overcooked_AI" "${OVERCOOKED_AI_REPO}" "${LEGACY_REPO_BRANCH}"
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
  # These are needed by Gridcraft rendering/env code even if the editable
  # Gridcraft checkout is absent or not initialized on a given machine.
  pip_install "${python}" numpy gymnasium pettingzoo pygame pillow pytest
  pip_install "${python}" "protobuf>=6.31.1,<7"

  install_editable_or_warn "${python}" "${GRIDCRAFT_DIR}" "Gridcraft"
  install_editable_or_warn "${python}" "${VGRIDCRAFT_DIR}" "vGridcraft"
  install_editable_or_warn "${python}" "${BENCHMARL_DIR}" "BenchMARL"

  if [[ "${INSTALL_WORLDMODELS}" == "1" ]]; then
    echo "[install] installing legacy World Models requirements in main venv"
    pip_install "${python}" -r "${ROOT_DIR}/requirements-worldmodels.txt"
    pip_install "${python}" "protobuf>=6.31.1,<7"
  fi
}

install_legacy_isolated() {
  echo "[install] installing legacy modules in isolated virtualenvs"

  if [[ -d "${SMAC_DIR}" ]]; then
    warn_uninitialized_project "${SMAC_DIR}" "SMAC"
    local smac_venv="${SMAC_DIR}/.venv_ns_mawm"
    create_venv "${smac_venv}" "${PYTHON_BIN}"
    pip_install "${smac_venv}/bin/python" --upgrade pip wheel "setuptools<82"
    install_editable_if_present "${smac_venv}/bin/python" "${SMAC_DIR}" "SMAC"
  else
    echo "[install] skipping SMAC: directory not found at ${SMAC_DIR}" >&2
  fi

  if [[ -d "${OVERCOOKED_AI_DIR}" ]]; then
    warn_uninitialized_project "${OVERCOOKED_AI_DIR}" "Overcooked_AI"
    local overcooked_venv="${OVERCOOKED_AI_DIR}/.venv_ns_mawm"
    create_venv "${overcooked_venv}" "${PYTHON_BIN}"
    pip_install "${overcooked_venv}/bin/python" --upgrade pip wheel "setuptools<82"
    install_editable_if_present "${overcooked_venv}/bin/python" "${OVERCOOKED_AI_DIR}" "Overcooked_AI"
  else
    echo "[install] skipping Overcooked_AI: directory not found at ${OVERCOOKED_AI_DIR}" >&2
  fi

  if [[ -d "${MAMBPO_DIR}" ]]; then
    warn_uninitialized_project "${MAMBPO_DIR}" "MAMBPO"
    local mambpo_venv="${MAMBPO_DIR}/.venv_ns_mawm"
    create_venv "${mambpo_venv}" "${PYTHON_BIN}"
    pip_install "${mambpo_venv}/bin/python" --upgrade pip wheel "setuptools<82"
    if [[ "${INSTALL_MAMBPO_LEGACY_REQUIREMENTS}" == "1" ]]; then
      echo "[install] installing MAMBPO legacy requirements; this may fail on modern Python/CUDA stacks"
      pip_install "${mambpo_venv}/bin/python" -r "${MAMBPO_DIR}/requirements.txt"
    else
      echo "[install] skipping MAMBPO/requirements.txt by default because it pins old torch/tensorflow/gym versions"
    fi
    install_editable_if_present "${mambpo_venv}/bin/python" "${MAMBPO_DIR}" "MAMBPO"
  else
    echo "[install] skipping MAMBPO: directory not found at ${MAMBPO_DIR}" >&2
  fi
}

install_legacy_in_main() {
  local python
  python="$(main_python)"
  cat >&2 <<'EOF'
[install] WARNING: installing legacy modules in the main venv can downgrade protobuf
          and break modern W&B. Prefer the default isolated install.
EOF
  install_editable_or_warn "${python}" "${SMAC_DIR}" "SMAC"
  install_editable_or_warn "${python}" "${OVERCOOKED_AI_DIR}" "Overcooked_AI"
  install_editable_or_warn "${python}" "${MAMBPO_DIR}" "MAMBPO"
  pip_install "${python}" "protobuf>=6.31.1,<7"
}

install_main_stack

ensure_legacy_forks

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
    if is_python_project "${SMAC_DIR}"; then
      verify_legacy "SMAC" "${SMAC_DIR}/.venv_ns_mawm/bin/python" "smac" || true
    fi
    if is_python_project "${OVERCOOKED_AI_DIR}"; then
      verify_legacy "Overcooked_AI" "${OVERCOOKED_AI_DIR}/.venv_ns_mawm/bin/python" "overcooked_ai_py" || true
    fi
    if is_python_project "${MAMBPO_DIR}"; then
      verify_legacy "MAMBPO" "${MAMBPO_DIR}/.venv_ns_mawm/bin/python" "decentralizedlearning" || true
    fi
  fi
fi

cat <<EOF
[install] NS-MAWM installation completed.

Main environment:
  source ${VENV_DIR}/bin/activate

Legacy isolated environments:
  SMAC:          ${SMAC_DIR}/.venv_ns_mawm
  Overcooked_AI: ${OVERCOOKED_AI_DIR}/.venv_ns_mawm
  MAMBPO:        ${MAMBPO_DIR}/.venv_ns_mawm
EOF
