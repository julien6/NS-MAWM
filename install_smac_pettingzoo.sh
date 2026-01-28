#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv-smac}"
SMAC_DIR="${SMAC_DIR:-./smac}"

# Dossier final souhaité (FLAT layout):
#   ~/StarCraftII/Versions
#   ~/StarCraftII/Maps
SC2PATH_FLAT_DEFAULT="${SC2PATH_FLAT_DEFAULT:-$HOME/StarCraftII}"

# Version SC2 Linux package (souvent 4.10 dans la pratique SMAC/PySC2)
SC2_VERSION="${SC2_VERSION:-4.10}"
SC2_ZIP="SC2.${SC2_VERSION}.zip"
SC2_URL="https://blzdistsc2-a.akamaihd.net/Linux/${SC2_ZIP}"
SC2_ZIP_PASSWORD="iagreetotheeula"

# =========================
# Helpers
# =========================
need_cmd() { command -v "$1" >/dev/null 2>&1 || {
	echo "Missing command: $1" >&2
	exit 1
}; }

# True si le chemin contient directement Versions/
is_sc2_flat() {
	local sc2path="$1"
	[[ -d "$sc2path/Versions" ]]
}

# True si le chemin contient StarCraftII/Versions (nested)
is_sc2_nested() {
	local sc2path="$1"
	[[ -d "$sc2path/StarCraftII/Versions" ]]
}

# True si maps SMAC présentes au bon endroit dans un layout flat
is_smac_maps_installed() {
	local sc2path="$1"
	[[ -d "$sc2path/Maps/SMAC_Maps" ]]
}

# Flatten: si ~/StarCraftII/StarCraftII/... alors on copie le contenu interne vers ~/StarCraftII/
flatten_sc2_if_nested() {
	local root="$1"
	if is_sc2_flat "$root"; then
		return 0
	fi
	if is_sc2_nested "$root"; then
		echo "==> Flattening existing SC2 layout: $root/StarCraftII -> $root"
		mkdir -p "$root"
		rsync -a "$root/StarCraftII/" "$root/"
		# Optionnel: supprimer le dossier nested pour éviter confusion
		rm -rf "$root/StarCraftII"
		return 0
	fi
	return 1
}

# Détection (sans export manuel):
# On veut retourner un SC2PATH "flat" final (ou le root à flatten).
detect_sc2_root_candidate() {
	# Si l'utilisateur a déjà SC2PATH dans son env, on le respecte,
	# mais on essaie de le "flatten" vers lui.
	if [[ -n "${SC2PATH:-}" ]]; then
		echo "$SC2PATH"
		return 0
	fi

	# Candidats usuels
	local cands=(
		"$HOME/StarCraftII"
		"$HOME/StarCraftII/StarCraftII"
	)

	for p in "${cands[@]}"; do
		if [[ -d "$p" ]]; then
			# flat ou nested détectable
			if [[ -d "$p/Versions" || -d "$p/StarCraftII/Versions" ]]; then
				echo "$p"
				return 0
			fi
		fi
	done

	return 1
}

download_and_install_sc2_flat() {
	local target_root="$1" # ex: ~/StarCraftII (flat souhaité)
	mkdir -p "$target_root"

	echo "==> StarCraft II not found (flat). Installing automatically to: $target_root"
	echo "==> Downloading SC2: $SC2_URL"
	echo "    Note: zip is password-protected; password='$SC2_ZIP_PASSWORD'"

	local tmpdir
	tmpdir="$(mktemp -d)"
	local zip_path="$tmpdir/$SC2_ZIP"

	if command -v curl >/dev/null 2>&1; then
		curl -L -o "$zip_path" "$SC2_URL"
	else
		need_cmd wget
		wget -O "$zip_path" "$SC2_URL"
	fi

	echo "==> Extracting SC2 zip to temp dir (then flatten to target)..."
	local extract_dir="$tmpdir/extract"
	mkdir -p "$extract_dir"
	unzip -q -P "$SC2_ZIP_PASSWORD" "$zip_path" -d "$extract_dir"

	# Détecte si l’archive a un unique dossier racine (souvent StarCraftII/)
	local top_entries
	mapfile -t top_entries < <(find "$extract_dir" -mindepth 1 -maxdepth 1 -printf "%f\n")

	if [[ ${#top_entries[@]} -eq 1 && -d "$extract_dir/${top_entries[0]}" ]]; then
		echo "==> Zip contains a single root folder: ${top_entries[0]} (flattening)"
		rsync -a "$extract_dir/${top_entries[0]}/" "$target_root/"
	else
		echo "==> Zip contains multiple root entries (copying as-is)"
		rsync -a "$extract_dir/" "$target_root/"
	fi

	rm -rf "$tmpdir"

	# Sécurité : s’il reste nested, flatten aussi
	flatten_sc2_if_nested "$target_root" || true

	if ! is_sc2_flat "$target_root"; then
		echo "ERROR: SC2 installation did not produce a flat layout with $target_root/Versions" >&2
		echo "Inspect: ls -la \"$target_root\" and find Versions/" >&2
		exit 1
	fi
}

# =========================
# Main
# =========================
echo "==> Checking required tools..."
need_cmd git
need_cmd "$PYTHON_BIN"
need_cmd rsync
need_cmd unzip

# curl ou wget pour download SC2 si besoin
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
	echo "Missing command: curl or wget (needed if SC2 must be downloaded automatically)" >&2
	exit 1
fi

echo "==> Creating venv: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip..."
python -m pip install -U pip setuptools wheel

echo "==> Cloning SMAC into: $SMAC_DIR"
if [[ ! -d "$SMAC_DIR/.git" ]]; then
	git clone https://github.com/oxwhirl/smac.git "$SMAC_DIR"
else
	echo "==> SMAC repo already exists, pulling latest..."
	git -C "$SMAC_DIR" pull
fi

echo "==> Installing SMAC (editable)..."
python -m pip install -e "$SMAC_DIR"

echo "==> Patching StarCraft2PZEnv.py for PettingZoo reset() compatibility..."

PZ_FILE="$SMAC_DIR/smac/env/pettingzoo/StarCraft2PZEnv.py"

if [[ ! -f "$PZ_FILE" ]]; then
	echo "ERROR: Cannot find $PZ_FILE"
	exit 1
fi

# Check if patch already applied
if grep -q "return self._observe_all(), {agent: {} for agent in self.agents}" "$PZ_FILE"; then
	echo "==> Patch already applied, skipping."
else
	# Replace ONLY the exact legacy return line
	sed -i \
		's/^[[:space:]]*return self._observe_all()[[:space:]]*$/        return self._observe_all(), {agent: {} for agent in self.agents}/' \
		"$PZ_FILE"

	echo "==> Patch applied successfully."
fi

# deps utiles pour PettingZoo example (si utilisé) + pygame
python -m pip install -U pettingzoo pygame numpy

# -------------------------
# SC2 install / detect / flatten
# -------------------------
echo "==> Detecting StarCraft II..."
SC2ROOT_CANDIDATE=""
if SC2ROOT_CANDIDATE="$(detect_sc2_root_candidate)"; then
	echo "==> Found SC2 candidate root: $SC2ROOT_CANDIDATE"
else
	echo "==> No SC2 candidate found."
	SC2ROOT_CANDIDATE="$SC2PATH_FLAT_DEFAULT"
fi

# On veut forcer un layout flat dans $SC2PATH_FLAT_DEFAULT
# - Si candidate est ailleurs, on peut soit l’utiliser, soit copier.
# Ici: on normalise directement dans $SC2PATH_FLAT_DEFAULT (ton souhait).
SC2PATH="$SC2PATH_FLAT_DEFAULT"
mkdir -p "$SC2PATH"

if [[ "$SC2ROOT_CANDIDATE" != "$SC2PATH" ]]; then
	# Si l'utilisateur a déjà une install ailleurs, on préfère ne rien casser :
	# mais comme ton objectif est un layout précis dans ~/StarCraftII,
	# on "synchronise" dans SC2PATH si Versions n'existe pas encore.
	if ! is_sc2_flat "$SC2PATH" && [[ -d "$SC2ROOT_CANDIDATE" ]]; then
		echo "==> Syncing existing SC2 candidate into $SC2PATH (no manual step)."
		rsync -a "$SC2ROOT_CANDIDATE/" "$SC2PATH/"
	fi
fi

# Flatten si nested
flatten_sc2_if_nested "$SC2PATH" || true

# Si toujours pas flat, on télécharge
if ! is_sc2_flat "$SC2PATH"; then
	download_and_install_sc2_flat "$SC2PATH"
fi

# Double-check final layout
if ! is_sc2_flat "$SC2PATH"; then
	echo "ERROR: SC2PATH is not flat: expected $SC2PATH/Versions" >&2
	exit 1
fi

echo "==> Using SC2PATH (flat) = $SC2PATH"

# -------------------------
# Ensure SMAC maps
# -------------------------
echo "==> Ensuring SMAC maps in: $SC2PATH/Maps/SMAC_Maps"
mkdir -p "$SC2PATH/Maps"

if is_smac_maps_installed "$SC2PATH"; then
	echo "==> SMAC_Maps already present."
else
	MAPS_SRC="$SMAC_DIR/smac/env/starcraft2/maps/SMAC_Maps"
	if [[ ! -d "$MAPS_SRC" ]]; then
		echo "ERROR: SMAC_Maps not found at: $MAPS_SRC" >&2
		exit 1
	fi
	rsync -a "$MAPS_SRC" "$SC2PATH/Maps/"
fi

# -------------------------
# Tests (no export required)
# -------------------------
echo "==> Running SMAC smoke test (random agents)..."
SC2PATH="$SC2PATH" python -m smac.examples.random_agents || {
	echo "!! SMAC smoke test failed even though SC2PATH looks correct."
	echo "   Inspect:"
	echo "     ls -la \"$SC2PATH\""
	echo "     ls -la \"$SC2PATH/Versions\""
	echo "     ls -la \"$SC2PATH/Maps/SMAC_Maps\""
	exit 1
}

echo "==> Running PettingZoo example from SMAC (best-effort)..."
PZ_DIR="$SMAC_DIR/smac/examples/pettingzoo"
if [[ -d "$PZ_DIR" ]]; then
	# pick first .py (prefer demo/test/example names if present)
	mapfile -t PZ_FILES < <(ls -1 "$PZ_DIR"/*.py 2>/dev/null || true)
	if [[ ${#PZ_FILES[@]} -eq 0 ]]; then
		echo "==> No .py files found in $PZ_DIR (skipping)."
	else
		TARGET=""
		for f in "${PZ_FILES[@]}"; do
			b="$(basename "$f" | tr '[:upper:]' '[:lower:]')"
			if [[ "$b" == *demo* || "$b" == *test* || "$b" == *example* || "$b" == *run* ]]; then
				TARGET="$f"
				break
			fi
		done
		if [[ -z "$TARGET" ]]; then
			TARGET="${PZ_FILES[0]}"
		fi
		echo "==> Running: $TARGET"
		SC2PATH="$SC2PATH" python "$TARGET" || {
			echo "!! PettingZoo example failed."
			echo "   SMAC is OK (random_agents passed). The PettingZoo example may require extra deps or a different entrypoint."
			exit 1
		}
	fi
else
	echo "==> PettingZoo examples directory not found at $PZ_DIR (skipping)."
fi

echo "==> ✅ Done."
echo "Venv:        $VENV_DIR"
echo "SC2PATH:     $SC2PATH"
echo "Re-run later (no export needed):"
echo "  source \"$VENV_DIR/bin/activate\""
echo "  SC2PATH=\"$SC2PATH\" python -m smac.examples.random_agents"
