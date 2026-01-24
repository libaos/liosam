#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_PREFIX_DEFAULT="$WS_DIR/conda_envs/randla39"
ENV_PREFIX="${RANDLA_ENV_PREFIX:-${ENV_PREFIX:-$ENV_PREFIX_DEFAULT}}"

ENV_FILE_DEFAULT="$WS_DIR/conda/envs/randla39.yml"
ENV_FILE_LEGACY="$WS_DIR/conda/envs/randlanet39.yml"
if [[ -n "${ENV_FILE:-}" ]]; then
  ENV_FILE="$ENV_FILE"
elif [[ -f "$ENV_FILE_DEFAULT" ]]; then
  ENV_FILE="$ENV_FILE_DEFAULT"
else
  ENV_FILE="$ENV_FILE_LEGACY"
fi

CONDARC_DEFAULT="$WS_DIR/conda/condarc.yml"

if [[ -d "$ENV_PREFIX/conda-meta" ]]; then
  echo "[OK] env already exists: $ENV_PREFIX"
  exit 0
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] env file not found: $ENV_FILE" >&2
  exit 2
fi

CONDA_EXE_CANDIDATES=()
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_EXE_CANDIDATES+=("$CONDA_EXE")
fi
CONDA_EXE_CANDIDATES+=("$WS_DIR/miniforge3/bin/conda")

CONDA=""
for candidate in "${CONDA_EXE_CANDIDATES[@]}"; do
  if [[ -x "$candidate" ]]; then
    CONDA="$candidate"
    break
  fi
done
if [[ -z "$CONDA" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA="$(command -v conda)"
  fi
fi
if [[ -z "$CONDA" ]]; then
  echo "[ERROR] conda not found." >&2
  echo "        Install Miniforge/Conda first, or set CONDA_EXE=/path/to/conda" >&2
  exit 3
fi

mkdir -p "$(dirname "$ENV_PREFIX")"

echo "[INFO] Workspace : $WS_DIR"
echo "[INFO] Conda     : $CONDA"
echo "[INFO] Env file  : $ENV_FILE"
echo "[INFO] Prefix    : $ENV_PREFIX"

CONDARC_TO_USE=""
if [[ -n "${CONDARC:-}" && -f "${CONDARC:-}" ]]; then
  CONDARC_TO_USE="$CONDARC"
elif [[ -f "$CONDARC_DEFAULT" ]]; then
  CONDARC_TO_USE="$CONDARC_DEFAULT"
fi

if [[ -n "$CONDARC_TO_USE" ]]; then
  echo "[INFO] CONDARC   : $CONDARC_TO_USE"
  CONDARC="$CONDARC_TO_USE" "$CONDA" --no-plugins env create -p "$ENV_PREFIX" -f "$ENV_FILE"
else
  "$CONDA" --no-plugins env create -p "$ENV_PREFIX" -f "$ENV_FILE"
fi

echo "[INFO] Sanity check..."
"$CONDA" --no-plugins run -p "$ENV_PREFIX" python -c "import sys,torch,numpy; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('numpy', numpy.__version__)"

echo "[OK] Ready."
echo "Next:"
echo "  export RANDLA_ENV_PREFIX=\"$ENV_PREFIX\""
echo "  # then run roslaunch / tools that use RANDLA_ENV_PREFIX"
