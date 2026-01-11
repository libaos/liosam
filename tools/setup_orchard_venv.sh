#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$WS_DIR/.venv_orchard}"

source /opt/ros/noetic/setup.bash

if [[ ! -d "$VENV_PATH" ]]; then
  echo "[INFO] Creating venv: $VENV_PATH"
  python3 -m venv --system-site-packages "$VENV_PATH"
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch (works without CUDA)
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "[OK] venv ready: $VENV_PATH"
echo "Next: tools/run_tree_map_from_bag_liorl.sh /path/to.bag"

