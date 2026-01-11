#!/usr/bin/env bash
set -euo pipefail

# Render per-frame BEV SVGs along the map4 rosbag trajectory.
#
# Outputs to: /mysda/w/w/lio_ws/maps/bev_sequence_YYYYmmdd_HHMMSS/
#
# Usage:
#   bash src/orchard_row_mapping/config/presets/map4_bev_sequence.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

OUT_DIR="${OUT_DIR:-${WS_DIR}/maps/bev_sequence_$(date +%Y%m%d_%H%M%S)}"

python3 "${WS_DIR}/src/orchard_row_mapping/tools/render_bev_sequence.py" \
  --bag "${WS_DIR}/rosbags/2025-10-29-16-05-00.bag" \
  --pcd "${WS_DIR}/maps/map4_label0.pcd" \
  --row-model "${WS_DIR}/maps/row_model_from_map4.json" \
  --out-dir "${OUT_DIR}" \
  --sample-rate 1.0 \
  --bev-span 25 \
  --z-min 0.9 --z-max 1.1 \
  --row-v-offsets '{"4":0.43}' \
  --row-v-yaw-offsets-deg '{"4":2.36}' \
  --u-bin 0.05 --smooth-window 5 \
  --peak-min-fraction 0.05 --min-separation 0.9 --refine-u-half-width 0.45 \
  --max-points 40000 --sample-seed 0 \
  --width 1800 --height 1400 --margin 40
