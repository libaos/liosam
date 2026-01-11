#!/usr/bin/env bash
set -euo pipefail

# Render per-frame BEV SVGs using /points_raw + /tf (vehicle-centric).
#
# Outputs to: /mysda/w/w/lio_ws/maps/bev_points_raw_YYYYmmdd_HHMMSS/
#
# Usage:
#   bash src/orchard_row_mapping/config/presets/map4_bev_sequence_points_raw.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

OUT_DIR="${OUT_DIR:-${WS_DIR}/maps/bev_points_raw_$(date +%Y%m%d_%H%M%S)}"

python3 "${WS_DIR}/src/orchard_row_mapping/tools/render_bev_sequence_points_raw.py" \
  --bag "${WS_DIR}/rosbags/2025-10-29-16-05-00.bag" \
  --out-dir "${OUT_DIR}" \
  --points-topic "/points_raw" \
  --tf-topic "/tf" \
  --row-model "${WS_DIR}/maps/row_model_from_map4.json" \
  --sample-rate 1.0 \
  --bev-span 20 \
  --line-length 20 \
  --show-robot-marker \
  --robot-color "#ff6f00" \
  --robot-size-m 0.6 \
  --z-min 0.9 --z-max 1.1 \
  --row-bandwidth 0.9 \
  --hist-bin-size 0.2 --hist-smooth-window 7 --hist-peak-min-fraction 0.15 \
  --u-bin 0.05 --smooth-window 5 \
  --peak-min-fraction 0.05 --min-separation 0.9 --refine-u-half-width 0.45 \
  --max-points 40000 --sample-seed 0 \
  --width 1800 --height 1400 --margin 40
