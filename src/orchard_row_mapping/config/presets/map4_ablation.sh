#!/usr/bin/env bash
set -euo pipefail

# Run an offline ablation comparing multiple tree-circle methods under priors.
#
# Outputs to: /mysda/w/w/lio_ws/maps/ablation_map4_YYYYmmdd_HHMMSS/
#
# Usage:
#   bash src/orchard_row_mapping/config/presets/map4_ablation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

OUT_DIR="${OUT_DIR:-${WS_DIR}/maps/ablation_map4_$(date +%Y%m%d_%H%M%S)}"

python3 "${WS_DIR}/src/orchard_row_mapping/tools/tree_circles_ablation.py" \
  --pcd "${WS_DIR}/maps/map4_label0.pcd" \
  --row-model "${WS_DIR}/maps/row_model_from_map4.json" \
  --out-dir "${OUT_DIR}" \
  --z-min 0.9 --z-max 1.1 \
  --row-bandwidth 0.9 \
  --row-v-offsets '{"4":0.43}' \
  --row-v-yaw-offsets-deg '{"4":2.36}' \
  --u-bin 0.05 --smooth-window 5 \
  --peak-min-fraction 0.05 --min-separation 1.1 --refine-u-half-width 0.45 \
  --cluster-cell-size 0.12 --cluster-neighbor-range 1 --min-points-per-tree 60 \
  --ransac-iters 250 --ransac-inlier-threshold 0.08 --ransac-min-inliers 40 --ransac-min-points 60

# Render BEV SVGs for each method with consistent bounds.
for method in A_row_peaks_median B_row_peaks_ransac C_row_cell_median D_row_cell_ransac; do
  python3 "${WS_DIR}/src/orchard_row_mapping/tools/render_bev_svg.py" \
    --pcd "${WS_DIR}/maps/map4_label0.pcd" \
    --row-model "${WS_DIR}/maps/row_model_from_map4.json" \
    --circles "${OUT_DIR}/${method}/tree_circles.csv" \
    --out "${OUT_DIR}/${method}/bev.svg" \
    --z-min 0.9 --z-max 1.1 \
    --row-v-offsets '{"4":0.43}' \
    --row-v-yaw-offsets-deg '{"4":2.36}' \
    --max-points 40000 --sample-seed 0 \
    --width 1800 --height 1400 --margin 40
done
