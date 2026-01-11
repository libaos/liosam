#!/usr/bin/env bash
set -euo pipefail

# Paper-quality preset for:
# - Tree-only map: maps/map4_label0.pcd
# - Row model:     maps/row_model_from_map4.json
# - Bag:           rosbags/2025-10-29-16-05-00.bag
#
# Usage:
#   bash src/orchard_row_mapping/config/presets/map4_best.sh
#   # optional: override BAG file
#   BAG=rosbags/xxx.bag bash src/orchard_row_mapping/config/presets/map4_best.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${WS_DIR}"

unset ROS_MASTER_URI ROS_IP ROS_HOSTNAME ROS_HOME || true

export ROS_PORT="${ROS_PORT:-auto}"
export ENABLE_RVIZ="${ENABLE_RVIZ:-1}"
export ROW_VIEW="${ROW_VIEW:-trees}"
export BAG_RATE="${BAG_RATE:-2}"
export BAG_DURATION="${BAG_DURATION:-60}"

export PCD_PATH="${PCD_PATH:-${WS_DIR}/maps/map4_label0.pcd}"
export ROW_MODEL_FILE="${ROW_MODEL_FILE:-${WS_DIR}/maps/row_model_from_map4.json}"

# Row 4 correction (fits map4 better).
export ROW_V_OFFSETS="${ROW_V_OFFSETS:-{\"4\":0.43}}"
export ROW_V_YAW_OFFSETS_DEG="${ROW_V_YAW_OFFSETS_DEG:-{\"4\":2.36}}"

# Tree circles (row_model_peaks).
export ENABLE_TREE_CIRCLES="${ENABLE_TREE_CIRCLES:-1}"
export TREE_CIRCLES_DETECTION_MODE="${TREE_CIRCLES_DETECTION_MODE:-row_model_peaks}"
export TREE_CIRCLES_SNAP_TO_ROW="${TREE_CIRCLES_SNAP_TO_ROW:-false}"
export TREE_CIRCLES_MAX_POINTS="${TREE_CIRCLES_MAX_POINTS:-0}"
export TREE_CIRCLES_Z_MIN="${TREE_CIRCLES_Z_MIN:-0.9}"
export TREE_CIRCLES_Z_MAX="${TREE_CIRCLES_Z_MAX:-1.1}"
export TREE_CIRCLES_ROW_BANDWIDTH="${TREE_CIRCLES_ROW_BANDWIDTH:-0.9}"
export TREE_CIRCLES_TREE_U_BIN_SIZE="${TREE_CIRCLES_TREE_U_BIN_SIZE:-0.05}"
export TREE_CIRCLES_TREE_SMOOTH_WINDOW="${TREE_CIRCLES_TREE_SMOOTH_WINDOW:-5}"
export TREE_CIRCLES_TREE_PEAK_MIN_FRACTION="${TREE_CIRCLES_TREE_PEAK_MIN_FRACTION:-0.05}"
export TREE_CIRCLES_TREE_MIN_SEPARATION="${TREE_CIRCLES_TREE_MIN_SEPARATION:-1.1}"
export TREE_CIRCLES_REFINE_U_HALF_WIDTH="${TREE_CIRCLES_REFINE_U_HALF_WIDTH:-0.45}"

BAG="${BAG:-rosbags/2025-10-29-16-05-00.bag}"

exec "${WS_DIR}/tools/run_row_prior_from_bag.sh" "${BAG}"

