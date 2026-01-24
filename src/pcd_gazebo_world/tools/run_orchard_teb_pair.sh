#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT="${1:-11347}"
MODE="${2:-velodyne}" # velodyne | static_map | no_obstacles

SKIP_BASELINE="${SKIP_BASELINE:-false}"
SKIP_VIAPOINTS="${SKIP_VIAPOINTS:-false}"

REF_JSON_DEFAULT="${WS_DIR}/src/pcd_gazebo_world/maps/runs/rosbag_path_map.json"
WORLD_DEFAULT="${WS_DIR}/src/pcd_gazebo_world/worlds/orchard_from_cloud_registered_full_0p7_1p3.world"
MAP_YAML_DEFAULT="${WS_DIR}/src/pcd_gazebo_world/maps/orchard_from_cloud_registered_full_0p7_1p3/map.yaml"
CIRCLES_DEFAULT="${WS_DIR}/rosbags/runs/tree_centers_from_cloud_registered_full_0p7_1p3.json"

REF_JSON="${REF_JSON:-${REF_JSON_DEFAULT}}"
WORLD_FILE="${WORLD_FILE:-${WORLD_DEFAULT}}"
MAP_YAML="${MAP_YAML:-${MAP_YAML_DEFAULT}}"
TREE_CIRCLES_JSON="${TREE_CIRCLES_JSON:-${CIRCLES_DEFAULT}}"

LIDAR_HZ="${LIDAR_HZ:-10}"
LIDAR_SAMPLES="${LIDAR_SAMPLES:-900}"
TREE_CLEARANCE="${TREE_CLEARANCE:-0.30}"
TREE_DEFAULT_RADIUS="${TREE_DEFAULT_RADIUS:-0.15}"
GOAL_TIMEOUT_S="${GOAL_TIMEOUT_S:-60}"
WAYPOINT_MIN_DIST="${WAYPOINT_MIN_DIST:-0.5}"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-1800}"
WAYPOINT_TOLERANCE_BASELINE="${WAYPOINT_TOLERANCE_BASELINE:-}"
WAYPOINT_TOLERANCE_VIAPOINTS="${WAYPOINT_TOLERANCE_VIAPOINTS:-}"

COMPARE_ONE="${WS_DIR}/src/pcd_gazebo_world/scripts/plot_reference_vs_replay.py"
COMPARE_TWO="${WS_DIR}/src/pcd_gazebo_world/scripts/plot_reference_vs_two_replays.py"

abs_path() {
  python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$1"
}

REF_JSON="$(abs_path "${REF_JSON}")"
WORLD_FILE="$(abs_path "${WORLD_FILE}")"
MAP_YAML="$(abs_path "${MAP_YAML}")"
TREE_CIRCLES_JSON="$(abs_path "${TREE_CIRCLES_JSON}")"

if [[ ! -f "${WS_DIR}/devel/setup.bash" ]]; then
  echo "ERROR: cannot find ${WS_DIR}/devel/setup.bash (did you build the workspace?)" >&2
  exit 1
fi
for f in "${REF_JSON}" "${WORLD_FILE}" "${MAP_YAML}" "${TREE_CIRCLES_JSON}" "${COMPARE_ONE}" "${COMPARE_TWO}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}" >&2
    exit 1
  fi
done

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${WS_DIR}/trajectory_data/pair_${RUN_TAG}_${MODE}_${LIDAR_HZ}hz_${LIDAR_SAMPLES}samples"
mkdir -p "${RUN_DIR}"

echo "[pair] run_dir=${RUN_DIR}"
echo "[pair] mode=${MODE}"
echo "[pair] ref=${REF_JSON}"
echo "[pair] world=${WORLD_FILE}"
echo "[pair] map=${MAP_YAML}"
echo "[pair] circles=${TREE_CIRCLES_JSON}"
echo "[pair] lidar=${LIDAR_HZ}Hz samples=${LIDAR_SAMPLES}"
echo "[pair] goal_timeout_s=${GOAL_TIMEOUT_S} tree_clearance=${TREE_CLEARANCE}"
echo

run_one() {
  local label="$1"
  local use_via_points="$2"
  local gazebo_port="$3"
  local ros_port="$4"

  local csv="${RUN_DIR}/${label}_odom.csv"
  local log="${RUN_DIR}/${label}.log"
  local svg="${RUN_DIR}/ref_vs_${label}.svg"
  local report="${RUN_DIR}/${label}_report.json"
  local summary="${RUN_DIR}/ref_vs_${label}.summary.json"

  echo "[run] ${label} (use_via_points=${use_via_points})"
  echo "      csv=${csv}"

  (
    export USE_VIA_POINTS="${use_via_points}"
    export RECORD_CSV="${csv}"
    export SHUTDOWN_ON_PATH_DONE="true"

    export ROS_MASTER_URI="http://localhost:${ros_port}"
    export ROS_IP="127.0.0.1"

    export LIDAR_HZ="${LIDAR_HZ}"
    export LIDAR_SAMPLES="${LIDAR_SAMPLES}"

    export MAP_YAML="${MAP_YAML}"
    export TREE_CIRCLES_JSON="${TREE_CIRCLES_JSON}"
    export TREE_CLEARANCE="${TREE_CLEARANCE}"
    export TREE_DEFAULT_RADIUS="${TREE_DEFAULT_RADIUS}"
    export GOAL_TIMEOUT_S="${GOAL_TIMEOUT_S}"
    export WAYPOINT_MIN_DIST="${WAYPOINT_MIN_DIST}"

    unset WAYPOINT_TOLERANCE || true
    if [[ "${label}" == "baseline" && -n "${WAYPOINT_TOLERANCE_BASELINE}" ]]; then
      export WAYPOINT_TOLERANCE="${WAYPOINT_TOLERANCE_BASELINE}"
    fi
    if [[ "${label}" == "viapoints" && -n "${WAYPOINT_TOLERANCE_VIAPOINTS}" ]]; then
      export WAYPOINT_TOLERANCE="${WAYPOINT_TOLERANCE_VIAPOINTS}"
    fi

    timeout -s INT "${RUN_TIMEOUT_S}" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_teb_server.sh" "${gazebo_port}" "${REF_JSON}" "${WORLD_FILE}" "${MODE}"
  ) >"${log}" 2>&1 || true

  if [[ ! -f "${csv}" ]]; then
    echo "[warn] missing trajectory csv: ${csv}"
    echo "       log: ${log}"
    return 0
  fi

  /usr/bin/python3 "${COMPARE_ONE}" --reference "${REF_JSON}" --replay "${csv}" --out "${svg}" --report "${report}" >"${summary}" || true
  echo "      svg=${svg}"
  echo "      report=${report}"
  echo
}

BASE_GZ_PORT="${PORT}"
VIA_GZ_PORT="$((PORT + 1))"
BASE_ROS_PORT="$((PORT + 1000))"
VIA_ROS_PORT="$((PORT + 1001))"

if [[ "${SKIP_BASELINE}" != "true" ]]; then
  run_one "baseline" "false" "${BASE_GZ_PORT}" "${BASE_ROS_PORT}"
  sleep 6
fi

if [[ "${SKIP_VIAPOINTS}" != "true" ]]; then
  run_one "viapoints" "true" "${VIA_GZ_PORT}" "${VIA_ROS_PORT}"
  sleep 2
fi

if [[ -f "${RUN_DIR}/baseline_odom.csv" && -f "${RUN_DIR}/viapoints_odom.csv" ]]; then
  TWO_SVG="${RUN_DIR}/ref_vs_two_replays.svg"
  TWO_REPORT="${RUN_DIR}/ref_vs_two_replays.report.json"
  TWO_SUMMARY="${RUN_DIR}/ref_vs_two_replays.summary.json"
  /usr/bin/python3 "${COMPARE_TWO}" \
    --reference "${REF_JSON}" \
    --replay-a "${RUN_DIR}/baseline_odom.csv" --label-a "baseline" \
    --replay-b "${RUN_DIR}/viapoints_odom.csv" --label-b "viapoints" \
    --circles "${TREE_CIRCLES_JSON}" \
    --out "${TWO_SVG}" \
    --report "${TWO_REPORT}" \
    --title "ref vs baseline vs viapoints | ${MODE} ${LIDAR_HZ}Hz ${LIDAR_SAMPLES} samples" >"${TWO_SUMMARY}" || true
  echo "[plot] two-replays"
  echo "       svg=${TWO_SVG}"
  echo "       report=${TWO_REPORT}"
  echo
fi

echo "[OK] done: ${RUN_DIR}"
