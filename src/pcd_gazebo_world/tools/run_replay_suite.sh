#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT="${1:-11347}"
DURATION_S="${2:-450}"

REF_JSON="${3:-${WS_DIR}/src/pcd_gazebo_world/maps/runs/rosbag_path.json}"
COMPARE_PY="${WS_DIR}/src/pcd_gazebo_world/scripts/plot_reference_vs_replay.py"
DEFAULT_WORLD_FILE="${WS_DIR}/src/pcd_gazebo_world/worlds/orchard_from_pcd_validated_by_bag.world"

if [[ ! -f "${REF_JSON}" ]]; then
  echo "ERROR: reference path json not found: ${REF_JSON}" >&2
  exit 1
fi

if [[ ! -f "${COMPARE_PY}" ]]; then
  echo "ERROR: compare script not found: ${COMPARE_PY}" >&2
  exit 1
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${WS_DIR}/trajectory_data/replay_suite_${RUN_TAG}"
mkdir -p "${RUN_DIR}"

echo "[suite] run_dir=${RUN_DIR}"
echo "[suite] duration=${DURATION_S}s"
echo

run_one() {
  local label="$1"
  shift

  local csv="${RUN_DIR}/${label}_odom.csv"
  local report="${RUN_DIR}/${label}_report.json"
  local svg="${RUN_DIR}/${label}_overlay.svg"
  local summary="${RUN_DIR}/${label}_summary.json"

  echo "[run] ${label}"
  echo "      csv=${csv}"

  # Run the ROS/Gazebo pipeline for a fixed duration; SIGINT triggers a graceful roslaunch shutdown.
  RECORD_CSV="${csv}" timeout -s INT "${DURATION_S}" "$@" || true

  if [[ ! -f "${csv}" ]]; then
    echo "[warn] missing trajectory csv: ${csv}"
    return 0
  fi

  /usr/bin/python3 "${COMPARE_PY}" --reference "${REF_JSON}" --replay "${csv}" --out "${svg}" --report "${report}" >"${summary}"
  echo "      report=${report}"
  echo "      svg=${svg}"
  echo

  # Give gzserver/roscore a moment to exit cleanly before the next run (avoid port conflicts).
  sleep 5
}

run_one "teb_velodyne" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_teb_server.sh" "${PORT}" "${REF_JSON}" "${DEFAULT_WORLD_FILE}" velodyne
run_one "teb_static_map" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_teb_server.sh" "${PORT}" "${REF_JSON}" "${DEFAULT_WORLD_FILE}" static_map
run_one "teb_no_obstacles" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_teb_server.sh" "${PORT}" no_obstacles "${REF_JSON}" "${WS_DIR}/src/pcd_gazebo_world/worlds/empty_orchard.world"
run_one "pid" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_pid_server.sh" "${PORT}" "${REF_JSON}" "${DEFAULT_WORLD_FILE}"

echo "[summary] metrics from *_report.json"
/usr/bin/python3 "${WS_DIR}/src/pcd_gazebo_world/tools/summarize_replay_suite.py" --dir "${RUN_DIR}" || true
echo

echo "[OK] suite finished: ${RUN_DIR}"
