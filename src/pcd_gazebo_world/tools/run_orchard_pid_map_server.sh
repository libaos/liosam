#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT="${1:-11347}"
DEFAULT_PATH_JSON="${WS_DIR}/src/pcd_gazebo_world/maps/runs/rosbag_path.json"
DEFAULT_WORLD_FILE="${WS_DIR}/src/pcd_gazebo_world/worlds/orchard_from_pcd_validated_by_bag.world"
USE_SKID_STEER="${USE_SKID_STEER:-false}"
USE_PLANAR_MOVE="${USE_PLANAR_MOVE:-true}"
LIDAR_HZ="${LIDAR_HZ:-10}"
LIDAR_SAMPLES="${LIDAR_SAMPLES:-440}"
SHUTDOWN_ON_PATH_DONE="${SHUTDOWN_ON_PATH_DONE:-false}"

PATH_JSON="${2:-${DEFAULT_PATH_JSON}}"
WORLD_FILE="${3:-${DEFAULT_WORLD_FILE}}"

# Normalize to absolute paths; roslaunch runs nodes with cwd=$ROS_HOME, so relative paths would break.
PATH_JSON="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${PATH_JSON}")"
WORLD_FILE="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${WORLD_FILE}")"

if [[ ! -f "${WS_DIR}/devel/setup.bash" ]]; then
  echo "ERROR: cannot find ${WS_DIR}/devel/setup.bash (did you build the workspace?)" >&2
  exit 1
fi

if [[ ! -f "${PATH_JSON}" ]]; then
  echo "ERROR: path json not found: ${PATH_JSON}" >&2
  exit 1
fi

if [[ ! -f "${WORLD_FILE}" ]]; then
  echo "ERROR: world file not found: ${WORLD_FILE}" >&2
  exit 1
fi

export GAZEBO_MASTER_URI="http://localhost:${PORT}"
# Gazebo writes logs under ~/.gazebo by default; in sandboxed runs HOME may be read-only.
# Redirect logs to a workspace-local directory so gzserver doesn't exit with code 255.
export GAZEBO_LOG_PATH="${WS_DIR}/.gazebo"
mkdir -p "${GAZEBO_LOG_PATH}"
export ROS_LOG_DIR="${WS_DIR}/.ros_log"
mkdir -p "${ROS_LOG_DIR}"
export ROS_HOME="${WS_DIR}/.ros"
mkdir -p "${ROS_HOME}"

# If a virtualenv is active, `/usr/bin/env python3` inside ROS nodes may pick the venv python.
# That often breaks ROS Python nodes (rospy not installed in the venv). Force /usr/bin first.
export PATH="/usr/bin:${PATH}"

if [[ -z "${RECORD_CSV:-}" ]]; then
  RECORD_CSV="${WS_DIR}/trajectory_data/pid_odom_map.csv"
fi

cd "${WS_DIR}"
source "${WS_DIR}/devel/setup.bash"

exec roslaunch pcd_gazebo_world orchard_pid_replay_map.launch \
  gui:=false \
  gazebo_master_uri:="${GAZEBO_MASTER_URI}" \
  gpu:=false organize_cloud:=false lidar_hz:="${LIDAR_HZ}" lidar_samples:="${LIDAR_SAMPLES}" \
  use_skid_steer:="${USE_SKID_STEER}" use_planar_move:="${USE_PLANAR_MOVE}" \
  world_name:="${WORLD_FILE}" \
  path_file:="${PATH_JSON}" \
  record_csv:="${RECORD_CSV}" \
  shutdown_on_path_done:="${SHUTDOWN_ON_PATH_DONE}"

