#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT="${1:-11347}"
WORLD_FILE="${2:-${WS_DIR}/src/pcd_gazebo_world/worlds/orchard_from_pcd_validated_by_bag.world}"

# Normalize to absolute paths; roslaunch runs nodes with cwd=$ROS_HOME, so relative paths would break.
WORLD_FILE="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${WORLD_FILE}")"

if [[ ! -f "${WS_DIR}/devel/setup.bash" ]]; then
  echo "ERROR: cannot find ${WS_DIR}/devel/setup.bash (did you build the workspace?)" >&2
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

export PATH="/usr/bin:${PATH}"

cd "${WS_DIR}"
source "${WS_DIR}/devel/setup.bash"

exec roslaunch pcd_gazebo_world orchard_pcd_sim.launch \
  gui:=false \
  gazebo_master_uri:="${GAZEBO_MASTER_URI}" \
  gpu:=false organize_cloud:=false lidar_hz:=2 lidar_samples:=60 \
  use_skid_steer:=false use_planar_move:=true \
  world_name:="${WORLD_FILE}"
