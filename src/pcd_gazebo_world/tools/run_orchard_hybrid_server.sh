#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT="${1:-11347}"
DEFAULT_PATH_JSON="${WS_DIR}/src/pcd_gazebo_world/maps/runs/rosbag_path.json"
DEFAULT_WORLD_FILE="${WS_DIR}/src/pcd_gazebo_world/worlds/orchard_from_pcd_validated_by_bag.world"
USE_VIA_POINTS="${USE_VIA_POINTS:-false}"
USE_SKID_STEER="${USE_SKID_STEER:-false}"
USE_PLANAR_MOVE="${USE_PLANAR_MOVE:-true}"
LIDAR_HZ="${LIDAR_HZ:-10}"
LIDAR_SAMPLES="${LIDAR_SAMPLES:-440}"
HYBRID_MODE="${HYBRID_MODE:-auto}" # auto|pid|teb
SHUTDOWN_ON_PATH_DONE="${SHUTDOWN_ON_PATH_DONE:-false}"

# Support both:
#   run_orchard_hybrid_server.sh PORT [PATH_JSON] [WORLD_FILE] [MODE]
# and a short form:
#   run_orchard_hybrid_server.sh PORT MODE
case "${2:-}" in
  velodyne|no_obstacles|no_obstacle|static_map)
    MODE="${2}"
    if [[ "${MODE}" == "no_obstacle" ]]; then
      MODE="no_obstacles"
    fi
    PATH_JSON="${3:-${DEFAULT_PATH_JSON}}"
    WORLD_FILE="${4:-${DEFAULT_WORLD_FILE}}"
    ;;
  *)
    PATH_JSON="${2:-${DEFAULT_PATH_JSON}}"
    WORLD_FILE="${3:-${DEFAULT_WORLD_FILE}}"
    MODE="${4:-velodyne}"
    ;;
esac

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
  case "${MODE}" in
    velodyne)
      RECORD_CSV="${WS_DIR}/trajectory_data/hybrid_odom.csv"
      ;;
    no_obstacles)
      RECORD_CSV="${WS_DIR}/trajectory_data/hybrid_no_obstacles_odom.csv"
      ;;
    static_map)
      RECORD_CSV="${WS_DIR}/trajectory_data/hybrid_static_odom.csv"
      ;;
  esac
fi

cd "${WS_DIR}"
source "${WS_DIR}/devel/setup.bash"

EXTRA_ARGS=()
case "${MODE}" in
  velodyne)
    ;;
  no_obstacles)
    EXTRA_ARGS+=(
      map_yaml:="${WS_DIR}/src/pcd_gazebo_world/maps/empty_orchard/map.yaml"
      costmap_common_params:="${WS_DIR}/src/pcd_gazebo_world/config/orchard_costmap_common_no_obstacles.yaml"
    )
    ;;
  static_map)
    EXTRA_ARGS+=(
      map_yaml:="${WS_DIR}/src/pcd_gazebo_world/maps/orchard_from_pcd_validated_by_bag/map.yaml"
      costmap_common_params:="${WS_DIR}/src/pcd_gazebo_world/config/orchard_costmap_common_static.yaml"
    )
    ;;
  *)
    echo "ERROR: unknown MODE '${MODE}' (supported: velodyne | no_obstacles | static_map)" >&2
    exit 2
    ;;
esac

exec roslaunch pcd_gazebo_world orchard_hybrid_replay.launch \
  gui:=false \
  gazebo_master_uri:="${GAZEBO_MASTER_URI}" \
  gpu:=false organize_cloud:=false lidar_hz:="${LIDAR_HZ}" lidar_samples:="${LIDAR_SAMPLES}" \
  controller_frequency:=5.0 planner_frequency:=2.0 local_costmap_update_frequency:=5.0 local_costmap_publish_frequency:=2.0 \
  use_skid_steer:="${USE_SKID_STEER}" use_planar_move:="${USE_PLANAR_MOVE}" \
  world_name:="${WORLD_FILE}" \
  path_file:="${PATH_JSON}" \
  record_csv:="${RECORD_CSV}" \
  use_via_points:="${USE_VIA_POINTS}" \
  hybrid_mode:="${HYBRID_MODE}" \
  shutdown_on_path_done:="${SHUTDOWN_ON_PATH_DONE}" \
  "${EXTRA_ARGS[@]}"
