#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ ! -f "${WS_DIR}/devel/setup.bash" ]]; then
  echo "ERROR: cannot find ${WS_DIR}/devel/setup.bash (did you build the workspace?)" >&2
  exit 1
fi

source "${WS_DIR}/devel/setup.bash"

if ! command -v rostopic >/dev/null 2>&1; then
  echo "ERROR: rostopic not found (did you source ROS?)" >&2
  exit 1
fi

if [[ -z "${ROS_HOME:-}" ]]; then
  export ROS_HOME="${WS_DIR}/.ros"
fi

echo "[env] ROS_MASTER_URI=${ROS_MASTER_URI:-<unset>}"
echo "[env] GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-<unset>}"
echo "[env] ROS_HOME=${ROS_HOME:-<unset>}"
echo

if ! timeout 2s rostopic list >/dev/null 2>&1; then
  echo "ERROR: cannot talk to ROS master (is the server launch running?)" >&2
  exit 2
fi

echo "[nodes] (grep gazebo|move_base|follow_path_goals|map_server)"
rosnode list 2>/dev/null | grep -E "gazebo|move_base|follow_path_goals|map_server" || true
echo

echo "[topics] expected: /clock /odom /move_base_simple/goal /cmd_vel"
for t in /clock /odom /move_base_simple/goal /cmd_vel; do
  echo "== ${t} =="
  rostopic info "${t}" 2>/dev/null || echo "(missing)"
done
echo

echo "[hint] /cmd_vel should have 1 publisher (move_base or your teleop). If there are 2+ publishers, the robot may '乱跑'."
echo

echo "[sample] /move_base_simple/goal (timeout 3s)"
timeout 3s rostopic echo -n 1 /move_base_simple/goal 2>/dev/null || echo "(no message)"
echo

echo "[sample] /cmd_vel (timeout 3s; default should be ~5Hz)"
timeout 3s rostopic hz -w 10 /cmd_vel 2>/dev/null || echo "(no message)"
