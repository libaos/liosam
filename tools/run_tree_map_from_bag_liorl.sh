#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS_DIR"

BAG_PATH="${1:-$WS_DIR/rosbags/2025-10-29-16-05-00.bag}"
ROS_PORT="${ROS_PORT:-11312}"
START_LIORL="${START_LIORL:-0}"  # 1 = run LIORL from raw topics, 0 = use bag's /liorl/* topics
BAG_RATE="${BAG_RATE:-1}"
BAG_DURATION="${BAG_DURATION:-}" # e.g. 60 (seconds). Empty = play full bag.
BAG_TOPICS="${BAG_TOPICS:-}"

OUTPUT_PCD="${OUTPUT_PCD:-$WS_DIR/maps/TreeMap_auto.pcd}"
VOXEL_SIZE="${VOXEL_SIZE:-0.10}"
ORCHARD_CONFIG="${ORCHARD_CONFIG:-}"

ENABLE_RVIZ="${ENABLE_RVIZ:-0}"
ENABLE_LASERSCAN="${ENABLE_LASERSCAN:-0}"
VENV_PATH="${VENV_PATH:-$WS_DIR/.venv_orchard}"

BAG_PATH="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$BAG_PATH")"
OUTPUT_PCD="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$OUTPUT_PCD")"
if [[ -n "$ORCHARD_CONFIG" ]]; then
  ORCHARD_CONFIG="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$ORCHARD_CONFIG")"
fi

if [[ ! -f "$BAG_PATH" ]]; then
  echo "[ERROR] rosbag not found: $BAG_PATH" >&2
  echo "Usage: $0 /path/to.bag" >&2
  exit 2
fi

if [[ "$ENABLE_RVIZ" == "0" || "$ENABLE_RVIZ" == "false" ]]; then
  RVIZ_ARG="false"
else
  RVIZ_ARG="true"
fi

if [[ "$ENABLE_LASERSCAN" == "0" || "$ENABLE_LASERSCAN" == "false" ]]; then
  LASERSCAN_ARG="false"
else
  LASERSCAN_ARG="true"
fi

if [[ -z "$BAG_TOPICS" ]]; then
  if [[ "$START_LIORL" == "1" || "$START_LIORL" == "true" ]]; then
    BAG_TOPICS="/points_raw /imu/data /initialpose"
  else
    BAG_TOPICS="/liorl/deskew/cloud_deskewed /tf"
  fi
fi

export ROS_HOME="${ROS_HOME:-$WS_DIR/.ros_tree_map_${ROS_PORT}}"
mkdir -p "$ROS_HOME"

source /opt/ros/noetic/setup.bash
source "$WS_DIR/devel/setup.bash"

PY_VER="$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
VENV_SITE_PACKAGES="$VENV_PATH/lib/python${PY_VER}/site-packages"
if [[ -d "$VENV_SITE_PACKAGES" ]]; then
  export PYTHONPATH="${VENV_SITE_PACKAGES}:${PYTHONPATH:-}"
fi

python3 -c "import torch" >/dev/null 2>&1 || {
  echo "[ERROR] Python module 'torch' not found." >&2
  echo "        Run: tools/setup_orchard_venv.sh (then re-run this script)" >&2
  exit 3
}

export ROS_MASTER_URI="http://127.0.0.1:${ROS_PORT}"
export ROS_IP="127.0.0.1"

ROSCORE_PID=""
LIORL_PID=""
ORCHARD_PID=""
RVIZ_PID=""

cleanup() {
  set +e
  if [[ -n "$ORCHARD_PID" ]] && kill -0 "$ORCHARD_PID" 2>/dev/null; then
    kill -INT "$ORCHARD_PID" 2>/dev/null || true
    wait "$ORCHARD_PID" 2>/dev/null || true
  fi
  if [[ -n "$LIORL_PID" ]] && kill -0 "$LIORL_PID" 2>/dev/null; then
    kill -INT "$LIORL_PID" 2>/dev/null || true
    wait "$LIORL_PID" 2>/dev/null || true
  fi
  if [[ -n "$RVIZ_PID" ]] && kill -0 "$RVIZ_PID" 2>/dev/null; then
    kill -INT "$RVIZ_PID" 2>/dev/null || true
    wait "$RVIZ_PID" 2>/dev/null || true
  fi
  if [[ -n "$ROSCORE_PID" ]] && kill -0 "$ROSCORE_PID" 2>/dev/null; then
    kill -INT "$ROSCORE_PID" 2>/dev/null || true
    wait "$ROSCORE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Workspace   : $WS_DIR"
echo "[INFO] ROS master  : $ROS_MASTER_URI"
echo "[INFO] ROS_HOME    : $ROS_HOME"
echo "[INFO] Bag         : $BAG_PATH"
echo "[INFO] START_LIORL : $START_LIORL"
echo "[INFO] Topics      : $BAG_TOPICS"
echo "[INFO] Output PCD  : $OUTPUT_PCD"
echo "[INFO] Voxel size  : $VOXEL_SIZE"

echo "[INFO] Starting roscore..."
roscore -p "$ROS_PORT" >"$ROS_HOME/roscore.log" 2>&1 &
ROSCORE_PID="$!"

echo "[INFO] Waiting for ROS master..."
for _ in $(seq 1 80); do
  if rosparam list >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done
rosparam set /use_sim_time true >/dev/null

if [[ "$START_LIORL" == "1" || "$START_LIORL" == "true" ]]; then
  echo "[INFO] Starting LIORL..."
  roslaunch liorl run_liorl.launch enable_rviz:="$RVIZ_ARG" enable_pointcloud_to_laserscan:="$LASERSCAN_ARG" >"$ROS_HOME/liorl.launch.log" 2>&1 &
  LIORL_PID="$!"
fi

echo "[INFO] Starting orchard segmentation + tree-map builder..."
orchard_args=(orchard_tree_map_liorl.launch output_pcd:="$OUTPUT_PCD" voxel_size:="$VOXEL_SIZE")
if [[ -n "$ORCHARD_CONFIG" ]]; then
  orchard_args+=(config:="$ORCHARD_CONFIG")
fi
roslaunch orchard_row_mapping "${orchard_args[@]}" >"$ROS_HOME/orchard_tree_map.launch.log" 2>&1 &
ORCHARD_PID="$!"

if [[ "$RVIZ_ARG" == "true" && ! ( "$START_LIORL" == "1" || "$START_LIORL" == "true" ) ]]; then
  if command -v rviz >/dev/null 2>&1; then
    RVIZ_CFG="$WS_DIR/src/lio_sam_move_base_tutorial/robot_gazebo/liorl/launch/include/config/rviz.rviz"
    if [[ -f "$RVIZ_CFG" ]]; then
      echo "[INFO] Starting RViz (standalone)..."
      rviz -d "$RVIZ_CFG" >"$ROS_HOME/rviz.log" 2>&1 &
      RVIZ_PID="$!"
    else
      echo "[WARN] RViz config not found: $RVIZ_CFG"
    fi
  else
    echo "[WARN] rviz executable not found in PATH."
  fi
fi

sleep 2

echo "[INFO] Playing bag..."
bag_args=(--clock -r "$BAG_RATE" -q "$BAG_PATH" --topics $BAG_TOPICS)
if [[ -n "$BAG_DURATION" ]]; then
  bag_args=(--clock -r "$BAG_RATE" -q --duration "$BAG_DURATION" "$BAG_PATH" --topics $BAG_TOPICS)
fi
rosbag play "${bag_args[@]}"

echo "[INFO] Bag finished. Shutting down nodes to save PCD..."
cleanup
trap - EXIT INT TERM

if [[ -f "$OUTPUT_PCD" ]]; then
  echo "[OK] Saved: $OUTPUT_PCD"
else
  echo "[WARN] Output PCD not found yet: $OUTPUT_PCD"
  echo "       Check logs: $ROS_HOME/orchard_tree_map.launch.log"
fi
