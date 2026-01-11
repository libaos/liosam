#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS_DIR"

BAG_PATH="${1:-$WS_DIR/rosbags/2025-10-29-16-05-00.bag}"
ROS_PORT_PARAM="${ROS_PORT:-11327}"
AUTO_PORT=0
if [[ "$ROS_PORT_PARAM" == "auto" ]]; then
  AUTO_PORT=1
fi
ROS_HOME_PARAM="${ROS_HOME:-}"
USER_ROS_HOME_SET=0
if [[ -n "${ROS_HOME_PARAM}" ]]; then
  USER_ROS_HOME_SET=1
fi
BAG_RATE="${BAG_RATE:-1}"
BAG_DURATION="${BAG_DURATION:-40}" # seconds; must be > ~12s to include first /tf
BAG_TOPICS="${BAG_TOPICS:-/tf}"

PCD_PATH="${PCD_PATH:-$WS_DIR/maps/TreeMap_full.pcd}"
CONFIG_PATH="${CONFIG_PATH:-$WS_DIR/src/orchard_row_mapping/config/liorl_tree_prior.yaml}"
PCD_TOPIC="${PCD_TOPIC:-/orchard_tree_map_builder/tree_map}"
PCD_FRAME_ID="${PCD_FRAME_ID:-map}"
PCD_PUBLISH_MAX_POINTS="${PCD_PUBLISH_MAX_POINTS:-0}" # 0=publish all points
ROW_MODEL_FILE="${ROW_MODEL_FILE:-$WS_DIR/src/orchard_row_mapping/config/row_model_pca_major.json}"

ROW_DIRECTION_MODE="${ROW_DIRECTION_MODE:-pca_major}"  # auto | pca_major | pca_minor | manual_yaw
ROW_DIRECTION_YAW_DEG="${ROW_DIRECTION_YAW_DEG:-0.0}"
FORCE_REBUILD_ROW_MODEL="${FORCE_REBUILD_ROW_MODEL:-0}"

ROW_DETECTION="${ROW_DETECTION:-histogram}"
ROW_CENTER_MIN_SEPARATION="${ROW_CENTER_MIN_SEPARATION:-2.0}"
ROW_V_OFFSETS="${ROW_V_OFFSETS:-}"
PUBLISH_ALL_ROWS="${PUBLISH_ALL_ROWS:-false}"
COLORIZE_ALL_ROWS="${COLORIZE_ALL_ROWS:-false}"
PUBLISH_NEAREST_ROWS="${PUBLISH_NEAREST_ROWS:-0}"
PUBLISH_ALL_CENTERLINES="${PUBLISH_ALL_CENTERLINES:-true}"
PUBLISH_NEAREST_CENTERLINES="${PUBLISH_NEAREST_CENTERLINES:-4}"
PUBLISH_CENTERLINE="${PUBLISH_CENTERLINE:-false}"
PUBLISH_ROW_BOUNDARIES="${PUBLISH_ROW_BOUNDARIES:-false}"
PUBLISH_CENTERLINE_LABELS="${PUBLISH_CENTERLINE_LABELS:-true}"
CENTERLINE_LABEL_MODE="${CENTERLINE_LABEL_MODE:-absolute}" # absolute | relative
CENTERLINE_LABEL_START_INDEX="${CENTERLINE_LABEL_START_INDEX:-1}"
CENTERLINE_LABEL_HEIGHT="${CENTERLINE_LABEL_HEIGHT:-0.6}"
CENTERLINE_LABEL_Z_OFFSET="${CENTERLINE_LABEL_Z_OFFSET:-0.6}"
LINE_LENGTH_USER_SET=0
if [[ -n "${LINE_LENGTH+x}" && -n "${LINE_LENGTH:-}" ]]; then
  LINE_LENGTH_USER_SET=1
  LINE_LENGTH="${LINE_LENGTH}"
else
  LINE_LENGTH="20.0"
fi

ROW_VIEW="${ROW_VIEW:-trees}" # trees | lanes
if [[ "$ROW_VIEW" == "trees" || "$ROW_VIEW" == "tree_rows" ]]; then
  # Show tree rows only (no lane centerlines).
  PUBLISH_ALL_ROWS="true"
  PUBLISH_NEAREST_ROWS="0"
  COLORIZE_ALL_ROWS="${COLORIZE_ALL_ROWS:-false}"
  PUBLISH_ALL_CENTERLINES="false"
  PUBLISH_NEAREST_CENTERLINES="0"
  PUBLISH_CENTERLINE="false"
  PUBLISH_ROW_BOUNDARIES="false"
  PUBLISH_CENTERLINE_LABELS="false"
  if [[ "$LINE_LENGTH_USER_SET" == "0" ]]; then
    LINE_LENGTH="200.0"
  fi
fi

ENABLE_RVIZ="${ENABLE_RVIZ:-1}"
RVIZ_CFG="${RVIZ_CFG:-$WS_DIR/src/orchard_row_mapping/config/orchard_row_mapping.rviz}"
ENABLE_TREE_CIRCLES="${ENABLE_TREE_CIRCLES:-1}"
TREE_CIRCLES_EXPORT_CSV="${TREE_CIRCLES_EXPORT_CSV:-}"
EXIT_ON_BAG_END="${EXIT_ON_BAG_END:-0}"
START_ROSCORE="${START_ROSCORE:-1}"

abspath() {
  python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$1"
}

BAG_PATH="$(abspath "$BAG_PATH")"
PCD_PATH="$(abspath "$PCD_PATH")"
CONFIG_PATH="$(abspath "$CONFIG_PATH")"
RVIZ_CFG="$(abspath "$RVIZ_CFG")"

if [[ ! -f "$BAG_PATH" ]]; then
  echo "[ERROR] rosbag not found: $BAG_PATH" >&2
  exit 2
fi
if [[ ! -f "$PCD_PATH" ]]; then
  echo "[ERROR] PCD not found: $PCD_PATH" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] config not found: $CONFIG_PATH" >&2
  exit 2
fi

source /opt/ros/noetic/setup.bash
source "$WS_DIR/devel/setup.bash"

export ROS_IP="127.0.0.1"

ROSCORE_PID=""
PRIOR_PID=""
PCD_PUB_PID=""
RVIZ_PID=""
TREE_CIRCLES_PID=""

port_is_listening() {
  local port="$1"
  python3 - "$port" <<'PY'
from pathlib import Path
import sys

port = int(sys.argv[1])
port_hex = f"{port:04X}"
for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = Path(path).read_text().splitlines()[1:]
    except Exception:
        continue
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[1]
        st = parts[3]
        try:
            _, p = local.split(":")
        except ValueError:
            continue
        if st == "0A" and p.upper() == port_hex:
            sys.exit(0)
sys.exit(1)
PY
}

configure_ros_env() {
  local port="$1"
  ROS_PORT="$port"
  export ROS_MASTER_URI="http://127.0.0.1:${ROS_PORT}"
  if [[ -z "${ROS_HOME:-}" ]]; then
    export ROS_HOME="$WS_DIR/.ros_row_prior_${ROS_PORT}"
  fi
  mkdir -p "$ROS_HOME"
}

master_is_responding() {
  timeout 1.0 rosparam list >/dev/null 2>&1
}

kill_tree() {
  local pid="$1"
  local timeout_s="${2:-3}"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  kill -TERM "$pid" 2>/dev/null || true
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if (( waited >= timeout_s * 10 )); then
      kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  set +e
  kill_tree "$RVIZ_PID" 3
  kill_tree "$PRIOR_PID" 3
  kill_tree "$TREE_CIRCLES_PID" 3
  kill_tree "$PCD_PUB_PID" 3
  kill_tree "$ROSCORE_PID" 5
}
trap cleanup EXIT INT TERM
trap 'echo "[WARN] Received Ctrl-Z (SIGTSTP); cleaning up."; exit 0' TSTP

echo "[INFO] Workspace   : $WS_DIR"
echo "[INFO] Bag         : $BAG_PATH"
echo "[INFO] Bag topics  : $BAG_TOPICS"
echo "[INFO] PCD         : $PCD_PATH"
echo "[INFO] PCD topic   : $PCD_TOPIC"
echo "[INFO] Config      : $CONFIG_PATH"
if [[ -n "${ROW_MODEL_FILE}" ]]; then
  echo "[INFO] Row model   : $ROW_MODEL_FILE"
fi

echo "[INFO] Checking ROS master..."
MASTER_OK=0

if [[ "$AUTO_PORT" == "1" ]]; then
  # Find a usable port even if some ports are occupied by non-ROS listeners.
  for port in $(seq 12000 12199); do
    if [[ "$USER_ROS_HOME_SET" == "1" ]]; then
      export ROS_HOME="$ROS_HOME_PARAM"
    else
      ROS_HOME=""
    fi
    configure_ros_env "$port"
    if master_is_responding; then
      echo "[INFO] Reusing existing ROS master: $ROS_MASTER_URI"
      MASTER_OK=1
      break
    fi
    if port_is_listening "$port"; then
      continue
    fi
    if [[ "$START_ROSCORE" != "1" && "$START_ROSCORE" != "true" ]]; then
      continue
    fi

    echo "[INFO] Starting roscore..."
    PYTHONUNBUFFERED=1 roscore -p "$port" >"$ROS_HOME/roscore.log" 2>&1 &
    ROSCORE_PID="$!"

    sleep 0.4
    if ! kill -0 "$ROSCORE_PID" 2>/dev/null; then
      continue
    fi

    echo "[INFO] Waiting for ROS master..."
    for _ in $(seq 1 60); do
      if master_is_responding; then
        MASTER_OK=1
        break
      fi
      if ! kill -0 "$ROSCORE_PID" 2>/dev/null; then
        break
      fi
      sleep 0.2
    done
    if [[ "$MASTER_OK" == "1" ]]; then
      break
    fi
    kill_tree "$ROSCORE_PID" 2
    ROSCORE_PID=""
  done
else
  configure_ros_env "$ROS_PORT_PARAM"
  echo "[INFO] ROS master  : $ROS_MASTER_URI"
  echo "[INFO] ROS_HOME    : $ROS_HOME"

  # If a master is already running on this URI, reuse it.
  if master_is_responding; then
    echo "[INFO] ROS master already running; reusing."
    MASTER_OK=1
  elif [[ "$START_ROSCORE" == "1" || "$START_ROSCORE" == "true" ]]; then
    if port_is_listening "$ROS_PORT"; then
      SUGGEST_PORT=$((ROS_PORT + 1))
      echo "[ERROR] Port ${ROS_PORT} already has a listener, but it's not responding as a ROS master at ${ROS_MASTER_URI}." >&2
      echo "        Try a different port: ROS_PORT=${SUGGEST_PORT} $0 $1" >&2
      echo "        Or kill old ROS processes: pkill -f \"roscore -p ${ROS_PORT}\"; pkill -f \"rosmaster --core -p ${ROS_PORT}\"" >&2
      exit 1
    fi

    echo "[INFO] Starting roscore..."
    PYTHONUNBUFFERED=1 roscore -p "$ROS_PORT" >"$ROS_HOME/roscore.log" 2>&1 &
    ROSCORE_PID="$!"

    sleep 0.4
    if ! kill -0 "$ROSCORE_PID" 2>/dev/null; then
      SUGGEST_PORT=$((ROS_PORT + 1))
      echo "[ERROR] Failed to start roscore on port ${ROS_PORT} (port in use or permissions)." >&2
      echo "        Check log: $ROS_HOME/roscore.log" >&2
      tail -n 80 "$ROS_HOME/roscore.log" >&2 || true
      echo "        Try a different port: ROS_PORT=${SUGGEST_PORT} $0 $1" >&2
      echo "        Or kill old roscore: pkill -f \"roscore -p ${ROS_PORT}\"" >&2
      exit 1
    fi

    echo "[INFO] Waiting for ROS master..."
    for _ in $(seq 1 60); do
      if master_is_responding; then
        MASTER_OK=1
        break
      fi
      if ! kill -0 "$ROSCORE_PID" 2>/dev/null; then
        break
      fi
      sleep 0.2
    done
  else
    echo "[INFO] START_ROSCORE=0; waiting for existing master at $ROS_MASTER_URI"
    for _ in $(seq 1 60); do
      if master_is_responding; then
        MASTER_OK=1
        break
      fi
      sleep 0.2
    done
  fi
fi
if [[ "$MASTER_OK" != "1" ]]; then
  echo "[INFO] ROS master  : $ROS_MASTER_URI"
  echo "[INFO] ROS_HOME    : $ROS_HOME"
  echo "[ERROR] Unable to communicate with ROS master at $ROS_MASTER_URI" >&2
  echo "        Check log: $ROS_HOME/roscore.log" >&2
  tail -n 80 "$ROS_HOME/roscore.log" >&2 || true
  exit 1
fi
echo "[INFO] ROS master  : $ROS_MASTER_URI"
echo "[INFO] ROS_HOME    : $ROS_HOME"
rosparam set /use_sim_time true >/dev/null

echo "[INFO] Publishing PCD for RViz..."
roslaunch orchard_row_mapping pcd_publisher.launch \
  pcd_path:="$PCD_PATH" \
  topic:="$PCD_TOPIC" \
  frame_id:="$PCD_FRAME_ID" \
  publish_max_points:="$PCD_PUBLISH_MAX_POINTS" \
  >"$ROS_HOME/pcd_publisher.launch.log" 2>&1 &
PCD_PUB_PID="$!"

echo "[INFO] Starting orchard_row_prior..."
roslaunch orchard_row_mapping orchard_row_prior.launch \
  config:="$CONFIG_PATH" \
  pcd_path:="$PCD_PATH" \
  row_model_file:="${ROW_MODEL_FILE}" \
  row_direction_mode:="$ROW_DIRECTION_MODE" \
  row_direction_yaw_deg:="$ROW_DIRECTION_YAW_DEG" \
  force_rebuild_model:="$FORCE_REBUILD_ROW_MODEL" \
  row_detection:="$ROW_DETECTION" \
  row_center_min_separation:="$ROW_CENTER_MIN_SEPARATION" \
  row_v_offsets:="$ROW_V_OFFSETS" \
  publish_nearest_rows:="$PUBLISH_NEAREST_ROWS" \
  publish_all_centerlines:="$PUBLISH_ALL_CENTERLINES" \
  publish_nearest_centerlines:="$PUBLISH_NEAREST_CENTERLINES" \
  publish_all_rows:="$PUBLISH_ALL_ROWS" \
  publish_centerline:="$PUBLISH_CENTERLINE" \
  publish_row_boundaries:="$PUBLISH_ROW_BOUNDARIES" \
  publish_centerline_labels:="$PUBLISH_CENTERLINE_LABELS" \
  centerline_label_mode:="$CENTERLINE_LABEL_MODE" \
  centerline_label_start_index:="$CENTERLINE_LABEL_START_INDEX" \
  centerline_label_height:="$CENTERLINE_LABEL_HEIGHT" \
  centerline_label_z_offset:="$CENTERLINE_LABEL_Z_OFFSET" \
  colorize_all_rows:="$COLORIZE_ALL_ROWS" \
  line_length:="$LINE_LENGTH" \
  >"$ROS_HOME/orchard_row_prior.launch.log" 2>&1 &
PRIOR_PID="$!"

if [[ "$ENABLE_TREE_CIRCLES" == "1" || "$ENABLE_TREE_CIRCLES" == "true" ]]; then
  echo "[INFO] Starting orchard_tree_circles..."
  roslaunch orchard_row_mapping orchard_tree_circles.launch \
    pcd_path:="$PCD_PATH" \
    map_frame:="$PCD_FRAME_ID" \
    row_model_file:="${ROW_MODEL_FILE}" \
    export_csv:="$TREE_CIRCLES_EXPORT_CSV" \
    >"$ROS_HOME/orchard_tree_circles.launch.log" 2>&1 &
  TREE_CIRCLES_PID="$!"
fi

if [[ "$ENABLE_RVIZ" == "1" || "$ENABLE_RVIZ" == "true" ]]; then
  if command -v rviz >/dev/null 2>&1; then
    if [[ -f "$RVIZ_CFG" ]]; then
      if [[ -z "${DISPLAY:-}" ]]; then
        echo "[WARN] DISPLAY is empty; skipping RViz (GUI not available)."
      elif command -v xdpyinfo >/dev/null 2>&1 && ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        echo "[WARN] Unable to open X display ($DISPLAY); skipping RViz."
        echo "       Tip: use X11 forwarding (ssh -X) or run RViz on a desktop/VNC session."
      else
        echo "[INFO] Starting RViz..."
        rviz -d "$RVIZ_CFG" >"$ROS_HOME/rviz.log" 2>&1 &
        RVIZ_PID="$!"
        sleep 0.8
        if ! kill -0 "$RVIZ_PID" 2>/dev/null; then
          echo "[WARN] RViz failed to start. Check log: $ROS_HOME/rviz.log"
          tail -n 40 "$ROS_HOME/rviz.log" 2>/dev/null || true
        fi
      fi
    else
      echo "[WARN] RViz config not found: $RVIZ_CFG"
    fi
  else
    echo "[WARN] rviz executable not found."
  fi
fi

sleep 2

echo "[INFO] Playing bag..."
rosbag play --clock -r "$BAG_RATE" -q --duration "$BAG_DURATION" "$BAG_PATH" --topics $BAG_TOPICS

if [[ "$EXIT_ON_BAG_END" == "1" || "$EXIT_ON_BAG_END" == "true" ]]; then
  echo "[OK] Bag finished. Exiting."
  exit 0
fi

echo "[OK] Bag finished. You can keep RViz open; press Ctrl-C to exit."
wait
