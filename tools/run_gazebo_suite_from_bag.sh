#!/usr/bin/env bash
set -euo pipefail

# One-command helper:
# 1) auto-detect a nav_msgs/Path topic inside a rosbag
# 2) export the reference path to map frame JSON (using bag /tf)
# 3) run the Gazebo replay suite and print a metrics summary
#
# Usage:
#   bash tools/run_gazebo_suite_from_bag.sh rosbags/xxx.bag
#   bash tools/run_gazebo_suite_from_bag.sh rosbags/xxx.bag /liorl/mapping/path 11347 450

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BAG_PATH="${1:-}"
REQUESTED_TOPIC="${2:-}"
PORT="${3:-11347}"
DURATION_S="${4:-450}"

if [[ -z "${BAG_PATH}" ]]; then
  echo "Usage: bash tools/run_gazebo_suite_from_bag.sh BAG [PATH_TOPIC] [PORT] [DURATION_S]" >&2
  exit 2
fi

BAG_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${BAG_PATH}")"

if [[ ! -f "${BAG_ABS}" ]]; then
  echo "ERROR: bag not found: ${BAG_ABS}" >&2
  exit 1
fi

if ! python3 -c "import rosbag" >/dev/null 2>&1; then
  echo "ERROR: python cannot import rosbag; did you source ROS first?" >&2
  echo "  source /opt/ros/noetic/setup.bash && source ${WS_DIR}/devel/setup.bash" >&2
  exit 1
fi

TOPIC="${REQUESTED_TOPIC}"
if [[ -z "${TOPIC}" ]]; then
  TOPIC="$(python3 - "${BAG_ABS}" <<'PY'
import os
import sys

import rosbag

bag_path = sys.argv[1]

with rosbag.Bag(bag_path) as bag:
    topics = bag.get_type_and_topic_info().topics

path_topics = [
    topic
    for topic, info in topics.items()
    if getattr(info, "msg_type", None) == "nav_msgs/Path"
]
if not path_topics:
    raise SystemExit("No nav_msgs/Path topics found in bag; pass PATH_TOPIC explicitly.")

priority = [
    "/lio_sam/mapping/path",
    "/liorl/mapping/path",
    "/liorf/mapping/path",
]
for candidate in priority:
    if candidate in path_topics:
        print(candidate)
        raise SystemExit(0)

mapping_paths = [t for t in path_topics if t.endswith("/mapping/path")]
if mapping_paths:
    print(sorted(mapping_paths)[0])
else:
    print(sorted(path_topics)[0])
PY
)"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="${WS_DIR}/trajectory_data/reference_from_bag_${STAMP}.json"
mkdir -p "${WS_DIR}/trajectory_data"

echo "[1/3] Export reference JSON (map frame)"
echo "      bag  : ${BAG_ABS}"
echo "      topic: ${TOPIC}"
echo "      out  : ${OUT_JSON}"
if python3 "${WS_DIR}/src/pcd_gazebo_world/scripts/rosbag_path_to_json.py" \
  --bag "${BAG_ABS}" \
  --topic "${TOPIC}" \
  --target-frame map \
  --tf-topic /tf \
  --min-dist 0.4 \
  --out "${OUT_JSON}"; then
  :
else
  echo "[warn] failed to transform Path to map using bag /tf; falling back to the Path's own frame" >&2
  python3 "${WS_DIR}/src/pcd_gazebo_world/scripts/rosbag_path_to_json.py" \
    --bag "${BAG_ABS}" \
    --topic "${TOPIC}" \
    --min-dist 0.4 \
    --out "${OUT_JSON}"
fi
echo

echo "[2/3] Run Gazebo replay suite"
echo "      port    : ${PORT}"
echo "      duration: ${DURATION_S}s"
echo

bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_replay_suite.sh" "${PORT}" "${DURATION_S}" "${OUT_JSON}"

echo
echo "[3/3] Done"
echo "      run outputs: ${WS_DIR}/trajectory_data/replay_suite_*"
