#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_recorded.sh OUT_BAG --topics TOPICS_FILE -- <command...>

Example:
  bash tools/navigation_experiments/run_recorded.sh rosbags/runs/run1.bag \
    --topics tools/navigation_experiments/topics_minimal.txt \
    -- roslaunch bag_route_replay replay_from_bag.launch bag_path:=/abs/path/to/ref.bag
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

OUT_BAG="$1"
shift

TOPICS_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --topics)
      TOPICS_FILE="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${TOPICS_FILE}" ]]; then
  echo "[ERR] --topics is required" >&2
  usage
  exit 2
fi
if [[ ! -f "${TOPICS_FILE}" ]]; then
  echo "[ERR] topics file not found: ${TOPICS_FILE}" >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "[ERR] Missing command after --" >&2
  usage
  exit 2
fi

mapfile -t TOPICS < <(grep -v '^[[:space:]]*#' "${TOPICS_FILE}" | awk 'NF{print $0}')
if [[ ${#TOPICS[@]} -eq 0 ]]; then
  echo "[ERR] No topics found in: ${TOPICS_FILE}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT_BAG}")"

echo "[INFO] Recording: ${OUT_BAG}"
echo "[INFO] Topics (${#TOPICS[@]}): ${TOPICS[*]}"

rosbag record -O "${OUT_BAG}" "${TOPICS[@]}" __name:=nav_exp_record >/dev/null 2>&1 &
REC_PID=$!

cleanup() {
  if kill -0 "${REC_PID}" >/dev/null 2>&1; then
    echo "[INFO] Stopping rosbag record (pid=${REC_PID})..."
    kill -INT "${REC_PID}" >/dev/null 2>&1 || true
    wait "${REC_PID}" || true
  fi
}
trap cleanup EXIT

sleep 0.8

echo "[INFO] Running command: $*"
"$@"

echo "[INFO] Command finished, closing bag..."
cleanup
trap - EXIT
echo "[OK] Done: ${OUT_BAG}"

