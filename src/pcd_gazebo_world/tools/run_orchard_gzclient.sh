#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT="${1:-11347}"
SOFTWARE_GL="${2:-}"

export GAZEBO_MASTER_URI="http://localhost:${PORT}"
# Gazebo writes logs under ~/.gazebo by default; in sandboxed runs HOME may be read-only.
# Redirect logs to a workspace-local directory so gzclient can start reliably.
export GAZEBO_LOG_PATH="${WS_DIR}/.gazebo"
mkdir -p "${GAZEBO_LOG_PATH}"
# Avoid gzclient trying to fetch models from the internet (Fuel). This environment is often offline/restricted.
export GAZEBO_MODEL_DATABASE_URI=""

if [[ "${SOFTWARE_GL}" == "--software" ]]; then
  export LIBGL_ALWAYS_SOFTWARE=1
fi

exec gzclient
