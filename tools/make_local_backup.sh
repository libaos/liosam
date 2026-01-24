#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

BACKUP_DIR="${ROOT_DIR}/backups"
mkdir -p "${BACKUP_DIR}"

TAR_PATH="${BACKUP_DIR}/lio_ws_snapshot_${TIMESTAMP}.tar.gz"
MANIFEST_PATH="${BACKUP_DIR}/lio_ws_snapshot_${TIMESTAMP}.manifest.txt"

INCLUDE_RESULTS="${INCLUDE_RESULTS:-true}"

items=()
for entry in README.md .gitignore docs tools src; do
  [[ -e "${ROOT_DIR}/${entry}" ]] && items+=("${entry}")
done
for entry in CHANGELOG.md ROADMAP.md LICENSES.md CONTRIBUTING.md; do
  [[ -e "${ROOT_DIR}/${entry}" ]] && items+=("${entry}")
done
[[ -e "${ROOT_DIR}/.codex/plans/计划.md" ]] && items+=(".codex/plans/计划.md")
if [[ "${INCLUDE_RESULTS}" == "true" && -d "${ROOT_DIR}/结果/Stage6_FSM多组对比" ]]; then
  items+=("结果/Stage6_FSM多组对比")
fi

tar -czf "${TAR_PATH}" -C "${ROOT_DIR}" "${items[@]}"

{
  echo "created_at: $(date -Is)"
  echo "root: ${ROOT_DIR}"
  echo "tar: ${TAR_PATH}"
  echo "include_results: ${INCLUDE_RESULTS}"
  echo
  echo "included_items:"
  printf -- "- %s\n" "${items[@]}"

  if command -v git >/dev/null 2>&1 && [[ -d "${ROOT_DIR}/.git" ]]; then
    echo
    echo "git_head: $(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"
    echo
    echo "git_status:"
    git -C "${ROOT_DIR}" status -sb || true
    echo
    echo "git_porcelain:"
    git -C "${ROOT_DIR}" status --porcelain=v1 || true
  fi
} > "${MANIFEST_PATH}"

echo "OK: ${TAR_PATH}"
echo "MANIFEST: ${MANIFEST_PATH}"
