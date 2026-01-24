#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT_BASE="${1:-11360}"
RUNS="${2:-3}"
DURATION_S="${3:-450}"

PATH_JSON="${4:-${WS_DIR}/src/pcd_gazebo_world/maps/runs/rosbag_path.json}"
WORLD_FILE="${5:-${WS_DIR}/src/pcd_gazebo_world/worlds/orchard_from_pcd_validated_by_bag.world}"
REF_FULLFRAMES_JSON="${6:-${WS_DIR}/src/pcd_gazebo_world/maps/runs/rosbag_pose_fullframes_map.json}"

START_INDEX="${START_INDEX:-1}"        # 追加跑：从第几组开始（默认 1）
RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE:-}" # 追加跑：指定已有 run_dir（例如 trajectory_data/fsm_benchmark_...）
ALGO_LIST="${ALGO_LIST:-pid,hybrid,teb,purepursuit,lqr,psolqr,dwa}" # 逗号/空格分隔
RESUME="${RESUME:-false}"             # 断点续跑：true=已有实验目录则继续补齐缺失文件

# Drive plugin selection (passed through to run_orchard_* scripts / roslaunch).
export USE_SKID_STEER="${USE_SKID_STEER:-false}"
export USE_PLANAR_MOVE="${USE_PLANAR_MOVE:-true}"

PATH_JSON="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${PATH_JSON}")"
WORLD_FILE="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${WORLD_FILE}")"
REF_FULLFRAMES_JSON="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${REF_FULLFRAMES_JSON}")"

if [[ ! -f "${REF_FULLFRAMES_JSON}" ]]; then
  echo "ERROR: fullframes reference json not found: ${REF_FULLFRAMES_JSON}" >&2
  exit 1
fi

END_INDEX=$((START_INDEX + RUNS - 1))

ALGO_LIST_NORM="${ALGO_LIST//,/ }"
read -r -a ALGO_ARR <<<"${ALGO_LIST_NORM}"
if [[ "${#ALGO_ARR[@]}" -lt 1 ]]; then
  echo "ERROR: empty ALGO_LIST (example: ALGO_LIST=pid,hybrid,teb)" >&2
  exit 2
fi

for a in "${ALGO_ARR[@]}"; do
  case "${a}" in
    pid|hybrid|teb|purepursuit|lqr|psolqr|dwa)
      ;;
    *)
      echo "ERROR: unsupported algorithm '${a}' (supported: pid, hybrid, teb, purepursuit, lqr, psolqr, dwa)" >&2
      exit 2
      ;;
  esac
done

ALGO_DISPLAY_PLUS="$(IFS=' + '; echo "${ALGO_ARR[*]}")"
ALGO_DISPLAY_SLASH="$(IFS='/'; echo "${ALGO_ARR[*]}")"

IS_NEW_RUN=true
if [[ -n "${RUN_DIR_OVERRIDE}" ]]; then
  IS_NEW_RUN=false
  RUN_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_DIR_OVERRIDE}")"
  mkdir -p "${RUN_DIR}"
  RUN_TAG="$(basename "${RUN_DIR}")"
else
  RUN_TAG="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="${WS_DIR}/trajectory_data/fsm_benchmark_${RUN_TAG}"
  mkdir -p "${RUN_DIR}"
fi

RUN_TIME_CN="$(date '+%Y年%m月%d日_%H时%M分%S秒')"
RUN_TIME_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %Z %z')"

if [[ ! -f "${RUN_DIR}/运行信息.md" ]]; then
  cat >"${RUN_DIR}/运行信息.md" <<EOF
# FSM 基准运行信息

- 时间：${RUN_TIME_HUMAN}（${RUN_TIME_CN}）
- RUN_DIR：${RUN_DIR}
- runs：${END_INDEX}
- duration：${DURATION_S}s
- reference_fullframes：${REF_FULLFRAMES_JSON}
- path_json：${PATH_JSON}
- world_file：${WORLD_FILE}
- use_skid_steer：${USE_SKID_STEER}
- use_planar_move：${USE_PLANAR_MOVE}

## 组编号（每组 = ${ALGO_DISPLAY_PLUS}）

EOF
else
  # Best-effort update runs count (keep history; only adjust the top metadata line).
  /usr/bin/python3 - "${RUN_DIR}/运行信息.md" "${END_INDEX}" <<'PY' || true
import sys
from pathlib import Path

path = Path(sys.argv[1])
end_idx = int(sys.argv[2])
text = path.read_text(encoding="utf-8").splitlines(True)
out = []
updated = False
for line in text:
    if line.startswith("- runs："):
        out.append(f"- runs：{end_idx}\n")
        updated = True
    else:
        out.append(line)
if not updated:
    out.insert(0, f"- runs：{end_idx}\n")
path.write_text("".join(out), encoding="utf-8")
PY
fi

# Also create/sync the Chinese results directory view (best-effort).
RESULTS_DIR="${WS_DIR}/结果/Stage6_FSM多组对比"
RESULTS_RUN_DIR="${RESULTS_DIR}/$(basename "${RUN_DIR}")"
REL_RUN_DIR="../../../trajectory_data/$(basename "${RUN_DIR}")"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_RUN_DIR}"
if [[ "${IS_NEW_RUN}" == "true" ]]; then
  ln -sfn "$(basename "${RESULTS_RUN_DIR}")" "${RESULTS_DIR}/${RUN_TIME_CN}_FSM基准" || true
fi

cat >"${RESULTS_RUN_DIR}/README.md" <<EOF
# $(basename "${RUN_DIR}")（${ALGO_DISPLAY_PLUS}，${END_INDEX} 组）

- 时间：${RUN_TIME_HUMAN}（${RUN_TIME_CN}）
- 原始输出：\`${RUN_DIR}\`
- 基准脚本：\`src/pcd_gazebo_world/tools/run_fsm_benchmark.sh\`
- 驱动：\`use_skid_steer=${USE_SKID_STEER} use_planar_move=${USE_PLANAR_MOVE}\`

建议先看：
- \`summary_groups.csv\`：按算法汇总（group=${ALGO_DISPLAY_SLASH}）
- \`summary_groups_plus.csv\`：按算法汇总（附 \`replay_length_m\` 等推进统计，便于识别“几乎没动”的样本）
- \`多组对比汇总.md\`：可选的“论文式”手工汇总入口（如存在）
- \`运行信息.md\`：本次基准参数 + 组编号映射
- \`实验01_一/\`、\`实验02_二/\` ...：每组 = ${ALGO_DISPLAY_PLUS}
EOF

# If we already materialized the file into a real one, don't overwrite it with a symlink
# on a resume run (it would temporarily re-introduce external links).
if [[ ! -e "${RESULTS_RUN_DIR}/运行信息.md" || -L "${RESULTS_RUN_DIR}/运行信息.md" ]]; then
  ln -sfn "${REL_RUN_DIR}/运行信息.md" "${RESULTS_RUN_DIR}/运行信息.md" || true
fi
# If we already materialized the files into real ones, don't overwrite them with symlinks
# on a resume run (it would temporarily re-introduce external links).
if [[ ! -e "${RESULTS_RUN_DIR}/summary.csv" || -L "${RESULTS_RUN_DIR}/summary.csv" ]]; then
  ln -sfn "${REL_RUN_DIR}/summary.csv" "${RESULTS_RUN_DIR}/summary.csv" || true
fi
if [[ ! -e "${RESULTS_RUN_DIR}/summary_groups.csv" || -L "${RESULTS_RUN_DIR}/summary_groups.csv" ]]; then
  ln -sfn "${REL_RUN_DIR}/summary_groups.csv" "${RESULTS_RUN_DIR}/summary_groups.csv" || true
fi
if [[ ! -e "${RESULTS_RUN_DIR}/summary_groups_plus.csv" || -L "${RESULTS_RUN_DIR}/summary_groups_plus.csv" ]]; then
  ln -sfn "${REL_RUN_DIR}/summary_groups_plus.csv" "${RESULTS_RUN_DIR}/summary_groups_plus.csv" || true
fi

echo "[fsm] run_dir=${RUN_DIR}"
echo "[fsm] runs=${RUNS} duration=${DURATION_S}s"
echo "[fsm] ref_fullframes=${REF_FULLFRAMES_JSON}"
echo

COMPARE_PY="${WS_DIR}/src/pcd_gazebo_world/scripts/plot_reference_vs_replay.py"
SUMMARIZE_PY="${WS_DIR}/src/pcd_gazebo_world/tools/summarize_replay_suite.py"

cn_num() {
  /usr/bin/python3 - "$1" <<'PY'
import sys

n = int(sys.argv[1])
digits = "零一二三四五六七八九"


def to_cn(x: int) -> str:
    if x <= 0:
        return str(x)
    if x < 10:
        return digits[x]
    if x == 10:
        return "十"
    if x < 20:
        return "十" + digits[x % 10]
    if x < 100:
        tens = x // 10
        ones = x % 10
        s = digits[tens] + "十"
        if ones:
            s += digits[ones]
        return s
    if x < 1000:
        hundreds = x // 100
        rest = x % 100
        s = digits[hundreds] + "百"
        if rest == 0:
            return s
        if rest < 10:
            s += "零"
        return s + to_cn(rest)
    return str(x)


print(to_cn(n))
PY
}

run_one() {
  local label="$1"
  local exp_dir="$2"
  local cmd_port="$3"
  shift 3

  local algo="${label%%_*}"
  local algo_dir="${RUN_DIR}/${exp_dir}/${algo}"
  local csv="${algo_dir}/${label}_odom_map.csv"
  local report="${algo_dir}/${label}_fullframes_report.json"
  local colored="${algo_dir}/${label}_fullframes_colored.svg"
  local details="${algo_dir}/${label}_fullframes_details.csv"
  local roslaunch_log="${algo_dir}/${label}_roslaunch.log"
  local info_md="${algo_dir}/${label}_运行信息.md"

  mkdir -p "${algo_dir}"

  # Resume: if the full report already exists, do nothing.
  if [[ -f "${report}" && -f "${details}" && -f "${colored}" ]]; then
    echo "[skip] ${label} already has report"
    return 0
  fi

  local start_human
  start_human="$(date '+%Y-%m-%d %H:%M:%S %Z %z')"
  local start_cn
  start_cn="$(date '+%Y年%m月%d日_%H时%M分%S秒')"

  cat >"${info_md}" <<EOF
# ${label} 运行信息

- 开始时间：${start_human}（${start_cn}）
- port：${cmd_port}
- reference_fullframes：${REF_FULLFRAMES_JSON}
- path_json：${PATH_JSON}
- world_file：${WORLD_FILE}
- csv：${csv}

EOF

  echo "[run] ${label} port=${cmd_port}"
  echo "      csv=${csv}"

  if [[ ! -f "${csv}" ]]; then
    RECORD_CSV="${csv}" SHUTDOWN_ON_PATH_DONE=true timeout -s INT "${DURATION_S}" "$@" "${cmd_port}" "${PATH_JSON}" "${WORLD_FILE}" >"${roslaunch_log}" 2>&1 || true
  else
    echo "[resume] csv already exists, skip roslaunch: ${csv}"
  fi

  if [[ ! -f "${csv}" ]]; then
    echo "[warn] missing trajectory csv: ${csv}"
    return 0
  fi

  /usr/bin/python3 "${COMPARE_PY}" \
    --reference "${REF_FULLFRAMES_JSON}" \
    --replay "${csv}" \
    --out "" \
    --out-colored "${colored}" \
    --report "${report}" \
    --details-csv "${details}" >/dev/null

  local end_human
  end_human="$(date '+%Y-%m-%d %H:%M:%S %Z %z')"
  local end_cn
  end_cn="$(date '+%Y年%m月%d日_%H时%M分%S秒')"
  {
    echo "- 结束时间：${end_human}（${end_cn}）"
    echo "- report：${report}"
    echo "- colored：${colored}"
    echo "- details：${details}"
    echo "- roslaunch_log：${roslaunch_log}"
  } >>"${info_md}"

  echo "      report=${report}"
  echo "      colored=${colored}"
  echo "      details=${details}"
  echo "      roslaunch_log=${roslaunch_log}"
  echo "      info=${info_md}"
  echo

  sleep 5
}

for i in $(seq "${START_INDEX}" "${END_INDEX}"); do
  exp_cn="$(cn_num "${i}")"
  exp_dir="实验$(printf '%02d' "${i}")_${exp_cn}"
  exp_root="${RUN_DIR}/${exp_dir}"
  if [[ -e "${exp_root}" ]]; then
    if [[ "${RESUME}" == "true" ]]; then
      echo "[resume] experiment dir exists: ${exp_root}"
    else
      echo "ERROR: experiment dir already exists, refuse to overwrite: ${exp_root}" >&2
      echo "Hint: set RESUME=true to continue and fill missing files." >&2
      exit 3
    fi
  else
    mkdir -p "${exp_root}"
    group_items=()
    for a in "${ALGO_ARR[@]}"; do
      group_items+=("${a}_${i}")
    done
    group_items_slash="$(IFS=' / '; echo "${group_items[*]}")"

    {
      echo "# ${exp_dir}"
      echo
      echo "本组编号：${i}（${group_items_slash}）"
      echo
      echo "建议先看（对 rosbag 全帧参考）："
      for a in "${ALGO_ARR[@]}"; do
        echo "- ${a}：${a}/${a}_${i}_fullframes_colored.svg"
      done
      echo
    } >"${exp_root}/README.md"
    cat >>"${RUN_DIR}/运行信息.md" <<EOF
- 第${i}组：${exp_dir}（${group_items_slash}）
EOF
  fi

  # If the results dir has already been materialized into a real directory, don't
  # re-link it on resume; `ln -sfn` would create a nested `${exp_dir}/${exp_dir}`
  # symlink (and it may become broken due to relative path differences).
  if [[ -d "${RESULTS_RUN_DIR}/${exp_dir}" && ! -L "${RESULTS_RUN_DIR}/${exp_dir}" ]]; then
    :
  else
    ln -sfn "${REL_RUN_DIR}/${exp_dir}" "${RESULTS_RUN_DIR}/${exp_dir}" || true
  fi
  if [[ -e "${RESULTS_DIR}/${exp_dir}" && ! -L "${RESULTS_DIR}/${exp_dir}" ]]; then
    echo "[warn] results entry exists and is not symlink, skip link: ${RESULTS_DIR}/${exp_dir}" >&2
  else
    ln -sfn "latest/${exp_dir}" "${RESULTS_DIR}/${exp_dir}" || true
  fi

  base_port=$((PORT_BASE + i * 10))
  for idx in "${!ALGO_ARR[@]}"; do
    algo="${ALGO_ARR[$idx]}"
    cmd_port=$((base_port + idx + 1))
    label="${algo}_${i}"

    case "${algo}" in
      pid)
        run_one "${label}" "${exp_dir}" "${cmd_port}" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_pid_map_server.sh"
        ;;
      hybrid)
        run_one "${label}" "${exp_dir}" "${cmd_port}" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_hybrid_server.sh"
        ;;
      teb)
        run_one "${label}" "${exp_dir}" "${cmd_port}" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_teb_server.sh"
        ;;
      purepursuit)
        run_one "${label}" "${exp_dir}" "${cmd_port}" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_pure_pursuit_map_server.sh"
        ;;
      lqr)
        run_one "${label}" "${exp_dir}" "${cmd_port}" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_lqr_map_server.sh"
        ;;
      psolqr)
        run_one "${label}" "${exp_dir}" "${cmd_port}" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_psolqr_server.sh"
        ;;
      dwa)
        run_one "${label}" "${exp_dir}" "${cmd_port}" bash "${WS_DIR}/src/pcd_gazebo_world/tools/run_orchard_dwa_server.sh"
        ;;
    esac
  done
done

echo "[summary] ${RUN_DIR}"
/usr/bin/python3 "${SUMMARIZE_PY}" \
  --dir "${RUN_DIR}" \
  --out-csv "${RUN_DIR}/summary.csv" \
  --out-group-csv "${RUN_DIR}/summary_groups.csv" \
  --out-group-plus-csv "${RUN_DIR}/summary_groups_plus.csv" || true
echo

# Make the Chinese results directory self-contained (copy external links into it).
/usr/bin/python3 "${WS_DIR}/src/pcd_gazebo_world/tools/materialize_results_dir.py" --results "${RESULTS_RUN_DIR}" >/dev/null || true

# Update the convenience pointer only after we have a usable summary.
if [[ -f "${RUN_DIR}/summary_groups.csv" && -f "${RUN_DIR}/summary_groups_plus.csv" ]]; then
  ln -sfn "$(basename "${RESULTS_RUN_DIR}")" "${RESULTS_DIR}/latest" || true
else
  echo "[warn] summary missing, keep ${RESULTS_DIR}/latest unchanged: ${RUN_DIR}" >&2
fi

# Clean stale top-level experiment links that point into latest but don't exist
# (e.g., when a shorter run replaces a longer run).
shopt -s nullglob
for link in "${RESULTS_DIR}"/实验??_*; do
  [[ -L "${link}" ]] || continue
  target="$(readlink "${link}" || true)"
  if [[ "${target}" == latest/* && ! -e "${RESULTS_DIR}/${target}" ]]; then
    rm -f "${link}" || true
  fi
done
shopt -u nullglob

echo "[OK] finished: ${RUN_DIR}"
