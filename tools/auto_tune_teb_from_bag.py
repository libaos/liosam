#!/usr/bin/env python3
"""
Auto-tune a small set of replay parameters in Gazebo to make the replay trajectory
match a reference rosbag Path as closely as possible.

What it does:
1) Export the reference nav_msgs/Path from a rosbag to a JSON path in `map` frame (using bag /tf).
2) For each candidate parameter set:
   - run `run_orchard_teb_server.sh` for a fixed duration (Gazebo + move_base_benchmark + TEB)
   - compare replay odom CSV vs reference JSON and compute a score
3) Print the best candidate + write a summary under `trajectory_data/auto_tune_*`

Notes:
- This script must be run on a machine where ROS Noetic + Gazebo are runnable (not inside a restricted sandbox).
- It uses only the Python standard library, but it calls ROS/Gazebo tools via subprocess.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    waypoint_min_dist: float
    waypoint_tolerance: float
    weight_viapoint: float
    use_via_points: bool

    def tag(self) -> str:
        md = f"{self.waypoint_min_dist:.3f}".replace(".", "p")
        tol = f"{self.waypoint_tolerance:.3f}".replace(".", "p")
        wv = f"{self.weight_viapoint:.1f}".replace(".", "p")
        vp = "vp1" if self.use_via_points else "vp0"
        return f"md{md}_tol{tol}_wv{wv}_{vp}"


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _write_teb_override_yaml(path: Path, *, weight_viapoint: float) -> None:
    # Keep this aligned with `src/pcd_gazebo_world/config/teb_via_points_override.yaml`.
    content = f"""TebLocalPlannerROS:
  global_plan_viapoint_sep: -1.0
  via_points_ordered: true
  # Bias toward the rosbag path so the global planner's straight-line segments don't cut corners.
  weight_viapoint: {float(weight_viapoint):.6f}

  # The default orchard params are conservative and can get stuck once pointcloud obstacles appear.
  # Relax a bit so the robot can keep moving in narrow rows.
  min_obstacle_dist: 0.2
  inflation_dist: 0.3

  # Use the real robot footprint (1.15m x 0.46m) instead of the default oversized footprint.
  footprint: [[0.575, 0.23], [0.575, -0.23], [-0.575, -0.23], [-0.575, 0.23]]
  footprint_model:
    type: "polygon"
    vertices: [[0.575, 0.23], [0.575, -0.23], [-0.575, -0.23], [-0.575, 0.23]]
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_process_for_duration(cmd: Sequence[str], *, duration_s: float, cwd: Path, env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[cmd] {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )

        try:
            proc.wait(timeout=float(duration_s))
            return int(proc.returncode or 0)
        except subprocess.TimeoutExpired:
            pass

        # Graceful shutdown (roslaunch/gzserver handle SIGINT).
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            return int(proc.returncode or 0)

        try:
            proc.wait(timeout=45.0)
            return int(proc.returncode or 0)
        except subprocess.TimeoutExpired:
            pass

        # Harder shutdown.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return int(proc.returncode or 0)

        try:
            proc.wait(timeout=15.0)
            return int(proc.returncode or 0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                return int(proc.returncode or 0)
            proc.wait(timeout=10.0)
            return int(proc.returncode or 0)


def _run_compare(compare_py: Path, *, reference: Path, replay_csv: Path, out_svg: Path, out_report: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(compare_py),
        "--reference",
        str(reference),
        "--replay",
        str(replay_csv),
        "--out",
        str(out_svg),
        "--report",
        str(out_report),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"compare failed: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    try:
        return json.loads(proc.stdout)
    except Exception as exc:
        raise RuntimeError(f"compare output is not JSON: {exc}\n{proc.stdout}") from exc


def _score_report(report: Dict[str, Any]) -> float:
    cover = float(report.get("coverage_ratio", 0.0) or 0.0)
    if cover != cover:  # NaN
        cover = 0.0
    stats = report.get("replay_to_reference", {}) or {}
    mean_m = float(stats.get("mean_m", 1e9) or 1e9)
    p95_m = float(stats.get("p95_m", 1e9) or 1e9)
    max_m = float(stats.get("max_m", 1e9) or 1e9)

    # Coverage dominates: we want to finish the whole route first.
    return (1.0 - cover) * 1000.0 + p95_m * 100.0 + mean_m * 40.0 + max_m * 10.0


def _default_world_for_mode(ws_dir: Path, mode: str) -> Path:
    if mode == "no_obstacles":
        return ws_dir / "src" / "pcd_gazebo_world" / "worlds" / "empty_orchard.world"
    return ws_dir / "src" / "pcd_gazebo_world" / "worlds" / "orchard_from_pcd_validated_by_bag.world"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ws_dir = Path(__file__).resolve().parents[1]
    run_teb = ws_dir / "src" / "pcd_gazebo_world" / "tools" / "run_orchard_teb_server.sh"
    export_ref = ws_dir / "src" / "pcd_gazebo_world" / "scripts" / "rosbag_path_to_json.py"
    compare_py = ws_dir / "src" / "pcd_gazebo_world" / "scripts" / "plot_reference_vs_replay.py"

    p = argparse.ArgumentParser(description="Auto-tune Gazebo TEB replay to match a rosbag Path.")
    p.add_argument("--bag", required=True, help="Input rosbag (reference)")
    p.add_argument("--topic", default="", help="nav_msgs/Path topic (default: auto-detect like bag_route_replay)")
    p.add_argument("--port", type=int, default=11347, help="GAZEBO_MASTER_URI port (default: 11347)")
    p.add_argument("--duration", type=float, default=450.0, help="Seconds per candidate run (default: 450)")
    p.add_argument("--mode", choices=["velodyne", "static_map", "no_obstacles"], default="no_obstacles")
    p.add_argument("--world", default="", help="Optional Gazebo world .world override")

    p.add_argument("--waypoint-min-dist", default="0.3,0.5,0.8", help="Comma-separated list (meters)")
    p.add_argument("--waypoint-tolerance", default="0.2,0.35,0.5", help="Comma-separated list (meters)")
    p.add_argument("--weight-viapoint", default="25", help="Comma-separated list (TEB weight_viapoint)")
    p.add_argument("--use-via-points", default="1", help="Comma-separated list: 0/1 (default: 1)")

    p.add_argument("--target-coverage", type=float, default=0.99, help="Early-stop target (default: 0.99)")
    p.add_argument("--target-p95", type=float, default=0.5, help="Early-stop target in meters (default: 0.5)")
    p.add_argument("--sleep-between", type=float, default=5.0, help="Seconds between runs (default: 5)")
    args = p.parse_args(argv)

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        print(f"ERROR: bag not found: {bag_path}", file=sys.stderr)
        return 2

    if not run_teb.is_file():
        print(f"ERROR: missing runner: {run_teb}", file=sys.stderr)
        return 2
    if not export_ref.is_file():
        print(f"ERROR: missing exporter: {export_ref}", file=sys.stderr)
        return 2
    if not compare_py.is_file():
        print(f"ERROR: missing comparer: {compare_py}", file=sys.stderr)
        return 2

    world_path = Path(args.world).expanduser().resolve() if str(args.world).strip() else _default_world_for_mode(ws_dir, str(args.mode))
    if not world_path.is_file():
        print(f"ERROR: world not found: {world_path}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ws_dir / "trajectory_data" / f"auto_tune_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    ref_json = out_root / "reference_path_map.json"
    topic_arg = str(args.topic).strip()

    export_cmd = [
        sys.executable,
        str(export_ref),
        "--bag",
        str(bag_path),
        "--out",
        str(ref_json),
        "--min-dist",
        "0.4",
        "--target-frame",
        "map",
        "--tf-topic",
        "/tf",
    ]
    if topic_arg:
        export_cmd += ["--topic", topic_arg]

    print(f"[export] reference json: {ref_json}")
    proc = subprocess.run(export_cmd, check=False, capture_output=True, text=True, cwd=str(ws_dir))
    if proc.returncode != 0:
        print("[warn] failed to export to map frame using /tf; falling back to Path frame", file=sys.stderr)
        export_cmd = [
            sys.executable,
            str(export_ref),
            "--bag",
            str(bag_path),
            "--out",
            str(ref_json),
            "--min-dist",
            "0.4",
        ]
        if topic_arg:
            export_cmd += ["--topic", topic_arg]
        proc = subprocess.run(export_cmd, check=False, capture_output=True, text=True, cwd=str(ws_dir))
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            print("ERROR: cannot export reference path json", file=sys.stderr)
            return 2

    waypoint_min_dists = _parse_csv_floats(args.waypoint_min_dist)
    waypoint_tolerances = _parse_csv_floats(args.waypoint_tolerance)
    weight_viapoints = _parse_csv_floats(args.weight_viapoint)
    use_via_points_raw = [int(x) for x in _parse_csv_floats(args.use_via_points)]
    use_via_points = [bool(x) for x in use_via_points_raw]

    candidates: List[Candidate] = []
    for md, tol, wv, vp in itertools.product(waypoint_min_dists, waypoint_tolerances, weight_viapoints, use_via_points):
        candidates.append(Candidate(float(md), float(tol), float(wv), bool(vp)))

    if not candidates:
        print("ERROR: no candidates", file=sys.stderr)
        return 2

    print(f"[tune] candidates: {len(candidates)}  mode={args.mode}  world={world_path.name}")
    print(f"[tune] duration per run: {float(args.duration):.1f}s  gazebo port: {int(args.port)}")

    results_path = out_root / "results.jsonl"
    best: Optional[Tuple[float, Candidate, Dict[str, Any], Path]] = None  # (score, cand, report, case_dir)

    for idx, cand in enumerate(candidates, start=1):
        case_dir = out_root / f"{idx:03d}_{cand.tag()}"
        case_dir.mkdir(parents=True, exist_ok=True)

        teb_override = case_dir / "teb_override.yaml"
        _write_teb_override_yaml(teb_override, weight_viapoint=float(cand.weight_viapoint))

        csv_out = case_dir / "teb_odom.csv"
        svg_out = case_dir / "overlay.svg"
        report_out = case_dir / "report.json"
        log_out = case_dir / "roslaunch.log"

        env = os.environ.copy()
        # Isolate ROS master per run to reduce interference with any existing roscore.
        # If the user already exported ROS_MASTER_URI, respect it.
        env.setdefault("ROS_MASTER_URI", f"http://localhost:{int(args.port) + 1000}")
        env.setdefault("ROS_IP", "127.0.0.1")
        env["RECORD_CSV"] = str(csv_out)
        env["USE_VIA_POINTS"] = "true" if cand.use_via_points else "false"
        env["WAYPOINT_MIN_DIST"] = str(cand.waypoint_min_dist)
        env["WAYPOINT_TOLERANCE"] = str(cand.waypoint_tolerance)
        env["LOCAL_PLANNER_OVERRIDE_PARAMS"] = str(teb_override)

        cmd = [
            "bash",
            str(run_teb),
            str(int(args.port)),
            str(args.mode),
            str(ref_json),
            str(world_path),
        ]

        print(f"[{idx:03d}/{len(candidates)}] {cand.tag()}  (log: {log_out.name})")
        _run_process_for_duration(cmd, duration_s=float(args.duration), cwd=ws_dir, env=env, log_path=log_out)

        if not csv_out.is_file():
            entry = {
                "candidate": cand.__dict__,
                "score": None,
                "ok": False,
                "error": "missing_csv",
                "case_dir": str(case_dir),
            }
            results_path.write_text("", encoding="utf-8") if not results_path.exists() else None
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"  -> FAIL (missing csv): {csv_out.name}")
            time.sleep(float(args.sleep_between))
            continue

        try:
            report = _run_compare(compare_py, reference=ref_json, replay_csv=csv_out, out_svg=svg_out, out_report=report_out)
        except Exception as exc:
            entry = {
                "candidate": cand.__dict__,
                "score": None,
                "ok": False,
                "error": f"compare_failed: {exc}",
                "case_dir": str(case_dir),
            }
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"  -> FAIL (compare): {exc}")
            time.sleep(float(args.sleep_between))
            continue

        score = float(_score_report(report))
        cover = float(report.get("coverage_ratio", 0.0) or 0.0)
        p95 = float((report.get("replay_to_reference", {}) or {}).get("p95_m", 1e9) or 1e9)

        entry = {
            "candidate": cand.__dict__,
            "score": score,
            "ok": True,
            "metrics": {
                "coverage_ratio": cover,
                "mean_m": float((report.get("replay_to_reference", {}) or {}).get("mean_m", 1e9) or 1e9),
                "p95_m": p95,
                "max_m": float((report.get("replay_to_reference", {}) or {}).get("max_m", 1e9) or 1e9),
            },
            "case_dir": str(case_dir),
        }
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  -> score={score:.2f} cover={cover*100:.1f}% p95={p95:.2f}m")

        if best is None or score < best[0]:
            best = (score, cand, report, case_dir)

        if cover >= float(args.target_coverage) and p95 <= float(args.target_p95):
            print("[tune] early stop: reached target thresholds")
            break

        time.sleep(float(args.sleep_between))

    if best is None:
        print(f"[tune] no successful runs; see: {out_root}", file=sys.stderr)
        return 3

    score, cand, report, case_dir = best
    best_path = out_root / "best.json"
    best_payload = {
        "score": float(score),
        "candidate": cand.__dict__,
        "case_dir": str(case_dir),
        "report": report,
    }
    best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n[best]")
    print(f"  score   : {score:.2f}")
    print(f"  cand    : {cand}")
    print(f"  case_dir: {case_dir}")
    print(f"  report  : {case_dir / 'report.json'}")
    print(f"  overlay : {case_dir / 'overlay.svg'}")
    print(f"  override: {case_dir / 'teb_override.yaml'}")
    print(f"\n[all results] {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
