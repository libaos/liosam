#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rosbag
import yaml


@dataclass(frozen=True)
class RefPath:
    s: np.ndarray  # (N,)
    xy: np.ndarray  # (N,2)
    frame_id: str


def _distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pick_best_path_topic(bag_path: str, requested_topic: Optional[str]) -> str:
    if requested_topic:
        return requested_topic

    with rosbag.Bag(bag_path) as bag:
        topics = bag.get_type_and_topic_info().topics

    path_topics = [
        topic
        for topic, info in topics.items()
        if getattr(info, "msg_type", None) == "nav_msgs/Path"
    ]
    if not path_topics:
        raise RuntimeError("No nav_msgs/Path topics found in bag; pass --run-topic explicitly.")

    priority = [
        "/lio_sam/mapping/path",
        "/liorl/mapping/path",
        "/liorf/mapping/path",
    ]
    for candidate in priority:
        if candidate in path_topics:
            return candidate

    mapping_paths = [t for t in path_topics if t.endswith("/mapping/path")]
    if mapping_paths:
        return sorted(mapping_paths)[0]

    return sorted(path_topics)[0]


def _load_last_path(bag_path: str, topic: str):
    last_msg = None
    last_t = None
    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            last_msg = msg
            last_t = t
    if last_msg is None:
        raise RuntimeError(f"Topic not found in bag: {topic}")
    if not hasattr(last_msg, "poses"):
        raise RuntimeError(f"Topic is not nav_msgs/Path (missing poses[]): {topic}")
    if not last_msg.poses:
        raise RuntimeError(f"Path is empty on topic: {topic}")
    return last_msg, last_t


def _load_path_xy_time_series(bag_path: str, topic: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a time series trajectory from a nav_msgs/Path topic.

    Many Path publishers emit the full path repeatedly. For evaluation and
    mode-time alignment, we take the *last pose* of each Path message as the
    robot position sample at that message time.

    Time source: use bag time (rosbag timestamp). This aligns with other topics
    recorded in the same bag (e.g. `/fsm/mode`), regardless of Path pose header
    stamp quirks.
    """

    bag_times: List[float] = []
    xy: List[Tuple[float, float]] = []

    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            if not hasattr(msg, "poses") or not msg.poses:
                continue

            ps = msg.poses[-1]
            try:
                x = float(ps.pose.position.x)
                y = float(ps.pose.position.y)
            except Exception:
                continue

            xy.append((x, y))
            bag_times.append(float(t.to_sec()))

    if not xy:
        raise RuntimeError(f"No usable Path samples found on topic: {topic}")

    bag_times_arr = np.asarray(bag_times, dtype=np.float64)
    xy_arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return bag_times_arr, xy_arr


def _downsample_xy_with_time(
    xy: np.ndarray,
    times: np.ndarray,
    min_dist_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if min_dist_m <= 0:
        return times, xy
    if xy.size == 0:
        return times, xy

    out_xy = [tuple(xy[0].tolist())]
    out_t = [float(times[0])]
    last = out_xy[0]

    for i in range(1, int(xy.shape[0]) - 1):
        p = (float(xy[i, 0]), float(xy[i, 1]))
        if _distance_xy(p, last) >= float(min_dist_m):
            out_xy.append(p)
            out_t.append(float(times[i]))
            last = p

    last_p = (float(xy[-1, 0]), float(xy[-1, 1]))
    if _distance_xy(last_p, out_xy[-1]) > 1.0e-9:
        out_xy.append(last_p)
        out_t.append(float(times[-1]))

    return np.asarray(out_t, dtype=np.float64), np.asarray(out_xy, dtype=np.float64).reshape(-1, 2)


def _downsample_xy(xy: Sequence[Tuple[float, float]], min_dist_m: float) -> List[Tuple[float, float]]:
    if min_dist_m <= 0:
        return list(xy)
    if not xy:
        return []

    out = [xy[0]]
    last = xy[0]
    for p in xy[1:-1]:
        if _distance_xy(p, last) >= min_dist_m:
            out.append(p)
            last = p
    if xy[-1] is not out[-1]:
        out.append(xy[-1])
    return out


def _read_ref_csv(path: Path) -> RefPath:
    s: List[float] = []
    x: List[float] = []
    y: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s.append(float(row["s"]))
            x.append(float(row["x"]))
            y.append(float(row["y"]))

    if not s:
        raise RuntimeError(f"Empty ref CSV: {path}")

    return RefPath(
        s=np.asarray(s, dtype=np.float64),
        xy=np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=1),
        frame_id="map",
    )


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    label: str


def _load_label_segments(path: Optional[Path]) -> Tuple[str, List[Segment]]:
    if path is None:
        return "straight", []
    obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    default_label = str(obj.get("default_label", "straight") or "straight")
    segments_raw = obj.get("segments", []) or []

    segments: List[Segment] = []
    for seg in segments_raw:
        if not isinstance(seg, dict):
            continue
        if "start_s" in seg and "end_s" in seg:
            segments.append(
                Segment(
                    start_s=float(seg["start_s"]),
                    end_s=float(seg["end_s"]),
                    label=str(seg.get("label", default_label) or default_label),
                )
            )
        else:
            raise ValueError(
                "labels.yaml segments must use {start_s,end_s,label}; "
                "use tools/navigation_experiments/export_reference_path.py to get s (meters)."
            )

    return default_label, segments


def _normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    if label in {"s", "straight", "line", "forward", "go"}:
        return "straight"
    if label in {"l", "left", "turn_left"}:
        return "left"
    if label in {"r", "right", "turn_right"}:
        return "right"
    if not label:
        return "unknown"
    return label


def _label_of_s(s: float, *, default_label: str, segments: Sequence[Segment]) -> str:
    for seg in segments:
        if seg.start_s <= s < seg.end_s:
            return _normalize_label(seg.label)
    return _normalize_label(default_label)


def _point_to_segment_distance_and_s(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, s_a: float, s_b: float
) -> Tuple[float, float]:
    v = b - a
    vv = float(np.dot(v, v))
    if vv < 1e-12:
        d = float(np.linalg.norm(p - a))
        return d, float(s_a)
    t = float(np.dot(p - a, v) / vv)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    proj = a + t * v
    d = float(np.linalg.norm(p - proj))
    s_proj = float(s_a + t * (s_b - s_a))
    return d, s_proj


def _nearest_ref_window(ref_xy: np.ndarray, p: np.ndarray, center: int, window: int) -> int:
    n = int(ref_xy.shape[0])
    lo = max(0, int(center) - int(window))
    hi = min(n, int(center) + int(window) + 1)
    d2 = np.sum((ref_xy[lo:hi] - p.reshape(1, 2)) ** 2, axis=1)
    return int(lo + int(np.argmin(d2)))


def _compute_match_errors(
    ref: RefPath,
    traj_xy: np.ndarray,
    *,
    default_label: str,
    segments: Sequence[Segment],
    window: int = 80,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    n_ref = int(ref.xy.shape[0])
    if n_ref < 2:
        raise RuntimeError("ref path too short")
    if traj_xy.size == 0:
        raise RuntimeError("trajectory is empty")

    errors: List[float] = []
    mapped_s: List[float] = []
    labels: List[str] = []

    prev_idx = int(np.argmin(np.sum((ref.xy - traj_xy[0].reshape(1, 2)) ** 2, axis=1)))
    for i in range(int(traj_xy.shape[0])):
        p = traj_xy[i]

        idx = _nearest_ref_window(ref.xy, p, prev_idx, window=window)
        prev_idx = idx

        # Evaluate distance to the closest of the 2 adjacent segments around idx.
        candidates: List[Tuple[float, float]] = []
        if idx > 0:
            d, s_proj = _point_to_segment_distance_and_s(p, ref.xy[idx - 1], ref.xy[idx], float(ref.s[idx - 1]), float(ref.s[idx]))
            candidates.append((d, s_proj))
        if idx < n_ref - 1:
            d, s_proj = _point_to_segment_distance_and_s(p, ref.xy[idx], ref.xy[idx + 1], float(ref.s[idx]), float(ref.s[idx + 1]))
            candidates.append((d, s_proj))

        if not candidates:
            d = float(np.linalg.norm(p - ref.xy[idx]))
            s_proj = float(ref.s[idx])
        else:
            d, s_proj = min(candidates, key=lambda t: t[0])

        errors.append(float(d))
        mapped_s.append(float(s_proj))
        labels.append(_label_of_s(float(s_proj), default_label=default_label, segments=segments))

    return np.asarray(errors, dtype=np.float64), np.asarray(mapped_s, dtype=np.float64), labels


def _stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"n": 0.0}
    v = values.astype(np.float64).reshape(-1)
    return {
        "n": float(v.size),
        "mean": float(np.mean(v)),
        "rmse": float(math.sqrt(float(np.mean(v * v)))),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
    }


def _read_mode_topic(bag_path: str, mode_topic: str) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=[mode_topic]):
            label = str(getattr(msg, "data", "")).strip()
            if not label:
                continue
            out.append((float(t.to_sec()), _normalize_label(label)))
    return out


def _mode_switch_stats(
    mode_samples: Sequence[Tuple[float, str]],
    traj_times: np.ndarray,
    traj_mapped_s: np.ndarray,
    gt_boundaries_s: Sequence[float],
) -> Dict[str, object]:
    if not mode_samples:
        return {}
    if traj_times.size == 0:
        return {}

    # Map each mode timestamp to closest trajectory sample -> s.
    s_events: List[Tuple[float, str]] = []
    for t, label in mode_samples:
        j = int(np.clip(np.searchsorted(traj_times, t), 0, traj_times.size - 1))
        if j > 0 and abs(traj_times[j - 1] - t) < abs(traj_times[j] - t):
            j -= 1
        s_events.append((float(traj_mapped_s[j]), label))

    # Extract boundaries from mode stream.
    boundaries: List[float] = []
    last = None
    for s, label in s_events:
        if last is None:
            last = label
            continue
        if label != last:
            boundaries.append(float(s))
            last = label

    # Compare with GT boundaries in order (same count assumption).
    delays: List[float] = []
    for i in range(min(len(boundaries), len(gt_boundaries_s))):
        delays.append(float(boundaries[i] - float(gt_boundaries_s[i])))

    return {
        "mode_switches": int(len(boundaries)),
        "mode_boundaries_s": boundaries,
        "gt_boundaries_s": [float(x) for x in gt_boundaries_s],
        "boundary_delay_m": delays,
        "boundary_delay_abs_mean_m": float(np.mean(np.abs(delays))) if delays else None,
    }


def _gt_boundaries_from_segments(default_label: str, segments: Sequence[Segment]) -> List[float]:
    # Boundaries are the start_s of each segment except the first, sorted.
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s.start_s)
    out: List[float] = []
    last_label = _normalize_label(default_label)
    first = True
    for seg in segs:
        label = _normalize_label(seg.label)
        if first:
            first = False
            last_label = label
            continue
        if label != last_label:
            out.append(float(seg.start_s))
            last_label = label
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a navigation run (bag) against a reference path CSV with straight/left/right labels.")
    p.add_argument("--ref-csv", required=True, help="Reference CSV from export_reference_path.py")
    p.add_argument("--labels", default="", help="Labels YAML (segments by s), optional")
    p.add_argument("--run-bag", required=True, help="Run bag to evaluate")
    p.add_argument("--run-topic", default="", help="nav_msgs/Path topic in run bag (default: auto-detect)")
    p.add_argument("--traj-min-dist", type=float, default=0.1, help="Downsample executed trajectory spacing (m)")
    p.add_argument("--goal-tolerance", type=float, default=0.8, help="Success tolerance to final ref point (m)")
    p.add_argument("--method", default="", help="Method name written into metrics.json (e.g. teb_only/fsm_n3)")
    p.add_argument("--out-dir", default="", help="Output dir (default: maps/nav_eval/<run_bag_stem>)")
    p.add_argument("--export-samples", action="store_true", help="Export per-sample error CSV")
    p.add_argument("--mode-topic", default="", help="Optional std_msgs/String topic for FSM mode (straight/left/right)")
    args = p.parse_args(argv)

    ref_csv = Path(args.ref_csv).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve() if str(args.labels).strip() else None
    run_bag = Path(args.run_bag).expanduser().resolve()
    if not ref_csv.is_file():
        raise RuntimeError(f"ref CSV not found: {ref_csv}")
    if labels_path is not None and not labels_path.is_file():
        raise RuntimeError(f"labels file not found: {labels_path}")
    if not run_bag.is_file():
        raise RuntimeError(f"run bag not found: {run_bag}")

    ref = _read_ref_csv(ref_csv)
    default_label, segments = _load_label_segments(labels_path)

    topic = _pick_best_path_topic(str(run_bag), str(args.run_topic).strip() or None)
    traj_times_abs, traj_xy = _load_path_xy_time_series(str(run_bag), topic)
    traj_times_abs, traj_xy = _downsample_xy_with_time(traj_xy, traj_times_abs, float(args.traj_min_dist))

    t0 = float(traj_times_abs[0]) if traj_times_abs.size else 0.0
    traj_times_rel = traj_times_abs - t0

    errors, mapped_s, labels = _compute_match_errors(ref, traj_xy, default_label=default_label, segments=segments)

    total_time_s = float(traj_times_rel[-1] - traj_times_rel[0]) if traj_times_rel.size >= 2 else 0.0
    travel_dist = float(np.sum(np.linalg.norm(np.diff(traj_xy, axis=0), axis=1))) if traj_xy.shape[0] >= 2 else 0.0
    ref_len = float(ref.s[-1])
    end_err = float(np.linalg.norm(traj_xy[-1] - ref.xy[-1]))
    success = bool(end_err <= float(args.goal_tolerance))

    per_label: Dict[str, Dict[str, float]] = {}
    for lab in sorted(set(labels)):
        mask = np.asarray([l == lab for l in labels], dtype=bool)
        per_label[lab] = _stats(errors[mask])

    metrics: Dict[str, object] = {
        "method": str(args.method).strip() or None,
        "ref_csv": str(ref_csv),
        "labels": str(labels_path) if labels_path is not None else None,
        "run_bag": str(run_bag),
        "run_topic": topic,
        "traj_min_dist": float(args.traj_min_dist),
        "goal_tolerance": float(args.goal_tolerance),
        "duration_s": total_time_s,
        "travel_distance_m": travel_dist,
        "ref_length_m": ref_len,
        "end_error_to_goal_m": end_err,
        "success": success,
        "cte_overall": _stats(errors),
        "cte_by_label": per_label,
    }

    if str(args.mode_topic).strip():
        mode_samples = _read_mode_topic(str(run_bag), str(args.mode_topic).strip())
        gt_boundaries = _gt_boundaries_from_segments(default_label, segments)
        metrics["mode_topic"] = str(args.mode_topic).strip()
        metrics["mode_stats"] = _mode_switch_stats(mode_samples, traj_times_abs, mapped_s, gt_boundaries)

    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else Path("maps/nav_eval") / run_bag.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.export_samples:
        with (out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "s_ref", "label", "cte"])
            for i in range(int(traj_xy.shape[0])):
                w.writerow(
                    [
                        f"{float(traj_times_rel[i]):.6f}",
                        f"{float(traj_xy[i, 0]):.6f}",
                        f"{float(traj_xy[i, 1]):.6f}",
                        f"{float(mapped_s[i]):.6f}",
                        labels[i],
                        f"{float(errors[i]):.6f}",
                    ]
                )

    print(f"[OK] Wrote: {out_dir / 'metrics.json'}  (success={success}, cte_rmse={metrics['cte_overall'].get('rmse')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
