#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import rosbag
import yaml


@dataclass(frozen=True)
class RefPoint:
    idx: int
    s: float
    x: float
    y: float
    yaw: float
    kappa: float


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
        raise RuntimeError("No nav_msgs/Path topics found in bag; pass --topic explicitly.")

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
    with rosbag.Bag(bag_path) as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            last_msg = msg
    if last_msg is None:
        raise RuntimeError(f"Topic not found in bag: {topic}")
    if not hasattr(last_msg, "poses"):
        raise RuntimeError(f"Topic is not nav_msgs/Path (missing poses[]): {topic}")
    if not last_msg.poses:
        raise RuntimeError(f"Path is empty on topic: {topic}")
    return last_msg


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


def _cap_uniform(xy: Sequence[Tuple[float, float]], max_points: int) -> List[Tuple[float, float]]:
    if max_points <= 0:
        return list(xy)
    if max_points < 2:
        raise ValueError("--max-points must be >= 2")
    if len(xy) <= max_points:
        return list(xy)

    out = [xy[0]]
    inner = xy[1:-1]
    need_inner = max_points - 2
    step = max(1, len(inner) // need_inner)
    out.extend(inner[::step][:need_inner])
    out.append(xy[-1])
    return out


def _yaw_from_direction(this_xy: Tuple[float, float], next_xy: Tuple[float, float]) -> float:
    return math.atan2(next_xy[1] - this_xy[1], next_xy[0] - this_xy[0])


def _signed_curvature(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    abx, aby = b[0] - a[0], b[1] - a[1]
    acx, acy = c[0] - a[0], c[1] - a[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]

    area2 = abx * acy - aby * acx  # signed, = 2*triangle_area
    ab = math.hypot(abx, aby)
    bc = math.hypot(bcx, bcy)
    ac = math.hypot(acx, acy)
    denom = ab * bc * ac
    if denom < 1e-9:
        return 0.0
    return 2.0 * area2 / denom


def _build_ref_points(xy: Sequence[Tuple[float, float]]) -> List[RefPoint]:
    if not xy:
        return []

    yaws: List[float] = []
    last_yaw = 0.0
    for i in range(len(xy)):
        if len(xy) >= 2:
            if i < len(xy) - 1:
                dx = xy[i + 1][0] - xy[i][0]
                dy = xy[i + 1][1] - xy[i][1]
                if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                    yaw = last_yaw
                else:
                    yaw = math.atan2(dy, dx)
            else:
                yaw = last_yaw
        else:
            yaw = 0.0
        last_yaw = yaw
        yaws.append(yaw)

    s_vals = [0.0]
    for i in range(1, len(xy)):
        s_vals.append(s_vals[-1] + _distance_xy(xy[i - 1], xy[i]))

    kappas = [0.0] * len(xy)
    for i in range(1, len(xy) - 1):
        kappas[i] = _signed_curvature(xy[i - 1], xy[i], xy[i + 1])

    out: List[RefPoint] = []
    for i, (p, s, yaw, kappa) in enumerate(zip(xy, s_vals, yaws, kappas)):
        out.append(RefPoint(idx=int(i), s=float(s), x=float(p[0]), y=float(p[1]), yaw=float(yaw), kappa=float(kappa)))
    return out


def _write_csv(path: Path, points: Sequence[RefPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "s", "x", "y", "yaw", "kappa"])
        for p in points:
            w.writerow([p.idx, f"{p.s:.6f}", f"{p.x:.6f}", f"{p.y:.6f}", f"{p.yaw:.6f}", f"{p.kappa:.6f}"])


def _write_labels_template(path: Path, *, frame_id: str, source_bag: str, source_topic: str, min_dist: float, max_points: int, n_points: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "version": 1,
        "frame_id": frame_id,
        "source": {"bag": source_bag, "topic": source_topic},
        "min_dist": float(min_dist),
        "max_points": int(max_points),
        "n_points": int(n_points),
        "default_label": "straight",
        "allowed_labels": ["straight", "left", "right"],
        "segments": [],
    }
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export a reference nav_msgs/Path from rosbag as CSV with arc-length and curvature.")
    p.add_argument("bag", help="Input .bag file (reference route)")
    p.add_argument("--topic", default=None, help="nav_msgs/Path topic (default: auto-detect)")
    p.add_argument("--min-dist", type=float, default=0.5, help="Downsample spacing in meters (default: 0.5)")
    p.add_argument("--max-points", type=int, default=0, help="Cap points uniformly (0 = no cap)")
    p.add_argument("--out-csv", default="maps/ref_route/ref_path.csv", help="Output CSV path")
    p.add_argument("--out-labels", default="", help="Output labels YAML template path (optional)")
    args = p.parse_args(argv)

    bag_path = str(Path(args.bag).expanduser().resolve())
    if not os.path.isfile(bag_path):
        raise RuntimeError(f"Bag file not found: {bag_path}")

    topic = _pick_best_path_topic(bag_path, args.topic)
    path_msg = _load_last_path(bag_path, topic)

    frame_id = str(getattr(path_msg.header, "frame_id", "") or "").strip()
    if not frame_id and getattr(path_msg, "poses", None):
        frame_id = str(getattr(path_msg.poses[0].header, "frame_id", "") or "").strip()
    if not frame_id:
        frame_id = "map"

    xy = [(ps.pose.position.x, ps.pose.position.y) for ps in path_msg.poses]
    xy = _downsample_xy(xy, float(args.min_dist))
    xy = _cap_uniform(xy, int(args.max_points))

    points = _build_ref_points(xy)
    if not points:
        raise RuntimeError("No points exported")

    out_csv = Path(args.out_csv).expanduser().resolve()
    _write_csv(out_csv, points)

    if str(args.out_labels).strip():
        out_labels = Path(args.out_labels).expanduser().resolve()
        _write_labels_template(
            out_labels,
            frame_id=frame_id,
            source_bag=bag_path,
            source_topic=topic,
            min_dist=float(args.min_dist),
            max_points=int(args.max_points),
            n_points=len(points),
        )

    print(f"[OK] CSV: {out_csv}  (points={len(points)}, frame={frame_id}, topic={topic})")
    if str(args.out_labels).strip():
        print(f"[OK] labels template: {Path(args.out_labels).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

