#!/usr/bin/env python3
"""Extract a nav_msgs/Odometry trajectory from a rosbag and save it as JSON/CSV.

This is useful as a dense reference trajectory (all frames) for comparing against
Gazebo closed-loop replay CSVs produced by `record_trajectory.py`.

Output JSON format matches `rosbag_path_to_json.py` (top-level `points` list of dicts
with x/y/z/yaw), so it can be consumed by `plot_reference_vs_replay.py` and
`publish_path.py`.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import rosbag
from tf.transformations import euler_from_quaternion


def _norm_frame(frame_id: str) -> str:
    return str(frame_id or "").strip().lstrip("/")


def _yaw_from_quat(q) -> float:
    return float(euler_from_quaternion([float(q.x), float(q.y), float(q.z), float(q.w)])[2])


def _load_tf_parent_child_series(
    bag_path: Path,
    tf_topic: str,
    parent_frame: str,
    child_frame: str,
) -> Tuple[List[float], List[Tuple[float, float, float]]]:
    parent_frame = _norm_frame(parent_frame)
    child_frame = _norm_frame(child_frame)
    if not parent_frame or not child_frame:
        raise ValueError("tf parent/child frame must be non-empty")

    times: List[float] = []
    tfs: List[Tuple[float, float, float]] = []  # (tx,ty,yaw)

    with rosbag.Bag(str(bag_path), "r") as bag:
        for _topic, msg, _t in bag.read_messages(topics=[tf_topic]):
            for tr in getattr(msg, "transforms", []) or []:
                parent = _norm_frame(getattr(getattr(tr, "header", None), "frame_id", ""))
                child = _norm_frame(getattr(tr, "child_frame_id", ""))
                if parent != parent_frame or child != child_frame:
                    continue
                stamp = getattr(getattr(tr, "header", None), "stamp", None)
                if stamp is None:
                    continue
                ts = float(stamp.to_sec())
                trans = tr.transform.translation
                rot = tr.transform.rotation
                times.append(ts)
                tfs.append((float(trans.x), float(trans.y), _yaw_from_quat(rot)))

    if not times:
        raise RuntimeError(
            f"TF not found in bag: topic={tf_topic} parent={parent_frame} child={child_frame} (need a direct transform)"
        )

    order = sorted(range(len(times)), key=times.__getitem__)
    times_sorted = [times[i] for i in order]
    tfs_sorted = [tfs[i] for i in order]
    return times_sorted, tfs_sorted


def _apply_tf_parent_child(
    points: List[Tuple[float, float, float, float]],
    point_times: List[float],
    tf_times: List[float],
    tf_xy_yaw: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    if len(points) != len(point_times):
        raise ValueError("points and point_times length mismatch")
    if not points:
        return []
    if not tf_times:
        raise ValueError("empty tf series")

    out: List[Tuple[float, float, float, float]] = []
    for (x, y, z, yaw), t in zip(points, point_times):
        idx = bisect.bisect_right(tf_times, float(t)) - 1
        idx = max(0, min(idx, len(tf_times) - 1))
        tx, ty, yaw_tf = tf_xy_yaw[idx]
        c = float(math.cos(yaw_tf))
        s = float(math.sin(yaw_tf))
        x_out = float(tx) + c * float(x) - s * float(y)
        y_out = float(ty) + s * float(x) + c * float(y)
        yaw_out = float(yaw_tf) + float(yaw)
        out.append((x_out, y_out, z, yaw_out))
    return out


def _apply_transform(
    points: List[Tuple[float, float, float, float]],
    yaw_offset: float,
    dx: float,
    dy: float,
) -> List[Tuple[float, float, float, float]]:
    if abs(yaw_offset) < 1.0e-9 and abs(dx) < 1.0e-9 and abs(dy) < 1.0e-9:
        return points
    c = math.cos(yaw_offset)
    s = math.sin(yaw_offset)
    out: List[Tuple[float, float, float, float]] = []
    for x, y, z, yaw in points:
        xr = c * x - s * y + dx
        yr = s * x + c * y + dy
        out.append((float(xr), float(yr), float(z), float(yaw + yaw_offset)))
    return out


def _downsample_points(
    points: List[Tuple[float, float, float, float]],
    times: List[float],
    min_dist: float,
) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    if min_dist <= 0.0:
        return points, times
    if len(points) != len(times):
        raise ValueError("points/times length mismatch")
    out_p: List[Tuple[float, float, float, float]] = []
    out_t: List[float] = []
    last_x = last_y = None
    for (x, y, z, yaw), t in zip(points, times):
        if last_x is None:
            out_p.append((x, y, z, yaw))
            out_t.append(float(t))
            last_x, last_y = x, y
            continue
        if math.hypot(x - last_x, y - last_y) >= float(min_dist):
            out_p.append((x, y, z, yaw))
            out_t.append(float(t))
            last_x, last_y = x, y
    return out_p, out_t


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract nav_msgs/Odometry from rosbag and save JSON/CSV")
    parser.add_argument("--bag", type=str, default="rosbags/2025-10-29-16-05-00.bag", help="输入 rosbag 路径")
    parser.add_argument("--topic", type=str, default="/liorl/mapping/odometry", help="Odometry topic")
    parser.add_argument("--out", type=str, default="maps/runs/rosbag_odometry.json", help="输出 JSON")
    parser.add_argument("--out-csv", type=str, default="", help="可选 CSV 输出（含 timestamp）")
    parser.add_argument("--min-dist", type=float, default=0.0, help="下采样距离阈值（m，0=不下采样）")
    parser.add_argument(
        "--target-frame",
        type=str,
        default="",
        help="把 odom 点从其 header.frame_id 转到该 frame（需 bag 内 /tf 里有 target->source 的直接 TF）",
    )
    parser.add_argument("--tf-topic", type=str, default="/tf", help="TF topic（用于 target-frame 变换）")
    parser.add_argument("--frame-id", type=str, default="", help="覆盖输出 frame_id")
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="整体旋转偏移（rad）")
    parser.add_argument("--offset-x", type=float, default=0.0, help="整体平移 x（m）")
    parser.add_argument("--offset-y", type=float, default=0.0, help="整体平移 y（m）")
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve() if str(args.out_csv).strip() else None

    if not bag_path.is_file():
        raise SystemExit(f"bag not found: {bag_path}")

    points: List[Tuple[float, float, float, float]] = []
    times: List[float] = []
    msg_source_frame_raw = ""
    source_frame = ""
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _topic, msg, _t in bag.read_messages(topics=[str(args.topic)]):
            if getattr(msg, "_type", "") != "nav_msgs/Odometry":
                continue
            if not source_frame:
                msg_source_frame_raw = str(getattr(getattr(msg, "header", None), "frame_id", "") or "")
                source_frame = _norm_frame(msg_source_frame_raw)
            stamp = getattr(getattr(msg, "header", None), "stamp", None)
            if stamp is None:
                continue
            ts = float(stamp.to_sec())
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            points.append((float(p.x), float(p.y), float(p.z), _yaw_from_quat(q)))
            times.append(ts)

    if not points:
        raise SystemExit(f"no Odometry messages found on: {args.topic}")
    if not source_frame:
        source_frame = "map"
    if not msg_source_frame_raw:
        msg_source_frame_raw = source_frame

    target_frame = _norm_frame(str(args.target_frame))
    if target_frame and target_frame != source_frame:
        tf_times, tf_xy_yaw = _load_tf_parent_child_series(
            bag_path,
            tf_topic=str(args.tf_topic),
            parent_frame=target_frame,
            child_frame=source_frame,
        )
        points = _apply_tf_parent_child(points, times, tf_times, tf_xy_yaw)
        source_frame = target_frame

    points = _apply_transform(points, float(args.yaw_offset), float(args.offset_x), float(args.offset_y))
    points, times = _downsample_points(points, times, float(args.min_dist))

    output_frame_id = str(args.frame_id).strip() or source_frame

    out = {
        "bag": str(bag_path),
        "topic": str(args.topic),
        "frame_id": output_frame_id,
        "source_frame": str(msg_source_frame_raw),
        "target_frame": str(args.target_frame).strip(),
        "tf_topic": str(args.tf_topic).strip() if str(args.target_frame).strip() else "",
        "min_dist": float(args.min_dist),
        "yaw_offset": float(args.yaw_offset),
        "offset_x": float(args.offset_x),
        "offset_y": float(args.offset_y),
        "points": [
            {"t": float(t), "x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw)}
            for (x, y, z, yaw), t in zip(points, times)
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] odom points: {len(points)} -> {out_path}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "timestamp", "x", "y", "z", "yaw"])
            for i, ((x, y, z, yaw), t) in enumerate(zip(points, times)):
                writer.writerow([i, float(t), x, y, z, yaw])
        print(f"[OK] csv: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
