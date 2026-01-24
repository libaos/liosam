#!/usr/bin/env python3
"""Generate a pose trajectory in a target frame at timestamps from a bag topic.

Typical use-case:
- You want a "full-frame" reference (e.g. 2822 lidar frames) instead of a downsampled Path.
- You have TF in the bag that provides: output_frame -> odom_frame -> base_frame.

Example (this bag):
  /liorl/deskew/cloud_deskewed has 2822 frames (header.frame_id=base_link_est)
  TF provides map->odom_est and odom_est->base_link_est
  => output base_link_est pose in map at each lidar frame time.
"""

from __future__ import annotations

import argparse
import bisect
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


def _lookup_series(times: List[float], values: List[Tuple[float, float, float]], t: float) -> Tuple[float, float, float]:
    if not times:
        return (0.0, 0.0, 0.0)
    idx = bisect.bisect_right(times, float(t)) - 1
    idx = max(0, min(idx, len(times) - 1))
    return values[idx]


def _compose(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Compose 2D transforms a∘b.

    Each transform is (tx, ty, yaw) meaning: p_out = R(yaw)*p_in + t
    """
    ax, ay, ayaw = a
    bx, by, byaw = b
    c = float(math.cos(ayaw))
    s = float(math.sin(ayaw))
    x = float(ax) + c * float(bx) - s * float(by)
    y = float(ay) + s * float(bx) + c * float(by)
    yaw = float(ayaw) + float(byaw)
    return (x, y, yaw)


def _load_tf_series(
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
    vals: List[Tuple[float, float, float]] = []

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
                vals.append((float(trans.x), float(trans.y), _yaw_from_quat(rot)))

    if not times:
        raise RuntimeError(f"TF not found in bag: {parent_frame} -> {child_frame} on {tf_topic}")

    order = sorted(range(len(times)), key=times.__getitem__)
    return ([times[i] for i in order], [vals[i] for i in order])


def _load_times(
    bag_path: Path,
    topic: str,
    min_dt: float,
    max_msgs: int,
) -> Tuple[List[float], Optional[str]]:
    times: List[float] = []
    last_t: Optional[float] = None
    frame_id: Optional[str] = None
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _topic, msg, _t in bag.read_messages(topics=[topic]):
            hdr = getattr(msg, "header", None)
            stamp = getattr(hdr, "stamp", None)
            if stamp is None:
                continue
            ts = float(stamp.to_sec())
            if last_t is not None and min_dt > 0.0 and (ts - last_t) < float(min_dt):
                continue
            times.append(ts)
            last_t = ts
            if frame_id is None:
                frame_id = str(getattr(hdr, "frame_id", "") or "") or None
            if max_msgs > 0 and len(times) >= max_msgs:
                break
    if not times:
        raise RuntimeError(f"No header.stamp found on topic: {topic}")
    return times, frame_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate pose points at timestamps from a bag topic")
    parser.add_argument("--bag", type=str, default="rosbags/2025-10-29-16-05-00.bag")
    parser.add_argument("--times-topic", type=str, default="/liorl/deskew/cloud_deskewed", help="用这个 topic 的时间戳当“全帧”基准")
    parser.add_argument("--tf-topic", type=str, default="/tf")
    parser.add_argument("--output-frame", type=str, default="map")
    parser.add_argument("--odom-frame", type=str, default="odom_est")
    parser.add_argument("--base-frame", type=str, default="base_link_est")
    parser.add_argument("--out", type=str, default="maps/runs/rosbag_pose_fullframes_map.json")
    parser.add_argument("--min-dt", type=float, default=0.0, help="按时间间隔下采样（0=不下采样）")
    parser.add_argument("--max-msgs", type=int, default=0, help="最多读取多少条（0=不限制）")
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not bag_path.is_file():
        raise SystemExit(f"bag not found: {bag_path}")

    output_frame = _norm_frame(args.output_frame)
    odom_frame = _norm_frame(args.odom_frame)
    base_frame = _norm_frame(args.base_frame)
    if not output_frame or not odom_frame or not base_frame:
        raise SystemExit("output/odom/base frames must be non-empty")

    times, times_topic_frame = _load_times(
        bag_path,
        topic=str(args.times_topic),
        min_dt=float(args.min_dt),
        max_msgs=int(args.max_msgs),
    )

    # Load TF series for output->odom and odom->base.
    tf1_t: List[float] = []
    tf1_v: List[Tuple[float, float, float]] = []
    if output_frame != odom_frame:
        tf1_t, tf1_v = _load_tf_series(bag_path, str(args.tf_topic), output_frame, odom_frame)

    tf2_t, tf2_v = _load_tf_series(bag_path, str(args.tf_topic), odom_frame, base_frame)

    points = []
    for t in times:
        a = (0.0, 0.0, 0.0) if output_frame == odom_frame else _lookup_series(tf1_t, tf1_v, t)
        b = _lookup_series(tf2_t, tf2_v, t)
        x, y, yaw = _compose(a, b)
        points.append({"t": float(t), "x": float(x), "y": float(y), "z": 0.0, "yaw": float(yaw)})

    out = {
        "bag": str(bag_path),
        "times_topic": str(args.times_topic),
        "times_topic_frame_id": str(times_topic_frame or ""),
        "tf_topic": str(args.tf_topic),
        "frame_id": output_frame,
        "output_frame": output_frame,
        "odom_frame": odom_frame,
        "base_frame": base_frame,
        "min_dt": float(args.min_dt),
        "max_msgs": int(args.max_msgs),
        "points": points,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] pose points: {len(points)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

