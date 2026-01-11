#!/usr/bin/env python3
"""Export a downsampled map-frame PCD from a bag pointcloud topic.

This is meant for manual prior drawing. It transforms /points_raw into map
using /tf, filters by z, samples per frame, and writes a binary PCD.
Optionally emits a bev_meta JSON for draw_manual_prior_cv2.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rosbag
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, zz = qx * x2, qy * y2, qz * z2
    xy, xz, yz = qx * y2, qx * z2, qy * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _transform_matrix(translation: Tuple[float, float, float], rotation: Tuple[float, float, float, float]) -> np.ndarray:
    tx, ty, tz = translation
    qx, qy, qz, qw = rotation
    rot = _quat_to_rot(qx, qy, qz, qw)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return mat


def _invert_transform(mat: np.ndarray) -> np.ndarray:
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def _update_tf_buffer(buffer: Dict[Tuple[str, str], np.ndarray], msg: TFMessage) -> None:
    for tr in msg.transforms:
        parent = str(tr.header.frame_id)
        child = str(tr.child_frame_id)
        t = tr.transform.translation
        q = tr.transform.rotation
        buffer[(parent, child)] = _transform_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def _lookup_transform(buffer: Dict[Tuple[str, str], np.ndarray], target: str, source: str) -> Optional[np.ndarray]:
    if target == source:
        return np.eye(4, dtype=np.float64)
    adjacency: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for (parent, child), mat in buffer.items():
        adjacency.setdefault(child, []).append((parent, mat))
        adjacency.setdefault(parent, []).append((child, _invert_transform(mat)))
    visited = set()
    queue: List[Tuple[str, np.ndarray]] = [(source, np.eye(4, dtype=np.float64))]
    visited.add(source)
    while queue:
        frame, mat = queue.pop(0)
        if frame == target:
            return mat
        for nxt, edge in adjacency.get(frame, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, edge @ mat))
    return None


def _points_from_cloud(msg: PointCloud2) -> np.ndarray:
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(points.shape[0], int(max_points), replace=False)
    return points[idx]


def _write_pcd_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.astype(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {pts.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {pts.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(pts.tobytes())


def _compute_bounds(points_xy: np.ndarray, pad: float) -> Tuple[float, float, float, float]:
    if points_xy.size == 0:
        return -10.0, 10.0, -10.0, 10.0
    xmin = float(np.min(points_xy[:, 0]))
    xmax = float(np.max(points_xy[:, 0]))
    ymin = float(np.min(points_xy[:, 1]))
    ymax = float(np.max(points_xy[:, 1]))
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=str)
    parser.add_argument("--points-topic", default="/points_raw", type=str)
    parser.add_argument("--tf-topic", default="/tf", type=str)
    parser.add_argument("--map-frame", default="map", type=str)
    parser.add_argument("--source-frame", default="lidar_link", type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--out-meta", default="", type=str)

    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-points-per-frame", type=int, default=1200)
    parser.add_argument("--z-min", type=float, default=0.2)
    parser.add_argument("--z-max", type=float, default=1.8)
    parser.add_argument("--bounds", type=str, default="")
    parser.add_argument("--pad", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--missing-tf-policy", choices=["skip", "hold", "first"], default="first")

    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--margin", type=int, default=40)
    parser.add_argument("--meta-max-points", type=int, default=200000)

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_meta = Path(args.out_meta).expanduser().resolve() if str(args.out_meta).strip() else None

    tf_buffer: Dict[Tuple[str, str], np.ndarray] = {}
    first_map_T_by_frame: Dict[str, np.ndarray] = {}
    last_map_T_by_frame: Dict[str, np.ndarray] = {}
    prefill_time: Optional[float] = None

    with rosbag.Bag(str(bag_path)) as bag:
        start_time = bag.get_start_time() + float(args.start_offset)
        end_time = bag.get_end_time() if float(args.duration) <= 0.0 else start_time + float(args.duration)

    if str(args.missing_tf_policy) == "first":
        with rosbag.Bag(str(bag_path)) as bag:
            source_frame = str(args.source_frame)
            for _topic, _pc_msg, _pc_t in bag.read_messages(topics=[args.points_topic]):
                if float(_pc_t.to_sec()) < float(start_time):
                    continue
                source_frame = str(getattr(_pc_msg.header, "frame_id", "")) or str(args.source_frame)
                break

            for _topic, _tf_msg, _tf_t in bag.read_messages(topics=[args.tf_topic]):
                _t_sec = float(_tf_t.to_sec())
                if _t_sec > float(end_time):
                    break
                _update_tf_buffer(tf_buffer, _tf_msg)
                if _t_sec < float(start_time):
                    continue
                map_T = _lookup_transform(tf_buffer, str(args.map_frame), str(source_frame))
                if map_T is None:
                    continue
                first_map_T_by_frame[str(source_frame)] = map_T
                last_map_T_by_frame[str(source_frame)] = map_T
                prefill_time = float(_t_sec)
                break
        if prefill_time is None:
            print("[WARN] missing-tf-policy=first but TF prefill failed; early frames may be skipped.", file=sys.stderr)
        else:
            print(f"[INFO] Prefilled TF at t={prefill_time:.3f}s (source_frame={source_frame!r})", file=sys.stderr)

    bounds = None
    if str(args.bounds).strip():
        parts = [float(x) for x in str(args.bounds).split(",")]
        if len(parts) != 4:
            raise ValueError("bounds must be xmin,xmax,ymin,ymax")
        bounds = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))

    all_points: List[np.ndarray] = []
    frame_count = 0
    next_time: Optional[float] = None

    with rosbag.Bag(str(bag_path)) as bag:
        for topic, msg, t in bag.read_messages(topics=[args.tf_topic, args.points_topic]):
            t_sec = float(t.to_sec())
            if t_sec < float(start_time):
                if topic == args.tf_topic:
                    _update_tf_buffer(tf_buffer, msg)
                continue
            if t_sec > float(end_time):
                break

            if topic == args.tf_topic:
                _update_tf_buffer(tf_buffer, msg)
                continue

            if next_time is None:
                next_time = t_sec
            if t_sec + 1.0e-6 < float(next_time):
                continue
            next_time = float(next_time) + 1.0 / float(args.sample_rate)

            source_frame = str(getattr(msg.header, "frame_id", "")) or str(args.source_frame)
            map_T = _lookup_transform(tf_buffer, str(args.map_frame), source_frame)
            if map_T is None:
                if str(args.missing_tf_policy) == "hold":
                    map_T = last_map_T_by_frame.get(source_frame)
                elif str(args.missing_tf_policy) == "first":
                    map_T = first_map_T_by_frame.get(source_frame)
                if map_T is None:
                    continue
            else:
                last_map_T_by_frame[source_frame] = map_T
                first_map_T_by_frame.setdefault(source_frame, map_T)

            pts = _points_from_cloud(msg)
            if pts.size == 0:
                continue

            pts_h = np.hstack([pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)])
            pts_map = (map_T @ pts_h.T).T[:, :3].astype(np.float32)

            z_mask = (pts_map[:, 2] >= float(args.z_min)) & (pts_map[:, 2] <= float(args.z_max))
            pts_map = pts_map[z_mask]
            if pts_map.shape[0] == 0:
                continue

            if bounds is not None:
                xmin, xmax, ymin, ymax = bounds
                xy = pts_map[:, :2]
                mask = (
                    (xy[:, 0] >= xmin)
                    & (xy[:, 0] <= xmax)
                    & (xy[:, 1] >= ymin)
                    & (xy[:, 1] <= ymax)
                )
                pts_map = pts_map[mask]
                if pts_map.shape[0] == 0:
                    continue

            pts_map = _sample_points(pts_map, int(args.max_points_per_frame), int(args.seed) + frame_count)
            all_points.append(pts_map)
            frame_count += 1

    if not all_points:
        raise RuntimeError("No points collected; check TF/filters.")

    pts_all = np.vstack(all_points).astype(np.float32)
    _write_pcd_xyz(out_path, pts_all)
    print(f"[OK] Wrote PCD: {out_path}")

    if out_meta is not None:
        bounds = _compute_bounds(pts_all[:, :2].astype(np.float32), float(args.pad))
        meta = {
            "bounds": [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])],
            "width": int(args.width),
            "height": int(args.height),
            "margin": int(args.margin),
            "pcd": str(out_path),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "max_points": int(args.meta_max_points),
        }
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] Wrote BEV meta: {out_meta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
