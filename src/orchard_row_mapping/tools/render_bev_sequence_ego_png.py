#!/usr/bin/env python3
"""Render ego-centric (vehicle-heading) BEV PNG sequence from a rosbag.

This is meant for quick visualization with a fixed ego-frame window:
  - Ego frame defaults to `base_link_est` (falls back to the cloud frame).
  - Points are transformed into map, optionally accumulated, then re-projected
    into the *current* ego frame so the view follows the vehicle heading.

Image coordinates:
  - forward (+X in ego) is up
  - left    (+Y in ego) is left
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rosbag


_POINTFIELD_DATATYPE_TO_DTYPE: Dict[int, np.dtype] = {
    1: np.dtype("int8"),
    2: np.dtype("uint8"),
    3: np.dtype("int16"),
    4: np.dtype("uint16"),
    5: np.dtype("int32"),
    6: np.dtype("uint32"),
    7: np.dtype("float32"),
    8: np.dtype("float64"),
}


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


def _update_tf_buffer(buffer: Dict[Tuple[str, str], np.ndarray], msg: Any) -> None:
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


def _build_struct_dtype(msg: Any, field_names: Sequence[str]) -> np.dtype:
    fields_by_name = {f.name: f for f in msg.fields}
    missing = [name for name in field_names if name not in fields_by_name]
    if missing:
        raise ValueError(f"PointCloud2 missing fields: {missing}. Available: {sorted(fields_by_name)}")

    endian = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
    names: List[str] = []
    formats: List[np.dtype] = []
    offsets: List[int] = []
    for name in field_names:
        field = fields_by_name[name]
        base = _POINTFIELD_DATATYPE_TO_DTYPE.get(int(field.datatype))
        if base is None:
            raise ValueError(f"Unsupported PointField datatype={field.datatype} for field '{name}'")
        if int(field.count) != 1:
            raise ValueError(f"Unsupported PointField count={field.count} for field '{name}' (only count=1 supported)")
        names.append(name)
        formats.append(base.newbyteorder(endian))
        offsets.append(int(field.offset))
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(msg.point_step)})


def _cloud_xyzi(msg: Any, *, skip_nans: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n_points = int(msg.width) * int(msg.height)
    if n_points <= 0 or not msg.data:
        return np.empty((0, 3), dtype=np.float32), None

    available = {f.name for f in msg.fields}
    fields: List[str] = ["x", "y", "z"]
    has_intensity = "intensity" in available
    if has_intensity:
        fields.append("intensity")

    dtype = _build_struct_dtype(msg, fields)
    raw = np.frombuffer(msg.data, dtype=dtype, count=n_points)
    xyz = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float32, copy=False)
    intensity = raw["intensity"].astype(np.float32, copy=False) if has_intensity else None

    if skip_nans and xyz.size:
        mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[mask]
        if intensity is not None:
            intensity = intensity[mask]
    return xyz, intensity


def _hex_to_rgb(value: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    text = (value or "").strip().lstrip("#")
    if len(text) != 6:
        return default
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except Exception:
        return default


def _render_density(
    px: np.ndarray,
    py: np.ndarray,
    width: int,
    height: int,
    bg_rgb: Tuple[int, int, int],
) -> np.ndarray:
    acc = np.zeros((int(height), int(width)), dtype=np.uint32)
    if px.size:
        np.add.at(acc, (py, px), 1)
    img = np.log1p(acc.astype(np.float32))
    m = float(np.max(img)) if img.size else 0.0
    out = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    out[:, :, 0] = np.uint8(bg_rgb[0])
    out[:, :, 1] = np.uint8(bg_rgb[1])
    out[:, :, 2] = np.uint8(bg_rgb[2])
    if not (m > 0.0):
        return out
    gray = (img / m * 255.0).astype(np.uint8)
    out[:, :, 0] = np.maximum(out[:, :, 0], gray)
    out[:, :, 1] = np.maximum(out[:, :, 1], gray)
    out[:, :, 2] = np.maximum(out[:, :, 2], gray)
    return out


def _render_intensity(
    px: np.ndarray,
    py: np.ndarray,
    intensity: np.ndarray,
    width: int,
    height: int,
    bg_rgb: Tuple[int, int, int],
    percentile: float,
) -> np.ndarray:
    count = np.zeros((int(height), int(width)), dtype=np.uint32)
    summ = np.zeros((int(height), int(width)), dtype=np.float32)
    if px.size:
        np.add.at(count, (py, px), 1)
        np.add.at(summ, (py, px), intensity.astype(np.float32, copy=False))

    out = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    out[:, :, 0] = np.uint8(bg_rgb[0])
    out[:, :, 1] = np.uint8(bg_rgb[1])
    out[:, :, 2] = np.uint8(bg_rgb[2])

    mask = count > 0
    if not np.any(mask):
        return out

    mean = np.zeros((int(height), int(width)), dtype=np.float32)
    mean[mask] = summ[mask] / count[mask].astype(np.float32)
    vals = mean[mask]
    p = float(max(0.0, min(percentile, 49.9)))
    if p > 0.0:
        vmin, vmax = np.percentile(vals.astype(np.float64, copy=False), [p, 100.0 - p]).tolist()
    else:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
    span = max(1.0e-6, float(vmax) - float(vmin))
    gray = np.clip((mean - float(vmin)) / span, 0.0, 1.0)
    u8 = (gray * 255.0).astype(np.uint8)
    out[:, :, 0] = np.maximum(out[:, :, 0], u8)
    out[:, :, 1] = np.maximum(out[:, :, 1], u8)
    out[:, :, 2] = np.maximum(out[:, :, 2], u8)
    return out


def _project_ego_xy_to_pixels(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dx = max(1.0e-6, float(x_max) - float(x_min))
    dy = max(1.0e-6, float(y_max) - float(y_min))

    # Ego BEV:
    #   +x forward -> up (smaller pixel y)
    #   +y left    -> left (smaller pixel x)
    px = ((float(y_max) - y.astype(np.float64, copy=False)) / dy * float(int(width) - 1)).astype(np.int32, copy=False)
    py = ((float(x_max) - x.astype(np.float64, copy=False)) / dx * float(int(height) - 1)).astype(np.int32, copy=False)

    px = np.clip(px, 0, int(width) - 1)
    py = np.clip(py, 0, int(height) - 1)
    return px, py


def _sample_points(points: np.ndarray, intensity: Optional[np.ndarray], max_points: int, seed: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, intensity
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(points.shape[0], int(max_points), replace=False)
    pts = points[idx]
    if intensity is None:
        return pts, None
    return pts, intensity[idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=str)
    parser.add_argument("--points-topic", type=str, default="/liorl/deskew/cloud_deskewed")
    parser.add_argument("--tf-topic", type=str, default="/tf")
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument("--base-frame", type=str, default="base_link_est", help="Ego frame (vehicle heading).")
    parser.add_argument("--source-frame", type=str, default="lidar", help="Fallback if PointCloud2 header.frame_id is empty.")

    parser.add_argument("--out-dir", type=str, default="", help="Output directory (can be Chinese).")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth pointcloud message.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)

    parser.add_argument("--accumulate-frames", type=int, default=1, help="Accumulate last N frames (in map, reproject to current ego).")
    parser.add_argument("--skip-nans", action="store_true")
    parser.add_argument("--keep-nans", action="store_true")

    parser.add_argument("--z-min", type=float, default=-1.0e9, help="Filter on map Z before ego reprojection.")
    parser.add_argument("--z-max", type=float, default=1.0e9, help="Filter on map Z before ego reprojection.")

    parser.add_argument("--x-min", type=float, default=-2.0, help="Ego window X (forward).")
    parser.add_argument("--x-max", type=float, default=25.0)
    parser.add_argument("--y-min", type=float, default=-10.0, help="Ego window Y (left).")
    parser.add_argument("--y-max", type=float, default=10.0)

    parser.add_argument("--mode", type=str, default="intensity", help="density/intensity")
    parser.add_argument("--intensity-percentile", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=0, help="Random-sample points before rendering (0 disables).")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument("--bg", type=str, default="#000000")

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"bag not found: {bag_path}")

    ws_dir = Path(__file__).resolve().parents[3]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (ws_dir / "output" / f"BEV_车体朝向_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    bg_rgb = _hex_to_rgb(str(args.bg), (0, 0, 0))
    skip_nans = bool(args.skip_nans) and not bool(args.keep_nans)

    with rosbag.Bag(str(bag_path)) as bag:
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())

    start_time = bag_start + float(args.start_offset)
    end_time = bag_end if float(args.duration) <= 0.0 else start_time + float(args.duration)

    tf_buffer: Dict[Tuple[str, str], np.ndarray] = {}
    accum_map: Deque[np.ndarray] = deque()

    meta = {
        "bag": str(bag_path),
        "points_topic": str(args.points_topic),
        "tf_topic": str(args.tf_topic),
        "map_frame": str(args.map_frame),
        "base_frame": str(args.base_frame),
        "source_frame_fallback": str(args.source_frame),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "every": int(args.every),
        "accumulate_frames": int(args.accumulate_frames),
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "ego_window": {"x_min": float(args.x_min), "x_max": float(args.x_max), "y_min": float(args.y_min), "y_max": float(args.y_max)},
        "render": {"mode": str(args.mode), "width": int(args.width), "height": int(args.height), "bg": str(args.bg)},
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "frames.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "t_sec", "points", "ego_frame_used", "png_path"])

        processed = 0
        msg_idx = 0
        with rosbag.Bag(str(bag_path)) as bag:
            for topic, msg, t in bag.read_messages(topics=[str(args.tf_topic), str(args.points_topic)]):
                t_sec = float(t.to_sec())
                if t_sec < start_time:
                    if topic == str(args.tf_topic):
                        _update_tf_buffer(tf_buffer, msg)
                    continue
                if t_sec > end_time:
                    break

                if topic == str(args.tf_topic):
                    _update_tf_buffer(tf_buffer, msg)
                    continue

                msg_idx += 1
                if int(args.every) > 1 and (msg_idx - 1) % int(args.every) != 0:
                    continue

                source_frame = str(getattr(msg.header, "frame_id", "")) or str(args.source_frame)
                map_T_src = _lookup_transform(tf_buffer, str(args.map_frame), source_frame)
                if map_T_src is None:
                    continue

                xyz_src, intensity = _cloud_xyzi(msg, skip_nans=skip_nans)
                if xyz_src.size == 0:
                    continue

                pts_h = np.hstack([xyz_src.astype(np.float64), np.ones((xyz_src.shape[0], 1), dtype=np.float64)])
                pts_map = (map_T_src @ pts_h.T).T[:, :3].astype(np.float32)
                if pts_map.size == 0:
                    continue

                z_mask = (pts_map[:, 2] >= float(args.z_min)) & (pts_map[:, 2] <= float(args.z_max))
                if not np.any(z_mask):
                    continue
                pts_map = pts_map[z_mask]
                if intensity is not None:
                    intensity = intensity[z_mask]

                accum_map.append(pts_map)
                while len(accum_map) > int(max(1, args.accumulate_frames)):
                    accum_map.popleft()
                pts_map_accum = np.vstack(list(accum_map)) if accum_map else pts_map

                map_T_base = _lookup_transform(tf_buffer, str(args.map_frame), str(args.base_frame))
                ego_frame_used = str(args.base_frame)
                if map_T_base is None:
                    map_T_base = map_T_src
                    ego_frame_used = source_frame
                base_T_map = _invert_transform(map_T_base)

                pts_acc_h = np.hstack([pts_map_accum.astype(np.float64), np.ones((pts_map_accum.shape[0], 1), dtype=np.float64)])
                pts_ego = (base_T_map @ pts_acc_h.T).T[:, :3].astype(np.float32)
                if pts_ego.shape[0] < 5:
                    continue

                mask_xy = (
                    (pts_ego[:, 0] >= float(args.x_min))
                    & (pts_ego[:, 0] <= float(args.x_max))
                    & (pts_ego[:, 1] >= float(args.y_min))
                    & (pts_ego[:, 1] <= float(args.y_max))
                )
                if not np.any(mask_xy):
                    continue
                pts_ego = pts_ego[mask_xy]

                intensity_ego: Optional[np.ndarray] = None
                if intensity is not None and int(args.accumulate_frames) <= 1:
                    # Intensity is only tracked for the current frame (no accumulation).
                    intensity_ego = intensity[mask_xy]

                pts_ego, intensity_ego = _sample_points(pts_ego, intensity_ego, int(args.max_points), seed=processed)

                px, py = _project_ego_xy_to_pixels(
                    pts_ego[:, 0],
                    pts_ego[:, 1],
                    x_min=float(args.x_min),
                    x_max=float(args.x_max),
                    y_min=float(args.y_min),
                    y_max=float(args.y_max),
                    width=int(args.width),
                    height=int(args.height),
                )

                mode = str(args.mode).strip().lower()
                if mode == "intensity" and intensity_ego is not None:
                    img = _render_intensity(
                        px,
                        py,
                        intensity_ego,
                        width=int(args.width),
                        height=int(args.height),
                        bg_rgb=bg_rgb,
                        percentile=float(args.intensity_percentile),
                    )
                else:
                    img = _render_density(px, py, width=int(args.width), height=int(args.height), bg_rgb=bg_rgb)

                out_png = png_dir / f"bev_{processed:06d}.png"
                # Use cv2 if available for speed, else fall back to PIL.
                try:
                    import cv2  # type: ignore

                    cv2.imwrite(str(out_png), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                except Exception:
                    from PIL import Image

                    Image.fromarray(img, mode="RGB").save(out_png)

                writer.writerow([processed, f"{t_sec:.6f}", int(pts_ego.shape[0]), ego_frame_used, str(out_png)])

                processed += 1
                if int(args.max_frames) > 0 and processed >= int(args.max_frames):
                    break
                if processed % 50 == 0:
                    print(f"[OK] rendered {processed} frames")

    print(f"[OK] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
