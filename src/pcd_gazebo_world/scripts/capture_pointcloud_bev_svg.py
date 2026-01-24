#!/usr/bin/env python3
"""Capture one PointCloud2 message and write a simple BEV SVG (no GUI required).

This is useful in headless / sandboxed environments where Gazebo camera rendering is unavailable.
"""

from __future__ import annotations

import argparse
import math
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from sensor_msgs.msg import PointCloud2


_DTYPE_TO_NP = {
    7: np.float32,  # sensor_msgs/PointField.FLOAT32
    8: np.float64,  # sensor_msgs/PointField.FLOAT64
}


def _pointcloud2_sample_xyz(msg: PointCloud2, max_points: int, seed: int) -> np.ndarray:
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 0) or 0)
    n = int(width) * int(height)
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)

    point_step = int(getattr(msg, "point_step", 0) or 0)
    if point_step <= 0:
        return np.empty((0, 3), dtype=np.float32)

    field_by_name: Dict[str, Any] = {str(f.name): f for f in getattr(msg, "fields", [])}
    required = ("x", "y", "z")
    if any(name not in field_by_name for name in required):
        raise RuntimeError(f"PointCloud2 missing xyz fields: {sorted(field_by_name.keys())}")

    fields = []
    for name in required:
        f = field_by_name[name]
        dt = _DTYPE_TO_NP.get(int(f.datatype), None)
        if dt is None:
            raise RuntimeError(f"Unsupported PointField datatype for {name}: {int(f.datatype)}")
        fields.append((name, int(f.offset), dt))

    endian = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
    formats = []
    offsets = []
    for _name, off, dt in fields:
        fmt = np.dtype(dt).newbyteorder(endian)
        formats.append(fmt)
        offsets.append(int(off))

    dtype = np.dtype(
        {
            "names": [f[0] for f in fields],
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(point_step),
        }
    )
    arr = np.frombuffer(getattr(msg, "data"), dtype=dtype, count=n)

    rng = np.random.default_rng(int(seed))
    if int(max_points) > 0 and n > int(max_points):
        idx = rng.integers(0, n, size=int(max_points), endpoint=False)
    else:
        idx = slice(None)

    x = np.asarray(arr["x"][idx], dtype=np.float64).reshape(-1)
    y = np.asarray(arr["y"][idx], dtype=np.float64).reshape(-1)
    z = np.asarray(arr["z"][idx], dtype=np.float64).reshape(-1)
    xyz = np.stack([x, y, z], axis=1)
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    return xyz.astype(np.float32)


def _svg_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _compute_bounds(xy: np.ndarray) -> Tuple[float, float, float, float]:
    x = xy[:, 0]
    y = xy[:, 1]
    x0 = float(np.percentile(x, 1))
    x1 = float(np.percentile(x, 99))
    y0 = float(np.percentile(y, 1))
    y1 = float(np.percentile(y, 99))
    if not math.isfinite(x0) or not math.isfinite(x1) or x1 <= x0:
        x0 = float(np.min(x))
        x1 = float(np.max(x))
    if not math.isfinite(y0) or not math.isfinite(y1) or y1 <= y0:
        y0 = float(np.min(y))
        y1 = float(np.max(y))
    pad = 0.5
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def _write_svg(out_path: Path, xy: np.ndarray, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1200
    height = 800
    margin = 50

    x0, x1, y0, y1 = _compute_bounds(xy)
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    dx = max(1.0e-6, x1 - x0)
    dy = max(1.0e-6, y1 - y0)
    scale = min(plot_w / dx, plot_h / dy)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        px = margin + (x - x0) * scale
        py = height - margin - (y - y0) * scale
        return float(px), float(py)

    pts = [to_px(float(p[0]), float(p[1])) for p in xy]
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append(f'  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    lines.append(f'  <text x="{margin}" y="{margin-18}" font-family="monospace" font-size="16" fill="#111">{_svg_escape(title)}</text>')
    lines.append(f'  <rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbb" stroke-width="1"/>')
    for px, py in pts:
        lines.append(f'  <circle cx="{px:.2f}" cy="{py:.2f}" r="0.70" fill="#1f77b4" opacity="0.60"/>')
    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_out = ws_dir / "src" / "pcd_gazebo_world" / "maps" / "runs" / "gazebo_velodyne_bev.svg"

    parser = argparse.ArgumentParser(description="Capture one PointCloud2 and write a BEV SVG")
    parser.add_argument("--topic", type=str, default="/velodyne_points", help="PointCloud2 topic")
    parser.add_argument("--out", type=str, default=str(default_out), help="输出 SVG 路径")
    parser.add_argument("--timeout", type=float, default=10.0, help="等待超时（秒）")
    parser.add_argument("--max-points", type=int, default=35000, help="最多渲染点数（0=全画，可能很大）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z-min", type=float, default=-1.0, help="只保留 z>=z_min")
    parser.add_argument("--z-max", type=float, default=3.0, help="只保留 z<=z_max")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()

    rospy.init_node("capture_pointcloud_bev_svg", anonymous=True, disable_signals=True)
    try:
        msg: PointCloud2 = rospy.wait_for_message(str(args.topic), PointCloud2, timeout=float(args.timeout))
    except rospy.ROSException as exc:
        raise SystemExit(f"Timeout waiting for {args.topic}: {exc}") from exc

    xyz = _pointcloud2_sample_xyz(msg, max_points=int(args.max_points), seed=int(args.seed))
    if xyz.size == 0:
        raise SystemExit(f"No points captured from topic: {args.topic}")

    z = xyz[:, 2]
    mask = (z >= float(args.z_min)) & (z <= float(args.z_max))
    xyz = xyz[mask]
    if xyz.size == 0:
        raise SystemExit("No points left after z filtering; adjust --z-min/--z-max.")

    xy = xyz[:, :2]
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 0) or 0)
    total_pts = int(width) * int(height)
    title = (
        f"topic={args.topic} frame={msg.header.frame_id} size={width}x{height} "
        f"points={total_pts} rendered={int(xy.shape[0])}"
    )
    _write_svg(out_path, xy=xy, title=title)
    print(f"[OK] wrote: {out_path}")
    print(f"[OK] {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
