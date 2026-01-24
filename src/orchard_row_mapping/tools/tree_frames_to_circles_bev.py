#!/usr/bin/env python3
"""Per-frame tree clustering -> circles + BEV PNGs.

Input: a directory produced by `segment_bag_to_tree_pcd.py`:
  <in-dir>/frames.csv
  <in-dir>/pcd/tree_000000.pcd ...

Output:
  <out-dir>/circles/circles_000000.json
  <out-dir>/centers_pcd/centers_000000.pcd
  <out-dir>/png/bev_000000.png
  <out-dir>/frames.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class _Circle:
    x: float
    y: float
    z: float
    radius: float


def _load_tree_circles_impl() -> Any:
    impl_path = Path(__file__).resolve().parents[1] / "scripts" / "orchard_tree_circles_node.py"
    if not impl_path.is_file():
        raise RuntimeError(f"Cannot find implementation: {impl_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("orchard_tree_circles_node", str(impl_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses expect the module to be present in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_frames_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"frames.csv not found: {path}")
    out: List[Dict[str, str]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            out.append({str(k): str(v) for k, v in row.items() if k is not None})
    if not out:
        raise RuntimeError(f"frames.csv is empty: {path}")
    return out


def _write_pcd_xyzr(path: Path, xyzr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyzr = np.asarray(xyzr, dtype=np.float32).reshape((-1, 4))
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z radius\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {xyzr.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {xyzr.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        if xyzr.size:
            handle.write(xyzr.astype(np.float32, copy=False).tobytes())


def _hex_to_bgr(value: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    text = (value or "").strip().lstrip("#")
    if len(text) != 6:
        return default
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
        return (b, g, r)
    except Exception:
        return default


def _choose_grid_step(span: float) -> float:
    span = float(max(span, 1.0e-6))
    target = span / 8.0
    candidates = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    return min(candidates, key=lambda s: abs(float(s) - float(target)))


def _render_bev_cv2(
    out_path: Path,
    pts_xy: np.ndarray,
    circles_xyr: np.ndarray,
    *,
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
    margin_px: int,
    bg: Tuple[int, int, int],
    point_color: Tuple[int, int, int],
    circle_color: Tuple[int, int, int],
    draw_grid: bool,
    draw_title: str,
    draw_axes: bool,
    draw_radius: bool,
) -> None:
    import cv2  # type: ignore

    xmin, xmax, ymin, ymax = bounds
    dx = max(1.0e-6, float(xmax) - float(xmin))
    dy = max(1.0e-6, float(ymax) - float(ymin))
    scale = min((float(width - 2 * margin_px) / dx), (float(height - 2 * margin_px) / dy))

    def xy_to_px(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        px = margin_px + (x.astype(np.float64, copy=False) - float(xmin)) * float(scale)
        py = margin_px + (float(ymax) - y.astype(np.float64, copy=False)) * float(scale)
        return px.astype(np.int32), py.astype(np.int32)

    img = np.full((int(height), int(width), 3), bg, dtype=np.uint8)

    plot_x0 = int(margin_px)
    plot_y0 = int(margin_px)
    plot_x1 = int(width) - int(margin_px)
    plot_y1 = int(height) - int(margin_px)

    if draw_grid:
        grid_color = (235, 235, 235)
        axis_color = (80, 80, 80)
        grid_step_x = _choose_grid_step(float(xmax) - float(xmin))
        grid_step_y = _choose_grid_step(float(ymax) - float(ymin))

        def frange_start(vmin: float, step: float) -> float:
            return math.ceil(vmin / step) * step

        x0 = frange_start(float(xmin), grid_step_x)
        x_vals = np.arange(x0, float(xmax) + 0.5 * grid_step_x, grid_step_x, dtype=np.float64)
        for xv in x_vals.tolist():
            px, _ = xy_to_px(np.asarray([xv], dtype=np.float64), np.asarray([ymin], dtype=np.float64))
            xpx = int(px[0])
            if plot_x0 <= xpx <= plot_x1:
                cv2.line(img, (xpx, plot_y0), (xpx, plot_y1), grid_color, 1, lineType=cv2.LINE_AA)

        y0 = frange_start(float(ymin), grid_step_y)
        y_vals = np.arange(y0, float(ymax) + 0.5 * grid_step_y, grid_step_y, dtype=np.float64)
        for yv in y_vals.tolist():
            _, py = xy_to_px(np.asarray([xmin], dtype=np.float64), np.asarray([yv], dtype=np.float64))
            ypx = int(py[0])
            if plot_y0 <= ypx <= plot_y1:
                cv2.line(img, (plot_x0, ypx), (plot_x1, ypx), grid_color, 1, lineType=cv2.LINE_AA)

        cv2.rectangle(img, (plot_x0, plot_y0), (plot_x1, plot_y1), axis_color, 1, lineType=cv2.LINE_AA)

    if pts_xy.size:
        px, py = xy_to_px(pts_xy[:, 0], pts_xy[:, 1])
        inside = (px >= plot_x0) & (px <= plot_x1) & (py >= plot_y0) & (py <= plot_y1)
        px = px[inside]
        py = py[inside]
        img[py, px] = point_color

    if circles_xyr.size:
        cx, cy = xy_to_px(circles_xyr[:, 0], circles_xyr[:, 1])
        for (x_m, y_m, r_m), cx_i, cy_i in zip(circles_xyr.tolist(), cx.tolist(), cy.tolist()):
            if not (plot_x0 <= int(cx_i) <= plot_x1 and plot_y0 <= int(cy_i) <= plot_y1):
                continue
            cv2.circle(img, (int(cx_i), int(cy_i)), 5, circle_color, 2, lineType=cv2.LINE_AA)
            if draw_radius:
                rp = int(round(max(float(r_m), 0.0) * float(scale)))
                if rp > 0:
                    cv2.circle(img, (int(cx_i), int(cy_i)), int(rp), circle_color, 2, lineType=cv2.LINE_AA)

    if draw_title:
        cv2.putText(img, str(draw_title), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2, cv2.LINE_AA)

    if draw_axes:
        axis_color = (80, 80, 80)
        cv2.putText(img, "x [m]", (plot_x1 - 80, plot_y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, axis_color, 2, cv2.LINE_AA)
        cv2.putText(img, "y [m]", (plot_x0 - 55, plot_y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, axis_color, 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def _parse_bounds(text: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--bounds must be 'xmin,xmax,ymin,ymax'")
    xmin, xmax, ymin, ymax = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    return float(xmin), float(xmax), float(ymin), float(ymax)


def _compute_radius(
    pts_xy: np.ndarray,
    center_xy: np.ndarray,
    *,
    mode: str,
    radius_constant: float,
    radius_quantile: float,
    radius_min: float,
    radius_max: float,
) -> float:
    mode = (mode or "constant").strip().lower()
    radius_min = float(radius_min)
    radius_max = float(radius_max)
    if radius_max > 0.0 and radius_max < radius_min:
        radius_min, radius_max = radius_max, radius_min

    if mode in ("constant", "fixed"):
        radius = float(radius_constant)
    else:
        d = np.linalg.norm(pts_xy - center_xy.reshape(1, 2), axis=1)
        if d.size == 0:
            radius = float(radius_constant)
        elif mode in ("quantile", "percentile"):
            q = float(max(0.0, min(radius_quantile, 1.0)))
            radius = float(np.quantile(d, q))
        else:
            radius = float(np.median(d))

    if radius_min > 0.0:
        radius = max(radius, radius_min)
    if radius_max > 0.0:
        radius = min(radius, radius_max)
    return float(radius)


def _reindex_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32).reshape((-1,))
    if labels.size == 0:
        return labels
    uniq = sorted(int(v) for v in set(labels.tolist()) if int(v) >= 0)
    mapping = {old: new for new, old in enumerate(uniq)}
    out = np.full_like(labels, -1)
    for old, new in mapping.items():
        out[labels == int(old)] = int(new)
    out[labels < 0] = -1
    return out


def _filter_small_clusters(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32).reshape((-1,))
    min_cluster_size = int(min_cluster_size)
    if labels.size == 0 or min_cluster_size <= 1:
        return labels
    keep = labels.copy()
    uniq, counts = np.unique(keep[keep >= 0], return_counts=True)
    for cid, cnt in zip(uniq.tolist(), counts.tolist()):
        if int(cnt) < min_cluster_size:
            keep[keep == int(cid)] = -1
    return keep


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


def _update_tf_buffer(buffer: Dict[Tuple[str, str], np.ndarray], tf_msg: Any) -> None:
    for tr in getattr(tf_msg, "transforms", []) or []:
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


def _apply_transform(points_xyz: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz.astype(np.float32, copy=False).reshape((-1, 3))
    pts = points_xyz.astype(np.float64, copy=False).reshape((-1, 3))
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out = pts @ rot.T + trans.reshape((1, 3))
    return out.astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, type=str, help="Folder with frames.csv + pcd/*.pcd (tree frames).")
    parser.add_argument("--out-dir", default="", type=str, help="Output directory (can be Chinese).")
    parser.add_argument("--resume", action="store_true", help="Skip frames whose outputs already exist.")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--accumulate-frames", type=int, default=1)

    parser.add_argument("--z-min", type=float, default=-0.7)
    parser.add_argument("--z-max", type=float, default=0.7)
    parser.add_argument("--x-min", type=float, default=-2.0)
    parser.add_argument("--x-max", type=float, default=25.0)
    parser.add_argument("--y-abs-max", type=float, default=10.0)

    parser.add_argument(
        "--cluster-algo",
        choices=["grid", "dbscan", "euclid"],
        default="grid",
        help="Clustering algorithm: grid=original cell-cluster method; dbscan/euclid=sklearn DBSCAN variants on XY.",
    )
    parser.add_argument("--cell-size", type=float, default=0.12)
    parser.add_argument("--neighbor-range", type=int, default=1)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--max-clusters", type=int, default=0)

    parser.add_argument("--dbscan-eps", type=float, default=0.25)
    parser.add_argument("--dbscan-min-samples", type=int, default=10)
    parser.add_argument("--dbscan-min-cluster-size", type=int, default=40)

    parser.add_argument("--euclid-eps", type=float, default=0.25)
    parser.add_argument("--euclid-min-cluster-size", type=int, default=40)

    parser.add_argument(
        "--bag",
        type=str,
        default="",
        help="If set, use TF from this bag to align frames before accumulation (recommended when vehicle moves).",
    )
    parser.add_argument("--tf-topic", type=str, default="/tf")
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument("--base-frame", type=str, default="base_link_est", help="Frame of the input tree PCDs and output BEV.")
    parser.add_argument("--missing-tf-policy", choices=["skip", "hold", "first"], default="first")

    parser.add_argument("--marker-z", type=float, default=0.0)
    parser.add_argument("--radius-mode", choices=["constant", "median", "quantile"], default="quantile")
    parser.add_argument("--radius-constant", type=float, default=0.15)
    parser.add_argument("--radius-quantile", type=float, default=0.7)
    parser.add_argument("--radius-min", type=float, default=0.08)
    parser.add_argument("--radius-max", type=float, default=1.2)

    parser.add_argument("--no-png", action="store_true", help="Do not write BEV PNGs (only circles json + centers pcd).")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--margin-px", type=int, default=50)
    parser.add_argument("--bounds", type=str, default="", help="xmin,xmax,ymin,ymax (default uses x/y bounds).")
    parser.add_argument("--bg", type=str, default="#ffffff")
    parser.add_argument("--point-color", type=str, default="#b3b3b3")
    parser.add_argument("--circle-color", type=str, default="#2ca02c")
    parser.add_argument("--draw-grid", type=int, default=1)
    parser.add_argument("--draw-axes", type=int, default=1)
    parser.add_argument("--draw-title", type=int, default=0)
    parser.add_argument("--draw-radius", type=int, default=1)

    args = parser.parse_args()

    accumulate_frames = int(max(1, int(args.accumulate_frames)))
    tf_align_requested = bool(str(args.bag).strip())
    tf_align_enabled = tf_align_requested and accumulate_frames > 1

    in_dir = Path(args.in_dir).expanduser().resolve()
    frames_path = in_dir / "frames.csv"
    frames = _read_frames_csv(frames_path)

    ws_dir = Path(__file__).resolve().parents[3]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (ws_dir / "output" / f"每帧聚类圆圈_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    circles_dir = out_dir / "circles"
    centers_dir = out_dir / "centers_pcd"
    png_dir = out_dir / "png"
    circles_dir.mkdir(parents=True, exist_ok=True)
    centers_dir.mkdir(parents=True, exist_ok=True)
    save_png = not bool(args.no_png)
    if save_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    impl = _load_tree_circles_impl()

    bounds = (
        _parse_bounds(str(args.bounds))
        if str(args.bounds).strip()
        else (float(args.x_min), float(args.x_max), -float(args.y_abs_max), float(args.y_abs_max))
    )

    run_meta = {
        "in_dir": str(in_dir),
        "frames_csv": str(frames_path),
        "accumulate_frames": int(accumulate_frames),
        "filter": {"z_min": float(args.z_min), "z_max": float(args.z_max), "x_min": float(args.x_min), "x_max": float(args.x_max), "y_abs_max": float(args.y_abs_max)},
        "cluster": {
            "algo": str(args.cluster_algo),
            "grid": {"cell_size": float(args.cell_size), "neighbor_range": int(args.neighbor_range), "min_points": int(args.min_points), "max_clusters": int(args.max_clusters)},
            "dbscan": {"eps": float(args.dbscan_eps), "min_samples": int(args.dbscan_min_samples), "min_cluster_size": int(args.dbscan_min_cluster_size)},
            "euclid": {"eps": float(args.euclid_eps), "min_cluster_size": int(args.euclid_min_cluster_size), "note": "Implemented via sklearn DBSCAN(min_samples=1) + cluster size filter."},
        },
        "radius": {
            "marker_z": float(args.marker_z),
            "radius_mode": str(args.radius_mode),
            "radius_constant": float(args.radius_constant),
            "radius_quantile": float(args.radius_quantile),
            "radius_min": float(args.radius_min),
            "radius_max": float(args.radius_max),
        },
        "tf_align": {
            "enabled": bool(tf_align_enabled),
            "requested": bool(tf_align_requested),
            "bag": str(Path(str(args.bag)).expanduser().resolve()) if str(args.bag).strip() else "",
            "tf_topic": str(args.tf_topic),
            "map_frame": str(args.map_frame),
            "base_frame": str(args.base_frame),
            "missing_tf_policy": str(args.missing_tf_policy),
            "note": "If enabled: each frame is transformed to map with TF, accumulated in map, then reprojected to current base for clustering/BEV.",
        },
        "render": {
            "save_png": bool(save_png),
            "bounds": [float(v) for v in bounds],
            "width": int(args.width),
            "height": int(args.height),
            "margin_px": int(args.margin_px),
            "bg": str(args.bg),
            "point_color": str(args.point_color),
            "circle_color": str(args.circle_color),
            "draw_grid": bool(int(args.draw_grid)),
            "draw_axes": bool(int(args.draw_axes)),
            "draw_title": bool(int(args.draw_title)),
            "draw_radius": bool(int(args.draw_radius)),
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tf_buffer: Dict[Tuple[str, str], np.ndarray] = {}
    tf_bag: Any = None
    tf_iter: Any = None
    tf_next: Any = None
    last_map_T_base: Optional[np.ndarray] = None
    first_map_T_base: Optional[np.ndarray] = None

    def _tf_setup() -> None:
        nonlocal tf_bag, tf_iter, tf_next
        if not tf_align_enabled:
            return
        try:
            import rosbag  # type: ignore
        except Exception as exc:
            raise RuntimeError("TF-align requires ROS1 python 'rosbag' to be available") from exc

        bag_path = Path(str(args.bag)).expanduser().resolve()
        if not bag_path.is_file():
            raise FileNotFoundError(f"--bag not found: {bag_path}")
        tf_bag = rosbag.Bag(str(bag_path))
        tf_iter = tf_bag.read_messages(topics=[str(args.tf_topic)])
        try:
            tf_next = next(tf_iter)
        except StopIteration:
            tf_next = None

    def _tf_teardown() -> None:
        nonlocal tf_bag
        if tf_bag is not None:
            try:
                tf_bag.close()
            except Exception:
                pass
            tf_bag = None

    def _advance_tf_to(t_sec: float) -> None:
        nonlocal tf_next
        if not tf_align_enabled:
            return
        while tf_next is not None and float(tf_next[2].to_sec()) <= float(t_sec):
            _update_tf_buffer(tf_buffer, tf_next[1])
            try:
                tf_next = next(tf_iter)
            except StopIteration:
                tf_next = None

    def _map_T_base_at(t_sec: float) -> Optional[np.ndarray]:
        nonlocal first_map_T_base, last_map_T_base, tf_next
        if not tf_align_enabled:
            return np.eye(4, dtype=np.float64)

        _advance_tf_to(float(t_sec))
        mat = _lookup_transform(tf_buffer, str(args.map_frame), str(args.base_frame))
        if mat is not None:
            last_map_T_base = mat
            if first_map_T_base is None:
                first_map_T_base = mat
            return mat

        policy = str(args.missing_tf_policy)
        if policy == "hold":
            return last_map_T_base
        if policy == "first":
            if first_map_T_base is not None:
                return first_map_T_base
            while tf_next is not None:
                _update_tf_buffer(tf_buffer, tf_next[1])
                try:
                    tf_next = next(tf_iter)
                except StopIteration:
                    tf_next = None
                mat2 = _lookup_transform(tf_buffer, str(args.map_frame), str(args.base_frame))
                if mat2 is not None:
                    first_map_T_base = mat2
                    last_map_T_base = mat2
                    return mat2
        return None

    _tf_setup()

    csv_out = out_dir / "frames.csv"
    try:
        with csv_out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "t_sec", "points_in", "points_used", "clusters", "circles_json", "centers_pcd", "png"])

            every = max(1, int(args.every))
            max_frames = int(args.max_frames)
            max_frames = max_frames if max_frames > 0 else 0

            # When TF-align is enabled, we accumulate in map-frame for stability, then reproject into current base.
            accum: Deque[np.ndarray] = deque()

            processed = 0
            for i, row in enumerate(frames):
                if i % every != 0:
                    continue
                idx_str = (row.get("index") or row.get("frame") or row.get("idx") or "").strip()
                try:
                    frame_idx = int(idx_str) if idx_str else int(i)
                except Exception:
                    frame_idx = int(i)

                t_sec_text = (row.get("t_sec") or row.get("t") or "").strip()
                t_sec: Optional[float]
                try:
                    t_sec = float(t_sec_text) if t_sec_text else None
                except Exception:
                    t_sec = None

                if tf_align_enabled and t_sec is None:
                    print(f"[WARN] frame {frame_idx}: missing t_sec; cannot TF-align -> skipped")
                    continue

                pcd_path_str = (row.get("pcd_path") or row.get("pcd") or "").strip()
                if not pcd_path_str:
                    # Fall back: guess from standard name.
                    pcd_path = in_dir / "pcd" / f"tree_{frame_idx:06d}.pcd"
                else:
                    pcd_path_candidate = Path(pcd_path_str).expanduser()
                    pcd_path = (
                        pcd_path_candidate.resolve()
                        if pcd_path_candidate.is_absolute()
                        else (in_dir / pcd_path_candidate).resolve()
                    )
                    if not pcd_path.is_file():
                        # If the input folder was moved, frames.csv may still contain old absolute paths.
                        # Recover by matching only the filename under <in-dir>/pcd/.
                        fallback = (in_dir / "pcd" / Path(pcd_path_str).name).resolve()
                        if fallback.is_file():
                            pcd_path = fallback

                out_circles = circles_dir / f"circles_{frame_idx:06d}.json"
                out_centers = centers_dir / f"centers_{frame_idx:06d}.pcd"
                out_png = png_dir / f"bev_{frame_idx:06d}.png" if save_png else Path("")

                def _accum_push(points_base: np.ndarray) -> Optional[np.ndarray]:
                    if accumulate_frames <= 1:
                        return points_base

                    if tf_align_enabled:
                        map_T_base = _map_T_base_at(float(t_sec or 0.0))
                        if map_T_base is None:
                            return None
                        points_map = _apply_transform(points_base, map_T_base)
                        accum.append(points_map)
                        while len(accum) > int(accumulate_frames):
                            accum.popleft()
                        pts_map_accum = np.vstack(list(accum)) if accum else points_map
                        base_T_map = _invert_transform(map_T_base)
                        pts_base_accum = _apply_transform(pts_map_accum, base_T_map)
                        return pts_base_accum

                    accum.append(points_base)
                    while len(accum) > int(accumulate_frames):
                        accum.popleft()
                    return np.vstack(list(accum)) if accum else points_base

                # Resume: if outputs exist, just record them (and keep accumulation if requested).
                if args.resume and out_circles.is_file() and out_centers.is_file() and (not save_png or out_png.is_file()):
                    try:
                        obj = json.loads(out_circles.read_text(encoding="utf-8"))
                        points_in = int(obj.get("points_in", 0))
                        points_used = int(obj.get("points_used", 0))
                        clusters = int(len(obj.get("circles", []) or []))
                    except Exception:
                        points_in = 0
                        points_used = 0
                        clusters = 0

                    # Maintain accumulation window if requested (>1): load this frame's points.
                    if accumulate_frames > 1 and pcd_path.is_file():
                        pts = impl._load_pcd_xyz(pcd_path).astype(np.float32)
                        pts = impl._filter_points(
                            pts,
                            z_min=float(args.z_min),
                            z_max=float(args.z_max),
                            x_min=float(args.x_min),
                            x_max=float(args.x_max),
                            y_abs_max=float(args.y_abs_max),
                        )
                        _accum_push(pts)

                    writer.writerow(
                        [
                            int(frame_idx),
                            t_sec_text,
                            int(points_in),
                            int(points_used),
                            int(clusters),
                            str(out_circles),
                            str(out_centers),
                            str(out_png) if save_png else "",
                        ]
                    )
                    processed += 1
                    if processed % 200 == 0:
                        print(f"[OK] recorded {processed} existing frames")
                    if max_frames > 0 and processed >= max_frames:
                        break
                    continue

                if not pcd_path.is_file():
                    continue

                pts = impl._load_pcd_xyz(pcd_path).astype(np.float32)
                points_in = int(pts.shape[0])

                pts = impl._filter_points(
                    pts,
                    z_min=float(args.z_min),
                    z_max=float(args.z_max),
                    x_min=float(args.x_min),
                    x_max=float(args.x_max),
                    y_abs_max=float(args.y_abs_max),
                )

                pts_used = _accum_push(pts)
                if pts_used is None:
                    print(f"[WARN] frame {frame_idx}: missing TF ({args.map_frame}->{args.base_frame}); skipped")
                    continue
                pts_used = impl._filter_points(
                    pts_used,
                    z_min=float(args.z_min),
                    z_max=float(args.z_max),
                    x_min=float(args.x_min),
                    x_max=float(args.x_max),
                    y_abs_max=float(args.y_abs_max),
                )
                points_used = int(pts_used.shape[0])

                algo = str(args.cluster_algo).strip().lower()
                if algo == "grid":
                    circles, _labels = impl._tree_circles_and_labels_from_cell_clusters(
                        points_xyz=pts_used,
                        cell_size=float(args.cell_size),
                        neighbor_range=int(args.neighbor_range),
                        min_points=int(args.min_points),
                        max_clusters=int(args.max_clusters),
                        marker_z=float(args.marker_z),
                        radius_mode=str(args.radius_mode),
                        radius_constant=float(args.radius_constant),
                        radius_quantile=float(args.radius_quantile),
                        radius_min=float(args.radius_min),
                        radius_max=float(args.radius_max),
                    )
                    circles_out: List[Dict[str, float]] = [asdict(c) for c in circles]
                else:
                    try:
                        from sklearn.cluster import DBSCAN  # type: ignore
                    except Exception as exc:
                        raise RuntimeError("cluster-algo=dbscan/euclid requires sklearn to be available") from exc

                    xy = pts_used[:, :2].astype(np.float32, copy=False) if pts_used.size else np.zeros((0, 2), dtype=np.float32)
                    if xy.shape[0] == 0:
                        labels = np.empty((0,), dtype=np.int32)
                    elif algo == "dbscan":
                        labels = DBSCAN(
                            eps=float(args.dbscan_eps),
                            min_samples=int(args.dbscan_min_samples),
                            n_jobs=-1,
                        ).fit_predict(xy)
                        labels = _filter_small_clusters(labels, int(args.dbscan_min_cluster_size))
                        labels = _reindex_labels(labels)
                    elif algo == "euclid":
                        labels = DBSCAN(eps=float(args.euclid_eps), min_samples=1, n_jobs=-1).fit_predict(xy)
                        labels = _filter_small_clusters(labels, int(args.euclid_min_cluster_size))
                        labels = _reindex_labels(labels)
                    else:
                        raise ValueError(f"Unsupported --cluster-algo: {algo}")

                    uniq = sorted(int(v) for v in set(labels.tolist()) if int(v) >= 0)
                    circles2: List[_Circle] = []
                    for cid in uniq:
                        pts_c = pts_used[labels == int(cid)]
                        if pts_c.size == 0:
                            continue
                        center_xy = np.median(pts_c[:, :2], axis=0)
                        z = float(args.marker_z) if float(args.marker_z) != 0.0 else float(np.median(pts_c[:, 2]))
                        radius = _compute_radius(
                            pts_xy=pts_c[:, :2],
                            center_xy=center_xy,
                            mode=str(args.radius_mode),
                            radius_constant=float(args.radius_constant),
                            radius_quantile=float(args.radius_quantile),
                            radius_min=float(args.radius_min),
                            radius_max=float(args.radius_max),
                        )
                        circles2.append(_Circle(x=float(center_xy[0]), y=float(center_xy[1]), z=float(z), radius=float(radius)))
                    circles2.sort(key=lambda c: (c.y, c.x))
                    circles_out = [asdict(c) for c in circles2]

                clusters = int(len(circles_out))

                out_obj = {
                    "frame_index": int(frame_idx),
                    "t_sec": float(t_sec) if t_sec is not None else None,
                    "pcd": str(pcd_path),
                    "points_in": int(points_in),
                    "points_used": int(points_used),
                    "cluster_algo": str(args.cluster_algo),
                    "circles": circles_out,
                }
                out_circles.parent.mkdir(parents=True, exist_ok=True)
                out_circles.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

                if circles_out:
                    centers = np.array(
                        [[float(c["x"]), float(c["y"]), float(c["z"]), float(c.get("radius", 0.0))] for c in circles_out],
                        dtype=np.float32,
                    )
                else:
                    centers = np.zeros((0, 4), dtype=np.float32)
                _write_pcd_xyzr(out_centers, centers)

                if save_png:
                    pts_xy = pts_used[:, :2].astype(np.float32) if pts_used.size else np.zeros((0, 2), dtype=np.float32)
                    circles_xyr = centers[:, :3].copy() if centers.size else np.zeros((0, 3), dtype=np.float32)
                    if circles_xyr.size:
                        circles_xyr[:, 2] = centers[:, 3]
                    _render_bev_cv2(
                        out_png,
                        pts_xy,
                        circles_xyr,
                        bounds=bounds,
                        width=int(args.width),
                        height=int(args.height),
                        margin_px=int(args.margin_px),
                        bg=_hex_to_bgr(str(args.bg), (255, 255, 255)),
                        point_color=_hex_to_bgr(str(args.point_color), (180, 180, 180)),
                        circle_color=_hex_to_bgr(str(args.circle_color), (44, 160, 44)),
                        draw_grid=bool(int(args.draw_grid)),
                        draw_title=f"frame {frame_idx}" if bool(int(args.draw_title)) else "",
                        draw_axes=bool(int(args.draw_axes)),
                        draw_radius=bool(int(args.draw_radius)),
                    )

                writer.writerow(
                    [
                        int(frame_idx),
                        t_sec_text,
                        int(points_in),
                        int(points_used),
                        int(clusters),
                        str(out_circles),
                        str(out_centers),
                        str(out_png) if save_png else "",
                    ]
                )

                processed += 1
                if processed % 50 == 0:
                    print(f"[OK] processed {processed} frames")
                if max_frames > 0 and processed >= max_frames:
                    break
    finally:
        _tf_teardown()

    print(f"[OK] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
