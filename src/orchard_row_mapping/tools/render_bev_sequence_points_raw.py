#!/usr/bin/env python3
"""Render per-frame BEV SVGs from /points_raw using /tf transforms."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import rosbag
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage


def _load_tree_circles_impl() -> Any:
    impl_path = Path(__file__).resolve().parents[1] / "scripts" / "orchard_tree_circles_node.py"
    if not impl_path.is_file():
        raise RuntimeError(f"Cannot find implementation: {impl_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("orchard_tree_circles_node", str(impl_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_indexed_float_map(text: str) -> Dict[int, float]:
    text = (text or "").strip()
    if not text:
        return {}
    decoded = json.loads(text)
    out: Dict[int, float] = {}
    for k, v in decoded.items():
        out[int(k)] = float(v)
    return out


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
        mat = _transform_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
        buffer[(parent, child)] = mat


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


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return values.astype(np.float32)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _find_row_centers_histogram(
    v: np.ndarray,
    bin_size: float,
    smooth_window: int,
    peak_min_fraction: float,
    min_center_separation: float,
) -> np.ndarray:
    v = v.astype(np.float32)
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        return np.empty((0,), dtype=np.float32)

    bin_size = float(bin_size)
    edges = np.arange(v_min - bin_size, v_max + bin_size * 2.0, bin_size, dtype=np.float32)
    if edges.size < 4:
        return np.empty((0,), dtype=np.float32)
    hist, _ = np.histogram(v, bins=edges)
    smooth = _moving_average(hist.astype(np.float32), int(smooth_window))

    peak_min_fraction = float(max(0.0, min(peak_min_fraction, 1.0)))
    threshold = max(1.0, float(np.max(smooth)) * peak_min_fraction)
    candidate_peaks = [
        i
        for i in range(1, int(smooth.size) - 1)
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and float(smooth[i]) >= threshold
    ]
    if not candidate_peaks:
        return np.empty((0,), dtype=np.float32)

    min_sep_bins = max(1, int(round(float(min_center_separation) / bin_size)))
    candidate_peaks.sort(key=lambda i: float(smooth[i]), reverse=True)
    selected: List[int] = []
    for idx in candidate_peaks:
        if all(abs(idx - keep) > min_sep_bins for keep in selected):
            selected.append(idx)
    selected.sort()
    centers = np.array([(float(edges[i]) + float(edges[i + 1])) * 0.5 for i in selected], dtype=np.float32)
    return centers


def _row_lines(
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    centers: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    for center, u0, u1 in zip(centers.tolist(), u_min.tolist(), u_max.tolist()):
        if u1 <= u0:
            continue
        p0 = direction_xy * float(u0) + perp_xy * float(center)
        p1 = direction_xy * float(u1) + perp_xy * float(center)
        lines.append((p0.astype(np.float32), p1.astype(np.float32)))
    return lines


def _filter_points_by_bounds(points_xy: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    if points_xy.size == 0:
        return points_xy
    mask = (
        (points_xy[:, 0] >= float(xmin))
        & (points_xy[:, 0] <= float(xmax))
        & (points_xy[:, 1] >= float(ymin))
        & (points_xy[:, 1] <= float(ymax))
    )
    return points_xy[mask]


def _mask_points_by_bounds(points_xy: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    if points_xy.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (points_xy[:, 0] >= float(xmin))
        & (points_xy[:, 0] <= float(xmax))
        & (points_xy[:, 1] >= float(ymin))
        & (points_xy[:, 1] <= float(ymax))
    )


def _sample_points_with_index(points: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        idx = np.arange(points.shape[0], dtype=np.int64)
        return points, idx
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(points.shape[0], int(max_points), replace=False)
    return points[idx], idx


def _filter_circles_by_bounds(circles: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    if circles.size == 0:
        return circles
    mask = (
        (circles[:, 0] >= float(xmin))
        & (circles[:, 0] <= float(xmax))
        & (circles[:, 1] >= float(ymin))
        & (circles[:, 1] <= float(ymax))
    )
    return circles[mask]


def _line_bbox_intersects(p0: np.ndarray, p1: np.ndarray, bounds: Tuple[float, float, float, float]) -> bool:
    xmin, xmax, ymin, ymax = bounds
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    if max(x0, x1) < float(xmin) or min(x0, x1) > float(xmax):
        return False
    if max(y0, y1) < float(ymin) or min(y0, y1) > float(ymax):
        return False
    return True


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(points.shape[0], int(max_points), replace=False)
    return points[idx]


def _svg_header(width: int, height: int, bg: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">\n'
        f'  <rect width="100%" height="100%" fill="{bg}" />\n'
    )


def _map_point(
    x: float,
    y: float,
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax = bounds
    dx = max(1.0e-6, xmax - xmin)
    dy = max(1.0e-6, ymax - ymin)
    scale = min((width - 2 * margin) / dx, (height - 2 * margin) / dy)
    px = margin + (x - xmin) * scale
    py = margin + (ymax - y) * scale
    return float(px), float(py)


def _render_svg(
    out_path: Path,
    points_xy: np.ndarray,
    row_lines: List[Tuple[np.ndarray, np.ndarray]],
    circles: np.ndarray,
    bounds: Tuple[float, float, float, float],
    robot_xy: Optional[Tuple[float, float]],
    max_points: int,
    sample_seed: int,
    width: int,
    height: int,
    margin: int,
    bg: str,
    point_color: str,
    row_color: str,
    circle_color: str,
    point_size: float,
    row_width: float,
    circle_width: float,
    circle_scale: float,
    robot_color: str,
    robot_size_m: float,
    show_robot_marker: bool,
) -> None:
    pts = _filter_points_by_bounds(points_xy, bounds)
    pts = _sample_points(pts, int(max_points), int(sample_seed))
    cir = _filter_circles_by_bounds(circles, bounds)

    svg: List[str] = []
    svg.append(_svg_header(int(width), int(height), bg))

    svg.append(f'  <g id="points" fill="{point_color}" fill-opacity="0.9" stroke="none">')
    r = max(0.1, float(point_size))
    for x, y in pts.tolist():
        px, py = _map_point(float(x), float(y), bounds, width, height, margin)
        svg.append(f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{r:.2f}" />')
    svg.append("  </g>")

    svg.append(f'  <g id="rows" fill="none" stroke="{row_color}" stroke-width="{float(row_width):.2f}">')
    for p0, p1 in row_lines:
        if not _line_bbox_intersects(p0, p1, bounds):
            continue
        px0, py0 = _map_point(float(p0[0]), float(p0[1]), bounds, width, height, margin)
        px1, py1 = _map_point(float(p1[0]), float(p1[1]), bounds, width, height, margin)
        svg.append(f'    <line x1="{px0:.2f}" y1="{py0:.2f}" x2="{px1:.2f}" y2="{py1:.2f}" />')
    svg.append("  </g>")

    svg.append(
        f'  <g id="circles" fill="none" stroke="{circle_color}" stroke-width="{float(circle_width):.2f}">'
    )
    for x, y, radius in cir.tolist():
        px, py = _map_point(float(x), float(y), bounds, width, height, margin)
        pr = float(radius) * float(circle_scale)
        px_r, _ = _map_point(float(x) + pr, float(y), bounds, width, height, margin)
        r_px = abs(px_r - px)
        svg.append(f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{r_px:.2f}" />')
    svg.append("  </g>")

    if show_robot_marker and robot_xy is not None:
        rx, ry = robot_xy
        px, py = _map_point(float(rx), float(ry), bounds, width, height, margin)
        px_r, _ = _map_point(float(rx) + float(robot_size_m), float(ry), bounds, width, height, margin)
        r_px = max(2.0, abs(px_r - px))
        svg.append(f'  <g id="robot" fill="{robot_color}" stroke="none">')
        svg.append(f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{r_px:.2f}" />')
        svg.append("  </g>")

    svg.append("</svg>\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def _points_from_cloud(msg: PointCloud2) -> np.ndarray:
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--points-topic", type=str, default="/points_raw")
    parser.add_argument("--tf-topic", type=str, default="/tf")
    parser.add_argument("--source-frame", type=str, default="lidar_link")
    parser.add_argument("--base-frame", type=str, default="base_link_est")
    parser.add_argument("--map-frame", type=str, default="map")

    parser.add_argument("--row-model", type=str, default=str(ws_dir / "maps" / "row_model_from_map4.json"))
    parser.add_argument("--out-dir", type=str, default=str(ws_dir / "maps" / "bev_points_raw"))
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--bev-span", type=float, default=25.0)
    parser.add_argument("--line-length", type=float, default=20.0)
    parser.add_argument("--accumulate-frames", type=int, default=1)
    parser.add_argument("--max-fit-points", type=int, default=200000)

    parser.add_argument("--z-min", type=float, default=0.9)
    parser.add_argument("--z-max", type=float, default=1.1)
    parser.add_argument("--row-bandwidth", type=float, default=0.9)
    parser.add_argument("--min-points-per-row", type=int, default=40)
    parser.add_argument("--u-percentile", type=float, default=2.0)
    parser.add_argument("--hist-bin-size", type=float, default=0.2)
    parser.add_argument("--hist-smooth-window", type=int, default=7)
    parser.add_argument("--hist-peak-min-fraction", type=float, default=0.15)
    parser.add_argument("--row-center-min-separation", type=float, default=2.0)

    parser.add_argument("--row-v-offsets", type=str, default="")
    parser.add_argument("--row-v-yaw-offsets-deg", type=str, default="")
    parser.add_argument("--use-prior-row-centers", action="store_true")

    parser.add_argument("--u-bin", type=float, default=0.05)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--peak-min-fraction", type=float, default=0.05)
    parser.add_argument("--min-separation", type=float, default=0.9)
    parser.add_argument("--refine-u-half-width", type=float, default=0.45)

    parser.add_argument("--center-refine-mode", type=str, default="median")
    parser.add_argument("--ransac-iters", type=int, default=250)
    parser.add_argument("--ransac-inlier-threshold", type=float, default=0.08)
    parser.add_argument("--ransac-min-inliers", type=int, default=40)
    parser.add_argument("--ransac-min-points", type=int, default=60)

    parser.add_argument("--max-points", type=int, default=40000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--margin", type=int, default=40)
    parser.add_argument("--bg", type=str, default="#ffffff")
    parser.add_argument("--point-color", type=str, default="#cfcfcf")
    parser.add_argument("--row-color", type=str, default="#00a8ff")
    parser.add_argument("--circle-color", type=str, default="#ff2bd6")
    parser.add_argument("--point-size", type=float, default=1.1)
    parser.add_argument("--row-width", type=float, default=2.2)
    parser.add_argument("--circle-width", type=float, default=1.8)
    parser.add_argument("--circle-scale", type=float, default=1.0)
    parser.add_argument("--skip-circles", action="store_true")
    parser.add_argument("--show-robot-marker", action="store_true")
    parser.add_argument("--robot-color", type=str, default="#ff6f00")
    parser.add_argument("--robot-size-m", type=float, default=0.6)

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    row_model_path = Path(args.row_model).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    impl = _load_tree_circles_impl()
    direction_xy, perp_xy, rows = impl._load_row_model_file(row_model_path)
    rows = impl._apply_row_overrides(
        rows,
        row_v_offsets=_parse_indexed_float_map(args.row_v_offsets) if str(args.row_v_offsets).strip() else {},
        row_u_offsets={},
        row_v_slopes={},
        row_v_yaw_offsets_deg=_parse_indexed_float_map(args.row_v_yaw_offsets_deg)
        if str(args.row_v_yaw_offsets_deg).strip()
        else {},
    )
    if rows:
        row_centers_prior = np.array([float(r["v_center"]) for r in rows], dtype=np.float32)
    else:
        row_centers_prior = np.empty((0,), dtype=np.float32)

    tf_buffer: Dict[Tuple[str, str], np.ndarray] = {}
    accum: Deque[np.ndarray] = deque()
    frames: List[Dict[str, Any]] = []
    max_frames = int(args.max_frames)
    next_time = None

    with rosbag.Bag(str(bag_path)) as bag:
        start_time = bag.get_start_time() + float(args.start_offset)
        end_time = bag.get_end_time() if float(args.duration) <= 0.0 else start_time + float(args.duration)

        for topic, msg, t in bag.read_messages(topics=[args.tf_topic, args.points_topic]):
            t_sec = float(t.to_sec())
            if t_sec < start_time:
                if topic == args.tf_topic:
                    _update_tf_buffer(tf_buffer, msg)
                continue
            if t_sec > end_time:
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
            map_T_lidar = _lookup_transform(tf_buffer, str(args.map_frame), source_frame)
            if map_T_lidar is None:
                continue
            map_T_base = _lookup_transform(tf_buffer, str(args.map_frame), str(args.base_frame))
            if map_T_base is None:
                map_T_base = map_T_lidar

            pts = _points_from_cloud(msg)
            if pts.size == 0:
                continue

            pts_h = np.hstack([pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)])
            pts_map = (map_T_lidar @ pts_h.T).T[:, :3].astype(np.float32)

            z_mask = (pts_map[:, 2] >= float(args.z_min)) & (pts_map[:, 2] <= float(args.z_max))
            pts_map = pts_map[z_mask]
            if pts_map.shape[0] < 5:
                continue

            accum.append(pts_map)
            while len(accum) > int(max(1, args.accumulate_frames)):
                accum.popleft()

            pts_accum = np.vstack(list(accum)) if accum else pts_map
            if pts_accum.shape[0] < 5:
                continue

            robot_xy = map_T_base[:2, 3].astype(np.float32)
            half = 0.5 * float(args.bev_span)
            bounds = (
                float(robot_xy[0] - half),
                float(robot_xy[0] + half),
                float(robot_xy[1] - half),
                float(robot_xy[1] + half),
            )

            pts_xy = pts_accum[:, :2].astype(np.float32)
            mask_bounds = _mask_points_by_bounds(pts_xy, bounds)
            if mask_bounds.shape[0] != pts_xy.shape[0]:
                continue
            pts_map = pts_accum[mask_bounds]
            pts_xy = pts_xy[mask_bounds]
            if pts_xy.shape[0] < 5:
                continue

            pts_map_fit, fit_idx = _sample_points_with_index(
                pts_map, int(args.max_fit_points), int(args.sample_seed) + len(frames)
            )
            pts_xy_fit = pts_map_fit[:, :2].astype(np.float32)

            u = pts_xy_fit @ direction_xy.astype(np.float32)
            v = pts_xy_fit @ perp_xy.astype(np.float32)

            if args.use_prior_row_centers and row_centers_prior.size >= 2:
                centers = row_centers_prior.copy()
            else:
                centers = _find_row_centers_histogram(
                    v=v,
                    bin_size=float(args.hist_bin_size),
                    smooth_window=int(args.hist_smooth_window),
                    peak_min_fraction=float(args.hist_peak_min_fraction),
                    min_center_separation=float(args.row_center_min_separation),
                )
                if centers.size < 2 and row_centers_prior.size >= 2:
                    centers = row_centers_prior.copy()

            centers = np.sort(centers.astype(np.float32))
            u_min_list: List[float] = []
            u_max_list: List[float] = []
            z_list: List[float] = []
            centers_kept: List[float] = []

            for center in centers.tolist():
                mask = np.abs(v - float(center)) <= float(args.row_bandwidth) * 0.5
                if np.count_nonzero(mask) < int(args.min_points_per_row):
                    continue
                u_vals = u[mask]
                if float(args.u_percentile) > 0.0:
                    u_min_val = float(np.percentile(u_vals, float(args.u_percentile)))
                    u_max_val = float(np.percentile(u_vals, 100.0 - float(args.u_percentile)))
                else:
                    u_min_val = float(np.min(u_vals))
                    u_max_val = float(np.max(u_vals))
                if u_max_val <= u_min_val:
                    continue
                z_list.append(float(np.median(pts_map_fit[mask, 2])))
                centers_kept.append(float(center))
                u_min_list.append(u_min_val)
                u_max_list.append(u_max_val)

            if len(u_min_list) < 2:
                continue

            centers = np.asarray(centers_kept, dtype=np.float32)
            u_min_arr = np.asarray(u_min_list, dtype=np.float32)
            u_max_arr = np.asarray(u_max_list, dtype=np.float32)

            u_robot = float(robot_xy @ direction_xy.astype(np.float32))
            half_len = 0.5 * float(args.line_length)
            u0 = np.maximum(u_min_arr, u_robot - half_len)
            u1 = np.minimum(u_max_arr, u_robot + half_len)

            row_lines = _row_lines(direction_xy, perp_xy, centers, u0, u1)

            circles_arr = np.empty((0, 3), dtype=np.float32)
            if not args.skip_circles:
                rows_local = [
                    {"v_center": float(c), "u_min": float(u0i), "u_max": float(u1i), "z": float(z)}
                    for c, u0i, u1i, z in zip(centers.tolist(), u0.tolist(), u1.tolist(), z_list)
                ]
                ransac_cfg = impl.CircleRansacConfig(
                    enabled=str(args.center_refine_mode).strip().lower() == "circle_ransac",
                    max_iterations=int(args.ransac_iters),
                    inlier_threshold=float(args.ransac_inlier_threshold),
                    min_inliers=int(args.ransac_min_inliers),
                    min_points=int(args.ransac_min_points),
                    use_inliers_for_radius=True,
                    set_radius=False,
                    seed=0,
                )
                circles, _ = impl._tree_circles_and_labels_from_row_model(
                    points_xyz=pts_map_fit,
                    direction_xy=direction_xy,
                    perp_xy=perp_xy,
                    rows=rows_local,
                    row_bandwidth=float(args.row_bandwidth),
                    u_bin_size=float(args.u_bin),
                    smooth_window=int(args.smooth_window),
                    peak_min_fraction=float(args.peak_min_fraction),
                    min_separation=float(args.min_separation),
                    u_padding=0.0,
                    refine_u_half_width=float(args.refine_u_half_width),
                    max_trees_per_row=0,
                    max_trees=0,
                    snap_to_row=False,
                    circle_ransac=ransac_cfg,
                    marker_z=0.0,
                    radius_mode="constant",
                    radius_constant=0.35,
                    radius_quantile=0.8,
                    radius_min=0.15,
                    radius_max=1.5,
                )
                circles_arr = np.array([(c.x, c.y, c.radius) for c in circles], dtype=np.float32)

            frame_path = out_dir / f"frame_{len(frames):04d}.svg"
            _render_svg(
                out_path=frame_path,
                points_xy=pts_xy,
                row_lines=row_lines,
                circles=circles_arr,
                bounds=bounds,
                robot_xy=(float(robot_xy[0]), float(robot_xy[1])),
                max_points=int(args.max_points),
                sample_seed=int(args.sample_seed) + len(frames),
                width=int(args.width),
                height=int(args.height),
                margin=int(args.margin),
                bg=str(args.bg),
                point_color=str(args.point_color),
                row_color=str(args.row_color),
                circle_color=str(args.circle_color),
                point_size=float(args.point_size),
                row_width=float(args.row_width),
                circle_width=float(args.circle_width),
                circle_scale=float(args.circle_scale),
                robot_color=str(args.robot_color),
                robot_size_m=float(args.robot_size_m),
                show_robot_marker=bool(args.show_robot_marker),
            )
            frames.append(
                {
                    "frame": int(len(frames) - 1),
                    "time": float(t_sec),
                    "x": float(robot_xy[0]),
                    "y": float(robot_xy[1]),
                    "bounds": [float(b) for b in bounds],
                    "accumulate_frames": int(max(1, args.accumulate_frames)),
                    "file": str(frame_path),
                }
            )

            if max_frames > 0 and len(frames) >= max_frames:
                break

    meta = {
        "bag": str(bag_path),
        "points_topic": str(args.points_topic),
        "tf_topic": str(args.tf_topic),
        "sample_rate": float(args.sample_rate),
        "start_offset": float(args.start_offset),
        "duration": float(args.duration),
        "bev_span": float(args.bev_span),
        "accumulate_frames": int(max(1, args.accumulate_frames)),
        "use_prior_row_centers": bool(args.use_prior_row_centers),
        "frames": frames,
    }
    (out_dir / "frames.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {len(frames)} frames to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
