#!/usr/bin/env python3
"""Render a BEV SVG sequence along a rosbag trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rosbag


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


def _load_odometry(bag_path: Path, topic: str) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    poses: List[Tuple[float, float]] = []
    with rosbag.Bag(str(bag_path)) as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            if not hasattr(msg, "pose"):
                continue
            pos = msg.pose.pose.position
            times.append(float(t.to_sec()))
            poses.append((float(pos.x), float(pos.y)))
    if not times:
        raise RuntimeError(f"No odometry messages found on {topic}")
    return np.asarray(times, dtype=np.float64), np.asarray(poses, dtype=np.float64)


def _downsample(times: np.ndarray, poses: np.ndarray, rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return times, poses
    step = 1.0 / float(rate_hz)
    t0 = float(times[0])
    t_end = float(times[-1])
    target = t0
    idx = 0
    out_times: List[float] = []
    out_poses: List[Tuple[float, float]] = []
    while target <= t_end and idx < times.size:
        while idx < times.size and times[idx] < target:
            idx += 1
        if idx >= times.size:
            break
        out_times.append(float(times[idx]))
        out_poses.append((float(poses[idx, 0]), float(poses[idx, 1])))
        target += step
    return np.asarray(out_times, dtype=np.float64), np.asarray(out_poses, dtype=np.float64)


def _row_lines(
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    rows: List[Dict[str, float]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        v_center = float(row["v_center"])
        v_slope = float(row.get("v_slope", 0.0))
        u_offset = float(row.get("u_offset", 0.0))
        u_min = float(row["u_min"]) + float(u_offset)
        u_max = float(row["u_max"]) + float(u_offset)
        if u_max <= u_min:
            continue
        u_anchor = 0.5 * (float(row["u_min"]) + float(row["u_max"])) + float(u_offset)

        v_min = v_center + v_slope * (u_min - u_anchor)
        v_max = v_center + v_slope * (u_max - u_anchor)

        p0 = direction_xy * float(u_min) + perp_xy * float(v_min)
        p1 = direction_xy * float(u_max) + perp_xy * float(v_max)
        lines.append((p0.astype(np.float32), p1.astype(np.float32)))
    return lines


def _filter_points_by_bounds(
    points_xy: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy
    mask = (
        (points_xy[:, 0] >= float(xmin))
        & (points_xy[:, 0] <= float(xmax))
        & (points_xy[:, 1] >= float(ymin))
        & (points_xy[:, 1] <= float(ymax))
    )
    return points_xy[mask]


def _filter_circles_by_bounds(
    circles: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> np.ndarray:
    if circles.size == 0:
        return circles
    mask = (
        (circles[:, 0] >= float(xmin))
        & (circles[:, 0] <= float(xmax))
        & (circles[:, 1] >= float(ymin))
        & (circles[:, 1] <= float(ymax))
    )
    return circles[mask]


def _line_bbox_intersects(
    p0: np.ndarray,
    p1: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> bool:
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
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    margin: int,
) -> Tuple[float, float]:
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
) -> None:
    xmin, xmax, ymin, ymax = bounds
    pts = _filter_points_by_bounds(points_xy, xmin, xmax, ymin, ymax)
    pts = _sample_points(pts, int(max_points), int(sample_seed))
    cir = _filter_circles_by_bounds(circles, xmin, xmax, ymin, ymax)

    svg: List[str] = []
    svg.append(_svg_header(int(width), int(height), bg))

    svg.append(f'  <g id="points" fill="{point_color}" fill-opacity="0.9" stroke="none">')
    r = max(0.1, float(point_size))
    for x, y in pts.tolist():
        px, py = _map_point(float(x), float(y), xmin, xmax, ymin, ymax, width, height, margin)
        svg.append(f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{r:.2f}" />')
    svg.append("  </g>")

    svg.append(f'  <g id="rows" fill="none" stroke="{row_color}" stroke-width="{float(row_width):.2f}">')
    for p0, p1 in row_lines:
        if not _line_bbox_intersects(p0, p1, xmin, xmax, ymin, ymax):
            continue
        px0, py0 = _map_point(float(p0[0]), float(p0[1]), xmin, xmax, ymin, ymax, width, height, margin)
        px1, py1 = _map_point(float(p1[0]), float(p1[1]), xmin, xmax, ymin, ymax, width, height, margin)
        svg.append(f'    <line x1="{px0:.2f}" y1="{py0:.2f}" x2="{px1:.2f}" y2="{py1:.2f}" />')
    svg.append("  </g>")

    svg.append(
        f'  <g id="circles" fill="none" stroke="{circle_color}" stroke-width="{float(circle_width):.2f}">'
    )
    for x, y, radius in cir.tolist():
        px, py = _map_point(float(x), float(y), xmin, xmax, ymin, ymax, width, height, margin)
        pr = float(radius) * float(circle_scale)
        px_r, _ = _map_point(float(x) + pr, float(y), xmin, xmax, ymin, ymax, width, height, margin)
        r_px = abs(px_r - px)
        svg.append(f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{r_px:.2f}" />')
    svg.append("  </g>")

    svg.append("</svg>\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--odom-topic", type=str, default="/liorl/mapping/odometry")

    parser.add_argument("--pcd", type=str, default=str(ws_dir / "maps" / "map4_label0.pcd"))
    parser.add_argument("--row-model", type=str, default=str(ws_dir / "maps" / "row_model_from_map4.json"))

    parser.add_argument("--out-dir", type=str, default=str(ws_dir / "maps" / "bev_sequence"))
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--bev-span", type=float, default=25.0)

    parser.add_argument("--z-min", type=float, default=0.9)
    parser.add_argument("--z-max", type=float, default=1.1)
    parser.add_argument("--row-bandwidth", type=float, default=0.9)

    parser.add_argument("--row-v-offsets", type=str, default='{"4":0.43}')
    parser.add_argument("--row-u-offsets", type=str, default="")
    parser.add_argument("--row-v-slopes", type=str, default="")
    parser.add_argument("--row-v-yaw-offsets-deg", type=str, default='{"4":2.36}')

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
    parser.add_argument("--point-size", type=float, default=1.2)
    parser.add_argument("--row-width", type=float, default=2.4)
    parser.add_argument("--circle-width", type=float, default=2.0)
    parser.add_argument("--circle-scale", type=float, default=1.0)

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    pcd_path = Path(args.pcd).expanduser().resolve()
    row_model_path = Path(args.row_model).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    times, poses = _load_odometry(bag_path, args.odom_topic)
    times, poses = _downsample(times, poses, rate_hz=float(args.sample_rate))

    if poses.shape[0] < 2:
        raise RuntimeError("Not enough odometry samples after downsampling.")

    t0 = float(times[0])
    start_time = t0 + float(args.start_offset)
    end_time = times[-1] if float(args.duration) <= 0.0 else start_time + float(args.duration)
    mask = (times >= start_time) & (times <= end_time)
    times = times[mask]
    poses = poses[mask]
    if poses.shape[0] == 0:
        raise RuntimeError("No samples left after applying time range.")

    impl = _load_tree_circles_impl()
    points = impl._load_pcd_xyz(pcd_path).astype(np.float32)
    points = impl._filter_points(points, float(args.z_min), float(args.z_max), -1.0e9, 1.0e9, 1.0e9)
    points_xy = points[:, :2].astype(np.float32)

    direction_xy, perp_xy, rows = impl._load_row_model_file(row_model_path)
    rows = impl._apply_row_overrides(
        rows,
        row_v_offsets=_parse_indexed_float_map(args.row_v_offsets),
        row_u_offsets=_parse_indexed_float_map(args.row_u_offsets) if args.row_u_offsets.strip() else {},
        row_v_slopes=_parse_indexed_float_map(args.row_v_slopes) if args.row_v_slopes.strip() else {},
        row_v_yaw_offsets_deg=_parse_indexed_float_map(args.row_v_yaw_offsets_deg),
    )
    row_lines = _row_lines(direction_xy, perp_xy, rows)

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
        points_xyz=points,
        direction_xy=direction_xy,
        perp_xy=perp_xy,
        rows=rows,
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

    frames: List[Dict[str, Any]] = []
    max_frames = int(args.max_frames)
    for idx, (t, pose) in enumerate(zip(times.tolist(), poses.tolist())):
        if max_frames > 0 and idx >= max_frames:
            break
        cx, cy = float(pose[0]), float(pose[1])
        half = 0.5 * float(args.bev_span)
        bounds = (cx - half, cx + half, cy - half, cy + half)
        frame_path = out_dir / f"frame_{idx:04d}.svg"
        _render_svg(
            out_path=frame_path,
            points_xy=points_xy,
            row_lines=row_lines,
            circles=circles_arr,
            bounds=bounds,
            max_points=int(args.max_points),
            sample_seed=int(args.sample_seed) + idx,
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
        )
        frames.append(
            {
                "frame": int(idx),
                "time": float(t),
                "x": float(cx),
                "y": float(cy),
                "bounds": [float(b) for b in bounds],
                "file": str(frame_path),
            }
        )

    meta = {
        "bag": str(bag_path),
        "odom_topic": str(args.odom_topic),
        "sample_rate": float(args.sample_rate),
        "start_offset": float(args.start_offset),
        "duration": float(args.duration),
        "bev_span": float(args.bev_span),
        "frames": frames,
    }
    (out_dir / "frames.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {len(frames)} frames to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
