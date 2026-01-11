#!/usr/bin/env python3
"""Select typical segments (straight/turn/sparse) from a rosbag and render BEV SVGs.

Assumes a prior tree-only map and row model are available. We use odometry for
segment selection and the tree map for BEV visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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


def _heading_angles(poses: np.ndarray) -> np.ndarray:
    if poses.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    dx = np.diff(poses[:, 0])
    dy = np.diff(poses[:, 1])
    return np.arctan2(dy, dx)


def _angle_diff(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    return np.abs(np.arctan2(np.sin(np.diff(a)), np.cos(np.diff(a))))


def _window_stats(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or values.size < window:
        return np.zeros((0,), dtype=np.float64)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def _select_window(scores: np.ndarray, prefer_min: bool, used: List[Tuple[int, int]], gap: int) -> int:
    if scores.size == 0:
        return -1
    order = np.argsort(scores) if prefer_min else np.argsort(scores)[::-1]
    for idx in order.tolist():
        start = int(idx)
        end = int(idx + 1)
        ok = True
        for u0, u1 in used:
            if not (end + gap <= u0 or start >= u1 + gap):
                ok = False
                break
        if ok:
            return start
    return int(order[0]) if order.size else -1


def _compute_density(points_xy: np.ndarray, poses: np.ndarray, radius: float) -> np.ndarray:
    if points_xy.size == 0 or poses.size == 0:
        return np.zeros((poses.shape[0],), dtype=np.float64)
    r2 = float(radius) ** 2
    counts = np.zeros((poses.shape[0],), dtype=np.float64)
    for i, (x, y) in enumerate(poses.tolist()):
        d2 = (points_xy[:, 0] - float(x)) ** 2 + (points_xy[:, 1] - float(y)) ** 2
        counts[i] = float(np.sum(d2 <= r2))
    return counts


def _export_circles_csv(path: Path, circles: List[Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "z", "radius"])
        for i, c in enumerate(circles):
            writer.writerow([int(i), f"{float(c.x):.6f}", f"{float(c.y):.6f}", f"{float(c.z):.6f}", f"{float(c.radius):.6f}"])


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--odom-topic", type=str, default="/liorl/mapping/odometry")

    parser.add_argument("--pcd", type=str, default=str(ws_dir / "maps" / "map4_label0.pcd"))
    parser.add_argument("--row-model", type=str, default=str(ws_dir / "maps" / "row_model_from_map4.json"))

    parser.add_argument("--out-dir", type=str, default=str(ws_dir / "maps" / "typical_segments"))
    parser.add_argument("--sample-rate", type=float, default=2.0)
    parser.add_argument("--segment-length", type=float, default=10.0)
    parser.add_argument("--min-gap", type=float, default=8.0)
    parser.add_argument("--density-radius", type=float, default=10.0)
    parser.add_argument("--bev-span", type=float, default=40.0)

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
    parser.add_argument("--min-separation", type=float, default=1.1)
    parser.add_argument("--refine-u-half-width", type=float, default=0.45)

    parser.add_argument("--center-refine-mode", type=str, default="circle_ransac")
    parser.add_argument("--ransac-iters", type=int, default=250)
    parser.add_argument("--ransac-inlier-threshold", type=float, default=0.08)
    parser.add_argument("--ransac-min-inliers", type=int, default=40)
    parser.add_argument("--ransac-min-points", type=int, default=60)

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    pcd_path = Path(args.pcd).expanduser().resolve()
    row_model_path = Path(args.row_model).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load odometry.
    times, poses = _load_odometry(bag_path, args.odom_topic)
    times, poses = _downsample(times, poses, rate_hz=float(args.sample_rate))
    if poses.shape[0] < 5:
        raise RuntimeError("Not enough odometry samples after downsampling.")

    headings = _heading_angles(poses)
    heading_diff = _angle_diff(headings)

    # Per-sample density using tree map points.
    impl = _load_tree_circles_impl()
    points = impl._load_pcd_xyz(pcd_path).astype(np.float32)
    points = impl._filter_points(points, z_min=float(args.z_min), z_max=float(args.z_max), x_min=-1.0e9, x_max=1.0e9, y_abs_max=1.0e9)
    points_xy = points[:, :2].astype(np.float32)
    density = _compute_density(points_xy, poses, radius=float(args.density_radius))

    window = max(3, int(round(float(args.segment_length) * float(args.sample_rate))))
    curvature_score = _window_stats(np.abs(heading_diff), window - 1)
    density_score = _window_stats(density, window)

    used: List[Tuple[int, int]] = []
    gap = int(round(float(args.min_gap) * float(args.sample_rate)))

    straight_idx = _select_window(curvature_score, prefer_min=True, used=used, gap=gap)
    if straight_idx >= 0:
        used.append((straight_idx, straight_idx + window))

    turn_idx = _select_window(curvature_score, prefer_min=False, used=used, gap=gap)
    if turn_idx >= 0:
        used.append((turn_idx, turn_idx + window))

    sparse_idx = _select_window(density_score, prefer_min=True, used=used, gap=gap)
    if sparse_idx >= 0:
        used.append((sparse_idx, sparse_idx + window))

    segments = [
        ("straight", straight_idx),
        ("turn", turn_idx),
        ("sparse", sparse_idx),
    ]

    # Compute global circles once (for visualization).
    rows = impl._apply_row_overrides(
        impl._load_row_model_file(row_model_path)[2],
        row_v_offsets=_parse_indexed_float_map(args.row_v_offsets),
        row_u_offsets=_parse_indexed_float_map(args.row_u_offsets) if args.row_u_offsets.strip() else {},
        row_v_slopes=_parse_indexed_float_map(args.row_v_slopes) if args.row_v_slopes.strip() else {},
        row_v_yaw_offsets_deg=_parse_indexed_float_map(args.row_v_yaw_offsets_deg),
    )
    direction_xy, perp_xy, _ = impl._load_row_model_file(row_model_path)

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
    circles_csv = out_dir / "tree_circles.csv"
    _export_circles_csv(circles_csv, circles)

    info: Dict[str, Any] = {
        "bag": str(bag_path),
        "odom_topic": str(args.odom_topic),
        "sample_rate": float(args.sample_rate),
        "segment_length": float(args.segment_length),
        "min_gap": float(args.min_gap),
        "density_radius": float(args.density_radius),
        "bev_span": float(args.bev_span),
        "segments": [],
    }

    for name, idx in segments:
        if idx < 0:
            continue
        start = int(idx)
        end = int(idx + window)
        seg_times = times[start:end]
        seg_poses = poses[start:end]
        center = np.mean(seg_poses, axis=0)
        bounds = [
            float(center[0] - float(args.bev_span) * 0.5),
            float(center[0] + float(args.bev_span) * 0.5),
            float(center[1] - float(args.bev_span) * 0.5),
            float(center[1] + float(args.bev_span) * 0.5),
        ]
        seg_dir = out_dir / name
        seg_dir.mkdir(parents=True, exist_ok=True)

        bev_out = seg_dir / "bev.svg"
        cmd = [
            "python3",
            str(ws_dir / "src" / "orchard_row_mapping" / "tools" / "render_bev_svg.py"),
            "--pcd",
            str(pcd_path),
            "--row-model",
            str(row_model_path),
            "--circles",
            str(circles_csv),
            "--out",
            str(bev_out),
            "--z-min",
            str(args.z_min),
            "--z-max",
            str(args.z_max),
            "--row-v-offsets",
            str(args.row_v_offsets),
            "--row-v-yaw-offsets-deg",
            str(args.row_v_yaw_offsets_deg),
            "--max-points",
            "40000",
            "--sample-seed",
            "0",
            f"--bounds={','.join(f'{b:.3f}' for b in bounds)}",
            "--width",
            "1800",
            "--height",
            "1400",
            "--margin",
            "40",
        ]
        subprocess.run(cmd, check=True)

        info["segments"].append(
            {
                "name": name,
                "start_time": float(seg_times[0]),
                "end_time": float(seg_times[-1]),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "curvature_mean": float(np.mean(np.abs(heading_diff[start : end - 1]))) if end - start > 1 else 0.0,
                "density_mean": float(np.mean(density[start:end])),
                "bounds": bounds,
                "bev_svg": str(bev_out),
            }
        )

    (out_dir / "segments.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[OK] Wrote segments to: {out_dir}")
    print(f"     segments.json: {out_dir / 'segments.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
