#!/usr/bin/env python3
"""Compare multiple tree-row fitting methods against a manual prior (lines drawn by user).

This is an offline evaluator:
- Reads a bag with a point cloud topic + /tf
- Transforms points into map frame, optionally accumulates frames
- For each frame, selects the two model rows that surround the robot (in v)
- Runs one or more methods to estimate each row v_center, e.g.:
  - Circle methods: row_model_cell_clusters (grid connected components), row_model_peaks (+ optional circle_ransac)
  - Points-only baselines: use gated point v distribution (median/mean/madmean/mode)
- Estimates each row v_center from detected circles
- Reports per-frame errors vs the manual prior v_center (ground truth), plus optional renderings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

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
    # Dataclasses expect the module to be present in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _circles_to_v(circles: Sequence[Any], perp_xy: np.ndarray) -> np.ndarray:
    if not circles:
        return np.empty((0,), dtype=np.float32)
    xy = np.array([[float(c.x), float(c.y)] for c in circles], dtype=np.float32)
    return xy.dot(perp_xy.astype(np.float32).reshape(2))


def _circles_to_xy(circles: Sequence[Any]) -> np.ndarray:
    if not circles:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[float(c.x), float(c.y)] for c in circles], dtype=np.float32)


def _filter_circles_local(
    circles: Sequence[Any],
    direction_xy: np.ndarray,
    u_robot: float,
    u_window: float,
    k_nearest: int,
) -> List[Any]:
    if not circles:
        return []
    use_window = float(u_window) > 0.0
    use_k = int(k_nearest) > 0
    if not use_window and not use_k:
        return list(circles)

    direction = direction_xy.astype(np.float32).reshape(2)
    pairs: List[Tuple[Any, float]] = []
    for c in circles:
        u = float(np.dot(np.array([float(c.x), float(c.y)], dtype=np.float32), direction))
        if use_window and abs(u - float(u_robot)) > float(u_window):
            continue
        pairs.append((c, u))

    if use_k and len(pairs) > int(k_nearest):
        pairs.sort(key=lambda cu: abs(float(cu[1]) - float(u_robot)))
        pairs = pairs[: int(k_nearest)]
    return [c for c, _ in pairs]


def _filter_points_local_mask(
    u_vals: np.ndarray,
    u_robot: float,
    u_window: float,
    k_nearest: int,
) -> np.ndarray:
    if u_vals.size == 0:
        return np.zeros((0,), dtype=bool)
    mask = np.ones((u_vals.size,), dtype=bool)
    use_window = float(u_window) > 0.0
    use_k = int(k_nearest) > 0
    if use_window:
        mask = np.abs(u_vals - float(u_robot)) <= float(u_window)
    if use_k:
        idx = np.where(mask)[0]
        if idx.size > int(k_nearest):
            order = np.argsort(np.abs(u_vals[idx] - float(u_robot)))
            keep = idx[order[: int(k_nearest)]]
            new_mask = np.zeros_like(mask)
            new_mask[keep] = True
            mask = new_mask
    return mask


def _fit_direction_from_xy(xy: np.ndarray) -> Optional[np.ndarray]:
    if xy.shape[0] < 2:
        return None
    center = xy.mean(axis=0)
    centered = xy - center
    cov = (centered.T @ centered) / float(max(1, xy.shape[0] - 1))
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    idx = int(np.argmax(vals))
    direction = vecs[:, idx]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        return None
    return (direction / norm).astype(np.float32)


def _angle_error_rad(dir_est: Optional[np.ndarray], dir_ref: np.ndarray) -> float:
    if dir_est is None:
        return float("nan")
    ref = np.array(dir_ref, dtype=np.float64).reshape(2)
    est = np.array(dir_est, dtype=np.float64).reshape(2)
    ref_norm = float(np.linalg.norm(ref))
    est_norm = float(np.linalg.norm(est))
    if ref_norm <= 1.0e-9 or est_norm <= 1.0e-9:
        return float("nan")
    dot = float(abs(np.dot(ref / ref_norm, est / est_norm)))
    dot = float(max(-1.0, min(1.0, dot)))
    return float(math.acos(dot))


def _ransac_line_inliers(
    points_xy: np.ndarray,
    *,
    max_iters: int,
    inlier_threshold: float,
    min_inliers: int,
    seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if points_xy.shape[0] < 2:
        return None, None
    rng = np.random.default_rng(int(seed))
    n = points_xy.shape[0]
    best_mask = None
    best_dir = None
    best_count = 0
    for _ in range(int(max_iters)):
        i, j = rng.choice(n, size=2, replace=False)
        p1 = points_xy[i]
        p2 = points_xy[j]
        d = p2 - p1
        norm = float(np.linalg.norm(d))
        if norm <= 1.0e-6:
            continue
        d = (d / norm).astype(np.float32)
        normal = np.array([-float(d[1]), float(d[0])], dtype=np.float32)
        dist = np.abs((points_xy - p1) @ normal)
        mask = dist <= float(inlier_threshold)
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
            best_dir = d
            if best_count == n:
                break
    if best_mask is None:
        return None, None
    if best_count >= int(min_inliers):
        dir_refined = _fit_direction_from_xy(points_xy[best_mask])
        if dir_refined is not None:
            best_dir = dir_refined
    return best_mask, best_dir


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(int(window), dtype=np.float32)
    kernel /= float(kernel.sum())
    return np.convolve(values, kernel, mode="same")


def _find_peak_indices(values: np.ndarray, min_value: float, min_separation: int) -> List[int]:
    if values.size < 3:
        return []
    peaks = []
    for i in range(1, int(values.size) - 1):
        if values[i] < float(min_value):
            continue
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            peaks.append(int(i))
    if not peaks:
        return []
    peaks = sorted(peaks, key=lambda idx: float(values[idx]), reverse=True)
    selected: List[int] = []
    for idx in peaks:
        if all(abs(idx - s) >= int(min_separation) for s in selected):
            selected.append(idx)
    return selected


def _segments_from_circles(
    circles: Sequence[Any],
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    max_gap: float,
    min_count: int,
) -> List[Dict[str, Any]]:
    if not circles:
        return []
    direction = direction_xy.astype(np.float32).reshape(2)
    perp = perp_xy.astype(np.float32).reshape(2)
    data: List[Tuple[float, float]] = []
    for c in circles:
        xy = np.array([float(c.x), float(c.y)], dtype=np.float32)
        u = float(np.dot(xy, direction))
        v = float(np.dot(xy, perp))
        data.append((u, v))
    if not data:
        return []
    data.sort(key=lambda uv: uv[0])
    segments: List[Dict[str, Any]] = []
    seg_u: List[float] = [data[0][0]]
    seg_v: List[float] = [data[0][1]]
    for u, v in data[1:]:
        if float(max_gap) > 0.0 and abs(float(u) - float(seg_u[-1])) > float(max_gap):
            if len(seg_u) >= int(min_count):
                segments.append(
                    {
                        "u_min": float(min(seg_u)),
                        "u_max": float(max(seg_u)),
                        "v_center": float(np.median(np.array(seg_v, dtype=np.float32))),
                        "count": int(len(seg_u)),
                    }
                )
            seg_u = [u]
            seg_v = [v]
            continue
        seg_u.append(u)
        seg_v.append(v)
    if len(seg_u) >= int(min_count):
        segments.append(
            {
                "u_min": float(min(seg_u)),
                "u_max": float(max(seg_u)),
                "v_center": float(np.median(np.array(seg_v, dtype=np.float32))),
                "count": int(len(seg_u)),
            }
        )
    return segments


def _segments_from_circles_window(
    circles: Sequence[Any],
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    u_min: float,
    u_max: float,
    window_length: float,
    window_stride: float,
    min_count: int,
) -> List[Dict[str, Any]]:
    if not circles:
        return []
    length = float(window_length)
    stride = float(window_stride)
    if length <= 0.0:
        length = max(1.0e-3, float(u_max) - float(u_min))
    if stride <= 0.0:
        stride = length

    direction = direction_xy.astype(np.float32).reshape(2)
    perp = perp_xy.astype(np.float32).reshape(2)
    data: List[Tuple[float, float]] = []
    for c in circles:
        xy = np.array([float(c.x), float(c.y)], dtype=np.float32)
        u = float(np.dot(xy, direction))
        v = float(np.dot(xy, perp))
        data.append((u, v))
    if not data:
        return []

    u_vals = np.array([u for u, _ in data], dtype=np.float32)
    v_vals = np.array([v for _, v in data], dtype=np.float32)

    segments: List[Dict[str, Any]] = []
    u_start = float(u_min)
    u_stop = float(u_max)
    while u_start < u_stop - 1.0e-6:
        u_end = u_start + length
        if u_end > u_stop:
            u_end = u_stop
        mask = (u_vals >= float(u_start)) & (u_vals <= float(u_end))
        if int(np.count_nonzero(mask)) >= int(min_count):
            v_center = float(np.median(v_vals[mask]))
            segments.append(
                {
                    "u_min": float(u_start),
                    "u_max": float(u_end),
                    "v_center": v_center,
                    "count": int(np.count_nonzero(mask)),
                }
            )
        u_start += stride
    return segments


def _segments_from_circles_window_adaptive(
    circles: Sequence[Any],
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    u_min: float,
    u_max: float,
    window_length: float,
    window_stride: float,
    min_count: int,
    trim_quantile: float,
    min_span: float,
) -> List[Dict[str, Any]]:
    if not circles:
        return []
    length = float(window_length)
    stride = float(window_stride)
    if length <= 0.0:
        length = max(1.0e-3, float(u_max) - float(u_min))
    if stride <= 0.0:
        stride = length

    direction = direction_xy.astype(np.float32).reshape(2)
    perp = perp_xy.astype(np.float32).reshape(2)
    data: List[Tuple[float, float]] = []
    for c in circles:
        xy = np.array([float(c.x), float(c.y)], dtype=np.float32)
        u = float(np.dot(xy, direction))
        v = float(np.dot(xy, perp))
        data.append((u, v))
    if not data:
        return []

    u_vals = np.array([u for u, _ in data], dtype=np.float32)
    v_vals = np.array([v for _, v in data], dtype=np.float32)

    segments: List[Dict[str, Any]] = []
    u_start = float(u_min)
    u_stop = float(u_max)
    q = float(trim_quantile)
    if q < 0.0:
        q = 0.0
    if q > 0.49:
        q = 0.49
    min_span = max(0.0, float(min_span))

    while u_start < u_stop - 1.0e-6:
        u_end = u_start + length
        if u_end > u_stop:
            u_end = u_stop
        mask = (u_vals >= float(u_start)) & (u_vals <= float(u_end))
        if int(np.count_nonzero(mask)) >= int(min_count):
            u_in = u_vals[mask]
            v_in = v_vals[mask]
            if u_in.size == 0:
                u_start += stride
                continue
            if q > 0.0:
                u_lo = float(np.quantile(u_in, q))
                u_hi = float(np.quantile(u_in, 1.0 - q))
            else:
                u_lo = float(u_in.min())
                u_hi = float(u_in.max())

            if float(u_hi) - float(u_lo) < min_span:
                center = float(np.median(u_in))
                half = 0.5 * max(min_span, 1.0e-3)
                u_lo = max(float(u_start), center - half)
                u_hi = min(float(u_end), center + half)

            if float(u_hi) <= float(u_lo) + 1.0e-6:
                u_start += stride
                continue

            v_center = float(np.median(v_in))
            segments.append(
                {
                    "u_min": float(u_lo),
                    "u_max": float(u_hi),
                    "v_center": float(v_center),
                    "count": int(u_in.size),
                }
            )
        u_start += stride
    return segments


def _segments_from_circles_circle(
    circles: Sequence[Any],
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    v_row: float,
    shrink: float,
    min_len: float,
    max_len: float,
    default_len: float,
    use_circle_v: bool,
) -> List[Dict[str, Any]]:
    if not circles:
        return []
    direction = direction_xy.astype(np.float32).reshape(2)
    perp = perp_xy.astype(np.float32).reshape(2)
    data: List[Tuple[float, float]] = []
    for c in circles:
        xy = np.array([float(c.x), float(c.y)], dtype=np.float32)
        u = float(np.dot(xy, direction))
        v = float(np.dot(xy, perp))
        data.append((u, v))
    if not data:
        return []
    data.sort(key=lambda uv: uv[0])
    u_vals = [u for u, _ in data]
    v_vals = [v for _, v in data]

    shrink = float(shrink)
    if shrink <= 0.0 or shrink > 1.0:
        shrink = 0.8
    min_len = max(0.1, float(min_len))
    max_len = max(min_len, float(max_len))
    default_len = max(min_len, float(default_len))

    segments: List[Dict[str, Any]] = []
    n = len(u_vals)
    for i, (u, v) in enumerate(zip(u_vals, v_vals)):
        if n == 1:
            base = default_len
        else:
            d_prev = u - u_vals[i - 1] if i > 0 else float("nan")
            d_next = u_vals[i + 1] - u if i < n - 1 else float("nan")
            if math.isfinite(d_prev) and math.isfinite(d_next):
                base = 0.5 * (d_prev + d_next)
            elif math.isfinite(d_prev):
                base = d_prev
            elif math.isfinite(d_next):
                base = d_next
            else:
                base = default_len
        if not math.isfinite(base) or base <= 0.0:
            base = default_len
        seg_len = max(min_len, min(max_len, base * shrink))
        u_min = float(u - 0.5 * seg_len)
        u_max = float(u + 0.5 * seg_len)
        v_center = float(v) if bool(use_circle_v) else float(v_row)
        segments.append(
            {
                "u_min": u_min,
                "u_max": u_max,
                "v_center": v_center,
                "count": 1,
            }
        )
    return segments


def _cluster_centers_from_points_cells(
    *,
    points_xy: np.ndarray,
    impl: Any,
    cell_size: float,
    neighbor_range: int,
    min_points: int,
    max_clusters: int,
    max_cluster_span: float,
) -> np.ndarray:
    if points_xy.shape[0] < int(max(1, min_points)):
        return np.empty((0, 2), dtype=np.float32)
    clusters = impl._cluster_cells(
        points_xy.astype(np.float32),
        float(cell_size),
        int(neighbor_range),
        int(min_points),
        int(max_clusters),
    )
    centers: List[np.ndarray] = []
    span_max = float(max_cluster_span)
    for idxs in clusters:
        pts = points_xy[np.asarray(idxs, dtype=np.int32)]
        if pts.shape[0] == 0:
            continue
        if span_max > 0.0:
            span = float(max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))))
            if span > span_max:
                continue
        centers.append(np.median(pts, axis=0).astype(np.float32))
    if not centers:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(centers).astype(np.float32)


def _pick_two_v_peaks(
    v_vals: np.ndarray,
    *,
    bin_size: float,
    smooth_window: int,
    peak_min_fraction: float,
    min_separation: float,
) -> List[float]:
    if v_vals.size < 2:
        return []
    bin_size = float(max(1.0e-3, bin_size))
    v_min = float(v_vals.min())
    v_max = float(v_vals.max())
    bins = np.arange(v_min, v_max + bin_size * 1.0001, bin_size, dtype=np.float32)
    if bins.size < 2:
        return []
    hist, edges = np.histogram(v_vals, bins=bins)
    hist = hist.astype(np.float32)
    hist_smooth = _smooth_1d(hist, int(max(1, smooth_window)))
    peak_min = float(hist_smooth.max()) * float(peak_min_fraction)
    min_sep_bins = int(max(1.0, float(min_separation) / bin_size))
    peak_idx = _find_peak_indices(hist_smooth, peak_min, min_sep_bins)
    if not peak_idx:
        return []
    centers = 0.5 * (edges[:-1] + edges[1:])
    peaks = [float(centers[i]) for i in peak_idx]
    peaks.sort(key=lambda v: abs(float(v)))
    chosen: List[float] = []
    for v in peaks:
        if not chosen:
            chosen.append(float(v))
            continue
        if float(v) * float(chosen[0]) < 0.0:
            chosen.append(float(v))
            break
    if len(chosen) < 2:
        for v in peaks:
            if len(chosen) >= 2:
                break
            if all(abs(float(v) - float(c)) >= float(min_separation) for c in chosen):
                chosen.append(float(v))
    return chosen[:2]


def _median_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _stats(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0.0, "mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": float(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _choose_row_pair(v_robot: float, rows_sorted: Sequence[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    v_centers = [float(r["v_center"]) for r in rows_sorted]
    for i in range(len(v_centers) - 1):
        if float(v_centers[i]) <= float(v_robot) <= float(v_centers[i + 1]):
            return i, i + 1
    return None


@dataclass(frozen=True)
class _MethodSpec:
    name: str
    kind: str  # "cell" | "peaks" | "points" | "points_pca"
    center_refine: str  # circle: "median" | "circle_ransac"; points: "median" | "mean" | "madmean" | "mode"
    snap_to_row: bool
    line_bgr: Tuple[int, int, int]
    circle_bgr: Tuple[int, int, int]


def _try_import_cv2() -> Any:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _yaw_from_rot(rot: np.ndarray) -> float:
    # rot is 3x3
    return float(math.atan2(float(rot[1, 0]), float(rot[0, 0])))


def _load_rows_from_json(path: Optional[Path]) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
    if path is None:
        return [], None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    direction = None
    perp = None
    if "direction_xy" in data and "perp_xy" in data:
        direction = np.array(data["direction_xy"], dtype=np.float32).reshape(2)
        perp = np.array(data["perp_xy"], dtype=np.float32).reshape(2)

    rows_in = data.get("rows", data.get("rows_uv", []))
    rows: List[Dict[str, Any]] = []
    for r in rows_in:
        if r is None:
            continue
        if "v_center" not in r:
            continue
        rows.append(
            {
                "v_center": float(r["v_center"]),
                "u_min": float(r.get("u_min", r.get("u_start", 0.0))),
                "u_max": float(r.get("u_max", r.get("u_end", 0.0))),
                "z": float(r.get("z", 0.0)),
            }
        )
    rows.sort(key=lambda rr: float(rr["v_center"]))
    return rows, direction, perp


def _uv_to_xy(direction_xy: np.ndarray, perp_xy: np.ndarray, u: float, v: float) -> np.ndarray:
    return direction_xy.astype(np.float32) * float(u) + perp_xy.astype(np.float32) * float(v)


def _xy_to_px(
    xy: np.ndarray, *, xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int
) -> Tuple[int, int]:
    x, y = float(xy[0]), float(xy[1])
    u = 0.0 if xmax == xmin else (x - xmin) / (xmax - xmin)
    v = 0.0 if ymax == ymin else (ymax - y) / (ymax - ymin)
    px = int(round(u * (width - 1)))
    py = int(round(v * (height - 1)))
    return int(np.clip(px, 0, width - 1)), int(np.clip(py, 0, height - 1))


def _render_density_bev(
    *,
    cv2: Any,
    points_xy: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_points: int,
    seed: int,
    point_darkness: int,
) -> np.ndarray:
    if points_xy.size == 0:
        return np.full((height, width, 3), 255, dtype=np.uint8)
    pts = _sample_points(points_xy.astype(np.float32), int(max_points), int(seed))
    x = pts[:, 0].astype(np.float32)
    y = pts[:, 1].astype(np.float32)
    if xmax == xmin or ymax == ymin:
        return np.full((height, width, 3), 255, dtype=np.uint8)

    ix = ((x - float(xmin)) / (float(xmax) - float(xmin)) * float(width - 1)).astype(np.int32)
    iy = ((float(ymax) - y) / (float(ymax) - float(ymin)) * float(height - 1)).astype(np.int32)
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)

    grid = np.zeros((height, width), dtype=np.uint16)
    np.add.at(grid, (iy, ix), 1)

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    if not np.any(grid):
        return img

    val = np.log1p(grid.astype(np.float32))
    nz = val[val > 0]
    if nz.size == 0:
        return img
    vmax = float(np.quantile(nz, 0.99))
    if vmax <= 0.0:
        return img
    norm = np.clip(val / vmax, 0.0, 1.0)
    darkness = float(max(0, min(int(point_darkness), 255)))
    grey = (255.0 - norm * darkness).astype(np.uint8)
    img[:, :, 0] = grey
    img[:, :, 1] = grey
    img[:, :, 2] = grey
    return img


def _draw_row_line(
    *,
    cv2: Any,
    img: np.ndarray,
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    v_center: float,
    u_min: float,
    u_max: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    color_bgr: Tuple[int, int, int],
    thickness: int,
) -> None:
    p0 = _uv_to_xy(direction_xy, perp_xy, float(u_min), float(v_center))
    p1 = _uv_to_xy(direction_xy, perp_xy, float(u_max), float(v_center))
    x0, y0 = _xy_to_px(p0, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=img.shape[1], height=img.shape[0])
    x1, y1 = _xy_to_px(p1, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=img.shape[1], height=img.shape[0])
    cv2.line(img, (x0, y0), (x1, y1), color_bgr, int(thickness), lineType=cv2.LINE_AA)


def _draw_circles(
    *,
    cv2: Any,
    img: np.ndarray,
    circles: Sequence[Any],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    color_bgr: Tuple[int, int, int],
    radius_px: int,
    thickness: int,
) -> None:
    for c in circles:
        xy = np.array([float(c.x), float(c.y)], dtype=np.float32)
        x, y = _xy_to_px(xy, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=img.shape[1], height=img.shape[0])
        cv2.circle(img, (x, y), int(radius_px), color_bgr, int(thickness), lineType=cv2.LINE_AA)


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_out = ws_dir / "maps" / f"manual_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=str)
    parser.add_argument("--points-topic", default="/orchard_segmentation/tree_cloud", type=str)
    parser.add_argument("--tf-topic", default="/tf", type=str)
    parser.add_argument("--source-frame", default="lidar_link", type=str)
    parser.add_argument("--base-frame", default="base_link_est", type=str)
    parser.add_argument("--map-frame", default="map", type=str)
    parser.add_argument(
        "--missing-tf-policy",
        default="skip",
        choices=["skip", "hold", "first"],
        help="When TF lookup fails: skip frame; hold=use last known TF; first=use first available TF for early frames.",
    )

    parser.add_argument("--manual-json", required=True, type=str, help="Output of extract_manual_prior_lines.py")
    parser.add_argument(
        "--row-model-json",
        default="",
        type=str,
        help="Row model prior used for circle detection (if empty, uses manual rows).",
    )
    parser.add_argument("--out-dir", default=str(default_out), type=str)

    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--accumulate-frames", type=int, default=10)
    parser.add_argument("--bev-span", type=float, default=55.0, help="Crop span (meters) around robot.")
    parser.add_argument("--z-min", type=float, default=-1.0)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--max-fit-points", type=int, default=400000)
    parser.add_argument("--sample-seed", type=int, default=0)

    # Shared row gating.
    parser.add_argument("--row-bandwidth", type=float, default=1.4)
    parser.add_argument("--u-padding", type=float, default=0.0)

    # Traditional (cell clusters) params.
    parser.add_argument("--cell-size", type=float, default=0.12)
    parser.add_argument("--cell-neighbor-range", type=int, default=1)
    parser.add_argument("--cell-min-points", type=int, default=40)

    # Latest (peaks) params.
    parser.add_argument("--peaks-u-bin", type=float, default=0.03)
    parser.add_argument("--peaks-smooth-window", type=int, default=5)
    parser.add_argument("--peaks-peak-min-fraction", type=float, default=0.01)
    parser.add_argument("--peaks-min-separation", type=float, default=0.45)
    parser.add_argument("--peaks-refine-u-half-width", type=float, default=1.0)

    # Auto-tree (no-prior) params: cluster trunks -> tree centers -> fit two nearest rows.
    parser.add_argument("--auto-tree-max-points", type=int, default=80000, help="Max points used for auto-tree clustering.")
    parser.add_argument(
        "--auto-tree-z-min",
        type=float,
        default=0.7,
        help="Z-min (map frame) used for auto-tree clustering slice.",
    )
    parser.add_argument(
        "--auto-tree-z-max",
        type=float,
        default=1.3,
        help="Z-max (map frame) used for auto-tree clustering slice.",
    )
    parser.add_argument("--auto-tree-cell-size", type=float, default=0.15, help="Grid cell size (m) for auto-tree clustering.")
    parser.add_argument("--auto-tree-neighbor-range", type=int, default=1, help="Neighbor range for connected-components clustering.")
    parser.add_argument("--auto-tree-min-points", type=int, default=12, help="Min points per cluster to keep (auto-tree).")
    parser.add_argument("--auto-tree-max-clusters", type=int, default=0, help="Max clusters to keep (0=unlimited).")
    parser.add_argument(
        "--auto-tree-max-cluster-span",
        type=float,
        default=1.2,
        help="Drop clusters whose XY span exceeds this (m); 0 disables.",
    )
    parser.add_argument("--auto-tree-v-bin", type=float, default=0.25, help="Bin size (m) for auto-tree v histogram peaks.")
    parser.add_argument(
        "--auto-tree-assign-max-dist",
        type=float,
        default=1.6,
        help="Max |v - v_peak| (m) to assign a tree center to a row peak.",
    )

    # Circle RANSAC (latest).
    parser.add_argument("--latest-center-refine", type=str, default="circle_ransac", choices=["median", "circle_ransac"])
    parser.add_argument("--ransac-iters", type=int, default=300)
    parser.add_argument("--ransac-inlier-threshold", type=float, default=0.12)
    parser.add_argument("--ransac-min-inliers", type=int, default=20)
    parser.add_argument("--ransac-min-points", type=int, default=25)

    # Fit settings.
    parser.add_argument("--min-circles-per-row", type=int, default=3)

    # Local short-segment selection (for *_local methods).
    parser.add_argument("--local-u-window", type=float, default=0.0, help="Keep circles within |u-u_robot| < L (meters).")
    parser.add_argument("--local-k-nearest", type=int, default=0, help="Keep K nearest circles to u_robot (per row).")
    parser.add_argument(
        "--segment-max-gap",
        type=float,
        default=1.5,
        help="Max u-gap (m) between circles before splitting into segments (for *_segments).",
    )
    parser.add_argument(
        "--segment-min-circles",
        type=int,
        default=3,
        help="Min circles per segment to keep (for *_segments).",
    )
    parser.add_argument(
        "--segment-mode",
        type=str,
        default="gap",
        choices=["gap", "window", "window_adaptive", "circle"],
        help="Segmenting mode for *_segments: gap, window, window_adaptive, or circle.",
    )
    parser.add_argument(
        "--segment-window-length",
        type=float,
        default=4.0,
        help="Sliding window length (m) when --segment-mode=window.",
    )
    parser.add_argument(
        "--segment-window-stride",
        type=float,
        default=2.0,
        help="Sliding window stride (m) when --segment-mode=window.",
    )
    parser.add_argument(
        "--segment-window-trim",
        type=float,
        default=0.1,
        help="Trim quantile (0-0.49) for adaptive window segments.",
    )
    parser.add_argument(
        "--segment-min-span",
        type=float,
        default=0.6,
        help="Min segment length (m) for adaptive window segments.",
    )
    parser.add_argument(
        "--segment-circle-shrink",
        type=float,
        default=0.8,
        help="Length scale (0-1) for circle-centered segments.",
    )
    parser.add_argument(
        "--segment-circle-min-len",
        type=float,
        default=0.6,
        help="Min length (m) for circle-centered segments.",
    )
    parser.add_argument(
        "--segment-circle-max-len",
        type=float,
        default=3.0,
        help="Max length (m) for circle-centered segments.",
    )
    parser.add_argument(
        "--segment-circle-default-len",
        type=float,
        default=1.5,
        help="Default length (m) when neighbor spacing is unavailable.",
    )
    parser.add_argument(
        "--segment-circle-use-circle-v",
        action="store_true",
        help="Use each circle's v for its segment (otherwise use row v).",
    )

    # Points-only v estimation settings.
    parser.add_argument("--points-mad-k", type=float, default=2.5, help="Outlier cutoff multiplier for points_madmean.")
    parser.add_argument("--points-v-bin", type=float, default=0.05, help="Bin size (meters) for points_mode.")

    # Methods to evaluate.
    parser.add_argument(
        "--methods",
        type=str,
        default="cell_median,cell_ransac,peaks_median,peaks_ransac,points_median",
        help=(
            "Comma-separated method names. Available: "
            "cell_median,cell_ransac,cell_median_snap,cell_ransac_snap,"
            "peaks_median,peaks_ransac,peaks_median_snap,peaks_ransac_snap,"
            "points_median,points_mean,points_madmean,points_mode,"
            "points_pca_ransac,points_pca_ransac_local,"
            "auto_pca_median,auto_pca_median_local,"
            "auto_pca_ransac,auto_pca_ransac_local,"
            "auto_tree_median,auto_tree_ransac,auto_tree_ransac_segments"
        ),
    )

    # Rendering / plots.
    parser.add_argument("--render-dir", default="", type=str, help="If set, write per-frame BEV PNGs to this directory.")
    parser.add_argument("--render-every", default=0, type=int, help="Render every N evaluated frames (0 disables).")
    parser.add_argument("--render-size", default=900, type=int, help="Square render size in pixels.")
    parser.add_argument("--render-max-points", default=200000, type=int, help="Max points to draw per frame.")
    parser.add_argument(
        "--render-accumulate-frames",
        default=0,
        type=int,
        help="Accumulation window for renders (0 uses --accumulate-frames).",
    )
    parser.add_argument(
        "--render-z-min",
        default=None,
        type=float,
        help="Z-min for render background (defaults to --z-min).",
    )
    parser.add_argument(
        "--render-z-max",
        default=None,
        type=float,
        help="Z-max for render background (defaults to --z-max).",
    )
    parser.add_argument(
        "--render-point-darkness",
        default=220,
        type=int,
        help="0..255; larger draws point density darker (default=220).",
    )
    parser.add_argument("--render-line-thickness", default=3, type=int, help="Line thickness for row renders.")
    parser.add_argument(
        "--render-local-window",
        action="store_true",
        help="Render only a short line segment around the robot (uses --local-u-window).",
    )
    parser.add_argument(
        "--render-circles-from",
        default="",
        type=str,
        help=(
            "Overlay circles from a circle-based method on all renders. "
            "Use a method name (e.g. peaks_ransac), 'auto' to pick the first circle method, "
            "or 'all' to merge all circle methods."
        ),
    )
    parser.add_argument(
        "--render-no-manual",
        action="store_true",
        help="Do not draw manual (GT) lines in renders (helps visualize method segments).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plot PNGs.")

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    manual_path = Path(args.manual_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    if "rows_uv" not in manual or "direction_xy" not in manual or "perp_xy" not in manual:
        raise ValueError("manual-json must include direction_xy/perp_xy/rows_uv (run extractor with --row-model).")

    direction_xy = np.array(manual["direction_xy"], dtype=np.float32).reshape(2)
    perp_xy = np.array(manual["perp_xy"], dtype=np.float32).reshape(2)
    manual_rows = list(manual["rows_uv"])
    manual_rows.sort(key=lambda r: float(r["v_center"]))

    row_model_path: Optional[Path] = None
    if str(args.row_model_json).strip():
        row_model_path = Path(str(args.row_model_json)).expanduser().resolve()

    model_rows: List[Dict[str, Any]] = []
    model_dir = None
    model_perp = None
    if row_model_path is not None and row_model_path.is_file():
        model_rows, model_dir, model_perp = _load_rows_from_json(row_model_path)
    if not model_rows:
        model_rows = [dict(r) for r in manual_rows]

    if model_dir is not None and float(np.linalg.norm(model_dir - direction_xy)) > 0.05:
        print(f"[WARN] model direction_xy differs from manual by {float(np.linalg.norm(model_dir - direction_xy)):.3f}", file=sys.stderr)
    if model_perp is not None and float(np.linalg.norm(model_perp - perp_xy)) > 0.05:
        print(f"[WARN] model perp_xy differs from manual by {float(np.linalg.norm(model_perp - perp_xy)):.3f}", file=sys.stderr)

    impl = _load_tree_circles_impl()

    available_methods: Dict[str, _MethodSpec] = {
        "cell_median": _MethodSpec(
            name="cell_median",
            kind="cell",
            center_refine="median",
            snap_to_row=False,
            line_bgr=(0, 160, 0),
            circle_bgr=(0, 160, 0),
        ),
        "cell_ransac": _MethodSpec(
            name="cell_ransac",
            kind="cell",
            center_refine="circle_ransac",
            snap_to_row=False,
            line_bgr=(0, 110, 0),
            circle_bgr=(0, 110, 0),
        ),
        "cell_median_snap": _MethodSpec(
            name="cell_median_snap",
            kind="cell",
            center_refine="median",
            snap_to_row=True,
            line_bgr=(60, 200, 60),
            circle_bgr=(60, 200, 60),
        ),
        "cell_ransac_snap": _MethodSpec(
            name="cell_ransac_snap",
            kind="cell",
            center_refine="circle_ransac",
            snap_to_row=True,
            line_bgr=(40, 150, 40),
            circle_bgr=(40, 150, 40),
        ),
        "peaks_median": _MethodSpec(
            name="peaks_median",
            kind="peaks",
            center_refine="median",
            snap_to_row=False,
            line_bgr=(255, 200, 0),
            circle_bgr=(255, 0, 255),
        ),
        "peaks_ransac": _MethodSpec(
            name="peaks_ransac",
            kind="peaks",
            center_refine="circle_ransac",
            snap_to_row=False,
            line_bgr=(255, 150, 0),
            circle_bgr=(255, 0, 255),
        ),
        "peaks_ransac_segments": _MethodSpec(
            name="peaks_ransac_segments",
            kind="peaks",
            center_refine="circle_ransac",
            snap_to_row=False,
            line_bgr=(0, 0, 255),
            circle_bgr=(255, 0, 255),
        ),
        "peaks_ransac_local": _MethodSpec(
            name="peaks_ransac_local",
            kind="peaks",
            center_refine="circle_ransac",
            snap_to_row=False,
            line_bgr=(0, 120, 255),
            circle_bgr=(255, 0, 255),
        ),
        "peaks_median_snap": _MethodSpec(
            name="peaks_median_snap",
            kind="peaks",
            center_refine="median",
            snap_to_row=True,
            line_bgr=(255, 220, 40),
            circle_bgr=(255, 0, 255),
        ),
        "peaks_ransac_snap": _MethodSpec(
            name="peaks_ransac_snap",
            kind="peaks",
            center_refine="circle_ransac",
            snap_to_row=True,
            line_bgr=(255, 120, 40),
            circle_bgr=(255, 0, 255),
        ),
        "points_median": _MethodSpec(
            name="points_median",
            kind="points",
            center_refine="median",
            snap_to_row=False,
            line_bgr=(50, 50, 50),
            circle_bgr=(50, 50, 50),
        ),
        "points_mean": _MethodSpec(
            name="points_mean",
            kind="points",
            center_refine="mean",
            snap_to_row=False,
            line_bgr=(120, 120, 120),
            circle_bgr=(120, 120, 120),
        ),
        "points_madmean": _MethodSpec(
            name="points_madmean",
            kind="points",
            center_refine="madmean",
            snap_to_row=False,
            line_bgr=(128, 0, 128),
            circle_bgr=(128, 0, 128),
        ),
        "points_mode": _MethodSpec(
            name="points_mode",
            kind="points",
            center_refine="mode",
            snap_to_row=False,
            line_bgr=(255, 0, 0),
            circle_bgr=(255, 0, 0),
        ),
        "points_pca_ransac": _MethodSpec(
            name="points_pca_ransac",
            kind="points_pca",
            center_refine="pca_ransac",
            snap_to_row=False,
            line_bgr=(0, 80, 200),
            circle_bgr=(0, 80, 200),
        ),
        "points_pca_ransac_local": _MethodSpec(
            name="points_pca_ransac_local",
            kind="points_pca",
            center_refine="pca_ransac",
            snap_to_row=False,
            line_bgr=(0, 120, 200),
            circle_bgr=(0, 120, 200),
        ),
        "auto_pca_ransac": _MethodSpec(
            name="auto_pca_ransac",
            kind="auto_pca",
            center_refine="pca_ransac",
            snap_to_row=False,
            line_bgr=(200, 80, 0),
            circle_bgr=(200, 80, 0),
        ),
        "auto_pca_median": _MethodSpec(
            name="auto_pca_median",
            kind="auto_pca",
            center_refine="median",
            snap_to_row=False,
            line_bgr=(200, 0, 200),
            circle_bgr=(200, 0, 200),
        ),
        "auto_pca_ransac_local": _MethodSpec(
            name="auto_pca_ransac_local",
            kind="auto_pca",
            center_refine="pca_ransac",
            snap_to_row=False,
            line_bgr=(200, 120, 0),
            circle_bgr=(200, 120, 0),
        ),
        "auto_pca_median_local": _MethodSpec(
            name="auto_pca_median_local",
            kind="auto_pca",
            center_refine="median",
            snap_to_row=False,
            line_bgr=(200, 0, 255),
            circle_bgr=(200, 0, 255),
        ),
        "auto_tree_median": _MethodSpec(
            name="auto_tree_median",
            kind="auto_tree",
            center_refine="median",
            snap_to_row=False,
            line_bgr=(180, 0, 180),
            circle_bgr=(255, 0, 255),
        ),
        "auto_tree_ransac": _MethodSpec(
            name="auto_tree_ransac",
            kind="auto_tree",
            center_refine="pca_ransac",
            snap_to_row=False,
            line_bgr=(255, 0, 0),
            circle_bgr=(255, 0, 255),
        ),
        "auto_tree_ransac_segments": _MethodSpec(
            name="auto_tree_ransac_segments",
            kind="auto_tree",
            center_refine="pca_ransac",
            snap_to_row=False,
            line_bgr=(0, 165, 255),
            circle_bgr=(255, 0, 255),
        ),
    }
    selected_method_names = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    if not selected_method_names:
        raise ValueError("--methods must include at least one method name")
    methods: List[_MethodSpec] = []
    for name in selected_method_names:
        if name not in available_methods:
            raise ValueError(f"Unknown method {name!r}. Available: {', '.join(sorted(available_methods.keys()))}")
        methods.append(available_methods[name])

    # Optional: compute circles for rendering even if that circle method is not being evaluated.
    render_circles_spec: Optional[_MethodSpec] = None
    render_circles_key = str(args.render_circles_from).strip().lower()
    if render_circles_key and render_circles_key not in ("auto", "all"):
        maybe = available_methods.get(render_circles_key)
        if maybe is not None and maybe.kind in ("cell", "peaks") and all(m.name != maybe.name for m in methods):
            render_circles_spec = maybe

    # Configs
    ransac_cfg_enabled = impl.CircleRansacConfig(
        enabled=True,
        max_iterations=int(args.ransac_iters),
        inlier_threshold=float(args.ransac_inlier_threshold),
        min_inliers=int(args.ransac_min_inliers),
        min_points=int(args.ransac_min_points),
        use_inliers_for_radius=True,
        set_radius=False,
        seed=int(args.sample_seed),
    )
    ransac_cfg_disabled = impl.CircleRansacConfig(
        enabled=False,
        max_iterations=0,
        inlier_threshold=0.0,
        min_inliers=0,
        min_points=0,
        use_inliers_for_radius=True,
        set_radius=False,
        seed=int(args.sample_seed),
    )

    # Resolve time window once (used for optional TF prefill and evaluation).
    with rosbag.Bag(str(bag_path)) as _bag:
        start_time = _bag.get_start_time() + float(args.start_offset)
        end_time = _bag.get_end_time() if float(args.duration) <= 0.0 else start_time + float(args.duration)

    tf_buffer: Dict[Tuple[str, str], np.ndarray] = {}
    first_map_T_by_frame: Dict[str, np.ndarray] = {}
    last_map_T_by_frame: Dict[str, np.ndarray] = {}
    prefill_time: Optional[float] = None
    prefill_source_frame: Optional[str] = None

    if str(args.missing_tf_policy) == "first":
        # Prefill TF buffer with the first available transforms so we can evaluate early point clouds
        # even if the bag does not contain /tf yet.
        prefill_source_frame = str(args.source_frame)
        with rosbag.Bag(str(bag_path)) as _bag:
            for _topic, _pc_msg, _pc_t in _bag.read_messages(topics=[args.points_topic]):
                _t_sec = float(_pc_t.to_sec())
                if _t_sec < float(start_time):
                    continue
                if _t_sec > float(end_time):
                    break
                prefill_source_frame = str(getattr(_pc_msg.header, "frame_id", "")) or str(args.source_frame)
                break

            for _topic, _tf_msg, _tf_t in _bag.read_messages(topics=[args.tf_topic]):
                _t_sec = float(_tf_t.to_sec())
                if _t_sec > float(end_time):
                    break
                _update_tf_buffer(tf_buffer, _tf_msg)
                if _t_sec < float(start_time):
                    continue
                map_T_lidar0 = _lookup_transform(tf_buffer, str(args.map_frame), str(prefill_source_frame))
                if map_T_lidar0 is None:
                    continue
                map_T_base0 = _lookup_transform(tf_buffer, str(args.map_frame), str(args.base_frame))
                if map_T_base0 is None:
                    map_T_base0 = map_T_lidar0
                first_map_T_by_frame[str(prefill_source_frame)] = map_T_lidar0
                first_map_T_by_frame[str(args.base_frame)] = map_T_base0
                last_map_T_by_frame[str(prefill_source_frame)] = map_T_lidar0
                last_map_T_by_frame[str(args.base_frame)] = map_T_base0
                prefill_time = float(_t_sec)
                break

        if prefill_time is None:
            print(
                "[WARN] missing-tf-policy=first requested, but TF could not be prefetched; early frames may be skipped.",
                file=sys.stderr,
            )
        else:
            print(
                f"[INFO] Prefilled TF at t={prefill_time:.3f}s (source_frame={prefill_source_frame!r})",
                file=sys.stderr,
            )

    accum_fit: Deque[np.ndarray] = deque()
    accum_render: Deque[np.ndarray] = deque()

    render_z_min = float(args.z_min) if args.render_z_min is None else float(args.render_z_min)
    render_z_max = float(args.z_max) if args.render_z_max is None else float(args.render_z_max)
    render_accum_n = int(args.render_accumulate_frames) if int(args.render_accumulate_frames) > 0 else int(args.accumulate_frames)
    render_accum_n = int(max(1, render_accum_n))

    per_frame_rows: List[Dict[str, Any]] = []

    render_dir: Optional[Path] = Path(str(args.render_dir)).expanduser().resolve() if str(args.render_dir).strip() else None
    cv2 = _try_import_cv2() if render_dir is not None or not bool(args.no_plots) else None
    if render_dir is not None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for --render-dir but could not be imported.")
        for m in methods:
            (render_dir / m.name).mkdir(parents=True, exist_ok=True)

    with rosbag.Bag(str(bag_path)) as bag:
        next_time: Optional[float] = None
        frame_count = 0

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

            missing_policy = str(args.missing_tf_policy)
            source_frame = str(getattr(msg.header, "frame_id", "")) or str(args.source_frame)
            map_T_lidar = _lookup_transform(tf_buffer, str(args.map_frame), source_frame)
            tf_lidar_src = "tf"
            if map_T_lidar is None:
                if missing_policy == "hold":
                    map_T_lidar = last_map_T_by_frame.get(source_frame)
                    tf_lidar_src = "hold"
                elif missing_policy == "first":
                    map_T_lidar = first_map_T_by_frame.get(source_frame)
                    tf_lidar_src = "first"
                if map_T_lidar is None:
                    continue
            else:
                last_map_T_by_frame[source_frame] = map_T_lidar
                first_map_T_by_frame.setdefault(source_frame, map_T_lidar)
                if missing_policy == "first" and prefill_time is not None and float(t_sec) < float(prefill_time):
                    tf_lidar_src = "first"

            base_frame = str(args.base_frame)
            map_T_base = _lookup_transform(tf_buffer, str(args.map_frame), base_frame)
            tf_base_src = "tf"
            if map_T_base is None:
                if missing_policy == "hold":
                    map_T_base = last_map_T_by_frame.get(base_frame)
                    tf_base_src = "hold" if map_T_base is not None else tf_lidar_src
                elif missing_policy == "first":
                    map_T_base = first_map_T_by_frame.get(base_frame)
                    tf_base_src = "first" if map_T_base is not None else tf_lidar_src
                if map_T_base is None:
                    map_T_base = map_T_lidar
                    tf_base_src = tf_lidar_src
            else:
                last_map_T_by_frame[base_frame] = map_T_base
                first_map_T_by_frame.setdefault(base_frame, map_T_base)
                if missing_policy == "first" and prefill_time is not None and float(t_sec) < float(prefill_time):
                    tf_base_src = "first"

            pts = _points_from_cloud(msg)
            if pts.size == 0:
                continue

            pts_h = np.hstack([pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)])
            pts_map_all = (map_T_lidar @ pts_h.T).T[:, :3].astype(np.float32)
            z_mask_fit = (pts_map_all[:, 2] >= float(args.z_min)) & (pts_map_all[:, 2] <= float(args.z_max))
            pts_map_fit = pts_map_all[z_mask_fit]
            z_mask_render = (pts_map_all[:, 2] >= float(render_z_min)) & (pts_map_all[:, 2] <= float(render_z_max))
            pts_map_render = pts_map_all[z_mask_render]
            # Allow occasional sparse/empty frames after z-slicing: keep evaluating using accumulated history.
            # This avoids dropping frames during turns/occlusions where few points survive the z filter.
            if pts_map_fit.shape[0] > 0:
                accum_fit.append(pts_map_fit)
            while len(accum_fit) > int(max(1, args.accumulate_frames)):
                accum_fit.popleft()
            if pts_map_render.shape[0] > 0:
                accum_render.append(pts_map_render)
            while len(accum_render) > render_accum_n:
                accum_render.popleft()

            if not accum_fit:
                continue
            pts_accum = np.vstack(list(accum_fit))
            pts_accum_render = np.vstack(list(accum_render)) if accum_render else pts_accum

            robot_xy = map_T_base[:2, 3].astype(np.float32)
            half = 0.5 * float(args.bev_span)
            xmin, xmax = float(robot_xy[0] - half), float(robot_xy[0] + half)
            ymin, ymax = float(robot_xy[1] - half), float(robot_xy[1] + half)
            pts_xy = pts_accum[:, :2].astype(np.float32)
            mask = (
                (pts_xy[:, 0] >= xmin)
                & (pts_xy[:, 0] <= xmax)
                & (pts_xy[:, 1] >= ymin)
                & (pts_xy[:, 1] <= ymax)
            )
            pts_accum = pts_accum[mask]
            pts_xy_r = pts_accum_render[:, :2].astype(np.float32)
            mask_r = (
                (pts_xy_r[:, 0] >= xmin)
                & (pts_xy_r[:, 0] <= xmax)
                & (pts_xy_r[:, 1] >= ymin)
                & (pts_xy_r[:, 1] <= ymax)
            )
            pts_accum_render = pts_accum_render[mask_r]
            if pts_accum.shape[0] < 20:
                continue

            pts_fit = _sample_points(pts_accum, int(args.max_fit_points), int(args.sample_seed) + frame_count)
            if pts_fit.shape[0] < 20:
                continue

            v_robot = float(robot_xy.dot(perp_xy.astype(np.float32)))
            u_robot = float(robot_xy.dot(direction_xy.astype(np.float32)))
            manual_pair = _choose_row_pair(v_robot, manual_rows)
            model_pair = _choose_row_pair(v_robot, model_rows)
            if manual_pair is None or model_pair is None:
                continue
            m0, m1 = manual_pair
            r0, r1 = model_pair
            manual_eval = [manual_rows[m0], manual_rows[m1]]
            model_eval = [model_rows[r0], model_rows[r1]]

            yaw = _yaw_from_rot(map_T_base[:3, :3])
            frame_rec: Dict[str, Any] = {
                "frame": int(frame_count),
                "time": float(t_sec),
                "source_frame": str(source_frame),
                "tf_lidar": str(tf_lidar_src),
                "tf_base": str(tf_base_src),
                "robot_x": float(robot_xy[0]),
                "robot_y": float(robot_xy[1]),
                "robot_yaw": float(yaw),
                "v_robot": float(v_robot),
                "u_robot": float(u_robot),
                "manual_pair": [int(m0), int(m1)],
                "model_pair": [int(r0), int(r1)],
            }

            circles_by_method: Dict[str, List[Any]] = {m.name: [] for m in methods}
            if render_circles_spec is not None:
                circles_by_method.setdefault(render_circles_spec.name, [])
            segments_by_method: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = {m.name: {} for m in methods}
            auto_cache: Dict[str, Dict[int, Dict[str, float]]] = {}
            auto_tree_circles_cache: Dict[str, Dict[int, List[Any]]] = {}

            for slot, (manual_row, model_row) in enumerate(zip(manual_eval, model_eval)):
                v_manual = float(manual_row["v_center"])
                row_local = {
                    "v_center": float(model_row["v_center"]),
                    "u_min": float(model_row["u_min"]),
                    "u_max": float(model_row["u_max"]),
                    "z": float(model_row.get("z", 0.0)),
                }

                frame_rec[f"row{slot}_v_manual"] = float(v_manual)
                frame_rec[f"row{slot}_v_model"] = float(row_local["v_center"])
                frame_rec[f"row{slot}_u_min"] = float(row_local["u_min"])
                frame_rec[f"row{slot}_u_max"] = float(row_local["u_max"])
                frame_rec[f"row{slot}_u_min_manual"] = float(manual_row.get("u_min", row_local["u_min"]))
                frame_rec[f"row{slot}_u_max_manual"] = float(manual_row.get("u_max", row_local["u_max"]))

                # Per-method evaluation. Cache heavy computations per-row (same inputs, only config toggles).
                cell_cache: Dict[Tuple[bool, bool], Tuple[List[Any], np.ndarray]] = {}
                peaks_cache: Dict[Tuple[bool, bool], Tuple[List[Any], np.ndarray]] = {}

                xy_fit = pts_fit[:, :2].astype(np.float32)
                u_fit = xy_fit.dot(direction_xy.astype(np.float32).reshape(2))
                v_fit = xy_fit.dot(perp_xy.astype(np.float32).reshape(2))
                u_min = float(row_local["u_min"]) - float(args.u_padding)
                u_max = float(row_local["u_max"]) + float(args.u_padding)
                v_pred = float(row_local["v_center"]) * np.ones_like(u_fit, dtype=np.float32)
                keep = (u_fit >= float(u_min)) & (u_fit <= float(u_max))
                keep &= np.abs(v_fit - v_pred) <= float(args.row_bandwidth)
                points_vv = v_fit[keep].astype(np.float32)
                points_xy_keep = xy_fit[keep].astype(np.float32)
                points_u_keep = u_fit[keep].astype(np.float32)

                if render_circles_spec is not None:
                    circles_render: List[Any] = []
                    if render_circles_spec.kind == "cell":
                        ransac_on = bool(render_circles_spec.center_refine == "circle_ransac")
                        key = (bool(render_circles_spec.snap_to_row), bool(ransac_on))
                        if key not in cell_cache:
                            circles_cell, _ = impl._tree_circles_and_labels_from_row_model_cell_clusters(
                                points_xyz=pts_fit,
                                direction_xy=direction_xy,
                                perp_xy=perp_xy,
                                rows=[row_local],
                                row_bandwidth=float(args.row_bandwidth),
                                u_padding=float(args.u_padding),
                                cell_size=float(args.cell_size),
                                neighbor_range=int(args.cell_neighbor_range),
                                min_points=int(args.cell_min_points),
                                max_trees_per_row=0,
                                max_trees=0,
                                snap_to_row=bool(render_circles_spec.snap_to_row),
                                circle_ransac=ransac_cfg_enabled if ransac_on else ransac_cfg_disabled,
                                marker_z=0.0,
                                radius_mode="constant",
                                radius_constant=0.35,
                                radius_quantile=0.8,
                                radius_min=0.15,
                                radius_max=1.5,
                            )
                            v_vals = _circles_to_v(circles_cell, perp_xy)
                            cell_cache[key] = (list(circles_cell), v_vals)
                        circles_render = list(cell_cache[key][0])
                    elif render_circles_spec.kind == "peaks":
                        ransac_on = bool(render_circles_spec.center_refine == "circle_ransac")
                        key = (bool(render_circles_spec.snap_to_row), bool(ransac_on))
                        if key not in peaks_cache:
                            circles_peaks, _ = impl._tree_circles_and_labels_from_row_model(
                                points_xyz=pts_fit,
                                direction_xy=direction_xy,
                                perp_xy=perp_xy,
                                rows=[row_local],
                                row_bandwidth=float(args.row_bandwidth),
                                u_bin_size=float(args.peaks_u_bin),
                                smooth_window=int(args.peaks_smooth_window),
                                peak_min_fraction=float(args.peaks_peak_min_fraction),
                                min_separation=float(args.peaks_min_separation),
                                u_padding=float(args.u_padding),
                                refine_u_half_width=float(args.peaks_refine_u_half_width),
                                max_trees_per_row=0,
                                max_trees=0,
                                snap_to_row=bool(render_circles_spec.snap_to_row),
                                circle_ransac=ransac_cfg_enabled if ransac_on else ransac_cfg_disabled,
                                marker_z=0.0,
                                radius_mode="constant",
                                radius_constant=0.35,
                                radius_quantile=0.8,
                                radius_min=0.15,
                                radius_max=1.5,
                            )
                            v_vals = _circles_to_v(circles_peaks, perp_xy)
                            peaks_cache[key] = (list(circles_peaks), v_vals)
                        circles_render = list(peaks_cache[key][0])
                    if circles_render:
                        circles_by_method[render_circles_spec.name].extend(circles_render)

                for method in methods:
                    support_count = 0
                    v_est = float("nan")
                    circles: List[Any] = []
                    theta_err = float("nan")
                    use_local = str(method.name).endswith("_local")
                    use_segments = str(method.name).endswith("_segments")

                    if method.kind == "cell":
                        ransac_on = bool(method.center_refine == "circle_ransac")
                        key = (bool(method.snap_to_row), bool(ransac_on))
                        if key not in cell_cache:
                            circles_cell, _ = impl._tree_circles_and_labels_from_row_model_cell_clusters(
                                points_xyz=pts_fit,
                                direction_xy=direction_xy,
                                perp_xy=perp_xy,
                                rows=[row_local],
                                row_bandwidth=float(args.row_bandwidth),
                                u_padding=float(args.u_padding),
                                cell_size=float(args.cell_size),
                                neighbor_range=int(args.cell_neighbor_range),
                                min_points=int(args.cell_min_points),
                                max_trees_per_row=0,
                                max_trees=0,
                                snap_to_row=bool(method.snap_to_row),
                                circle_ransac=ransac_cfg_enabled if ransac_on else ransac_cfg_disabled,
                                marker_z=0.0,
                                radius_mode="constant",
                                radius_constant=0.35,
                                radius_quantile=0.8,
                                radius_min=0.15,
                                radius_max=1.5,
                            )
                            v_vals = _circles_to_v(circles_cell, perp_xy)
                            cell_cache[key] = (list(circles_cell), v_vals)
                        circles_full, v_vals_full = cell_cache[key]
                        circles = (
                            _filter_circles_local(
                                circles_full,
                                direction_xy,
                                u_robot,
                                float(args.local_u_window),
                                int(args.local_k_nearest),
                            )
                            if use_local
                            else list(circles_full)
                        )
                        v_vals = _circles_to_v(circles, perp_xy)
                        support_count = int(v_vals.size)
                        v_est = _median_or_nan(v_vals)
                        frame_rec[f"row{slot}_{method.name}_circles"] = int(v_vals.size)
                        if support_count >= int(args.min_circles_per_row):
                            circles_theta = circles_full if use_local else circles
                            theta_err = _angle_error_rad(_fit_direction_from_xy(_circles_to_xy(circles_theta)), direction_xy)
                    elif method.kind == "peaks":
                        ransac_on = bool(method.center_refine == "circle_ransac")
                        key = (bool(method.snap_to_row), bool(ransac_on))
                        if key not in peaks_cache:
                            circles_peaks, _ = impl._tree_circles_and_labels_from_row_model(
                                points_xyz=pts_fit,
                                direction_xy=direction_xy,
                                perp_xy=perp_xy,
                                rows=[row_local],
                                row_bandwidth=float(args.row_bandwidth),
                                u_bin_size=float(args.peaks_u_bin),
                                smooth_window=int(args.peaks_smooth_window),
                                peak_min_fraction=float(args.peaks_peak_min_fraction),
                                min_separation=float(args.peaks_min_separation),
                                u_padding=float(args.u_padding),
                                refine_u_half_width=float(args.peaks_refine_u_half_width),
                                max_trees_per_row=0,
                                max_trees=0,
                                snap_to_row=bool(method.snap_to_row),
                                circle_ransac=ransac_cfg_enabled if ransac_on else ransac_cfg_disabled,
                                marker_z=0.0,
                                radius_mode="constant",
                                radius_constant=0.35,
                                radius_quantile=0.8,
                                radius_min=0.15,
                                radius_max=1.5,
                            )
                            v_vals = _circles_to_v(circles_peaks, perp_xy)
                            peaks_cache[key] = (list(circles_peaks), v_vals)
                        circles_full, v_vals_full = peaks_cache[key]
                        circles = (
                            _filter_circles_local(
                                circles_full,
                                direction_xy,
                                u_robot,
                                float(args.local_u_window),
                                int(args.local_k_nearest),
                            )
                            if use_local
                            else list(circles_full)
                        )
                        if use_segments:
                            if str(args.segment_mode) == "circle":
                                v_row = float(row_local["v_center"])
                                if circles:
                                    v_vals = _circles_to_v(circles, perp_xy)
                                    if v_vals.size:
                                        v_row = float(np.median(v_vals))
                                segments = _segments_from_circles_circle(
                                    circles,
                                    direction_xy,
                                    perp_xy,
                                    v_row,
                                    float(args.segment_circle_shrink),
                                    float(args.segment_circle_min_len),
                                    float(args.segment_circle_max_len),
                                    float(args.segment_circle_default_len),
                                    bool(args.segment_circle_use_circle_v),
                                )
                            elif str(args.segment_mode) == "window_adaptive":
                                segments = _segments_from_circles_window_adaptive(
                                    circles,
                                    direction_xy,
                                    perp_xy,
                                    float(u_min),
                                    float(u_max),
                                    float(args.segment_window_length),
                                    float(args.segment_window_stride),
                                    int(args.segment_min_circles),
                                    float(args.segment_window_trim),
                                    float(args.segment_min_span),
                                )
                            elif str(args.segment_mode) == "window":
                                segments = _segments_from_circles_window(
                                    circles,
                                    direction_xy,
                                    perp_xy,
                                    float(u_min),
                                    float(u_max),
                                    float(args.segment_window_length),
                                    float(args.segment_window_stride),
                                    int(args.segment_min_circles),
                                )
                            else:
                                segments = _segments_from_circles(
                                    circles,
                                    direction_xy,
                                    perp_xy,
                                    float(args.segment_max_gap),
                                    int(args.segment_min_circles),
                                )
                            segments_by_method[method.name][slot] = [
                                (s["u_min"], s["u_max"], s["v_center"]) for s in segments
                            ]
                            if segments:
                                seg = min(segments, key=lambda s: abs(0.5 * (s["u_min"] + s["u_max"]) - u_robot))
                                v_est = float(seg["v_center"])
                                support_count = int(seg["count"])
                                frame_rec[f"row{slot}_{method.name}_segments"] = int(len(segments))
                                frame_rec[f"row{slot}_{method.name}_seg_u_min"] = float(seg["u_min"])
                                frame_rec[f"row{slot}_{method.name}_seg_u_max"] = float(seg["u_max"])
                                frame_rec[f"row{slot}_{method.name}_seg_len"] = float(seg["u_max"] - seg["u_min"])
                            else:
                                v_est = float("nan")
                                support_count = 0
                                frame_rec[f"row{slot}_{method.name}_segments"] = int(0)
                        else:
                            v_vals = _circles_to_v(circles, perp_xy)
                            support_count = int(v_vals.size)
                            v_est = _median_or_nan(v_vals)
                        frame_rec[f"row{slot}_{method.name}_circles"] = int(support_count)
                        if support_count >= int(args.min_circles_per_row):
                            circles_theta = circles_full if use_local else circles
                            theta_err = _angle_error_rad(_fit_direction_from_xy(_circles_to_xy(circles_theta)), direction_xy)
                    elif method.kind == "points":
                        vv = points_vv
                        support_count = int(vv.size)
                        frame_rec[f"row{slot}_{method.name}_points"] = int(vv.size)

                        if method.center_refine == "mean":
                            v_est = float(vv.mean()) if vv.size else float("nan")
                        elif method.center_refine == "madmean":
                            if vv.size == 0:
                                v_est = float("nan")
                                frame_rec[f"row{slot}_{method.name}_points_used"] = 0
                            else:
                                med = float(np.median(vv))
                                mad = float(np.median(np.abs(vv - med)))
                                if mad <= 1.0e-6:
                                    inliers = vv
                                else:
                                    inliers = vv[np.abs(vv - med) <= float(args.points_mad_k) * mad]
                                frame_rec[f"row{slot}_{method.name}_points_used"] = int(inliers.size)
                                v_est = float(inliers.mean()) if inliers.size else float("nan")
                        elif method.center_refine == "mode":
                            if vv.size == 0:
                                v_est = float("nan")
                                frame_rec[f"row{slot}_{method.name}_points_used"] = 0
                            else:
                                bin_size = max(1.0e-3, float(args.points_v_bin))
                                lo = float(vv.min())
                                hi = float(vv.max())
                                edges = np.arange(lo, hi + bin_size * 1.0001, bin_size, dtype=np.float32)
                                if edges.size < 2:
                                    frame_rec[f"row{slot}_{method.name}_points_used"] = int(vv.size)
                                    v_est = float(vv.mean())
                                else:
                                    hist = np.histogram(vv, bins=edges)[0]
                                    idx = int(np.argmax(hist))
                                    c0 = float(edges[idx])
                                    c1 = float(edges[idx + 1])
                                    inliers = vv[(vv >= c0) & (vv < c1)]
                                    frame_rec[f"row{slot}_{method.name}_points_used"] = int(inliers.size)
                                    v_est = float(inliers.mean()) if inliers.size else float(0.5 * (c0 + c1))
                        else:
                            v_est = _median_or_nan(vv)
                        if points_xy_keep.shape[0] >= 2:
                            theta_err = _angle_error_rad(_fit_direction_from_xy(points_xy_keep), direction_xy)
                    elif method.kind == "points_pca":
                        use_local = bool(use_local)
                        local_mask = (
                            _filter_points_local_mask(
                                points_u_keep,
                                u_robot,
                                float(args.local_u_window),
                                int(args.local_k_nearest),
                            )
                            if use_local
                            else np.ones((points_u_keep.size,), dtype=bool)
                        )
                        points_xy_sel = points_xy_keep[local_mask]
                        v_sel = points_vv[local_mask]
                        frame_rec[f"row{slot}_{method.name}_points"] = int(points_xy_sel.shape[0])
                        if points_xy_sel.shape[0] < 2:
                            support_count = 0
                            v_est = float("nan")
                            theta_err = float("nan")
                            frame_rec[f"row{slot}_{method.name}_points_used"] = 0
                        else:
                            mask_inliers, dir_est = _ransac_line_inliers(
                                points_xy_sel,
                                max_iters=int(args.ransac_iters),
                                inlier_threshold=float(args.ransac_inlier_threshold),
                                min_inliers=int(args.ransac_min_inliers),
                                seed=int(args.sample_seed) + int(frame_count) * 97 + int(slot),
                            )
                            if mask_inliers is None:
                                support_count = 0
                                v_est = float("nan")
                                theta_err = float("nan")
                                frame_rec[f"row{slot}_{method.name}_points_used"] = 0
                            else:
                                support_count = int(mask_inliers.sum())
                                v_est = _median_or_nan(v_sel[mask_inliers])
                                frame_rec[f"row{slot}_{method.name}_points_used"] = int(support_count)
                                if dir_est is None:
                                    dir_est = _fit_direction_from_xy(points_xy_sel[mask_inliers])
                                if support_count >= int(args.min_circles_per_row):
                                    theta_err = _angle_error_rad(dir_est, direction_xy)
                    elif method.kind == "auto_pca":
                        if method.name not in auto_cache:
                            xy_all = pts_fit[:, :2].astype(np.float32)
                            xy_local_all = xy_all - robot_xy.reshape(1, 2)
                            dir_est = _fit_direction_from_xy(xy_local_all)
                            if dir_est is None:
                                # No prior fallback: use robot heading from TF.
                                dir_est = np.array([float(math.cos(yaw)), float(math.sin(yaw))], dtype=np.float32)
                            perp_est = np.array([-float(dir_est[1]), float(dir_est[0])], dtype=np.float32)

                            u_all = xy_local_all.dot(dir_est.astype(np.float32).reshape(2))
                            v_all = xy_local_all.dot(perp_est.astype(np.float32).reshape(2))
                            if bool(use_local) and float(args.local_u_window) > 0.0:
                                mask_all = np.abs(u_all) <= float(args.local_u_window)
                                u_all = u_all[mask_all]
                                v_all = v_all[mask_all]
                                xy_local_all = xy_local_all[mask_all]

                            auto_rows: List[Dict[str, float]] = []
                            if v_all.size >= 2:
                                bin_size = max(1.0e-3, float(args.points_v_bin))
                                v_min = float(v_all.min())
                                v_max = float(v_all.max())
                                bins = np.arange(v_min, v_max + bin_size * 1.0001, bin_size, dtype=np.float32)
                                if bins.size >= 2:
                                    hist, edges = np.histogram(v_all, bins=bins)
                                    hist = hist.astype(np.float32)
                                    hist_smooth = _smooth_1d(hist, int(max(1, args.peaks_smooth_window)))
                                    peak_min = float(hist_smooth.max()) * float(args.peaks_peak_min_fraction)
                                    min_sep_bins = int(max(1.0, float(args.peaks_min_separation) / bin_size))
                                    peak_idx = _find_peak_indices(hist_smooth, peak_min, min_sep_bins)
                                    centers = 0.5 * (edges[:-1] + edges[1:])
                                    neg_peaks = [float(centers[i]) for i in peak_idx if float(centers[i]) < 0.0]
                                    pos_peaks = [float(centers[i]) for i in peak_idx if float(centers[i]) > 0.0]
                                    v_neg = max(neg_peaks, key=lambda v: v) if neg_peaks else float("nan")
                                    v_pos = min(pos_peaks, key=lambda v: abs(v)) if pos_peaks else float("nan")
                                else:
                                    v_neg = float("nan")
                                    v_pos = float("nan")
                                if not math.isfinite(v_neg):
                                    neg_vals = v_all[v_all < 0.0]
                                    if neg_vals.size:
                                        v_neg = float(np.median(neg_vals))
                                if not math.isfinite(v_pos):
                                    pos_vals = v_all[v_all > 0.0]
                                    if pos_vals.size:
                                        v_pos = float(np.median(pos_vals))

                                for v_center_local in (v_neg, v_pos):
                                    if not math.isfinite(v_center_local):
                                        continue
                                    band = np.abs(v_all - float(v_center_local)) <= float(args.row_bandwidth)
                                    if int(band.sum()) < 2:
                                        continue
                                    xy_row = xy_local_all[band]
                                    v_row = v_all[band]
                                    mask_inliers = None
                                    dir_row = None
                                    if str(method.center_refine) == "pca_ransac":
                                        mask_inliers, dir_row = _ransac_line_inliers(
                                            xy_row,
                                            max_iters=int(args.ransac_iters),
                                            inlier_threshold=float(args.ransac_inlier_threshold),
                                            min_inliers=int(args.ransac_min_inliers),
                                            seed=int(args.sample_seed) + int(frame_count) * 131,
                                        )
                                        if mask_inliers is None or int(mask_inliers.sum()) == 0:
                                            continue
                                        v_est_local = float(np.median(v_row[mask_inliers]))
                                    else:
                                        # Median baseline: no RANSAC, just robust v-estimate within the band.
                                        v_est_local = float(np.median(v_row))
                                    # Convert the local row offset (along perp_est) back into manual v-coordinate,
                                    # so the error metric (v_est - v_manual) stays consistent.
                                    center_xy = (robot_xy.reshape(2) + perp_est.reshape(2) * float(v_est_local)).astype(np.float32)
                                    v_est_global = float(np.dot(center_xy, perp_xy.astype(np.float32)))
                                    support = int(mask_inliers.sum()) if mask_inliers is not None else int(band.sum())
                                    if dir_row is None:
                                        dir_row = _fit_direction_from_xy(xy_row)
                                        if dir_row is None:
                                            dir_row = dir_est
                                    theta = (
                                        _angle_error_rad(dir_row, direction_xy)
                                        if support >= int(args.min_circles_per_row)
                                        else float("nan")
                                    )
                                    auto_rows.append(
                                        {
                                            "v_est": v_est_global,
                                            "support": float(support),
                                            "points_used": float(support),
                                            "theta_err": float(theta),
                                        }
                                    )
                            auto_rows.sort(key=lambda r: float(r["v_est"]))
                            auto_cache[method.name] = {}
                            for idx, row in enumerate(auto_rows[:2]):
                                auto_cache[method.name][idx] = row

                        row = auto_cache.get(method.name, {}).get(slot)
                        if row is None:
                            support_count = 0
                            v_est = float("nan")
                            theta_err = float("nan")
                            frame_rec[f"row{slot}_{method.name}_points_used"] = 0
                        else:
                            support_count = int(row.get("support", 0))
                            v_est = float(row.get("v_est", float("nan")))
                            theta_err = float(row.get("theta_err", float("nan")))
                            frame_rec[f"row{slot}_{method.name}_points_used"] = int(row.get("points_used", support_count))
                        frame_rec[f"row{slot}_{method.name}_points"] = int(
                            row.get("points_used", support_count) if row is not None else 0
                        )
                    elif method.kind == "auto_tree":
                        if method.name not in auto_cache:
                            pts_cluster = _sample_points(
                                pts_fit,
                                int(args.auto_tree_max_points),
                                int(args.sample_seed) + int(frame_count) * 29,
                            )
                            z_mask_tree = (pts_cluster[:, 2] >= float(args.auto_tree_z_min)) & (
                                pts_cluster[:, 2] <= float(args.auto_tree_z_max)
                            )
                            pts_cluster = pts_cluster[z_mask_tree]
                            xy_all = pts_cluster[:, :2].astype(np.float32)
                            xy_local_all = xy_all - robot_xy.reshape(1, 2)
                            dir_est = _fit_direction_from_xy(xy_local_all)
                            if dir_est is None:
                                dir_est = np.array([float(math.cos(yaw)), float(math.sin(yaw))], dtype=np.float32)
                            perp_est = np.array([-float(dir_est[1]), float(dir_est[0])], dtype=np.float32)

                            if bool(use_local) and float(args.local_u_window) > 0.0 and xy_local_all.shape[0] > 0:
                                u_local = xy_local_all.dot(dir_est.astype(np.float32).reshape(2))
                                xy_local_all = xy_local_all[np.abs(u_local) <= float(args.local_u_window)]

                            centers_local = _cluster_centers_from_points_cells(
                                points_xy=xy_local_all,
                                impl=impl,
                                cell_size=float(args.auto_tree_cell_size),
                                neighbor_range=int(args.auto_tree_neighbor_range),
                                min_points=int(args.auto_tree_min_points),
                                max_clusters=int(args.auto_tree_max_clusters),
                                max_cluster_span=float(args.auto_tree_max_cluster_span),
                            )

                            circles_all: List[Any] = []
                            if centers_local.shape[0] > 0:
                                circles_all = [
                                    impl.TreeCircle(
                                        x=float(robot_xy[0] + float(c[0])),
                                        y=float(robot_xy[1] + float(c[1])),
                                        z=0.0,
                                        radius=0.35,
                                    )
                                    for c in centers_local
                                ]

                            auto_rows: List[Dict[str, float]] = []
                            auto_circles: List[List[Any]] = []
                            if centers_local.shape[0] >= 2:
                                u_cent = centers_local.dot(dir_est.astype(np.float32).reshape(2))
                                v_cent = centers_local.dot(perp_est.astype(np.float32).reshape(2))
                                peaks = _pick_two_v_peaks(
                                    v_cent,
                                    bin_size=float(args.auto_tree_v_bin),
                                    smooth_window=int(args.peaks_smooth_window),
                                    peak_min_fraction=float(args.peaks_peak_min_fraction),
                                    min_separation=float(args.peaks_min_separation),
                                )
                                if len(peaks) < 2:
                                    neg_vals = v_cent[v_cent < 0.0]
                                    pos_vals = v_cent[v_cent > 0.0]
                                    if neg_vals.size:
                                        peaks.append(float(np.median(neg_vals)))
                                    if len(peaks) < 2 and pos_vals.size:
                                        peaks.append(float(np.median(pos_vals)))
                                peaks = peaks[:2]

                                if peaks:
                                    groups_uv: List[List[Tuple[np.ndarray, float, float, Any]]] = [
                                        [] for _ in range(len(peaks))
                                    ]
                                    assign_max = float(args.auto_tree_assign_max_dist)
                                    for c_local, u, v, circle in zip(centers_local, u_cent, v_cent, circles_all):
                                        diffs = [abs(float(v) - float(p)) for p in peaks]
                                        idx = int(np.argmin(np.array(diffs, dtype=np.float32)))
                                        if assign_max > 0.0 and float(diffs[idx]) > assign_max:
                                            continue
                                        groups_uv[idx].append((c_local, float(u), float(v), circle))

                                    for group in groups_uv:
                                        if len(group) < 2:
                                            continue
                                        xy_row = np.vstack([g[0] for g in group]).astype(np.float32)
                                        v_row = np.array([g[2] for g in group], dtype=np.float32)
                                        circles_row = [g[3] for g in group]

                                        mask_inliers = None
                                        dir_row = None
                                        if str(method.center_refine) == "pca_ransac":
                                            mask_inliers, dir_row = _ransac_line_inliers(
                                                xy_row,
                                                max_iters=int(args.ransac_iters),
                                                inlier_threshold=float(args.ransac_inlier_threshold),
                                                min_inliers=int(args.ransac_min_inliers),
                                                seed=int(args.sample_seed) + int(frame_count) * 131 + 17,
                                            )
                                            if mask_inliers is None or int(mask_inliers.sum()) == 0:
                                                continue
                                            v_est_local = float(np.median(v_row[mask_inliers]))
                                            circles_row = [
                                                circle for circle, keep in zip(circles_row, mask_inliers) if bool(keep)
                                            ]
                                            xy_in = xy_row[mask_inliers]
                                            if dir_row is None:
                                                dir_row = _fit_direction_from_xy(xy_in)
                                        else:
                                            v_est_local = float(np.median(v_row))
                                            xy_in = xy_row

                                        if dir_row is None:
                                            dir_row = _fit_direction_from_xy(xy_in)
                                        if dir_row is None:
                                            dir_row = dir_est

                                        center_xy = (
                                            robot_xy.reshape(2) + perp_est.reshape(2) * float(v_est_local)
                                        ).astype(np.float32)
                                        v_est_global = float(np.dot(center_xy, perp_xy.astype(np.float32)))
                                        support = int(len(circles_row))
                                        theta = (
                                            _angle_error_rad(dir_row, direction_xy)
                                            if support >= int(args.min_circles_per_row)
                                            else float("nan")
                                        )
                                        auto_rows.append(
                                            {
                                                "v_est": float(v_est_global),
                                                "support": float(support),
                                                "theta_err": float(theta),
                                            }
                                        )
                                        auto_circles.append(list(circles_row))

                            order = list(range(len(auto_rows)))
                            order.sort(key=lambda i: float(auto_rows[i]["v_est"]) if i < len(auto_rows) else 0.0)
                            auto_cache[method.name] = {}
                            auto_tree_circles_cache[method.name] = {}
                            for out_idx, src_idx in enumerate(order[:2]):
                                auto_cache[method.name][out_idx] = auto_rows[src_idx]
                                auto_tree_circles_cache[method.name][out_idx] = auto_circles[src_idx]

                            if use_segments and method.name in auto_tree_circles_cache:
                                for out_idx in (0, 1):
                                    circles_seg = auto_tree_circles_cache[method.name].get(out_idx, [])
                                    if not circles_seg:
                                        continue
                                    u_vals_seg = np.dot(_circles_to_xy(circles_seg), direction_xy.astype(np.float32).reshape(2))
                                    u_min_seg = float(u_vals_seg.min()) if u_vals_seg.size else float(0.0)
                                    u_max_seg = float(u_vals_seg.max()) if u_vals_seg.size else float(0.0)
                                    if str(args.segment_mode) == "circle":
                                        v_vals = _circles_to_v(circles_seg, perp_xy)
                                        v_row = float(np.median(v_vals)) if v_vals.size else float("nan")
                                        segments = _segments_from_circles_circle(
                                            circles_seg,
                                            direction_xy,
                                            perp_xy,
                                            float(v_row),
                                            float(args.segment_circle_shrink),
                                            float(args.segment_circle_min_len),
                                            float(args.segment_circle_max_len),
                                            float(args.segment_circle_default_len),
                                            bool(args.segment_circle_use_circle_v),
                                        )
                                    elif str(args.segment_mode) == "window_adaptive":
                                        segments = _segments_from_circles_window_adaptive(
                                            circles_seg,
                                            direction_xy,
                                            perp_xy,
                                            float(u_min_seg),
                                            float(u_max_seg),
                                            float(args.segment_window_length),
                                            float(args.segment_window_stride),
                                            int(args.segment_min_circles),
                                            float(args.segment_window_trim),
                                            float(args.segment_min_span),
                                        )
                                    elif str(args.segment_mode) == "window":
                                        segments = _segments_from_circles_window(
                                            circles_seg,
                                            direction_xy,
                                            perp_xy,
                                            float(u_min_seg),
                                            float(u_max_seg),
                                            float(args.segment_window_length),
                                            float(args.segment_window_stride),
                                            int(args.segment_min_circles),
                                        )
                                    else:
                                        segments = _segments_from_circles(
                                            circles_seg,
                                            direction_xy,
                                            perp_xy,
                                            float(args.segment_max_gap),
                                            int(args.segment_min_circles),
                                        )
                                    segments_by_method[method.name][out_idx] = [
                                        (s["u_min"], s["u_max"], s["v_center"]) for s in segments
                                    ]

                        row = auto_cache.get(method.name, {}).get(slot)
                        circles = list(auto_tree_circles_cache.get(method.name, {}).get(slot, []))
                        if row is None:
                            support_count = 0
                            v_est = float("nan")
                            theta_err = float("nan")
                        else:
                            support_count = int(row.get("support", 0))
                            v_est = float(row.get("v_est", float("nan")))
                            theta_err = float(row.get("theta_err", float("nan")))
                        frame_rec[f"row{slot}_{method.name}_circles"] = int(support_count)
                    else:
                        raise RuntimeError(f"Unsupported method kind: {method.kind}")

                    if method.kind in ("cell", "peaks", "auto_tree"):
                        circles_by_method[method.name].extend(list(circles))

                    ok = (support_count >= int(args.min_circles_per_row)) and math.isfinite(float(v_est))
                    frame_rec[f"row{slot}_{method.name}_v_est"] = float(v_est) if math.isfinite(float(v_est)) else float("nan")
                    frame_rec[f"row{slot}_{method.name}_v_err"] = float(v_est - v_manual) if ok else float("nan")
                    frame_rec[f"row{slot}_{method.name}_theta_err"] = float(theta_err) if math.isfinite(float(theta_err)) else float("nan")

            if render_dir is not None and int(args.render_every) > 0 and (frame_count % int(args.render_every) == 0):
                width = int(args.render_size)
                height = int(args.render_size)
                pts_xy_accum = pts_accum_render[:, :2].astype(np.float32)
                base_img = _render_density_bev(
                    cv2=cv2,
                    points_xy=pts_xy_accum,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    width=width,
                    height=height,
                    max_points=int(args.render_max_points),
                    seed=int(args.sample_seed) + frame_count,
                    point_darkness=int(args.render_point_darkness),
                )
                # Robot marker.
                rx, ry = _xy_to_px(robot_xy, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=width, height=height)
                cv2.circle(base_img, (rx, ry), 10, (0, 140, 255), -1, lineType=cv2.LINE_AA)

                use_local_window = bool(args.render_local_window) and float(args.local_u_window) > 0.0
                u_center = float(frame_rec.get("u_robot", 0.0))
                u_half = float(args.local_u_window)

                def _local_u_range(u_min_val: float, u_max_val: float) -> Tuple[float, float]:
                    if not use_local_window:
                        return float(u_min_val), float(u_max_val)
                    u0 = max(float(u_min_val), u_center - u_half)
                    u1 = min(float(u_max_val), u_center + u_half)
                    if u1 <= u0 + 1.0e-6:
                        return float(u_min_val), float(u_max_val)
                    return float(u0), float(u1)

                # Manual (GT) lines: red.
                if not bool(args.render_no_manual):
                    for slot in (0, 1):
                        u_min_gt, u_max_gt = _local_u_range(
                            float(frame_rec.get(f"row{slot}_u_min_manual", frame_rec[f"row{slot}_u_min"])),
                            float(frame_rec.get(f"row{slot}_u_max_manual", frame_rec[f"row{slot}_u_max"])),
                        )
                        _draw_row_line(
                            cv2=cv2,
                            img=base_img,
                            direction_xy=direction_xy,
                            perp_xy=perp_xy,
                            v_center=float(frame_rec[f"row{slot}_v_manual"]),
                            u_min=u_min_gt,
                            u_max=u_max_gt,
                            xmin=xmin,
                            xmax=xmax,
                            ymin=ymin,
                            ymax=ymax,
                            color_bgr=(0, 0, 255),
                            thickness=int(args.render_line_thickness),
                        )

                circles_overlay: List[Any] = []
                circles_overlay_color = (255, 0, 255)
                if str(args.render_circles_from).strip():
                    source = str(args.render_circles_from).strip().lower()
                    if source == "auto":
                        for m in methods:
                            if m.kind in ("cell", "peaks", "auto_tree") and circles_by_method.get(m.name):
                                circles_overlay = list(circles_by_method.get(m.name, []))
                                circles_overlay_color = m.circle_bgr
                                break
                    elif source == "all":
                        for m in methods:
                            if m.kind in ("cell", "peaks", "auto_tree"):
                                circles_overlay.extend(list(circles_by_method.get(m.name, [])))
                    else:
                        circles_overlay = list(circles_by_method.get(source, []))
                        for m in methods:
                            if m.name == source:
                                circles_overlay_color = m.circle_bgr
                                break

                for method in methods:
                    img_m = base_img.copy()
                    for slot in (0, 1):
                        v_est = float(frame_rec.get(f"row{slot}_{method.name}_v_est", float("nan")))
                        segs = segments_by_method.get(method.name, {}).get(slot)
                        if segs:
                            for u_min_seg, u_max_seg, v_seg in segs:
                                u_min_est, u_max_est = _local_u_range(float(u_min_seg), float(u_max_seg))
                                _draw_row_line(
                                    cv2=cv2,
                                    img=img_m,
                                    direction_xy=direction_xy,
                                    perp_xy=perp_xy,
                                    v_center=float(v_seg),
                                    u_min=u_min_est,
                                    u_max=u_max_est,
                                    xmin=xmin,
                                    xmax=xmax,
                                    ymin=ymin,
                                    ymax=ymax,
                                    color_bgr=method.line_bgr,
                                    thickness=int(args.render_line_thickness),
                                )
                        elif math.isfinite(v_est):
                            u_min_est, u_max_est = _local_u_range(
                                float(frame_rec[f"row{slot}_u_min"]),
                                float(frame_rec[f"row{slot}_u_max"]),
                            )
                            _draw_row_line(
                                cv2=cv2,
                                img=img_m,
                                direction_xy=direction_xy,
                                perp_xy=perp_xy,
                                v_center=v_est,
                                u_min=u_min_est,
                                u_max=u_max_est,
                                xmin=xmin,
                                xmax=xmax,
                                ymin=ymin,
                                ymax=ymax,
                                color_bgr=method.line_bgr,
                                thickness=int(args.render_line_thickness),
                            )
                    if circles_overlay:
                        _draw_circles(
                            cv2=cv2,
                            img=img_m,
                            circles=circles_overlay,
                            xmin=xmin,
                            xmax=xmax,
                            ymin=ymin,
                            ymax=ymax,
                            color_bgr=circles_overlay_color,
                            radius_px=7,
                            thickness=2,
                        )
                    elif method.kind in ("cell", "peaks", "auto_tree"):
                        _draw_circles(
                            cv2=cv2,
                            img=img_m,
                            circles=circles_by_method.get(method.name, []),
                            xmin=xmin,
                            xmax=xmax,
                            ymin=ymin,
                            ymax=ymax,
                            color_bgr=method.circle_bgr,
                            radius_px=7,
                            thickness=2,
                        )
                    cv2.putText(
                        img_m,
                        f"{method.name}  frame={frame_count}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.imwrite(str(render_dir / method.name / f"frame_{frame_count:04d}.png"), img_m)

            per_frame_rows.append(frame_rec)
            frame_count += 1
            if int(args.max_frames) > 0 and frame_count >= int(args.max_frames):
                break

    if not per_frame_rows:
        raise RuntimeError("No frames evaluated. Check tf/topic names and time range.")

    # Export per-frame CSV
    csv_path = out_dir / "per_frame.csv"
    keys: List[str] = sorted({k for row in per_frame_rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in per_frame_rows:
            w.writerow([row.get(k, "") for k in keys])

    # Summary stats: abs(v_err)
    def collect_err(prefix: str) -> np.ndarray:
        vals: List[float] = []
        for row in per_frame_rows:
            for k, v in row.items():
                if k.endswith(f"_{prefix}_v_err") and isinstance(v, (int, float)) and math.isfinite(float(v)):
                    vals.append(abs(float(v)))
        return np.asarray(vals, dtype=np.float64)

    def collect_theta(prefix: str) -> np.ndarray:
        vals: List[float] = []
        for row in per_frame_rows:
            for k, v in row.items():
                if k.endswith(f"_{prefix}_theta_err") and isinstance(v, (int, float)) and math.isfinite(float(v)):
                    vals.append(abs(float(v)))
        return np.asarray(vals, dtype=np.float64)

    method_summaries: Dict[str, Any] = {}
    for method in methods:
        prefix = method.name
        errs = collect_err(prefix)
        theta_errs = collect_theta(prefix)
        theta_errs_deg = np.rad2deg(theta_errs) if theta_errs.size else np.asarray([], dtype=np.float64)

        frames_total = int(len(per_frame_rows))
        frames_any = 0
        frames_both = 0
        rows_valid = 0
        for row in per_frame_rows:
            e0 = row.get(f"row0_{prefix}_v_err", float("nan"))
            e1 = row.get(f"row1_{prefix}_v_err", float("nan"))
            v0 = float(e0) if isinstance(e0, (int, float)) else float("nan")
            v1 = float(e1) if isinstance(e1, (int, float)) else float("nan")
            good0 = math.isfinite(v0)
            good1 = math.isfinite(v1)
            frames_any += int(good0 or good1)
            frames_both += int(good0 and good1)
            rows_valid += int(good0) + int(good1)

        method_summaries[prefix] = {
            "abs_v_err_m": _stats(errs),
            "abs_theta_err_rad": _stats(theta_errs),
            "abs_theta_err_deg": _stats(theta_errs_deg),
            "valid": {
                "frames_total": frames_total,
                "frames_any_row": int(frames_any),
                "frames_both_rows": int(frames_both),
                "rows_valid": int(rows_valid),
            },
            "spec": {
                "kind": method.kind,
                "center_refine": method.center_refine,
                "snap_to_row": bool(method.snap_to_row),
            },
        }

    summary = {
        "bag": str(bag_path),
        "points_topic": str(args.points_topic),
        "manual_json": str(manual_path),
        "row_model_json": str(row_model_path) if row_model_path is not None else "",
        "frames": float(len(per_frame_rows)),
        "min_circles_per_row": int(args.min_circles_per_row),
        # Back-compat keys (when present)
        "cell_abs_v_err_m": _stats(collect_err("cell_median")) if "cell_median" in selected_method_names else _stats(np.asarray([], dtype=np.float64)),
        "peaks_abs_v_err_m": _stats(collect_err("peaks_ransac")) if "peaks_ransac" in selected_method_names else _stats(np.asarray([], dtype=np.float64)),
        "cell_abs_theta_err_rad": _stats(collect_theta("cell_median")) if "cell_median" in selected_method_names else _stats(np.asarray([], dtype=np.float64)),
        "peaks_abs_theta_err_rad": _stats(collect_theta("peaks_ransac")) if "peaks_ransac" in selected_method_names else _stats(np.asarray([], dtype=np.float64)),
        "cell_abs_theta_err_deg": _stats(np.rad2deg(collect_theta("cell_median"))) if "cell_median" in selected_method_names else _stats(np.asarray([], dtype=np.float64)),
        "peaks_abs_theta_err_deg": _stats(np.rad2deg(collect_theta("peaks_ransac"))) if "peaks_ransac" in selected_method_names else _stats(np.asarray([], dtype=np.float64)),
        "methods": [m.name for m in methods],
        "method_results": method_summaries,
        "params": {
            "sample_rate": float(args.sample_rate),
            "start_offset": float(args.start_offset),
            "duration": float(args.duration),
            "tf": {
                "missing_tf_policy": str(args.missing_tf_policy),
                "prefill_time": float(prefill_time) if prefill_time is not None else float("nan"),
                "prefill_source_frame": str(prefill_source_frame) if prefill_source_frame is not None else "",
            },
            "accumulate_frames": int(args.accumulate_frames),
            "bev_span": float(args.bev_span),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "row_bandwidth": float(args.row_bandwidth),
            "u_padding": float(args.u_padding),
            "cell": {"cell_size": float(args.cell_size), "neighbor_range": int(args.cell_neighbor_range), "min_points": int(args.cell_min_points)},
            "peaks": {
                "u_bin": float(args.peaks_u_bin),
                "smooth_window": int(args.peaks_smooth_window),
                "peak_min_fraction": float(args.peaks_peak_min_fraction),
                "min_separation": float(args.peaks_min_separation),
                "refine_u_half_width": float(args.peaks_refine_u_half_width),
                "center_refine": str(args.latest_center_refine),
                "ransac": asdict(ransac_cfg_enabled),
            },
            "points": {"mad_k": float(args.points_mad_k), "v_bin": float(args.points_v_bin)},
            "local": {
                "u_window": float(args.local_u_window),
                "k_nearest": int(args.local_k_nearest),
            },
            "segments": {
                "max_gap": float(args.segment_max_gap),
                "min_circles": int(args.segment_min_circles),
                "mode": str(args.segment_mode),
                "window_length": float(args.segment_window_length),
                "window_stride": float(args.segment_window_stride),
                "window_trim": float(args.segment_window_trim),
                "min_span": float(args.segment_min_span),
                "circle_shrink": float(args.segment_circle_shrink),
                "circle_min_len": float(args.segment_circle_min_len),
                "circle_max_len": float(args.segment_circle_max_len),
                "circle_default_len": float(args.segment_circle_default_len),
                "circle_use_circle_v": bool(args.segment_circle_use_circle_v),
            },
            "render": {
                "enabled": bool(render_dir is not None),
                "render_dir": str(render_dir) if render_dir is not None else "",
                "render_every": int(args.render_every),
                "render_size": int(args.render_size),
                "render_max_points": int(args.render_max_points),
                "render_line_thickness": int(args.render_line_thickness),
                "render_local_window": bool(args.render_local_window),
                "render_circles_from": str(args.render_circles_from),
                "render_no_manual": bool(args.render_no_manual),
            },
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Plot PNGs (no matplotlib needed).
    if not bool(args.no_plots):
        if cv2 is None:
            print("[WARN] cv2 not available; skipping plots.", file=sys.stderr)
        else:
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            def per_frame_abs(prefix: str) -> np.ndarray:
                vals: List[float] = []
                for row in per_frame_rows:
                    e0 = row.get(f"row0_{prefix}_v_err", float("nan"))
                    e1 = row.get(f"row1_{prefix}_v_err", float("nan"))
                    frame_vals = [abs(float(e)) for e in (e0, e1) if isinstance(e, (int, float)) and math.isfinite(float(e))]
                    vals.append(float(np.mean(frame_vals)) if frame_vals else float("nan"))
                return np.asarray(vals, dtype=np.float32)

            per_method_pf: Dict[str, np.ndarray] = {m.name: per_frame_abs(m.name) for m in methods}
            x = np.arange(next(iter(per_method_pf.values())).size, dtype=np.int32)

            # Time series plot.
            w, h = 1200, 520
            margin = 60
            img = np.full((h, w, 3), 255, dtype=np.uint8)
            cv2.rectangle(img, (margin, margin), (w - margin, h - margin), (0, 0, 0), 1)

            all_pf = np.hstack([v for v in per_method_pf.values()]) if per_method_pf else np.asarray([], dtype=np.float32)
            finite = all_pf[np.isfinite(all_pf)]
            ymax_plot = float(np.quantile(finite, 0.95)) if finite.size else 1.0
            ymax_plot = max(0.5, float(ymax_plot) * 1.2)

            def to_xy(i: int, y: float) -> Tuple[int, int]:
                xx = int(margin + (w - 2 * margin) * (float(i) / float(max(1, len(x) - 1))))
                yy = int((h - margin) - (h - 2 * margin) * (float(y) / float(ymax_plot)))
                return xx, yy

            def draw_series(series: np.ndarray, color: Tuple[int, int, int]) -> None:
                pts: List[Tuple[int, int]] = []
                for i, v in enumerate(series):
                    if not math.isfinite(float(v)):
                        if len(pts) >= 2:
                            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, lineType=cv2.LINE_AA)
                        pts = []
                        continue
                    pts.append(to_xy(i, float(v)))
                if len(pts) >= 2:
                    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, lineType=cv2.LINE_AA)

            for m in methods:
                draw_series(per_method_pf[m.name], m.line_bgr)
            cv2.putText(img, f"abs(v_err) per frame (mean of 2 rows), y_max~{ymax_plot:.2f}m", (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            lx = w - margin - 10
            for idx, m in enumerate(reversed(methods)):
                lx_text = max(margin, lx - 12 * len(m.name))
                ly = 30 + 22 * idx
                cv2.putText(img, m.name, (lx_text, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, m.line_bgr, 2)
            cv2.imwrite(str(plots_dir / "abs_v_err_per_frame.png"), img)

            # Histogram plot.
            errs_by_method = {m.name: collect_err(m.name) for m in methods}
            bins = 40
            max_err = 1.0
            for v in errs_by_method.values():
                if v.size:
                    max_err = max(max_err, float(np.max(v)))
            edges = np.linspace(0.0, max_err, bins + 1, dtype=np.float32)
            hists = {name: np.histogram(vals, bins=edges)[0] for name, vals in errs_by_method.items()}
            hist_h = 520
            hist_w = 1200
            img_h = np.full((hist_h, hist_w, 3), 255, dtype=np.uint8)
            cv2.rectangle(img_h, (margin, margin), (hist_w - margin, hist_h - margin), (0, 0, 0), 1)
            max_count = int(max([int(h.max()) for h in hists.values() if h.size] + [1]))
            for i in range(bins):
                x0 = int(margin + (hist_w - 2 * margin) * (float(i) / float(bins)))
                x1 = int(margin + (hist_w - 2 * margin) * (float(i + 1) / float(bins)))
                span = max(1, x1 - x0 - 2)
                bar_w = max(1, span // max(1, len(methods)))
                start = x0 + 1 + (span - bar_w * len(methods)) // 2
                for j, m in enumerate(methods):
                    count = int(hists[m.name][i]) if m.name in hists else 0
                    y_top = int((hist_h - margin) - (hist_h - 2 * margin) * (float(count) / float(max_count)))
                    xl = int(start + j * bar_w)
                    xr = int(min(xl + bar_w - 1, x1 - 1))
                    cv2.rectangle(img_h, (xl, y_top), (xr, hist_h - margin), m.line_bgr, -1)

            cv2.putText(img_h, "abs(v_err) histogram", (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            lx = hist_w - margin - 10
            for idx, m in enumerate(reversed(methods)):
                lx_text = max(margin, lx - 12 * len(m.name))
                ly = 30 + 22 * idx
                cv2.putText(img_h, m.name, (lx_text, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, m.line_bgr, 2)
            cv2.imwrite(str(plots_dir / "abs_v_err_hist.png"), img_h)

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {out_dir / 'summary.json'}")
    for m in methods:
        stats = summary["method_results"][m.name]["abs_v_err_m"]
        print(f"{m.name:12s} | abs(v_err) mean={stats['mean']:.3f}m p95={stats['p95']:.3f}m n={int(stats['n'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
