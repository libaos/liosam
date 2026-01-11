#!/usr/bin/env python3
"""ROS1 (rospy) fruit-tree instancing + MOT tracking + online fitting.

Input:
  - sensor_msgs/PointCloud2 with fields at least: x, y, z, label
  - label==0 is fruit-tree; other labels ignored

Pipeline (per frame):
  1) ROI crop (x/y/z ranges)
  2) Voxel downsample
  3) 2D grid connected-components instancing (XY projection)
  4) Simple MOT tracking (nearest-neighbor with gating)
  5) Sliding-window (K) online fitting + EMA smoothing

Outputs:
  - visualization_msgs/MarkerArray (~markers, default /tree_markers)
  - std_msgs/String JSON (~json_out, default /tree_detections_json)
  - Optional CSV logging (~csv_path)

Also supports a pure-Python `--test_mode` that does not require ROS.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import struct
import subprocess
import sys
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class RoiParams:
    x_min: float = 0.0
    x_max: float = 10.0
    y_min: float = -4.0
    y_max: float = 4.0
    z_min: float = -0.5
    z_max: float = 2.5


@dataclass(frozen=True)
class GridParams:
    cell_size: float = 0.10
    count_threshold: int = 5


@dataclass(frozen=True)
class MotParams:
    gate_distance: float = 0.30
    max_missed: int = 10


@dataclass(frozen=True)
class FitParams:
    window_size: int = 20
    ema_alpha: float = 0.4


@dataclass(frozen=True)
class TrackerParams:
    roi: RoiParams = RoiParams()
    voxel_size: float = 0.03
    grid: GridParams = GridParams()
    mot: MotParams = MotParams()
    fit: FitParams = FitParams()


@dataclass(frozen=True)
class InstanceObservation:
    cx: float
    cy: float
    point_count: int
    points_xyz: np.ndarray  # (N,3) float32


@dataclass
class Track:
    tree_id: int
    missed: int = 0
    age: int = 0  # number of frames observed (hits)
    last_point_count: int = 0
    history: Deque[np.ndarray] = field(default_factory=deque)  # per-frame points (N,3)

    ema_cx: Optional[float] = None
    ema_cy: Optional[float] = None
    ema_height: Optional[float] = None
    ema_crown: Optional[float] = None
    ema_z_med: Optional[float] = None


@dataclass(frozen=True)
class TreeDetection:
    tree_id: int
    cx: float
    cy: float
    height: float
    crown: float
    conf: float
    point_count: int
    z_med: float
    recently_seen: bool


@dataclass(frozen=True)
class BevConfig:
    res: float
    width_px: int
    height_px: int
    roi_width_px: int
    roi_height_px: int


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _finite_xyz_mask(points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.isfinite(points_xyz).all(axis=1)


def _roi_crop(points_xyz: np.ndarray, roi: RoiParams) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    mask = (
        (x >= float(roi.x_min))
        & (x <= float(roi.x_max))
        & (y >= float(roi.y_min))
        & (y <= float(roi.y_max))
        & (z >= float(roi.z_min))
        & (z <= float(roi.z_max))
    )
    return points_xyz[mask]


def _voxel_downsample(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    voxel_size = float(voxel_size)
    if points_xyz.shape[0] == 0 or voxel_size <= 0.0:
        return points_xyz
    coords = np.floor(points_xyz / voxel_size).astype(np.int32, copy=False)
    coords = np.ascontiguousarray(coords)
    key_dtype = np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))
    keys = coords.view(key_dtype).reshape(-1)
    _, unique_idx = np.unique(keys, return_index=True)
    return points_xyz[unique_idx]


_CC_NEIGHBORS_8: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _connected_components_8(binary_grid: np.ndarray) -> Tuple[np.ndarray, int]:
    labels = -np.ones(binary_grid.shape, dtype=np.int32)
    nx, ny = int(binary_grid.shape[0]), int(binary_grid.shape[1])
    comp_id = 0
    for ix in range(nx):
        for iy in range(ny):
            if not bool(binary_grid[ix, iy]) or labels[ix, iy] >= 0:
                continue
            queue: Deque[Tuple[int, int]] = deque([(ix, iy)])
            labels[ix, iy] = comp_id
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in _CC_NEIGHBORS_8:
                    nx2 = cx + dx
                    ny2 = cy + dy
                    if nx2 < 0 or nx2 >= nx or ny2 < 0 or ny2 >= ny:
                        continue
                    if not bool(binary_grid[nx2, ny2]) or labels[nx2, ny2] >= 0:
                        continue
                    labels[nx2, ny2] = comp_id
                    queue.append((nx2, ny2))
            comp_id += 1
    return labels, int(comp_id)


def _grid_instances(points_xyz: np.ndarray, roi: RoiParams, grid: GridParams) -> List[InstanceObservation]:
    if points_xyz.shape[0] == 0:
        return []

    cell_size = float(grid.cell_size)
    if cell_size <= 0.0:
        raise ValueError("grid.cell_size must be > 0")
    threshold = max(1, int(grid.count_threshold))

    x_min, x_max = float(roi.x_min), float(roi.x_max)
    y_min, y_max = float(roi.y_min), float(roi.y_max)
    if x_max <= x_min or y_max <= y_min:
        return []

    nx = int(math.floor((x_max - x_min) / cell_size)) + 1
    ny = int(math.floor((y_max - y_min) / cell_size)) + 1
    if nx <= 0 or ny <= 0:
        return []

    ix = np.floor((points_xyz[:, 0] - x_min) / cell_size).astype(np.int32)
    iy = np.floor((points_xyz[:, 1] - y_min) / cell_size).astype(np.int32)
    in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not bool(np.any(in_bounds)):
        return []

    ix = ix[in_bounds]
    iy = iy[in_bounds]
    pts = points_xyz[in_bounds]

    lin = (ix.astype(np.int64) * int(ny) + iy.astype(np.int64)).astype(np.int64, copy=False)
    counts = np.bincount(lin, minlength=int(nx * ny))
    occ = (counts.reshape((nx, ny)) >= threshold)
    if not bool(np.any(occ)):
        return []

    cell_labels, n_comp = _connected_components_8(occ)
    pt_comp = cell_labels[ix, iy]
    valid = pt_comp >= 0
    if not bool(np.any(valid)):
        return []

    pts = pts[valid]
    pt_comp = pt_comp[valid]

    order = np.argsort(pt_comp, kind="mergesort")
    pt_comp_sorted = pt_comp[order]
    pts_sorted = pts[order]
    change = np.nonzero(np.diff(pt_comp_sorted))[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [pt_comp_sorted.shape[0]]))

    observations: List[InstanceObservation] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e <= s:
            continue
        cluster_pts = pts_sorted[s:e]
        if cluster_pts.shape[0] == 0:
            continue
        cx = float(np.median(cluster_pts[:, 0]))
        cy = float(np.median(cluster_pts[:, 1]))
        observations.append(
            InstanceObservation(
                cx=cx,
                cy=cy,
                point_count=int(cluster_pts.shape[0]),
                points_xyz=cluster_pts.astype(np.float32, copy=False),
            )
        )

    return observations


def _fit_from_points(points_xyz: np.ndarray) -> Tuple[float, float, float, float, float]:
    if points_xyz.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    cx = float(np.median(x))
    cy = float(np.median(y))
    z_med = float(np.median(z))

    z_p5, z_p99 = np.percentile(z, [5.0, 99.0])
    height = float(z_p99 - z_p5)
    x_p5, x_p95 = np.percentile(x, [5.0, 95.0])
    y_p5, y_p95 = np.percentile(y, [5.0, 95.0])
    crown = float(((x_p95 - x_p5) + (y_p95 - y_p5)) / 2.0)
    return cx, cy, height, crown, z_med


def fit_line_pca(
    points_xy: np.ndarray,
    *,
    inlier_dist: float,
    min_points: int,
    iters: int,
) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, float]]:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    total = int(points_xy.shape[0])
    stats: Dict[str, float] = {"total": float(total), "inliers": 0.0, "rms": 0.0}

    if total < int(max(2, min_points)):
        return False, np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32), stats

    inlier_dist = float(max(0.0, inlier_dist))
    inlier2 = inlier_dist * inlier_dist
    mask = np.ones((total,), dtype=bool)

    p0 = np.zeros((2,), dtype=np.float32)
    v = np.zeros((2,), dtype=np.float32)
    for _ in range(max(1, int(iters))):
        pts = points_xy[mask]
        if int(pts.shape[0]) < 2:
            break
        mu = np.mean(pts, axis=0)
        x = pts - mu[None, :]
        cov = (x.T @ x).astype(np.float32, copy=False)
        evals, evecs = np.linalg.eigh(cov)
        v0 = evecs[:, int(np.argmax(evals))]
        n = float(np.linalg.norm(v0))
        if not (n > 1e-12):
            break
        v0 = (v0 / n).astype(np.float32, copy=False)
        if float(v0[0]) < 0.0 or (abs(float(v0[0])) < 1e-12 and float(v0[1]) < 0.0):
            v0 = (-v0).astype(np.float32, copy=False)

        w = points_xy - mu[None, :]
        cross = v0[0] * w[:, 1] - v0[1] * w[:, 0]
        d2 = (cross * cross).astype(np.float32, copy=False)
        new_mask = d2 <= float(inlier2)
        if bool(np.array_equal(new_mask, mask)):
            p0 = mu.astype(np.float32, copy=False)
            v = v0
            break
        mask = new_mask
        p0 = mu.astype(np.float32, copy=False)
        v = v0

    inliers = int(np.count_nonzero(mask))
    stats["inliers"] = float(inliers)
    if inliers < int(min_points):
        return False, p0, v, stats

    w_in = points_xy[mask] - p0[None, :]
    cross_in = v[0] * w_in[:, 1] - v[1] * w_in[:, 0]
    rms = float(np.sqrt(float(np.mean((cross_in * cross_in).astype(np.float32, copy=False)))))
    stats["rms"] = float(rms)
    return True, p0, v, stats


def clip_line_to_roi(p0: np.ndarray, v: np.ndarray, roi: RoiParams) -> Tuple[bool, np.ndarray, np.ndarray]:
    p0 = np.asarray(p0, dtype=np.float32).reshape((2,))
    v = np.asarray(v, dtype=np.float32).reshape((2,))
    vx, vy = float(v[0]), float(v[1])
    eps = 1e-9
    tol = 1e-6

    x_min, x_max = float(roi.x_min), float(roi.x_max)
    y_min, y_max = float(roi.y_min), float(roi.y_max)

    pts: List[Tuple[float, float]] = []

    if abs(vx) > eps:
        for x_edge in (x_min, x_max):
            t = (float(x_edge) - float(p0[0])) / float(vx)
            y = float(p0[1]) + t * float(vy)
            if (y >= y_min - tol) and (y <= y_max + tol):
                pts.append((float(x_edge), float(y)))

    if abs(vy) > eps:
        for y_edge in (y_min, y_max):
            t = (float(y_edge) - float(p0[1])) / float(vy)
            x = float(p0[0]) + t * float(vx)
            if (x >= x_min - tol) and (x <= x_max + tol):
                pts.append((float(x), float(y_edge)))

    uniq: List[Tuple[float, float]] = []
    for x, y in pts:
        if all(((x - ux) * (x - ux) + (y - uy) * (y - uy)) > (1e-6 * 1e-6) for ux, uy in uniq):
            uniq.append((float(x), float(y)))

    if len(uniq) < 2:
        return False, np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)

    best_i, best_j = 0, 1
    best_d2 = -1.0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            dx = float(uniq[i][0] - uniq[j][0])
            dy = float(uniq[i][1] - uniq[j][1])
            d2 = dx * dx + dy * dy
            if d2 > best_d2:
                best_d2 = d2
                best_i, best_j = i, j

    pA = np.array(uniq[best_i], dtype=np.float32)
    pB = np.array(uniq[best_j], dtype=np.float32)
    return True, pA, pB


def _confidence(point_count: int, track_age: int, missed: int, max_missed: int) -> float:
    pc_norm = _clamp(float(point_count) / 80.0, 0.0, 1.0)
    age_norm = _clamp(float(track_age) / 10.0, 0.0, 1.0)
    seen_factor = 1.0 if missed <= 0 else _clamp(1.0 - (float(missed) / float(max(1, max_missed + 1))), 0.0, 1.0)
    conf = pc_norm * (0.35 + 0.65 * age_norm) * seen_factor
    return _clamp(conf, 0.0, 1.0)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value) != 0)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "t", "yes", "y", "on"):
            return True
        if v in ("0", "false", "f", "no", "n", "off", ""):
            return False
    return bool(default)


def _stable_rgb_from_id(tree_id: int) -> Tuple[int, int, int]:
    x = (int(tree_id) * 1103515245 + 12345) & 0xFFFFFFFF
    r = (x >> 16) & 0xFF
    g = (x >> 8) & 0xFF
    b = x & 0xFF
    r = 64 + (int(r) * 191) // 255
    g = 64 + (int(g) * 191) // 255
    b = 64 + (int(b) * 191) // 255
    return int(r), int(g), int(b)


_FONT_5X7: Dict[str, Tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01111"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "g": ("00000", "00000", "01111", "10001", "01111", "00001", "01110"),
    "m": ("00000", "00000", "11010", "10101", "10101", "10101", "10101"),
    "n": ("00000", "00000", "11110", "10001", "10001", "10001", "10001"),
    "p": ("00000", "00000", "11110", "10001", "11110", "10000", "10000"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
}


def _draw_text_5x7(
    img_rgb: np.ndarray,
    *,
    u: int,
    v: int,
    text: str,
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: int = 2,
) -> None:
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return
    h, w, _ = img_rgb.shape
    x = int(u)
    y = int(v)
    scale = max(1, int(scale))
    for ch in str(text):
        if ch == "\n":
            x = int(u)
            y += (7 + 2) * scale
            continue
        glyph = _FONT_5X7.get(ch, _FONT_5X7.get(ch.lower(), _FONT_5X7[" "]))
        for gy in range(7):
            row = glyph[gy]
            yy0 = y + gy * scale
            yy1 = yy0 + scale
            if yy1 <= 0 or yy0 >= h:
                continue
            for gx in range(5):
                if row[gx] != "1":
                    continue
                xx0 = x + gx * scale
                xx1 = xx0 + scale
                if xx1 <= 0 or xx0 >= w:
                    continue
                img_rgb[max(0, yy0) : min(h, yy1), max(0, xx0) : min(w, xx1), :] = color
        x += (5 + 1) * scale


def _draw_disc(img_rgb: np.ndarray, *, u: int, v: int, r: int, color: Tuple[int, int, int]) -> None:
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return
    h, w, _ = img_rgb.shape
    r = int(max(0, r))
    if r == 0:
        if 0 <= v < h and 0 <= u < w:
            img_rgb[v, u, :] = color
        return

    u0 = max(0, int(u) - r)
    u1 = min(w - 1, int(u) + r)
    v0 = max(0, int(v) - r)
    v1 = min(h - 1, int(v) + r)
    rr = r * r
    for vv in range(v0, v1 + 1):
        dv = vv - int(v)
        rem = rr - dv * dv
        if rem < 0:
            continue
        du = int(math.sqrt(float(rem)))
        uu0 = max(u0, int(u) - du)
        uu1 = min(u1, int(u) + du)
        img_rgb[vv, uu0 : uu1 + 1, :] = color


def _draw_circle_outline(img_rgb: np.ndarray, *, u: int, v: int, r: int, color: Tuple[int, int, int]) -> None:
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return
    h, w, _ = img_rgb.shape
    r = int(r)
    if r <= 0:
        return

    x = r
    y = 0
    d = 1 - r
    while x >= y:
        pts = (
            (u + x, v + y),
            (u + y, v + x),
            (u - y, v + x),
            (u - x, v + y),
            (u - x, v - y),
            (u - y, v - x),
            (u + y, v - x),
            (u + x, v - y),
        )
        for uu, vv in pts:
            if 0 <= vv < h and 0 <= uu < w:
                img_rgb[int(vv), int(uu), :] = color
        y += 1
        if d < 0:
            d += 2 * y + 1
        else:
            x -= 1
            d += 2 * (y - x) + 1


def _draw_line(img_rgb: np.ndarray, *, u0: int, v0: int, u1: int, v1: int, color: Tuple[int, int, int]) -> None:
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return
    h, w, _ = img_rgb.shape
    x0 = int(u0)
    y0 = int(v0)
    x1 = int(u1)
    y1 = int(v1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= y0 < h and 0 <= x0 < w:
            img_rgb[y0, x0, :] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _apply_tf_xy(xy: np.ndarray, tf_msg) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    if xy.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    xy = xy.reshape((-1, 2)).astype(np.float32, copy=False)

    tr = tf_msg.transform.translation
    rot = tf_msg.transform.rotation

    tx = float(tr.x)
    ty = float(tr.y)
    qx = float(rot.x)
    qy = float(rot.y)
    qz = float(rot.z)
    qw = float(rot.w)

    x = xy[:, 0]
    y = xy[:, 1]

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)

    x2 = (r00 * x + r01 * y + tx).astype(np.float32, copy=False)
    y2 = (r10 * x + r11 * y + ty).astype(np.float32, copy=False)
    return np.stack([x2, y2], axis=1).astype(np.float32, copy=False)


def _bev_config_from_params(roi: RoiParams, *, bev_res: float, width_px: int, height_px: int) -> BevConfig:
    x_range = float(roi.x_max) - float(roi.x_min)
    y_range = float(roi.y_max) - float(roi.y_min)
    if not (x_range > 0.0 and y_range > 0.0):
        return BevConfig(res=0.05, width_px=1, height_px=1, roi_width_px=1, roi_height_px=1)

    width_px = int(width_px)
    height_px = int(height_px)
    bev_res = float(bev_res)

    if width_px > 0 and height_px > 0:
        res_w = y_range / float(width_px)
        res_h = x_range / float(height_px)
        res = float(max(res_w, res_h, 1e-6))
        out_w = int(width_px)
        out_h = int(height_px)
    elif bev_res > 0.0:
        res = float(max(bev_res, 1e-6))
        out_w = int(math.ceil(y_range / res))
        out_h = int(math.ceil(x_range / res))
    elif width_px > 0:
        res = float(max(y_range / float(width_px), 1e-6))
        out_w = int(width_px)
        out_h = int(math.ceil(x_range / res))
    elif height_px > 0:
        res = float(max(x_range / float(height_px), 1e-6))
        out_h = int(height_px)
        out_w = int(math.ceil(y_range / res))
    else:
        res = 0.02
        out_w = int(math.ceil(y_range / res))
        out_h = int(math.ceil(x_range / res))

    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    roi_w = int(math.ceil(y_range / res))
    roi_h = int(math.ceil(x_range / res))
    return BevConfig(res=float(res), width_px=out_w, height_px=out_h, roi_width_px=int(roi_w), roi_height_px=int(roi_h))


def _xy_to_uv_arrays(
    x: np.ndarray,
    y: np.ndarray,
    *,
    roi: RoiParams,
    bev: BevConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.floor((float(roi.y_max) - y) / float(bev.res)).astype(np.int32)
    v = np.floor((float(roi.x_max) - x) / float(bev.res)).astype(np.int32)
    mask = (u >= 0) & (u < int(bev.width_px)) & (v >= 0) & (v < int(bev.height_px))
    return u, v, mask


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack("!I", len(data))
        + chunk_type
        + data
        + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("empty image")
    raw = b"".join((b"\x00" + rgb[i].tobytes()) for i in range(h))
    comp = zlib.compress(raw, level=9)
    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", comp) + _png_chunk(b"IEND", b"")
    path.write_bytes(data)


class FruitTreeTracker:
    def __init__(self, params: TrackerParams):
        self.params = params
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def process_points(self, points_xyz: np.ndarray) -> List[TreeDetection]:
        points_xyz = points_xyz.astype(np.float32, copy=False)
        points_xyz = points_xyz[_finite_xyz_mask(points_xyz)]
        points_xyz = _roi_crop(points_xyz, self.params.roi)
        points_xyz = _voxel_downsample(points_xyz, self.params.voxel_size)
        observations = _grid_instances(points_xyz, self.params.roi, self.params.grid)
        self._update_tracks(observations)
        return self._collect_detections()

    def _update_tracks(self, observations: List[InstanceObservation]) -> None:
        gate2 = float(self.params.mot.gate_distance) ** 2
        max_missed = int(self.params.mot.max_missed)
        window = max(1, int(self.params.fit.window_size))
        alpha = _clamp(float(self.params.fit.ema_alpha), 0.0, 1.0)

        track_ids = sorted(self._tracks.keys())
        if track_ids and observations:
            tracks_xy = np.array(
                [[float(self._tracks[tid].ema_cx or 0.0), float(self._tracks[tid].ema_cy or 0.0)] for tid in track_ids],
                dtype=np.float32,
            )
            obs_xy = np.array([[float(o.cx), float(o.cy)] for o in observations], dtype=np.float32)
            d2 = np.sum((tracks_xy[:, None, :] - obs_xy[None, :, :]) ** 2, axis=2)
            ti, oi = np.where(d2 <= gate2)
            if ti.size:
                order = np.argsort(d2[ti, oi], kind="mergesort")
                matched_tracks: Set[int] = set()
                matched_obs: Set[int] = set()
                pairs: List[Tuple[int, int]] = []
                for k in order.tolist():
                    t_idx = int(ti[k])
                    o_idx = int(oi[k])
                    if t_idx in matched_tracks or o_idx in matched_obs:
                        continue
                    matched_tracks.add(t_idx)
                    matched_obs.add(o_idx)
                    pairs.append((t_idx, o_idx))
            else:
                pairs = []
                matched_tracks = set()
                matched_obs = set()
        else:
            pairs = []
            matched_tracks = set()
            matched_obs = set()

        matched_track_ids = {int(track_ids[t_idx]) for t_idx, _ in pairs}

        # Update matched tracks.
        for t_idx, o_idx in pairs:
            tid = int(track_ids[t_idx])
            obs = observations[int(o_idx)]
            tr = self._tracks[tid]
            tr.missed = 0
            tr.age += 1
            tr.last_point_count = int(obs.point_count)
            tr.history.append(obs.points_xyz)
            while len(tr.history) > window:
                tr.history.popleft()

            points = np.concatenate(list(tr.history), axis=0) if len(tr.history) > 1 else tr.history[0]
            # For association we want the center to follow motion quickly, so use the per-frame observation
            # center (median of current cluster). For size-related terms (height/crown), use a short history
            # window for robustness.
            meas_cx = float(obs.cx)
            meas_cy = float(obs.cy)
            _fit_cx, _fit_cy, height, crown, z_med = _fit_from_points(points)
            if tr.ema_cx is None:
                tr.ema_cx, tr.ema_cy = meas_cx, meas_cy
                tr.ema_height, tr.ema_crown = height, crown
                tr.ema_z_med = z_med
            else:
                tr.ema_cx = float(alpha * meas_cx + (1.0 - alpha) * float(tr.ema_cx))
                tr.ema_cy = float(alpha * meas_cy + (1.0 - alpha) * float(tr.ema_cy))
                tr.ema_height = float(alpha * height + (1.0 - alpha) * float(tr.ema_height or 0.0))
                tr.ema_crown = float(alpha * crown + (1.0 - alpha) * float(tr.ema_crown or 0.0))
                tr.ema_z_med = float(alpha * z_med + (1.0 - alpha) * float(tr.ema_z_med or 0.0))

        # Mark missed tracks.
        to_delete: List[int] = []
        for tid, tr in self._tracks.items():
            if int(tid) in matched_track_ids:
                continue
            tr.missed += 1
            tr.last_point_count = 0
            if tr.missed > max_missed:
                to_delete.append(int(tid))

        for tid in to_delete:
            self._tracks.pop(int(tid), None)

        # Create new tracks for unmatched observations.
        for idx, obs in enumerate(observations):
            if idx in matched_obs:
                continue
            tr = Track(tree_id=int(self._next_id), missed=0, age=1, last_point_count=int(obs.point_count))
            tr.history.append(obs.points_xyz)
            cx, cy, height, crown, z_med = _fit_from_points(obs.points_xyz)
            tr.ema_cx, tr.ema_cy = cx, cy
            tr.ema_height, tr.ema_crown = height, crown
            tr.ema_z_med = z_med
            self._tracks[int(tr.tree_id)] = tr
            self._next_id += 1

    def _collect_detections(self) -> List[TreeDetection]:
        max_missed = int(self.params.mot.max_missed)
        detections: List[TreeDetection] = []
        for tid in sorted(self._tracks.keys()):
            tr = self._tracks[int(tid)]
            cx = float(tr.ema_cx or 0.0)
            cy = float(tr.ema_cy or 0.0)
            height = float(tr.ema_height or 0.0)
            crown = float(tr.ema_crown or 0.0)
            z_med = float(tr.ema_z_med or 0.0)
            conf = _confidence(int(tr.last_point_count), int(tr.age), int(tr.missed), max_missed)
            detections.append(
                TreeDetection(
                    tree_id=int(tid),
                    cx=cx,
                    cy=cy,
                    height=max(0.0, height),
                    crown=max(0.0, crown),
                    conf=conf,
                    point_count=int(tr.last_point_count),
                    z_med=z_med,
                    recently_seen=(tr.missed == 0),
                )
            )
        return detections


def _simulate_tree_points(
    rng: np.random.Generator,
    center_xy: Tuple[float, float],
    n_points: int,
    crown_radius: float,
    height: float,
) -> np.ndarray:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    n = int(n_points)
    crown_radius = float(max(0.05, crown_radius))
    sigma = max(0.03, crown_radius / 4.0)
    xy = rng.normal(0.0, sigma, size=(n, 2)).astype(np.float32, copy=False)
    r = np.linalg.norm(xy, axis=1)
    over = r > crown_radius
    if bool(np.any(over)):
        xy[over] *= (crown_radius / r[over]).reshape((-1, 1)).astype(np.float32, copy=False)
    x = cx + xy[:, 0] + rng.normal(0.0, 0.01, size=n)
    y = cy + xy[:, 1] + rng.normal(0.0, 0.01, size=n)
    z = rng.uniform(0.0, float(height), size=n) + rng.normal(0.0, 0.01, size=n)
    return np.stack([x, y, z], axis=1).astype(np.float32, copy=False)


def _sample_tree_centers(
    rng: np.random.Generator,
    n_trees: int,
    *,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    min_center_dist: float,
    max_attempts: int = 10000,
) -> List[Tuple[float, float]]:
    n_trees = int(n_trees)
    if n_trees <= 0:
        return []

    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("invalid x_range/y_range for sampling centers")

    min_center_dist = float(max(0.0, min_center_dist))
    max_attempts = int(max(1, max_attempts))

    centers: List[Tuple[float, float]] = []
    attempts = 0
    while len(centers) < n_trees and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(x0, x1))
        y = float(rng.uniform(y0, y1))

        if min_center_dist > 0.0:
            ok = True
            for cx, cy in centers:
                if math.hypot(x - float(cx), y - float(cy)) < min_center_dist:
                    ok = False
                    break
            if not ok:
                continue

        centers.append((x, y))

    if len(centers) != n_trees:
        raise RuntimeError(
            f"Failed to sample {n_trees} tree centers with min_center_dist={min_center_dist:.3f} "
            f"within x_range={x_range}, y_range={y_range} (attempts={attempts}/{max_attempts}). "
            "Try reducing --trees or --min_center_dist."
        )

    return centers


def run_test_mode(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(int(args.seed))
    roi = RoiParams(
        x_min=float(args.roi_x_min),
        x_max=float(args.roi_x_max),
        y_min=float(args.roi_y_min),
        y_max=float(args.roi_y_max),
        z_min=float(args.roi_z_min),
        z_max=float(args.roi_z_max),
    )
    params = TrackerParams(
        roi=roi,
        voxel_size=float(args.voxel_size),
        grid=GridParams(cell_size=float(args.cell_size), count_threshold=int(args.grid_T)),
        mot=MotParams(gate_distance=float(args.gate), max_missed=int(args.max_missed)),
        fit=FitParams(window_size=int(args.K), ema_alpha=float(args.alpha)),
    )

    tracker = FruitTreeTracker(params)

    n_trees = int(args.trees)
    min_center_dist = float(getattr(args, "min_center_dist", 1.0))
    centers = _sample_tree_centers(
        rng,
        n_trees,
        x_range=(2.0, 8.0),
        y_range=(-2.5, 2.5),
        min_center_dist=min_center_dist,
        max_attempts=20000,
    )

    gt_to_id: Dict[int, int] = {}
    mismatches = 0
    total = 0
    drift_per_frame = float(args.drift)

    for t in range(int(args.frames)):
        drift = float(t) * drift_per_frame
        points_all = []
        for i, (cx, cy) in enumerate(centers):
            visible = True
            if args.drop_prob > 0.0 and rng.uniform() < float(args.drop_prob):
                visible = False
            if not visible:
                continue
            pts = _simulate_tree_points(
                rng=rng,
                center_xy=(cx - drift, cy),
                n_points=int(args.points_per_tree),
                crown_radius=float(args.crown_radius),
                height=float(args.height),
            )
            points_all.append(pts)
        frame_pts = np.concatenate(points_all, axis=0) if points_all else np.empty((0, 3), dtype=np.float32)
        detections = tracker.process_points(frame_pts)
        detections = [d for d in detections if int(d.point_count) > 0]

        det_xy = np.array([[d.cx, d.cy] for d in detections], dtype=np.float32) if detections else np.empty((0, 2), dtype=np.float32)
        det_id = [d.tree_id for d in detections]

        for i, (cx, cy) in enumerate(centers):
            gt_xy = np.array([cx - drift, cy], dtype=np.float32)
            if det_xy.shape[0] == 0:
                continue
            d2 = np.sum((det_xy - gt_xy[None, :]) ** 2, axis=1)
            j = int(np.argmin(d2))
            if float(d2[j]) > float(args.gate) ** 2:
                continue
            assigned = int(det_id[j])
            if i not in gt_to_id:
                gt_to_id[i] = assigned
            total += 1
            if assigned != int(gt_to_id[i]):
                mismatches += 1

        if args.verbose and (t % 10 == 0 or t == int(args.frames) - 1):
            print(f"[test_mode] frame={t:03d} detections={len(detections)} ids={[d.tree_id for d in detections]}")

    mismatch_rate = (float(mismatches) / float(max(1, total))) * 100.0
    print("[test_mode] gt_to_id:", gt_to_id)
    print(f"[test_mode] mismatches={mismatches}/{total} ({mismatch_rate:.2f}%)")
    return 0


class RosTreeTrackerNode:
    def __init__(self) -> None:
        import rospy
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray

        self.rospy = rospy
        self.PointCloud2 = PointCloud2
        self.String = String
        self.MarkerArray = MarkerArray

        rospy.init_node("fruit_tree_tracker", anonymous=False)

        self.input_topic = rospy.get_param("~input_topic", "/segmented_points")
        self.label_field = rospy.get_param("~label_field", "label")
        self.markers_topic = rospy.get_param("~markers", "/tree_markers")
        self.json_topic = rospy.get_param("~json_out", "/tree_detections_json")
        self.row_fit_json_topic = rospy.get_param("~row_fit_json", "/tree_row_fit_json")
        self.csv_path = rospy.get_param("~csv_path", "")
        self.csv_flush_interval = float(rospy.get_param("~csv_flush_interval", 1.0))
        self.log_summary_interval = float(rospy.get_param("~log_summary_interval", 1.0))
        self.publish_missed = _as_bool(rospy.get_param("~publish_missed", False), default=False)
        self.row_fit_min_points = int(rospy.get_param("~row_fit_min_points", 3))
        self.row_fit_inlier_dist = float(rospy.get_param("~row_fit_inlier_dist", 0.20))
        self.row_fit_iters = int(rospy.get_param("~row_fit_iters", 2))
        self.row_fit_history_frames = int(rospy.get_param("~row_fit_history_frames", 20))
        self.row_fit_min_conf = float(rospy.get_param("~row_fit_min_conf", 0.0))
        self.row_fit_fixed_frame = rospy.get_param("~row_fit_fixed_frame", "")
        self.row_fit_fixed_frame_timeout = float(rospy.get_param("~row_fit_fixed_frame_timeout", 0.05))
        self.export_dir = rospy.get_param("~export_dir", "")
        self.export_every_n = int(rospy.get_param("~export_every_n", 0))
        self.export_max_frames = int(rospy.get_param("~export_max_frames", 0))
        self.bev_res = float(rospy.get_param("~bev_res", 0.0))
        self.bev_width_px = int(rospy.get_param("~bev_width_px", 0))
        self.bev_height_px = int(rospy.get_param("~bev_height_px", 0))
        self.export_draw_ids = _as_bool(rospy.get_param("~export_draw_ids", True), default=True)
        self.export_draw_crowns = _as_bool(rospy.get_param("~export_draw_crowns", True), default=True)

        roi = RoiParams(
            x_min=float(rospy.get_param("~roi_x_min", 0.0)),
            x_max=float(rospy.get_param("~roi_x_max", 10.0)),
            y_min=float(rospy.get_param("~roi_y_min", -4.0)),
            y_max=float(rospy.get_param("~roi_y_max", 4.0)),
            z_min=float(rospy.get_param("~roi_z_min", -0.5)),
            z_max=float(rospy.get_param("~roi_z_max", 2.5)),
        )
        self.params = TrackerParams(
            roi=roi,
            voxel_size=float(rospy.get_param("~voxel_size", 0.03)),
            grid=GridParams(
                cell_size=float(rospy.get_param("~cell_size", 0.10)),
                count_threshold=int(rospy.get_param("~grid_T", 5)),
            ),
            mot=MotParams(
                gate_distance=float(rospy.get_param("~gate_distance", 0.30)),
                max_missed=int(rospy.get_param("~max_missed", 10)),
            ),
            fit=FitParams(
                window_size=int(rospy.get_param("~K", 20)),
                ema_alpha=float(rospy.get_param("~ema_alpha", 0.4)),
            ),
        )
        self.tracker = FruitTreeTracker(self.params)

        self.pub_markers = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=1)
        self.pub_json = rospy.Publisher(self.json_topic, String, queue_size=1)
        self.pub_row_fit_json = None
        if isinstance(self.row_fit_json_topic, str) and self.row_fit_json_topic.strip():
            self.pub_row_fit_json = rospy.Publisher(str(self.row_fit_json_topic).strip(), String, queue_size=1)

        self._prev_marker_ids: Set[int] = set()
        self._frame_index: int = 0
        self._export_dir_path: Optional[Path] = None
        self._export_bev: Optional[BevConfig] = None
        self._exported_frames: int = 0
        self._export_enabled: bool = False
        self._export_warned_max: bool = False
        self._row_fit_hist_left: Optional[Deque[np.ndarray]] = None
        self._row_fit_hist_right: Optional[Deque[np.ndarray]] = None
        self._row_fit_hist_fixed: Optional[Deque[np.ndarray]] = None
        self._tf_buffer = None
        self._tf_listener = None
        self._row_fit_fixed_frame: str = ""
        self._row_fit_fixed_timeout: float = float(max(0.0, self.row_fit_fixed_frame_timeout))
        if isinstance(self.row_fit_fixed_frame, str) and self.row_fit_fixed_frame.strip():
            self._row_fit_fixed_frame = str(self.row_fit_fixed_frame).strip()
            try:
                import tf2_ros

                self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
                self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
            except Exception as exc:
                rospy.logwarn("[orchard_tree_tracker] row_fit_fixed_frame disabled (tf2_ros unavailable): %s", str(exc))
                self._row_fit_fixed_frame = ""
        if int(self.row_fit_history_frames) > 1:
            self._row_fit_hist_left = deque(maxlen=int(self.row_fit_history_frames))
            self._row_fit_hist_right = deque(maxlen=int(self.row_fit_history_frames))
            if bool(self._row_fit_fixed_frame):
                self._row_fit_hist_fixed = deque(maxlen=int(self.row_fit_history_frames))
        self._csv_fp: Optional[object] = None
        self._csv_writer: Optional[csv.writer] = None
        self._last_csv_flush_time = rospy.Time(0)
        if isinstance(self.csv_path, str) and self.csv_path.strip():
            self._open_csv(self.csv_path.strip())
            rospy.on_shutdown(self._close_csv)

        self._dtype_cache_key: Optional[Tuple] = None
        self._dtype_cache: Optional[np.dtype] = None
        self._dtype_cache_fields: Optional[Tuple[str, str, str, str]] = None

        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self._on_cloud, queue_size=1, buff_size=2**24)

        rospy.loginfo("[orchard_tree_tracker] Listening on %s (label_field=%s)", self.input_topic, self.label_field)
        if self.pub_row_fit_json is not None:
            rospy.loginfo("[orchard_tree_tracker] row_fit_json=%s", str(self.row_fit_json_topic).strip())
        if bool(self._row_fit_fixed_frame):
            rospy.loginfo("[orchard_tree_tracker] row_fit_fixed_frame=%s (motion-compensated history)", str(self._row_fit_fixed_frame))
        self._setup_export()
        rospy.loginfo(
            "[orchard_tree_tracker] publish_missed=%s (published=%s)",
            str(bool(self.publish_missed)).lower(),
            "all_tracks" if bool(self.publish_missed) else "only_seen(point_count>0)",
        )

    def _open_csv(self, csv_path: str) -> None:
        p = Path(csv_path).expanduser()
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        file_exists = p.exists()
        self._csv_fp = p.open("a", encoding="utf-8", newline="")
        self._csv_writer = csv.writer(self._csv_fp)
        if (not file_exists) or p.stat().st_size == 0:
            self._csv_writer.writerow(["timestamp", "id", "cx", "cy", "height", "crown", "conf", "point_count"])
            self._csv_fp.flush()

    def _close_csv(self) -> None:
        if self._csv_fp is not None:
            try:
                self._csv_fp.flush()
                self._csv_fp.close()
            except Exception:
                pass
        self._csv_fp = None
        self._csv_writer = None

    def _try_get_git_hash(self) -> str:
        try:
            here = Path(__file__).resolve()
        except Exception:
            return ""
        for p in [here.parent, *here.parents]:
            if (p / ".git").exists():
                try:
                    out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(p), stderr=subprocess.DEVNULL)
                    s = out.decode("utf-8", errors="ignore").strip()
                    return s if len(s) >= 7 else ""
                except Exception:
                    return ""
        return ""

    def _setup_export(self) -> None:
        rospy = self.rospy
        if not isinstance(self.export_dir, str) or not self.export_dir.strip():
            return
        export_dir = str(self.export_dir).strip()
        try:
            p = Path(export_dir).expanduser()
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            rospy.logwarn("[orchard_tree_tracker] export_dir create failed: %s (%s)", export_dir, str(exc))
            return

        self._export_dir_path = p
        self._export_bev = _bev_config_from_params(
            self.params.roi,
            bev_res=float(self.bev_res),
            width_px=int(self.bev_width_px),
            height_px=int(self.bev_height_px),
        )
        self._export_enabled = True

        rospy.loginfo(
            "[orchard_tree_tracker] export_dir=%s every_n=%d max_frames=%d",
            export_dir,
            int(self.export_every_n),
            int(self.export_max_frames),
        )
        rospy.loginfo(
            "[orchard_tree_tracker] bev: res=%.6f width_px=%d height_px=%d (roi_width_px=%d roi_height_px=%d)",
            float(self._export_bev.res),
            int(self._export_bev.width_px),
            int(self._export_bev.height_px),
            int(self._export_bev.roi_width_px),
            int(self._export_bev.roi_height_px),
        )

        meta_path = p / "run_meta.json"
        if meta_path.exists():
            rospy.logwarn("[orchard_tree_tracker] run_meta.json already exists: %s (will not overwrite)", str(meta_path))
            return

        meta = {
            "generated_at": _dt.datetime.now().astimezone().isoformat(),
            "git_hash": self._try_get_git_hash(),
            "input_topic": str(self.input_topic),
            "label_field": str(self.label_field),
            "coordinate_system": {
                "vehicle": "+x forward, +y left, +z up",
                "bev_image": "image up = +x, image left = +y",
            },
            "bev_mapping": {
                "u": "(roi_y_max - y) / bev_res",
                "v": "(roi_x_max - x) / bev_res",
                "roi_y_min": float(self.params.roi.y_min),
                "roi_y_max": float(self.params.roi.y_max),
                "roi_x_min": float(self.params.roi.x_min),
                "roi_x_max": float(self.params.roi.x_max),
                "bev_res": float(self._export_bev.res),
                "width_px": int(self._export_bev.width_px),
                "height_px": int(self._export_bev.height_px),
                "roi_width_px": int(self._export_bev.roi_width_px),
                "roi_height_px": int(self._export_bev.roi_height_px),
            },
            "density_background": {"mode": "log1p_clip", "clip": 20},
            "tree_color": {"mode": "lcg_bright_rgb", "formula": "x=(id*1103515245+12345)&0xFFFFFFFF; rgb=(x>>16,x>>8,x) brightened"},
            "export": {
                "pattern": "frame_%06d.png",
                "export_every_n": int(self.export_every_n),
                "export_max_frames": int(self.export_max_frames),
                "draw_ids": bool(self.export_draw_ids),
                "draw_crowns": bool(self.export_draw_crowns),
            },
            "tracker_params": {
                "roi": {
                    "x_min": float(self.params.roi.x_min),
                    "x_max": float(self.params.roi.x_max),
                    "y_min": float(self.params.roi.y_min),
                    "y_max": float(self.params.roi.y_max),
                    "z_min": float(self.params.roi.z_min),
                    "z_max": float(self.params.roi.z_max),
                },
                "voxel_size": float(self.params.voxel_size),
                "cell_size": float(self.params.grid.cell_size),
                "grid_T": int(self.params.grid.count_threshold),
                "gate_distance": float(self.params.mot.gate_distance),
                "max_missed": int(self.params.mot.max_missed),
                "K": int(self.params.fit.window_size),
                "ema_alpha": float(self.params.fit.ema_alpha),
                "publish_missed": bool(self.publish_missed),
            },
            "row_fit": {
                "min_points": int(self.row_fit_min_points),
                "inlier_dist": float(self.row_fit_inlier_dist),
                "iters": int(self.row_fit_iters),
                "history_frames": int(self.row_fit_history_frames),
                "min_conf": float(self.row_fit_min_conf),
                "fixed_frame": str(self._row_fit_fixed_frame),
                "fixed_frame_timeout": float(self._row_fit_fixed_timeout),
                "grouping": {"left": "cy > 0", "right": "cy < 0", "cy == 0": "ignored"},
                "method": "PCA/TLS + deterministic inlier filtering",
            },
        }

        try:
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rospy.loginfo("[orchard_tree_tracker] wrote %s", str(meta_path))
        except Exception as exc:
            rospy.logwarn("[orchard_tree_tracker] failed to write run_meta.json: %s", str(exc))

    def _should_export_frame(self, frame_index: int) -> bool:
        if not bool(self._export_enabled) or self._export_dir_path is None or self._export_bev is None:
            return False
        if int(self.export_max_frames) > 0 and int(self._exported_frames) >= int(self.export_max_frames):
            if not bool(self._export_warned_max):
                self.rospy.logwarn(
                    "[orchard_tree_tracker] export_max_frames reached (%d), stop exporting",
                    int(self.export_max_frames),
                )
                self._export_warned_max = True
            return False
        if int(self.export_every_n) > 0 and (int(frame_index) % int(self.export_every_n) != 0):
            return False
        return True

    def _render_bev_rgb(
        self,
        *,
        frame_index: int,
        stamp_sec: float,
        points_xyz: np.ndarray,
        detections: List[TreeDetection],
        row_fit_payload: Optional[Dict],
    ) -> np.ndarray:
        bev = self._export_bev
        if bev is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        roi = self.params.roi
        w = int(bev.width_px)
        h = int(bev.height_px)
        img = np.zeros((h, w, 3), dtype=np.uint8)

        density_clip = 20.0
        pts = points_xyz.astype(np.float32, copy=False)
        pts = pts[_finite_xyz_mask(pts)]
        pts = _roi_crop(pts, roi)
        pts = _voxel_downsample(pts, float(self.params.voxel_size))
        if int(pts.shape[0]) > 0:
            u, v, mask = _xy_to_uv_arrays(pts[:, 0], pts[:, 1], roi=roi, bev=bev)
            if bool(np.any(mask)):
                uu = u[mask].astype(np.int64, copy=False)
                vv = v[mask].astype(np.int64, copy=False)
                lin = vv * int(w) + uu
                counts = np.bincount(lin, minlength=int(h * w)).reshape((h, w))
                counts_clip = np.minimum(counts.astype(np.float32, copy=False), float(density_clip))
                gray = (np.log1p(counts_clip) / float(math.log1p(density_clip)) * 255.0).astype(np.uint8)
                img[:, :, 0] = gray
                img[:, :, 1] = gray
                img[:, :, 2] = gray

        if isinstance(row_fit_payload, dict):
            left = row_fit_payload.get("left", {})
            right = row_fit_payload.get("right", {})
            for side, color in (("left", (0, 0, 255)), ("right", (255, 0, 0))):
                block = left if side == "left" else right
                seg = block.get("segment") if isinstance(block, dict) else None
                if not (isinstance(seg, dict) and isinstance(seg.get("pA"), list) and isinstance(seg.get("pB"), list)):
                    continue
                pA = seg["pA"]
                pB = seg["pB"]
                try:
                    u0 = int(math.floor((float(roi.y_max) - float(pA[1])) / float(bev.res)))
                    v0 = int(math.floor((float(roi.x_max) - float(pA[0])) / float(bev.res)))
                    u1 = int(math.floor((float(roi.y_max) - float(pB[1])) / float(bev.res)))
                    v1 = int(math.floor((float(roi.x_max) - float(pB[0])) / float(bev.res)))
                    _draw_line(img, u0=u0, v0=v0, u1=u1, v1=v1, color=color)
                except Exception:
                    continue

        for det in detections:
            try:
                u0 = int(math.floor((float(roi.y_max) - float(det.cy)) / float(bev.res)))
                v0 = int(math.floor((float(roi.x_max) - float(det.cx)) / float(bev.res)))
            except Exception:
                continue
            if not (0 <= u0 < w and 0 <= v0 < h):
                continue
            color = _stable_rgb_from_id(int(det.tree_id))
            if bool(self.export_draw_crowns):
                r_px = int(round((max(0.0, float(det.crown)) * 0.5) / float(bev.res)))
                if r_px > 1:
                    _draw_circle_outline(img, u=u0, v=v0, r=r_px, color=color)
            _draw_disc(img, u=u0, v=v0, r=2, color=color)
            if bool(self.export_draw_ids):
                _draw_text_5x7(img, u=u0 + 3, v=v0 - 8, text=str(int(det.tree_id)), color=(255, 255, 255), scale=1)

        file_text = f"frame_{int(frame_index):06d}.png"
        stamp_text = f"{float(stamp_sec):.6f}"
        _draw_text_5x7(img, u=4, v=4, text=file_text, color=(255, 255, 255), scale=2)
        _draw_text_5x7(img, u=4, v=4 + (7 + 2) * 2, text=stamp_text, color=(255, 255, 255), scale=2)
        return img

    def _export_bev_frame(
        self,
        *,
        frame_index: int,
        stamp_sec: float,
        points_xyz: np.ndarray,
        detections: List[TreeDetection],
        row_fit_payload: Optional[Dict],
    ) -> None:
        if self._export_dir_path is None:
            return
        out_path = self._export_dir_path / f"frame_{int(frame_index):06d}.png"
        try:
            img = self._render_bev_rgb(
                frame_index=int(frame_index),
                stamp_sec=float(stamp_sec),
                points_xyz=points_xyz,
                detections=detections,
                row_fit_payload=row_fit_payload,
            )
            _write_png_rgb(out_path, img)
            self._exported_frames += 1
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "[orchard_tree_tracker] export PNG failed: %s", str(exc))

    def _on_cloud(self, msg) -> None:
        rospy = self.rospy
        start = rospy.Time.now()
        frame_index = int(self._frame_index)
        stamp_sec = float(msg.header.stamp.to_sec())
        try:
            try:
                points_xyz = self._cloud_to_tree_points(msg)
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "[orchard_tree_tracker] Failed to parse PointCloud2: %s", str(exc))
                points_xyz = np.empty((0, 3), dtype=np.float32)

            detections_all = self.tracker.process_points(points_xyz)
            detections_seen = [d for d in detections_all if int(d.point_count) > 0]
            detections_pub = detections_all if bool(self.publish_missed) else detections_seen
            self._publish_outputs(msg, detections_pub)

            need_row_fit = (self.pub_row_fit_json is not None) or bool(self._export_enabled)
            row_fit_payload: Optional[Dict] = None
            if bool(need_row_fit):
                min_conf = float(max(0.0, self.row_fit_min_conf))
                det_fit = [
                    d
                    for d in detections_seen
                    if float(d.conf) >= float(min_conf)
                    and int(d.point_count) > 0
                    and math.isfinite(float(d.cx))
                    and math.isfinite(float(d.cy))
                ]
                roi = self.params.roi
                left_xy = np.empty((0, 2), dtype=np.float32)
                right_xy = np.empty((0, 2), dtype=np.float32)

                if bool(self._row_fit_fixed_frame) and self._tf_buffer is not None:
                    try:
                        tf_msg_to_fixed = self._tf_buffer.lookup_transform(
                            str(self._row_fit_fixed_frame),
                            str(msg.header.frame_id),
                            msg.header.stamp,
                            rospy.Duration(float(self._row_fit_fixed_timeout)),
                        )
                        xy_cur_msg = np.array([[float(d.cx), float(d.cy)] for d in det_fit], dtype=np.float32)
                        xy_cur_fixed = _apply_tf_xy(xy_cur_msg, tf_msg_to_fixed)
                        if self._row_fit_hist_fixed is not None:
                            self._row_fit_hist_fixed.append(xy_cur_fixed)
                            xy_fixed = (
                                np.concatenate(list(self._row_fit_hist_fixed), axis=0)
                                if len(self._row_fit_hist_fixed) > 0
                                else np.empty((0, 2), dtype=np.float32)
                            )
                        else:
                            xy_fixed = xy_cur_fixed

                        tf_fixed_to_msg = self._tf_buffer.lookup_transform(
                            str(msg.header.frame_id),
                            str(self._row_fit_fixed_frame),
                            msg.header.stamp,
                            rospy.Duration(float(self._row_fit_fixed_timeout)),
                        )
                        xy_msg = _apply_tf_xy(xy_fixed, tf_fixed_to_msg)

                        if int(xy_msg.shape[0]) > 0:
                            x = xy_msg[:, 0]
                            y = xy_msg[:, 1]
                            m = (
                                np.isfinite(x)
                                & np.isfinite(y)
                                & (x >= float(roi.x_min))
                                & (x <= float(roi.x_max))
                                & (y >= float(roi.y_min))
                                & (y <= float(roi.y_max))
                            )
                            xy_msg = xy_msg[m]

                        left_xy = xy_msg[xy_msg[:, 1] > 0.0] if int(xy_msg.shape[0]) > 0 else np.empty((0, 2), dtype=np.float32)
                        right_xy = xy_msg[xy_msg[:, 1] < 0.0] if int(xy_msg.shape[0]) > 0 else np.empty((0, 2), dtype=np.float32)
                    except Exception as exc:
                        rospy.logwarn_throttle(2.0, "[orchard_tree_tracker] row_fit_fixed_frame tf failed: %s", str(exc))

                if not (int(left_xy.shape[0]) > 0 or int(right_xy.shape[0]) > 0):
                    left_xy_cur = np.array([[float(d.cx), float(d.cy)] for d in det_fit if float(d.cy) > 0.0], dtype=np.float32)
                    right_xy_cur = np.array([[float(d.cx), float(d.cy)] for d in det_fit if float(d.cy) < 0.0], dtype=np.float32)

                    left_xy = left_xy_cur
                    right_xy = right_xy_cur
                    if self._row_fit_hist_left is not None and self._row_fit_hist_right is not None:
                        self._row_fit_hist_left.append(left_xy_cur)
                        self._row_fit_hist_right.append(right_xy_cur)
                        left_xy = (
                            np.concatenate(list(self._row_fit_hist_left), axis=0)
                            if len(self._row_fit_hist_left) > 0
                            else np.empty((0, 2), dtype=np.float32)
                        )
                        right_xy = (
                            np.concatenate(list(self._row_fit_hist_right), axis=0)
                            if len(self._row_fit_hist_right) > 0
                            else np.empty((0, 2), dtype=np.float32)
                        )

                left_valid, left_p0, left_v, left_stats = fit_line_pca(
                    left_xy,
                    inlier_dist=float(self.row_fit_inlier_dist),
                    min_points=int(self.row_fit_min_points),
                    iters=int(self.row_fit_iters),
                )
                right_valid, right_p0, right_v, right_stats = fit_line_pca(
                    right_xy,
                    inlier_dist=float(self.row_fit_inlier_dist),
                    min_points=int(self.row_fit_min_points),
                    iters=int(self.row_fit_iters),
                )

                left_seg_valid, left_pA, left_pB = (False, np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32))
                if bool(left_valid):
                    left_seg_valid, left_pA, left_pB = clip_line_to_roi(left_p0, left_v, self.params.roi)

                right_seg_valid, right_pA, right_pB = (False, np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32))
                if bool(right_valid):
                    right_seg_valid, right_pA, right_pB = clip_line_to_roi(right_p0, right_v, self.params.roi)

                row_fit_payload = {
                    "frame_index": frame_index,
                    "stamp": stamp_sec,
                    "frame_id": str(msg.header.frame_id),
                    "fixed_frame": str(self._row_fit_fixed_frame),
                    "left": {
                        "valid": bool(left_valid),
                        "p0": [float(left_p0[0]), float(left_p0[1])] if bool(left_valid) else None,
                        "v": [float(left_v[0]), float(left_v[1])] if bool(left_valid) else None,
                        "segment": (
                            {"pA": [float(left_pA[0]), float(left_pA[1])], "pB": [float(left_pB[0]), float(left_pB[1])]}
                            if bool(left_seg_valid)
                            else None
                        ),
                        "inliers": int(left_stats.get("inliers", 0.0)),
                        "total": int(left_stats.get("total", 0.0)),
                        "rms": float(left_stats.get("rms", 0.0)),
                    },
                    "right": {
                        "valid": bool(right_valid),
                        "p0": [float(right_p0[0]), float(right_p0[1])] if bool(right_valid) else None,
                        "v": [float(right_v[0]), float(right_v[1])] if bool(right_valid) else None,
                        "segment": (
                            {"pA": [float(right_pA[0]), float(right_pA[1])], "pB": [float(right_pB[0]), float(right_pB[1])]}
                            if bool(right_seg_valid)
                            else None
                        ),
                        "inliers": int(right_stats.get("inliers", 0.0)),
                        "total": int(right_stats.get("total", 0.0)),
                        "rms": float(right_stats.get("rms", 0.0)),
                    },
                }

                if self.pub_row_fit_json is not None:
                    self.pub_row_fit_json.publish(self.String(data=json.dumps(row_fit_payload, ensure_ascii=False)))

            if self._should_export_frame(frame_index):
                self._export_bev_frame(
                    frame_index=int(frame_index),
                    stamp_sec=float(stamp_sec),
                    points_xyz=points_xyz,
                    detections=detections_pub,
                    row_fit_payload=row_fit_payload,
                )

            if float(self.log_summary_interval) > 0.0:
                ids = [int(d.tree_id) for d in detections_pub[:20]]
                suffix = "" if len(detections_pub) <= 20 else "..."
                rospy.loginfo_throttle(
                    float(self.log_summary_interval),
                    "[orchard_tree_tracker] seen=%d total=%d published=%d ids=%s%s",
                    int(len(detections_seen)),
                    int(len(detections_all)),
                    int(len(detections_pub)),
                    str(ids),
                    suffix,
                )

            dt_ms = (rospy.Time.now() - start).to_sec() * 1000.0
            rospy.logdebug(
                "[orchard_tree_tracker] frame: in=%d tree_pts=%d tracks=%d (%.2f ms)",
                int(msg.width) * int(msg.height),
                int(points_xyz.shape[0]),
                int(len(detections_pub)),
                float(dt_ms),
            )
        finally:
            self._frame_index += 1

    def _cloud_to_tree_points(self, msg) -> np.ndarray:
        from sensor_msgs.msg import PointField

        required = ("x", "y", "z", str(self.label_field))
        fields = {f.name: f for f in msg.fields}
        for name in required:
            if name not in fields:
                raise KeyError(f"missing field: {name}")

        cache_key = (
            tuple((f.name, int(f.offset), int(f.datatype), int(f.count)) for f in msg.fields),
            int(msg.point_step),
            int(msg.row_step),
            int(msg.width),
            int(msg.height),
            bool(msg.is_bigendian),
            required,
        )
        if cache_key != self._dtype_cache_key:
            endian = ">" if bool(msg.is_bigendian) else "<"
            dt_map = {
                PointField.INT8: "i1",
                PointField.UINT8: "u1",
                PointField.INT16: "i2",
                PointField.UINT16: "u2",
                PointField.INT32: "i4",
                PointField.UINT32: "u4",
                PointField.FLOAT32: "f4",
                PointField.FLOAT64: "f8",
            }

            names: List[str] = []
            formats: List[np.dtype] = []
            offsets: List[int] = []
            for name in required:
                f = fields[name]
                if int(f.count) != 1:
                    raise ValueError(f"field {name} has count={int(f.count)}; expected 1")
                base = dt_map.get(int(f.datatype))
                if base is None:
                    raise ValueError(f"unsupported datatype for field {name}: {int(f.datatype)}")
                names.append(str(name))
                formats.append(np.dtype(endian + base))
                offsets.append(int(f.offset))
            self._dtype_cache = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(msg.point_step)})
            self._dtype_cache_key = cache_key
            self._dtype_cache_fields = tuple(required)

        dtype = self._dtype_cache
        if dtype is None:
            return np.empty((0, 3), dtype=np.float32)

        points_count = int(msg.width) * int(msg.height)
        if points_count <= 0:
            return np.empty((0, 3), dtype=np.float32)

        raw = memoryview(msg.data)
        if int(msg.height) == 1 or int(msg.row_step) == int(msg.point_step) * int(msg.width):
            arr = np.frombuffer(raw, dtype=dtype, count=points_count)
        else:
            rows = []
            width_bytes = int(msg.width) * int(msg.point_step)
            for r in range(int(msg.height)):
                start = int(r) * int(msg.row_step)
                row_view = raw[start : start + width_bytes]
                rows.append(np.frombuffer(row_view, dtype=dtype, count=int(msg.width)))
            arr = np.concatenate(rows, axis=0) if rows else np.empty((0,), dtype=dtype)

        x = arr["x"].astype(np.float32, copy=False)
        y = arr["y"].astype(np.float32, copy=False)
        z = arr["z"].astype(np.float32, copy=False)
        label = arr[str(self.label_field)]
        if label.dtype.kind == "f":
            label_i = np.rint(label).astype(np.int32)
        else:
            label_i = label.astype(np.int32, copy=False)

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        tree_mask = finite & (label_i == 0)
        if not bool(np.any(tree_mask)):
            return np.empty((0, 3), dtype=np.float32)

        pts = np.stack([x[tree_mask], y[tree_mask], z[tree_mask]], axis=1).astype(np.float32, copy=False)
        return pts

    def _publish_outputs(self, msg, detections: List[TreeDetection]) -> None:
        from std_msgs.msg import String
        from visualization_msgs.msg import Marker, MarkerArray

        header = msg.header

        marker_array = MarkerArray()
        active_ids = {int(d.tree_id) for d in detections}
        stale_ids = set(self._prev_marker_ids) - set(active_ids)

        def add_delete(ns: str, tid: int) -> None:
            m = Marker()
            m.header = header
            m.ns = ns
            m.id = int(tid)
            m.action = Marker.DELETE
            marker_array.markers.append(m)

        for tid in sorted(stale_ids):
            add_delete("tree_center", tid)
            add_delete("tree_crown", tid)
            add_delete("tree_text", tid)

        for det in detections:
            tid = int(det.tree_id)
            cx = float(det.cx)
            cy = float(det.cy)
            height = float(det.height)
            crown = float(det.crown)
            z_center = float(det.z_med)

            m_center = Marker()
            m_center.header = header
            m_center.ns = "tree_center"
            m_center.id = tid
            m_center.type = Marker.SPHERE
            m_center.action = Marker.ADD
            m_center.pose.orientation.w = 1.0
            m_center.pose.position.x = cx
            m_center.pose.position.y = cy
            m_center.pose.position.z = max(0.0, z_center)
            m_center.scale.x = 0.20
            m_center.scale.y = 0.20
            m_center.scale.z = 0.20
            m_center.color.r = 0.15
            m_center.color.g = 0.90
            m_center.color.b = 0.20
            m_center.color.a = 0.90
            marker_array.markers.append(m_center)

            m_crown = Marker()
            m_crown.header = header
            m_crown.ns = "tree_crown"
            m_crown.id = tid
            m_crown.type = Marker.CYLINDER
            m_crown.action = Marker.ADD
            m_crown.pose.orientation.w = 1.0
            m_crown.pose.position.x = cx
            m_crown.pose.position.y = cy
            m_crown.pose.position.z = max(0.0, z_center)
            m_crown.scale.x = max(0.05, crown)
            m_crown.scale.y = max(0.05, crown)
            m_crown.scale.z = 0.05
            m_crown.color.r = 0.10
            m_crown.color.g = 0.80
            m_crown.color.b = 0.10
            m_crown.color.a = 0.25
            marker_array.markers.append(m_crown)

            m_text = Marker()
            m_text.header = header
            m_text.ns = "tree_text"
            m_text.id = tid
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.orientation.w = 1.0
            m_text.pose.position.x = cx
            m_text.pose.position.y = cy
            m_text.pose.position.z = max(0.0, z_center) + max(0.3, height) + 0.2
            m_text.scale.z = 0.30
            m_text.color.r = 1.0
            m_text.color.g = 1.0
            m_text.color.b = 1.0
            m_text.color.a = 1.0
            m_text.text = f"id={tid} h={height:.2f} c={crown:.2f}"
            marker_array.markers.append(m_text)

        self.pub_markers.publish(marker_array)
        self._prev_marker_ids = set(active_ids)

        payload = [
            {"id": int(d.tree_id), "cx": float(d.cx), "cy": float(d.cy), "height": float(d.height), "crown": float(d.crown), "conf": float(d.conf)}
            for d in detections
        ]
        self.pub_json.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        if self._csv_writer is not None:
            t = float(header.stamp.to_sec())
            for d in detections:
                self._csv_writer.writerow(
                    [
                        f"{t:.6f}",
                        int(d.tree_id),
                        f"{float(d.cx):.3f}",
                        f"{float(d.cy):.3f}",
                        f"{float(d.height):.3f}",
                        f"{float(d.crown):.3f}",
                        f"{float(d.conf):.3f}",
                        int(d.point_count),
                    ]
                )
            if self._csv_fp is not None and float(self.csv_flush_interval) > 0.0:
                now = self.rospy.Time.now()
                if (now - self._last_csv_flush_time).to_sec() >= float(self.csv_flush_interval):
                    self._csv_fp.flush()
                    self._last_csv_flush_time = now


def run_ros() -> int:
    import rospy

    _ = RosTreeTrackerNode()
    rospy.spin()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true", help="Run pure-Python test mode (no ROS required).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--trees", type=int, default=6)
    parser.add_argument("--points_per_tree", type=int, default=500)
    parser.add_argument("--crown_radius", type=float, default=0.30)
    parser.add_argument("--height", type=float, default=1.6)
    parser.add_argument("--drift", type=float, default=0.02, help="Per-frame drift along +x (simulates vehicle motion).")
    parser.add_argument("--drop_prob", type=float, default=0.03, help="Randomly drop a tree observation with this prob.")
    parser.add_argument(
        "--min_center_dist",
        type=float,
        default=1.0,
        help="Minimum spacing between synthetic tree centers in test_mode (meters). Set 0 to disable.",
    )
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--roi_x_min", type=float, default=0.0)
    parser.add_argument("--roi_x_max", type=float, default=10.0)
    parser.add_argument("--roi_y_min", type=float, default=-4.0)
    parser.add_argument("--roi_y_max", type=float, default=4.0)
    parser.add_argument("--roi_z_min", type=float, default=-0.5)
    parser.add_argument("--roi_z_max", type=float, default=2.5)
    parser.add_argument("--voxel_size", type=float, default=0.03)
    parser.add_argument("--cell_size", type=float, default=0.10)
    parser.add_argument("--grid_T", type=int, default=5)
    parser.add_argument("--gate", type=float, default=0.30)
    parser.add_argument("--max_missed", type=int, default=10)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.4)

    args, _unknown = parser.parse_known_args(argv)
    if bool(args.test_mode):
        return run_test_mode(args)
    return run_ros()


if __name__ == "__main__":
    raise SystemExit(main())
