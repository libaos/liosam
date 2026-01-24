#!/usr/bin/env python3
"""Publish per-tree circle markers from a tree-only point cloud map.

This node is intended for paper-quality visualization: it clusters a tree-only
map into individual trees and draws a circle for each tree (LINE_STRIP or
CYLINDER markers).
"""

from __future__ import annotations

import csv
import json
import math
import struct
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class TreeCircle:
    x: float
    y: float
    z: float
    radius: float


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    h = float(h) % 1.0
    s = float(max(0.0, min(s, 1.0)))
    v = float(max(0.0, min(v, 1.0)))
    i = int(h * 6.0)
    f = h * 6.0 - float(i)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0))


def _cluster_id_to_rgb(cluster_id: int) -> Tuple[int, int, int]:
    # Spread hues using golden ratio for stable distinct colors.
    phi = 0.618033988749895
    h = (float(cluster_id) * phi) % 1.0
    return _hsv_to_rgb(h, 0.95, 1.0)


def _pack_rgb_float(r: int, g: int, b: int) -> float:
    r = int(max(0, min(r, 255)))
    g = int(max(0, min(g, 255)))
    b = int(max(0, min(b, 255)))
    rgb_uint32 = (r << 16) | (g << 8) | b
    return struct.unpack("f", struct.pack("I", int(rgb_uint32)))[0]


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    in_len = len(data)
    in_idx = 0
    out = bytearray()
    while in_idx < in_len:
        ctrl = data[in_idx]
        in_idx += 1
        if ctrl < 32:
            length = ctrl + 1
            if in_idx + length > in_len:
                raise RuntimeError("Invalid LZF data: literal run exceeds input length")
            out.extend(data[in_idx : in_idx + length])
            in_idx += length
            continue

        length = ctrl >> 5
        ref_offset = (ctrl & 0x1F) << 8
        if length == 7:
            if in_idx >= in_len:
                raise RuntimeError("Invalid LZF data: missing length byte")
            length += data[in_idx]
            in_idx += 1
        if in_idx >= in_len:
            raise RuntimeError("Invalid LZF data: missing offset byte")
        ref_offset += data[in_idx]
        in_idx += 1
        ref_offset += 1

        ref = len(out) - ref_offset
        if ref < 0:
            raise RuntimeError("Invalid LZF data: back reference before output start")
        for _ in range(length + 2):
            out.append(out[ref])
            ref += 1

    if expected_size >= 0 and len(out) != expected_size:
        raise RuntimeError(f"Invalid LZF data: expected {expected_size} bytes, got {len(out)} bytes")
    return bytes(out)


def _pcd_numpy_dtype(size: int, typ: str) -> np.dtype:
    typ = typ.upper()
    if typ == "F":
        if size == 4:
            return np.float32
        if size == 8:
            return np.float64
    if typ == "I":
        if size == 1:
            return np.int8
        if size == 2:
            return np.int16
        if size == 4:
            return np.int32
        if size == 8:
            return np.int64
    if typ == "U":
        if size == 1:
            return np.uint8
        if size == 2:
            return np.uint16
        if size == 4:
            return np.uint32
        if size == 8:
            return np.uint64
    raise RuntimeError(f"Unsupported PCD field type: TYPE={typ} SIZE={size}")


def _load_pcd_xyz(pcd_path: Path) -> np.ndarray:
    with pcd_path.open("rb") as handle:
        header_lines: List[str] = []
        data_mode: Optional[str] = None
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Invalid PCD header: {pcd_path}")
            decoded = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA"):
                parts = decoded.split()
                data_mode = parts[1].lower() if len(parts) >= 2 else None
                break

        header: Dict[str, str] = {}
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                header[parts[0].upper()] = parts[1]

        fields = header.get("FIELDS", "").split()
        sizes = [int(v) for v in header.get("SIZE", "").split()]
        types = header.get("TYPE", "").split()
        counts = [int(v) for v in header.get("COUNT", "").split()] if "COUNT" in header else [1] * len(fields)
        points_count = int(header.get("POINTS", header.get("WIDTH", "0"))) or 0

        if data_mode is None:
            raise RuntimeError(f"Missing DATA line in PCD: {pcd_path}")

        if data_mode == "ascii":
            data = np.loadtxt(handle, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            name_to_index = {name: idx for idx, name in enumerate(fields)}
            if not all(name in name_to_index for name in ("x", "y", "z")):
                raise RuntimeError(f"PCD missing xyz fields: {pcd_path} ({fields})")
            return data[:, [name_to_index["x"], name_to_index["y"], name_to_index["z"]]].astype(np.float32)

        if data_mode not in ("binary", "binary_compressed"):
            raise RuntimeError(f"Unsupported PCD DATA mode: {data_mode} ({pcd_path})")

        if not fields or not sizes or not types:
            raise RuntimeError(f"Incomplete PCD header (FIELDS/SIZE/TYPE): {pcd_path}")
        if len(fields) != len(sizes) or len(fields) != len(types) or len(fields) != len(counts):
            raise RuntimeError(f"PCD header length mismatch: {pcd_path}")

        dtype_fields: List[Tuple[str, np.dtype]] = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            if count != 1:
                for i in range(count):
                    dtype_fields.append((f"{name}_{i}", _pcd_numpy_dtype(size, typ)))
            else:
                dtype_fields.append((name, _pcd_numpy_dtype(size, typ)))
        dtype = np.dtype(dtype_fields)

        point_step = int(dtype.itemsize)
        expected_bytes = int(points_count) * point_step

        if data_mode == "binary":
            raw = handle.read(expected_bytes)
            if len(raw) < expected_bytes:
                raise RuntimeError(f"PCD data too short: expected {expected_bytes} bytes, got {len(raw)} ({pcd_path})")
            arr = np.frombuffer(raw, dtype=dtype, count=points_count)
        else:
            header_bytes = handle.read(8)
            if len(header_bytes) < 8:
                raise RuntimeError(f"PCD compressed header too short: {pcd_path}")
            compressed_size, uncompressed_size = struct.unpack("<II", header_bytes)
            compressed = handle.read(int(compressed_size))
            if len(compressed) < int(compressed_size):
                raise RuntimeError(
                    f"PCD compressed data too short: expected {compressed_size} bytes, got {len(compressed)} ({pcd_path})"
                )
            decompressed = _lzf_decompress(compressed, int(uncompressed_size))
            if int(uncompressed_size) < expected_bytes:
                raise RuntimeError(
                    f"PCD decompressed data too short: expected >= {expected_bytes} bytes, got {uncompressed_size} ({pcd_path})"
                )

            in_bytes = np.frombuffer(decompressed, dtype=np.uint8, count=expected_bytes)
            out_bytes = np.empty((points_count, point_step), dtype=np.uint8)

            in_offset = 0
            out_offset = 0
            for size, count in zip(sizes, counts):
                field_step = int(size) * int(count)
                block_len = field_step * int(points_count)
                field_block = in_bytes[in_offset : in_offset + block_len]
                if field_block.size != block_len:
                    raise RuntimeError(f"PCD compressed payload truncated for field block ({pcd_path})")
                out_bytes[:, out_offset : out_offset + field_step] = field_block.reshape(points_count, field_step)
                in_offset += block_len
                out_offset += field_step

            arr = np.frombuffer(out_bytes.tobytes(), dtype=dtype, count=points_count)

        if not all(name in arr.dtype.names for name in ("x", "y", "z")):
            raise RuntimeError(f"PCD missing xyz fields: {pcd_path} ({arr.dtype.names})")
        xyz = np.vstack([arr["x"], arr["y"], arr["z"]]).T
        return xyz.astype(np.float32)


def _cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _filter_points(
    points: np.ndarray,
    z_min: float,
    z_max: float,
    x_min: float,
    x_max: float,
    y_abs_max: float,
) -> np.ndarray:
    if points.size == 0:
        return points
    mask = (
        (points[:, 2] >= float(z_min))
        & (points[:, 2] <= float(z_max))
        & (points[:, 0] >= float(x_min))
        & (points[:, 0] <= float(x_max))
        & (np.abs(points[:, 1]) <= float(y_abs_max))
    )
    return points[mask]


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(points.shape[0], int(max_points), replace=False)
    return points[idx]


def _cluster_cells(
    xy: np.ndarray,
    cell_size: float,
    neighbor_range: int,
    min_points: int,
    max_clusters: int,
) -> List[np.ndarray]:
    cell_size = float(cell_size)
    if cell_size <= 0.0:
        raise ValueError("cell_size must be > 0")
    neighbor_range = max(0, int(neighbor_range))
    min_points = max(1, int(min_points))
    max_clusters = int(max_clusters)

    grid = np.floor(xy / cell_size).astype(np.int32)
    cells: Dict[Tuple[int, int], List[int]] = {}
    for idx, (cx, cy) in enumerate(grid.tolist()):
        cells.setdefault((int(cx), int(cy)), []).append(int(idx))

    visited: set[Tuple[int, int]] = set()
    clusters: List[np.ndarray] = []
    for cell in cells.keys():
        if cell in visited:
            continue
        queue: deque[Tuple[int, int]] = deque([cell])
        visited.add(cell)
        member_indices: List[int] = []
        while queue:
            cx, cy = queue.popleft()
            member_indices.extend(cells.get((cx, cy), []))
            for dx in range(-neighbor_range, neighbor_range + 1):
                for dy in range(-neighbor_range, neighbor_range + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in visited:
                        continue
                    if nb not in cells:
                        continue
                    visited.add(nb)
                    queue.append(nb)

        if len(member_indices) < min_points:
            continue
        clusters.append(np.asarray(member_indices, dtype=np.int32))
        if max_clusters > 0 and len(clusters) >= max_clusters:
            break

    return clusters


def _compute_radius(
    pts_xy: np.ndarray,
    center_xy: np.ndarray,
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


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or values.size == 0:
        return values
    if window % 2 == 0:
        window += 1
    if window > int(values.size):
        window = int(values.size)
        if window % 2 == 0:
            window -= 1
    if window <= 1:
        return values
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _find_peaks_1d(values: np.ndarray, min_height: float, min_distance: int) -> List[int]:
    values = values.astype(np.float32).reshape(-1)
    if values.size < 3:
        return []
    min_height = float(min_height)
    min_distance = max(0, int(min_distance))

    candidates: List[int] = []
    for i in range(1, int(values.size) - 1):
        if values[i] < min_height:
            continue
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            candidates.append(i)

    candidates.sort(key=lambda idx: float(values[idx]), reverse=True)
    selected: List[int] = []
    for idx in candidates:
        if all(abs(idx - j) >= min_distance for j in selected):
            selected.append(idx)
    selected.sort()
    return selected


def _load_row_model_file(path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    direction_xy = np.asarray(data.get("direction_xy", [1.0, 0.0]), dtype=np.float32).reshape(2)
    perp_xy = np.asarray(data.get("perp_xy", [0.0, 1.0]), dtype=np.float32).reshape(2)

    d_norm = float(np.linalg.norm(direction_xy))
    if d_norm <= 1.0e-8:
        raise RuntimeError(f"Invalid row_model direction_xy (zero norm): {path}")
    direction_xy = direction_xy / d_norm

    p_norm = float(np.linalg.norm(perp_xy))
    if p_norm <= 1.0e-8:
        perp_xy = np.asarray([-direction_xy[1], direction_xy[0]], dtype=np.float32)
        p_norm = float(np.linalg.norm(perp_xy))
    perp_xy = perp_xy / p_norm

    # Re-orthogonalize to be safe.
    perp_xy = perp_xy - direction_xy * float(np.dot(perp_xy, direction_xy))
    p_norm = float(np.linalg.norm(perp_xy))
    if p_norm <= 1.0e-8:
        perp_xy = np.asarray([-direction_xy[1], direction_xy[0]], dtype=np.float32)
    else:
        perp_xy = perp_xy / p_norm

    rows_in = data.get("rows", [])
    rows: List[Dict[str, float]] = []
    for row in rows_in:
        try:
            rows.append(
                {
                    "v_center": float(row["v_center"]),
                    "u_min": float(row["u_min"]),
                    "u_max": float(row["u_max"]),
                    "z": float(row.get("z", 0.0)),
                }
            )
        except Exception:
            continue

    if not rows:
        raise RuntimeError(f"row_model has no valid rows: {path}")
    return direction_xy, perp_xy, rows


def _tree_circles_from_row_model(
    points_xyz: np.ndarray,
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    rows: Sequence[Dict[str, float]],
    row_bandwidth: float,
    u_bin_size: float,
    smooth_window: int,
    peak_min_fraction: float,
    min_separation: float,
    u_padding: float,
    refine_u_half_width: float,
    max_trees_per_row: int,
    max_trees: int,
    marker_z: float,
    radius_mode: str,
    radius_constant: float,
    radius_quantile: float,
    radius_min: float,
    radius_max: float,
) -> List[TreeCircle]:
    row_bandwidth = float(row_bandwidth)
    u_bin_size = float(u_bin_size)
    if row_bandwidth <= 0.0:
        raise ValueError("row_bandwidth must be > 0")
    if u_bin_size <= 0.0:
        raise ValueError("tree_u_bin_size must be > 0")

    xy = points_xyz[:, :2].astype(np.float32)
    u = xy.dot(direction_xy.astype(np.float32).reshape(2))
    v = xy.dot(perp_xy.astype(np.float32).reshape(2))

    circles: List[TreeCircle] = []
    min_sep_bins = int(math.ceil(float(min_separation) / u_bin_size)) if float(min_separation) > 0.0 else 0
    max_trees_per_row = int(max_trees_per_row)
    max_trees = int(max_trees)

    for row in rows:
        v_center = float(row["v_center"])
        u_min = float(row["u_min"]) - float(u_padding)
        u_max = float(row["u_max"]) + float(u_padding)
        if u_max <= u_min:
            continue

        row_mask = (np.abs(v - v_center) <= row_bandwidth) & (u >= u_min) & (u <= u_max)
        if not np.any(row_mask):
            continue

        u_row = u[row_mask]
        edges = np.arange(u_min, u_max + u_bin_size, u_bin_size, dtype=np.float32)
        if edges.size < 4:
            continue

        hist, _ = np.histogram(u_row, bins=edges)
        hist = hist.astype(np.float32)
        hist_s = _smooth_1d(hist, smooth_window)
        h_max = float(hist_s.max()) if hist_s.size else 0.0
        if h_max <= 0.0:
            continue

        threshold = h_max * float(max(0.0, min(peak_min_fraction, 1.0)))
        peak_bins = _find_peaks_1d(hist_s, min_height=threshold, min_distance=min_sep_bins)
        if not peak_bins:
            continue

        if max_trees_per_row > 0:
            peak_bins = peak_bins[: max_trees_per_row]

        for bin_idx in peak_bins:
            if max_trees > 0 and len(circles) >= max_trees:
                return circles

            u_peak = float((edges[int(bin_idx)] + edges[int(bin_idx) + 1]) * 0.5)
            local_mask = row_mask & (np.abs(u - u_peak) <= float(refine_u_half_width))
            pts_local = points_xyz[local_mask]

            if pts_local.size:
                center_xy = np.median(pts_local[:, :2], axis=0)
                z = float(marker_z) if marker_z != 0.0 else float(np.median(pts_local[:, 2]))
                radius = _compute_radius(
                    pts_xy=pts_local[:, :2],
                    center_xy=center_xy,
                    mode=radius_mode,
                    radius_constant=radius_constant,
                    radius_quantile=radius_quantile,
                    radius_min=radius_min,
                    radius_max=radius_max,
                )
                circles.append(TreeCircle(x=float(center_xy[0]), y=float(center_xy[1]), z=float(z), radius=float(radius)))
                continue

            center_xy = direction_xy * float(u_peak) + perp_xy * float(v_center)
            z = float(marker_z) if marker_z != 0.0 else float(row.get("z", 0.0))
            radius = _compute_radius(
                pts_xy=np.empty((0, 2), dtype=np.float32),
                center_xy=center_xy.astype(np.float32),
                mode=radius_mode,
                radius_constant=radius_constant,
                radius_quantile=radius_quantile,
                radius_min=radius_min,
                radius_max=radius_max,
            )
            circles.append(TreeCircle(x=float(center_xy[0]), y=float(center_xy[1]), z=float(z), radius=float(radius)))

    circles.sort(key=lambda c: (c.y, c.x))
    return circles


def _tree_circles_and_labels_from_row_model(
    points_xyz: np.ndarray,
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    rows: Sequence[Dict[str, float]],
    row_bandwidth: float,
    u_bin_size: float,
    smooth_window: int,
    peak_min_fraction: float,
    min_separation: float,
    u_padding: float,
    refine_u_half_width: float,
    max_trees_per_row: int,
    max_trees: int,
    marker_z: float,
    radius_mode: str,
    radius_constant: float,
    radius_quantile: float,
    radius_min: float,
    radius_max: float,
) -> Tuple[List[TreeCircle], np.ndarray]:
    points_xyz = points_xyz.astype(np.float32)
    labels = np.full((points_xyz.shape[0],), -1, dtype=np.int32)

    row_bandwidth = float(row_bandwidth)
    u_bin_size = float(u_bin_size)
    if row_bandwidth <= 0.0:
        raise ValueError("row_bandwidth must be > 0")
    if u_bin_size <= 0.0:
        raise ValueError("tree_u_bin_size must be > 0")

    xy = points_xyz[:, :2].astype(np.float32)
    u = xy.dot(direction_xy.astype(np.float32).reshape(2))
    v = xy.dot(perp_xy.astype(np.float32).reshape(2))

    circles_raw: List[TreeCircle] = []
    clusters_raw: List[np.ndarray] = []
    min_sep_bins = int(math.ceil(float(min_separation) / u_bin_size)) if float(min_separation) > 0.0 else 0
    max_trees_per_row = int(max_trees_per_row)
    max_trees = int(max_trees)

    for row in rows:
        if max_trees > 0 and len(circles_raw) >= max_trees:
            break

        v_center = float(row["v_center"])
        u_min = float(row["u_min"]) - float(u_padding)
        u_max = float(row["u_max"]) + float(u_padding)
        if u_max <= u_min:
            continue

        row_mask = (np.abs(v - v_center) <= row_bandwidth) & (u >= u_min) & (u <= u_max)
        row_indices = np.flatnonzero(row_mask)
        if row_indices.size == 0:
            continue

        u_row = u[row_indices]
        edges = np.arange(u_min, u_max + u_bin_size, u_bin_size, dtype=np.float32)
        if edges.size < 4:
            continue

        hist, _ = np.histogram(u_row, bins=edges)
        hist = hist.astype(np.float32)
        hist_s = _smooth_1d(hist, smooth_window)
        h_max = float(hist_s.max()) if hist_s.size else 0.0
        if h_max <= 0.0:
            continue

        threshold = h_max * float(max(0.0, min(peak_min_fraction, 1.0)))
        peak_bins = _find_peaks_1d(hist_s, min_height=threshold, min_distance=min_sep_bins)
        if not peak_bins:
            continue

        if max_trees_per_row > 0:
            peak_bins = peak_bins[: max_trees_per_row]

        if max_trees > 0:
            remaining = int(max_trees) - int(len(circles_raw))
            if remaining <= 0:
                break
            peak_bins = peak_bins[:remaining]
            if not peak_bins:
                break

        peaks_u = np.array(
            [float((edges[int(bin_idx)] + edges[int(bin_idx) + 1]) * 0.5) for bin_idx in peak_bins], dtype=np.float32
        )
        if peaks_u.size == 0:
            continue

        if peaks_u.size == 1:
            assigned_local = np.zeros((row_indices.size,), dtype=np.int32)
        else:
            boundaries = (peaks_u[:-1] + peaks_u[1:]) * 0.5
            assigned_local = np.searchsorted(boundaries, u_row).astype(np.int32)

        for local_id, u_peak in enumerate(peaks_u.tolist()):
            local_mask = row_mask & (np.abs(u - float(u_peak)) <= float(refine_u_half_width))
            pts_local = points_xyz[local_mask]

            if pts_local.size:
                center_xy = np.median(pts_local[:, :2], axis=0)
                z = float(marker_z) if marker_z != 0.0 else float(np.median(pts_local[:, 2]))
                radius = _compute_radius(
                    pts_xy=pts_local[:, :2],
                    center_xy=center_xy,
                    mode=radius_mode,
                    radius_constant=radius_constant,
                    radius_quantile=radius_quantile,
                    radius_min=radius_min,
                    radius_max=radius_max,
                )
                circle = TreeCircle(x=float(center_xy[0]), y=float(center_xy[1]), z=float(z), radius=float(radius))
            else:
                center_xy = direction_xy * float(u_peak) + perp_xy * float(v_center)
                z = float(marker_z) if marker_z != 0.0 else float(row.get("z", 0.0))
                radius = _compute_radius(
                    pts_xy=np.empty((0, 2), dtype=np.float32),
                    center_xy=center_xy.astype(np.float32),
                    mode=radius_mode,
                    radius_constant=radius_constant,
                    radius_quantile=radius_quantile,
                    radius_min=radius_min,
                    radius_max=radius_max,
                )
                circle = TreeCircle(x=float(center_xy[0]), y=float(center_xy[1]), z=float(z), radius=float(radius))

            circles_raw.append(circle)
            clusters_raw.append(row_indices[assigned_local == int(local_id)].astype(np.int32))

    if not circles_raw:
        return [], labels

    order = sorted(range(len(circles_raw)), key=lambda i: (circles_raw[i].y, circles_raw[i].x))
    circles: List[TreeCircle] = []
    clusters: List[np.ndarray] = []
    for i in order:
        circles.append(circles_raw[i])
        clusters.append(clusters_raw[i])

    for new_id, idxs in enumerate(clusters):
        labels[idxs] = int(new_id)
    return circles, labels


def _tree_circles_and_labels_from_cell_clusters(
    points_xyz: np.ndarray,
    cell_size: float,
    neighbor_range: int,
    min_points: int,
    max_clusters: int,
    marker_z: float,
    radius_mode: str,
    radius_constant: float,
    radius_quantile: float,
    radius_min: float,
    radius_max: float,
) -> Tuple[List[TreeCircle], np.ndarray]:
    labels = np.full((points_xyz.shape[0],), -1, dtype=np.int32)
    xy = points_xyz[:, :2]
    clusters = _cluster_cells(
        xy,
        cell_size=float(cell_size),
        neighbor_range=int(neighbor_range),
        min_points=int(min_points),
        max_clusters=int(max_clusters),
    )
    if not clusters:
        return [], labels

    items: List[Tuple[TreeCircle, np.ndarray]] = []
    for cluster in clusters:
        pts = points_xyz[cluster]
        center_xy = np.median(pts[:, :2], axis=0)
        z = float(marker_z) if marker_z != 0.0 else float(np.median(pts[:, 2]))
        radius = _compute_radius(
            pts_xy=pts[:, :2],
            center_xy=center_xy,
            mode=radius_mode,
            radius_constant=radius_constant,
            radius_quantile=radius_quantile,
            radius_min=radius_min,
            radius_max=radius_max,
        )
        items.append((TreeCircle(x=float(center_xy[0]), y=float(center_xy[1]), z=float(z), radius=float(radius)), cluster))

    items.sort(key=lambda item: (item[0].y, item[0].x))
    circles = [item[0] for item in items]
    for idx, (_, cluster) in enumerate(items):
        labels[cluster] = int(idx)
    return circles, labels


def _parse_color(param: Sequence[float], default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if isinstance(param, str):
        try:
            maybe = json.loads(param)
            if isinstance(maybe, (list, tuple)):
                param = maybe  # type: ignore[assignment]
        except Exception:
            try:
                parts = [p.strip() for p in param.replace("[", "").replace("]", "").split(",") if p.strip()]
                if len(parts) >= 3:
                    param = [float(parts[0]), float(parts[1]), float(parts[2])]  # type: ignore[assignment]
            except Exception:
                return default
    try:
        if len(param) >= 3:
            r, g, b = float(param[0]), float(param[1]), float(param[2])
            return max(0.0, min(r, 1.0)), max(0.0, min(g, 1.0)), max(0.0, min(b, 1.0))
    except Exception:
        pass
    return default


class OrchardTreeCirclesNode:
    def __init__(self) -> None:
        self.map_frame = str(rospy.get_param("~map_frame", "map")).strip() or "map"
        self.input_topic = str(rospy.get_param("~input_topic", "/orchard_tree_map_builder/tree_map")).strip()

        self.detection_mode = str(rospy.get_param("~detection_mode", "row_model_peaks")).strip().lower()
        row_model_param = str(rospy.get_param("~row_model_file", "")).strip()
        self.row_model_file = Path(row_model_param).expanduser() if row_model_param else None
        self.row_bandwidth = float(rospy.get_param("~row_bandwidth", 0.8))
        self.tree_u_bin_size = float(rospy.get_param("~tree_u_bin_size", 0.2))
        self.tree_smooth_window = int(rospy.get_param("~tree_smooth_window", 9))
        self.tree_peak_min_fraction = float(rospy.get_param("~tree_peak_min_fraction", 0.12))
        self.tree_min_separation = float(rospy.get_param("~tree_min_separation", 2.0))
        self.tree_u_padding = float(rospy.get_param("~tree_u_padding", 0.0))
        self.refine_u_half_width = float(rospy.get_param("~refine_u_half_width", 0.8))
        self.max_trees_per_row = int(rospy.get_param("~max_trees_per_row", 0))

        pcd_param = str(rospy.get_param("~pcd_path", "")).strip()
        self.pcd_path = Path(pcd_param).expanduser() if pcd_param else None

        self.filter_z_min = float(rospy.get_param("~z_min", 0.2))
        self.filter_z_max = float(rospy.get_param("~z_max", 2.0))
        self.filter_x_min = float(rospy.get_param("~x_min", -1.0e9))
        self.filter_x_max = float(rospy.get_param("~x_max", 1.0e9))
        self.filter_y_abs_max = float(rospy.get_param("~y_abs_max", 1.0e9))
        self.max_points = int(rospy.get_param("~max_points", 80000))
        self.sample_seed = int(rospy.get_param("~sample_seed", 0))

        self.cluster_cell_size = float(rospy.get_param("~cluster_cell_size", 0.30))
        self.cluster_neighbor_range = int(rospy.get_param("~cluster_neighbor_range", 1))
        self.min_points_per_tree = int(rospy.get_param("~min_points_per_tree", 80))
        self.max_trees = int(rospy.get_param("~max_trees", 0))

        self.radius_mode = str(rospy.get_param("~radius_mode", "constant")).strip().lower()
        self.radius_constant = float(rospy.get_param("~radius_constant", 0.35))
        self.radius_quantile = float(rospy.get_param("~radius_quantile", 0.8))
        self.radius_min = float(rospy.get_param("~radius_min", 0.15))
        self.radius_max = float(rospy.get_param("~radius_max", 1.5))

        self.marker_type = str(rospy.get_param("~marker_type", "line_strip")).strip().lower()
        self.circle_segments = int(rospy.get_param("~circle_segments", 36))
        self.line_width = float(rospy.get_param("~line_width", 0.05))
        self.alpha = float(rospy.get_param("~alpha", 0.8))
        self.color = _parse_color(rospy.get_param("~color", [0.0, 0.8, 0.0]), (0.0, 0.8, 0.0))
        self.marker_z = float(rospy.get_param("~marker_z", 0.0))

        self.publish_labels = bool(rospy.get_param("~publish_labels", False))
        self.label_height = float(rospy.get_param("~label_height", 0.5))
        self.label_z_offset = float(rospy.get_param("~label_z_offset", 0.8))
        self.label_start_index = int(rospy.get_param("~label_start_index", 1))

        export_csv_param = str(rospy.get_param("~export_csv", "")).strip()
        self.export_csv = Path(export_csv_param).expanduser() if export_csv_param else None

        self._pub = rospy.Publisher("~tree_circles", MarkerArray, queue_size=1, latch=True)
        self.publish_clusters_cloud = bool(rospy.get_param("~publish_clusters_cloud", True))
        self._clusters_pub = (
            rospy.Publisher("~tree_clusters", PointCloud2, queue_size=1, latch=True)
            if self.publish_clusters_cloud
            else None
        )

        points = self._load_points()
        points_used, circles, labels = self._build_tree_result(points)
        if self.export_csv is not None:
            try:
                self._export_csv(circles, self.export_csv)
                rospy.loginfo("[orchard_row_mapping] Exported tree circles CSV: %s", self.export_csv)
            except Exception as exc:
                rospy.logwarn("[orchard_row_mapping] Failed to export tree circles CSV: %s", exc)

        markers = self._build_markers(circles)
        self._pub.publish(markers)
        rospy.loginfo("[orchard_row_mapping] Published %d tree circles on %s", len(circles), self._pub.resolved_name)
        if self._clusters_pub is not None:
            cloud = self._build_cluster_cloud(points_used, labels)
            if cloud is not None:
                self._clusters_pub.publish(cloud)
                rospy.loginfo(
                    "[orchard_row_mapping] Published tree clusters cloud with %d points on %s",
                    int(cloud.width) * int(cloud.height),
                    self._clusters_pub.resolved_name,
                )

    def _load_points(self) -> np.ndarray:
        if self.pcd_path is not None:
            if not self.pcd_path.is_file():
                raise RuntimeError(f"PCD not found: {self.pcd_path}")
            rospy.loginfo("[orchard_row_mapping] Loading tree map PCD: %s", self.pcd_path)
            return _load_pcd_xyz(self.pcd_path)

        rospy.loginfo("[orchard_row_mapping] Waiting for tree map topic: %s", self.input_topic)
        msg = rospy.wait_for_message(self.input_topic, PointCloud2, timeout=15.0)
        return _cloud_to_xyz(msg)

    def _build_tree_result(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, List[TreeCircle], np.ndarray]:
        points_xyz = points_xyz.astype(np.float32)
        points_xyz = _filter_points(
            points_xyz,
            z_min=self.filter_z_min,
            z_max=self.filter_z_max,
            x_min=self.filter_x_min,
            x_max=self.filter_x_max,
            y_abs_max=self.filter_y_abs_max,
        )
        if points_xyz.size == 0:
            rospy.logwarn("[orchard_row_mapping] No points after filtering; nothing to draw.")
            return points_xyz, [], np.empty((0,), dtype=np.int32)

        points_xyz = _sample_points(points_xyz, self.max_points, seed=self.sample_seed)

        if self.detection_mode in ("row_model", "row_model_peaks", "row_peaks", "row_model_peak"):
            if self.row_model_file is None:
                rospy.logwarn("[orchard_row_mapping] row_model_file is empty; falling back to cluster mode.")
            elif not self.row_model_file.is_file():
                rospy.logwarn("[orchard_row_mapping] row_model_file not found (%s); falling back to cluster mode.", self.row_model_file)
            else:
                direction_xy, perp_xy, rows = _load_row_model_file(self.row_model_file)
                circles, labels = _tree_circles_and_labels_from_row_model(
                    points_xyz=points_xyz,
                    direction_xy=direction_xy,
                    perp_xy=perp_xy,
                    rows=rows,
                    row_bandwidth=self.row_bandwidth,
                    u_bin_size=self.tree_u_bin_size,
                    smooth_window=self.tree_smooth_window,
                    peak_min_fraction=self.tree_peak_min_fraction,
                    min_separation=self.tree_min_separation,
                    u_padding=self.tree_u_padding,
                    refine_u_half_width=self.refine_u_half_width,
                    max_trees_per_row=self.max_trees_per_row,
                    max_trees=self.max_trees,
                    marker_z=self.marker_z,
                    radius_mode=self.radius_mode,
                    radius_constant=self.radius_constant,
                    radius_quantile=self.radius_quantile,
                    radius_min=self.radius_min,
                    radius_max=self.radius_max,
                )
                if circles:
                    return points_xyz, circles, labels
                rospy.logwarn("[orchard_row_mapping] No trees found from row model peaks; falling back to cluster mode.")

        circles, labels = _tree_circles_and_labels_from_cell_clusters(
            points_xyz=points_xyz,
            cell_size=self.cluster_cell_size,
            neighbor_range=self.cluster_neighbor_range,
            min_points=self.min_points_per_tree,
            max_clusters=self.max_trees,
            marker_z=self.marker_z,
            radius_mode=self.radius_mode,
            radius_constant=self.radius_constant,
            radius_quantile=self.radius_quantile,
            radius_min=self.radius_min,
            radius_max=self.radius_max,
        )
        if not circles:
            rospy.logwarn("[orchard_row_mapping] No tree clusters found (try lowering min_points_per_tree).")
            return points_xyz, [], labels

        return points_xyz, circles, labels

    def _build_cluster_cloud(self, points_xyz: np.ndarray, labels: np.ndarray) -> Optional[PointCloud2]:
        if points_xyz.size == 0 or labels.size == 0:
            return None
        labels = labels.reshape(-1).astype(np.int32)
        if labels.shape[0] != points_xyz.shape[0]:
            rospy.logwarn("[orchard_row_mapping] Cluster labels size mismatch; skipping clusters cloud.")
            return None

        mask = labels >= 0
        if not np.any(mask):
            return None

        pts = points_xyz[mask].astype(np.float32)
        ids = labels[mask].astype(np.int32)

        n_clusters = int(ids.max()) + 1 if ids.size else 0
        colors = [_pack_rgb_float(*_cluster_id_to_rgb(i)) for i in range(n_clusters)]

        header = Header()
        header.stamp = rospy.Time(0)
        header.frame_id = self.map_frame

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_points: List[Tuple[float, float, float, float, float]] = []
        for (x, y, z), cid in zip(pts.tolist(), ids.tolist()):
            rgb = colors[int(cid)] if 0 <= int(cid) < n_clusters else _pack_rgb_float(255, 255, 255)
            cloud_points.append((float(x), float(y), float(z), float(rgb), float(cid)))
        return pc2.create_cloud(header, fields, cloud_points)

    def _build_markers(self, circles: Sequence[TreeCircle]) -> MarkerArray:
        header = Header()
        header.stamp = rospy.Time(0)
        header.frame_id = self.map_frame

        out = MarkerArray()
        color = self.color
        segments = max(6, int(self.circle_segments))

        for i, circle in enumerate(circles):
            marker = Marker()
            marker.header = header
            marker.ns = "tree_circles"
            marker.id = int(i)
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = float(max(0.0, min(self.alpha, 1.0)))

            if self.marker_type in ("cylinder", "disc", "disk"):
                marker.type = Marker.CYLINDER
                marker.pose.position.x = float(circle.x)
                marker.pose.position.y = float(circle.y)
                marker.pose.position.z = float(circle.z)
                marker.scale.x = float(circle.radius * 2.0)
                marker.scale.y = float(circle.radius * 2.0)
                marker.scale.z = 0.05
            else:
                marker.type = Marker.LINE_STRIP
                marker.scale.x = float(max(0.001, self.line_width))
                points: List[Point] = []
                for k in range(segments + 1):
                    ang = (2.0 * math.pi * float(k)) / float(segments)
                    points.append(
                        Point(
                            x=float(circle.x + circle.radius * math.cos(ang)),
                            y=float(circle.y + circle.radius * math.sin(ang)),
                            z=float(circle.z),
                        )
                    )
                marker.points = points

            out.markers.append(marker)

            if self.publish_labels:
                text = Marker()
                text.header = header
                text.ns = "tree_circle_labels"
                text.id = int(i)
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = float(circle.x)
                text.pose.position.y = float(circle.y)
                text.pose.position.z = float(circle.z + float(self.label_z_offset))
                text.pose.orientation.w = 1.0
                text.scale.z = float(max(0.05, self.label_height))
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = float(max(0.0, min(self.alpha, 1.0)))
                text.text = str(int(i) + int(self.label_start_index))
                out.markers.append(text)

        if not out.markers:
            clear = Marker()
            clear.header = header
            clear.action = Marker.DELETEALL
            clear.pose.orientation.w = 1.0
            out.markers.append(clear)
        return out

    def _export_csv(self, circles: Sequence[TreeCircle], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z", "radius"])
            for i, c in enumerate(circles):
                writer.writerow([int(i), f"{c.x:.6f}", f"{c.y:.6f}", f"{c.z:.6f}", f"{c.radius:.6f}"])


def main() -> None:
    rospy.init_node("orchard_tree_circles")
    try:
        OrchardTreeCirclesNode()
    except rospy.ROSInterruptException:
        return
    rospy.spin()


if __name__ == "__main__":
    main()
