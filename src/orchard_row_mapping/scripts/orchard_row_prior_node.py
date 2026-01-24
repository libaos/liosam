#!/usr/bin/env python3
"""Publish stable orchard row lines from a fixed global map + localization."""

from __future__ import annotations

import json
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
import tf2_ros
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class RowLine:
    v_center: float
    u_min: float
    u_max: float
    z: float


@dataclass
class RowModel:
    direction_xy: np.ndarray  # (2,) unit vector
    perp_xy: np.ndarray  # (2,) unit vector (perpendicular to direction_xy)
    rows: List[RowLine]


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-9:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _normalize_direction_xy(direction_xy: np.ndarray) -> np.ndarray:
    direction_xy = _unit(direction_xy)
    if abs(float(direction_xy[0])) >= abs(float(direction_xy[1])):
        if float(direction_xy[0]) < 0.0:
            direction_xy = -direction_xy
    else:
        if float(direction_xy[1]) < 0.0:
            direction_xy = -direction_xy
    return direction_xy.astype(np.float32)


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
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
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def _parse_float_list(param: Any) -> List[float]:
    if param is None:
        return []
    if isinstance(param, (list, tuple)):
        out: List[float] = []
        for item in param:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out
    if isinstance(param, (int, float)):
        return [float(param)]

    text = str(param).strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
        return _parse_float_list(decoded)
    except Exception:
        pass

    parts = [p.strip() for p in text.replace("[", "").replace("]", "").split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


def _parse_indexed_offsets(param: Any, count: int) -> List[float]:
    offsets = [0.0 for _ in range(max(0, int(count)))]
    if param is None:
        return offsets

    if isinstance(param, str):
        text = str(param).strip()
        if text:
            try:
                decoded = json.loads(text)
                return _parse_indexed_offsets(decoded, count)
            except Exception:
                pass

    if isinstance(param, dict):
        for k, v in param.items():
            try:
                idx = int(k)
                if 0 <= idx < len(offsets):
                    offsets[idx] = float(v)
            except Exception:
                continue
        return offsets

    values = _parse_float_list(param)
    if not values:
        return offsets
    for i, val in enumerate(values[: len(offsets)]):
        offsets[i] = float(val)
    return offsets


def _cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    points = []
    for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append((x, y, z))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Decompress LZF data as used by PCL binary_compressed PCD files."""
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

        header: dict[str, str] = {}
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

            # PCL stores binary_compressed data in a field-major layout for better compression.
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


def _build_row_model(
    points_xyz: np.ndarray,
    z_min: float,
    z_max: float,
    x_min: float,
    x_max: float,
    y_abs_max: float,
    max_points: int,
    row_cluster_gap: float,
    min_points_per_row: int,
    u_percentile: float,
    row_direction_mode: str = "auto",
    row_direction_yaw_deg: float = 0.0,
    row_detection: str = "gap",
    hist_bin_size: float = 0.1,
    hist_smooth_window: int = 9,
    hist_peak_min_fraction: float = 0.08,
    row_center_min_separation: float = 2.0,
) -> RowModel:
    if points_xyz.size == 0:
        raise RuntimeError("Empty point cloud; cannot build row model")

    points_xyz = points_xyz.astype(np.float32)
    mask = (
        (points_xyz[:, 2] >= z_min)
        & (points_xyz[:, 2] <= z_max)
        & (points_xyz[:, 0] >= x_min)
        & (points_xyz[:, 0] <= x_max)
        & (np.abs(points_xyz[:, 1]) <= y_abs_max)
    )
    points_xyz = points_xyz[mask]
    if points_xyz.shape[0] < max(10, min_points_per_row * 2):
        raise RuntimeError(f"Not enough filtered points to build row model ({points_xyz.shape[0]})")

    if max_points > 0 and points_xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points_xyz.shape[0], max_points, replace=False)
        points_xyz = points_xyz[idx]

    xy = points_xyz[:, :2]
    centroid = np.mean(xy, axis=0)
    centered = xy - centroid
    cov = (centered.T @ centered) / float(max(centered.shape[0], 1))
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    row_direction_mode = (row_direction_mode or "auto").strip().lower()
    if row_direction_mode in ("auto", "max_rows"):
        candidates = [
            _normalize_direction_xy(eig_vecs[:, 0]),
            _normalize_direction_xy(eig_vecs[:, 1]),
        ]
    elif row_direction_mode in ("pca_major", "major"):
        candidates = [_normalize_direction_xy(eig_vecs[:, int(np.argmax(eig_vals))])]
    elif row_direction_mode in ("pca_minor", "minor"):
        candidates = [_normalize_direction_xy(eig_vecs[:, int(np.argmin(eig_vals))])]
    elif row_direction_mode in ("manual_yaw", "manual", "yaw"):
        yaw_rad = float(np.deg2rad(float(row_direction_yaw_deg)))
        candidates = [_normalize_direction_xy(np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=np.float32))]
    else:
        raise RuntimeError(f"Unknown row_direction_mode: {row_direction_mode}")
    u_percentile = float(max(0.0, min(u_percentile, 49.0)))
    row_detection = (row_detection or "gap").strip().lower()
    hist_bin_size = float(hist_bin_size)
    if hist_bin_size <= 0.0:
        raise RuntimeError("hist_bin_size must be > 0")
    hist_smooth_window = int(hist_smooth_window)
    if hist_smooth_window < 1:
        raise RuntimeError("hist_smooth_window must be >= 1")
    hist_peak_min_fraction = float(hist_peak_min_fraction)
    hist_peak_min_fraction = max(0.0, min(hist_peak_min_fraction, 1.0))
    row_center_min_separation = float(row_center_min_separation)
    if row_center_min_separation <= 0.0:
        raise RuntimeError("row_center_min_separation must be > 0")

    best: Optional[RowModel] = None
    best_row_count = -1
    for direction in candidates:
        direction = _normalize_direction_xy(direction)
        perp = _unit(np.array([-float(direction[1]), float(direction[0])], dtype=np.float32))
        u = xy @ direction
        v = xy @ perp

        rows: List[RowLine] = []
        if row_detection == "gap":
            sort_idx = np.argsort(v)
            v_sorted = v[sort_idx]
            gaps = np.diff(v_sorted)
            split_indices = np.where(gaps > float(row_cluster_gap))[0]

            clusters: List[np.ndarray] = []
            start = 0
            for split in split_indices:
                end = int(split) + 1
                clusters.append(sort_idx[start:end])
                start = end
            clusters.append(sort_idx[start:])

            for cluster in clusters:
                if cluster.size < int(min_points_per_row):
                    continue
                v_center = float(np.median(v[cluster]))
                u_vals = u[cluster]
                if u_percentile > 0.0:
                    u_min_val = float(np.percentile(u_vals, u_percentile))
                    u_max_val = float(np.percentile(u_vals, 100.0 - u_percentile))
                else:
                    u_min_val = float(np.min(u_vals))
                    u_max_val = float(np.max(u_vals))
                if u_max_val <= u_min_val:
                    continue
                z_med = float(np.median(points_xyz[cluster, 2]))
                rows.append(RowLine(v_center=v_center, u_min=u_min_val, u_max=u_max_val, z=z_med))
        elif row_detection in ("hist", "histogram", "density"):
            centers = _find_row_centers_histogram(
                v=v,
                bin_size=hist_bin_size,
                smooth_window=hist_smooth_window,
                peak_min_fraction=hist_peak_min_fraction,
                min_center_separation=row_center_min_separation,
            )
            if centers.size < 2:
                continue
            centers = np.sort(centers.astype(np.float32))
            boundaries = 0.5 * (centers[:-1] + centers[1:])
            row_idx = np.searchsorted(boundaries, v)

            for center_i, center_v in enumerate(centers):
                idxs = np.where(row_idx == center_i)[0]
                if idxs.size < int(min_points_per_row):
                    continue
                u_vals = u[idxs]
                if u_percentile > 0.0:
                    u_min_val = float(np.percentile(u_vals, u_percentile))
                    u_max_val = float(np.percentile(u_vals, 100.0 - u_percentile))
                else:
                    u_min_val = float(np.min(u_vals))
                    u_max_val = float(np.max(u_vals))
                if u_max_val <= u_min_val:
                    continue
                z_med = float(np.median(points_xyz[idxs, 2]))
                rows.append(RowLine(v_center=float(center_v), u_min=u_min_val, u_max=u_max_val, z=z_med))
        else:
            raise RuntimeError(f"Unknown row_detection method: {row_detection}")

        rows.sort(key=lambda r: r.v_center)
        if len(rows) > best_row_count:
            best_row_count = len(rows)
            best = RowModel(direction_xy=direction, perp_xy=perp, rows=rows)

    if best is None or len(best.rows) < 2:
        raise RuntimeError(
            f"Need at least 2 rows, got {0 if best is None else len(best.rows)} (try adjusting filters/cluster gap)"
        )

    return best


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


def _save_model(path: Path, model: RowModel) -> None:
    data = {
        "direction_xy": [float(model.direction_xy[0]), float(model.direction_xy[1])],
        "perp_xy": [float(model.perp_xy[0]), float(model.perp_xy[1])],
        "rows": [
            {"v_center": r.v_center, "u_min": r.u_min, "u_max": r.u_max, "z": r.z}
            for r in model.rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_model(path: Path) -> RowModel:
    raw = json.loads(path.read_text(encoding="utf-8"))
    direction = _normalize_direction_xy(np.asarray(raw["direction_xy"], dtype=np.float32))
    perp = _unit(np.array([-float(direction[1]), float(direction[0])], dtype=np.float32))
    rows = [
        RowLine(
            v_center=float(row["v_center"]),
            u_min=float(row["u_min"]),
            u_max=float(row["u_max"]),
            z=float(row.get("z", 0.0)),
        )
        for row in raw["rows"]
    ]
    rows.sort(key=lambda r: r.v_center)
    return RowModel(direction_xy=direction, perp_xy=perp, rows=rows)


class OrchardRowPriorNode:
    def __init__(self) -> None:
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link_est")
        self.global_map_topic = rospy.get_param("~global_map_topic", "/liorl/localization/global_map")
        pcd_param = str(rospy.get_param("~pcd_path", "")).strip()
        self.pcd_path = Path(pcd_param).expanduser() if pcd_param else None

        cache_param = str(rospy.get_param("~row_model_file", "")).strip()
        self.model_cache_path = Path(cache_param).expanduser() if cache_param else None

        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.2))

        self.map_z_min = float(rospy.get_param("~map_z_min", -1.0))
        self.map_z_max = float(rospy.get_param("~map_z_max", 3.0))
        self.map_x_min = float(rospy.get_param("~map_x_min", -1.0e9))
        self.map_x_max = float(rospy.get_param("~map_x_max", 1.0e9))
        self.map_y_abs_max = float(rospy.get_param("~map_y_abs_max", 1.0e9))
        self.max_map_points = int(rospy.get_param("~max_map_points", 200000))
        self.row_cluster_gap = float(rospy.get_param("~row_cluster_gap", 0.8))
        self.min_points_per_row = int(rospy.get_param("~min_points_per_row", 200))
        self.u_percentile = float(rospy.get_param("~u_percentile", 2.0))
        self.row_direction_mode = str(rospy.get_param("~row_direction_mode", "auto")).strip().lower()
        self.row_direction_yaw_deg = float(rospy.get_param("~row_direction_yaw_deg", 0.0))
        self.row_detection = str(rospy.get_param("~row_detection", "gap")).strip().lower()
        self.hist_bin_size = float(rospy.get_param("~hist_bin_size", 0.1))
        self.hist_smooth_window = int(rospy.get_param("~hist_smooth_window", 9))
        self.hist_peak_min_fraction = float(rospy.get_param("~hist_peak_min_fraction", 0.08))
        row_sep_param = rospy.get_param("~row_center_min_separation", None)
        if row_sep_param is None:
            self.row_center_min_separation = max(float(self.row_cluster_gap), 2.0)
        else:
            self.row_center_min_separation = float(row_sep_param)

        self.line_length = float(rospy.get_param("~line_length", 20.0))
        self.line_width = float(rospy.get_param("~line_width", 0.15))
        self.line_alpha = float(rospy.get_param("~line_alpha", 1.0))
        self.marker_z = float(rospy.get_param("~marker_z", 0.0))
        self.marker_lifetime = float(rospy.get_param("~marker_lifetime", 0.3))
        self.colorize_all_rows = bool(rospy.get_param("~colorize_all_rows", False))

        self.publish_all_rows = bool(rospy.get_param("~publish_all_rows", False))
        self.publish_centerline = bool(rospy.get_param("~publish_centerline", True))
        self.publish_nearest_rows = max(0, int(rospy.get_param("~publish_nearest_rows", 0)))
        self.publish_all_centerlines = bool(rospy.get_param("~publish_all_centerlines", False))
        self.publish_nearest_centerlines = max(0, int(rospy.get_param("~publish_nearest_centerlines", 0)))
        self.publish_row_boundaries = bool(rospy.get_param("~publish_row_boundaries", True))
        self.publish_centerline_labels = bool(rospy.get_param("~publish_centerline_labels", False))
        self.centerline_label_mode = str(rospy.get_param("~centerline_label_mode", "absolute")).strip().lower()
        self.centerline_label_start_index = int(rospy.get_param("~centerline_label_start_index", 1))
        self.centerline_label_height = float(rospy.get_param("~centerline_label_height", 0.6))
        self.centerline_label_z_offset = float(rospy.get_param("~centerline_label_z_offset", 0.6))
        self.row_v_offsets_param = rospy.get_param("~row_v_offsets", None)
        self.row_v_offset_direction = str(rospy.get_param("~row_v_offset_direction", "perp")).strip().lower()

        if self.centerline_label_mode not in ("absolute", "relative"):
            rospy.logwarn(
                "[orchard_row_mapping] Unknown centerline_label_mode=%s; using 'absolute'",
                self.centerline_label_mode,
            )
            self.centerline_label_mode = "absolute"
        self.centerline_label_height = max(0.05, float(self.centerline_label_height))
        self.force_rebuild_model = bool(rospy.get_param("~force_rebuild_model", False))

        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self.marker_pub = rospy.Publisher("~row_markers", MarkerArray, queue_size=1)

        self.model = self._load_or_build_model()
        self._apply_row_offsets()

        rospy.loginfo(
            "[orchard_row_mapping] Row prior ready: %d rows, dir=(%.3f, %.3f)",
            len(self.model.rows),
            float(self.model.direction_xy[0]),
            float(self.model.direction_xy[1]),
        )

        period = 1.0 / max(self.publish_rate, 0.1)
        self._timer = rospy.Timer(rospy.Duration(period), self._on_timer)

    def _load_or_build_model(self) -> RowModel:
        if not self.force_rebuild_model and self.model_cache_path is not None and self.model_cache_path.is_file():
            rospy.loginfo("[orchard_row_mapping] Loading cached row model: %s", self.model_cache_path)
            return _load_model(self.model_cache_path)

        points_xyz = self._get_map_points()
        model = _build_row_model(
            points_xyz=points_xyz,
            z_min=self.map_z_min,
            z_max=self.map_z_max,
            x_min=self.map_x_min,
            x_max=self.map_x_max,
            y_abs_max=self.map_y_abs_max,
            max_points=self.max_map_points,
            row_cluster_gap=self.row_cluster_gap,
            min_points_per_row=self.min_points_per_row,
            u_percentile=self.u_percentile,
            row_direction_mode=self.row_direction_mode,
            row_direction_yaw_deg=self.row_direction_yaw_deg,
            row_detection=self.row_detection,
            hist_bin_size=self.hist_bin_size,
            hist_smooth_window=self.hist_smooth_window,
            hist_peak_min_fraction=self.hist_peak_min_fraction,
            row_center_min_separation=self.row_center_min_separation,
        )
        if self.model_cache_path is not None:
            try:
                _save_model(self.model_cache_path, model)
                rospy.loginfo("[orchard_row_mapping] Saved row model cache: %s", self.model_cache_path)
            except Exception as exc:
                rospy.logwarn("[orchard_row_mapping] Failed to save row model cache: %s", exc)
        return model

    def _apply_row_offsets(self) -> None:
        if not self.model.rows:
            return
        if self.row_v_offset_direction not in ("perp", "v"):
            rospy.logwarn(
                "[orchard_row_mapping] Unsupported row_v_offset_direction=%s; using 'perp'",
                self.row_v_offset_direction,
            )
            self.row_v_offset_direction = "perp"

        offsets = _parse_indexed_offsets(self.row_v_offsets_param, len(self.model.rows))
        if not any(abs(float(v)) > 1.0e-9 for v in offsets):
            return

        for idx, (row, off) in enumerate(zip(self.model.rows, offsets)):
            row.v_center = float(row.v_center) + float(off)
            rospy.loginfo("[orchard_row_mapping] Row %d v_center offset %+0.3f (new v_center=%0.3f)", idx, off, row.v_center)

    def _get_map_points(self) -> np.ndarray:
        if self.pcd_path is not None:
            if not self.pcd_path.is_file():
                raise RuntimeError(f"PCD path not found: {self.pcd_path}")
            rospy.loginfo("[orchard_row_mapping] Loading PCD map: %s", self.pcd_path)
            return _load_pcd_xyz(self.pcd_path)

        rospy.loginfo("[orchard_row_mapping] Waiting for global map: %s", self.global_map_topic)
        msg = rospy.wait_for_message(self.global_map_topic, PointCloud2, timeout=10.0)
        return _cloud_to_xyz(msg)

    def _lookup_robot_xy(self) -> Optional[Tuple[np.ndarray, rospy.Time]]:
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rospy.Time(0), rospy.Duration(self.tf_timeout)
            )
        except Exception:
            return None
        t = tf_msg.transform.translation
        return np.array([float(t.x), float(t.y)], dtype=np.float32), tf_msg.header.stamp

    def _make_line_marker(
        self,
        header: Header,
        marker_id: int,
        ns: str,
        p0: np.ndarray,
        p1: np.ndarray,
        z: float,
        color: Tuple[float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.line_width
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = self.line_alpha
        marker.points = [Point(x=float(p0[0]), y=float(p0[1]), z=z), Point(x=float(p1[0]), y=float(p1[1]), z=z)]
        marker.lifetime = rospy.Duration(self.marker_lifetime)
        return marker

    def _make_text_marker(
        self,
        header: Header,
        marker_id: int,
        ns: str,
        position_xy: np.ndarray,
        z: float,
        text: str,
        color: Tuple[float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(position_xy[0])
        marker.pose.position.y = float(position_xy[1])
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0
        marker.scale.z = float(self.centerline_label_height)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(self.line_alpha)
        marker.text = str(text)
        marker.lifetime = rospy.Duration(self.marker_lifetime)
        return marker

    def _row_segment_near_robot(self, row: RowLine, u_robot: float) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        half = 0.5 * float(self.line_length)
        u0 = max(float(u_robot) - half, float(row.u_min))
        u1 = min(float(u_robot) + half, float(row.u_max))
        if u1 <= u0:
            return None
        p0 = self.model.direction_xy * u0 + self.model.perp_xy * float(row.v_center)
        p1 = self.model.direction_xy * u1 + self.model.perp_xy * float(row.v_center)
        return p0, p1, float(self.marker_z)

    def _on_timer(self, _: rospy.TimerEvent) -> None:
        lookup = self._lookup_robot_xy()
        if lookup is None:
            return
        robot_xy, stamp = lookup

        u_robot = float(robot_xy @ self.model.direction_xy)
        v_robot = float(robot_xy @ self.model.perp_xy)
        v_centers = np.array([row.v_center for row in self.model.rows], dtype=np.float32)
        idx = int(np.searchsorted(v_centers, v_robot))

        left: Optional[RowLine] = None
        right: Optional[RowLine] = None
        if idx <= 0:
            right = self.model.rows[0]
        elif idx >= len(self.model.rows):
            left = self.model.rows[-1]
        else:
            left = self.model.rows[idx - 1]
            right = self.model.rows[idx]

        header = Header()
        header.stamp = stamp
        header.frame_id = self.map_frame

        markers = MarkerArray()
        if self.publish_all_rows:
            selected_rows: List[Tuple[int, RowLine, Tuple[np.ndarray, np.ndarray, float]]] = []
            if self.publish_nearest_rows > 0:
                order = np.argsort(np.abs(v_centers - float(v_robot)))
                for row_idx in order.tolist():
                    row = self.model.rows[int(row_idx)]
                    seg = self._row_segment_near_robot(row, u_robot)
                    if seg is None:
                        continue
                    selected_rows.append((int(row_idx), row, seg))
                    if len(selected_rows) >= self.publish_nearest_rows:
                        break
            else:
                for row_idx, row in enumerate(self.model.rows):
                    seg = self._row_segment_near_robot(row, u_robot)
                    if seg is None:
                        continue
                    selected_rows.append((int(row_idx), row, seg))

            selected_rows.sort(key=lambda item: float(item[1].v_center))
            total_rows = max(1, len(self.model.rows))
            for row_idx, _, seg in selected_rows:
                p0, p1, z = seg
                if self.colorize_all_rows:
                    color = _hsv_to_rgb(float(row_idx) / float(total_rows), 0.85, 1.0)
                else:
                    color = (0.2, 0.7, 1.0)
                markers.markers.append(self._make_line_marker(header, int(row_idx), "row_all", p0, p1, z, color))

            if (self.publish_centerline or self.publish_centerline_labels) and len(selected_rows) >= 2:
                centers: List[Tuple[float, int, RowLine]] = []
                for (left_idx, left_row, _), (right_idx, right_row, _) in zip(selected_rows[:-1], selected_rows[1:]):
                    if int(right_idx) != int(left_idx) + 1:
                        continue
                    center = RowLine(
                        v_center=0.5 * (float(left_row.v_center) + float(right_row.v_center)),
                        u_min=max(float(left_row.u_min), float(right_row.u_min)),
                        u_max=min(float(left_row.u_max), float(right_row.u_max)),
                        z=0.5 * (float(left_row.z) + float(right_row.z)),
                    )
                    centers.append((abs(float(center.v_center) - float(v_robot)), int(left_idx), center))

                if self.publish_nearest_centerlines > 0:
                    centers.sort(key=lambda item: item[0])
                    centers = centers[: self.publish_nearest_centerlines]
                centers.sort(key=lambda item: float(item[2].v_center))

                current_lane_idx: Optional[int] = None
                if 0 < idx < len(self.model.rows):
                    current_lane_idx = int(idx - 1)
                else:
                    lane_centers_v = 0.5 * (v_centers[:-1] + v_centers[1:])
                    if lane_centers_v.size:
                        current_lane_idx = int(np.argmin(np.abs(lane_centers_v - float(v_robot))))

                for _, lane_idx, center in centers:
                    seg = self._row_segment_near_robot(center, u_robot)
                    if seg is None:
                        continue
                    p0, p1, z = seg
                    if self.publish_centerline:
                        markers.markers.append(
                            self._make_line_marker(header, lane_idx, "row_center_all", p0, p1, z, (0.2, 0.7, 1.0))
                        )
                    if self.publish_centerline_labels:
                        if self.centerline_label_mode == "relative" and current_lane_idx is not None:
                            rel = int(lane_idx - int(current_lane_idx))
                            label = "C0" if rel == 0 else f"C{rel:+d}"
                        else:
                            label = f"C{int(lane_idx) + int(self.centerline_label_start_index)}"
                        mid = 0.5 * (p0 + p1)
                        label_z = float(self.marker_z) + float(self.centerline_label_z_offset)
                        markers.markers.append(
                            self._make_text_marker(
                                header,
                                lane_idx,
                                "row_center_all_label",
                                mid,
                                label_z,
                                label,
                                (1.0, 1.0, 1.0),
                            )
                        )
        else:
            if self.publish_row_boundaries:
                if left is not None:
                    seg = self._row_segment_near_robot(left, u_robot)
                    if seg is not None:
                        p0, p1, z = seg
                        markers.markers.append(self._make_line_marker(header, 0, "row_left", p0, p1, z, (0.1, 0.8, 0.1)))
                if right is not None:
                    seg = self._row_segment_near_robot(right, u_robot)
                    if seg is not None:
                        p0, p1, z = seg
                        markers.markers.append(self._make_line_marker(header, 1, "row_right", p0, p1, z, (0.9, 0.6, 0.1)))
            if self.publish_centerline and left is not None and right is not None:
                center = RowLine(
                    v_center=0.5 * (float(left.v_center) + float(right.v_center)),
                    u_min=max(float(left.u_min), float(right.u_min)),
                    u_max=min(float(left.u_max), float(right.u_max)),
                    z=0.5 * (float(left.z) + float(right.z)),
                )
                seg = self._row_segment_near_robot(center, u_robot)
                if seg is not None:
                    p0, p1, z = seg
                    markers.markers.append(self._make_line_marker(header, 2, "row_center", p0, p1, z, (0.2, 0.7, 1.0)))

            if self.publish_all_centerlines and len(self.model.rows) >= 2:
                centers: List[Tuple[float, int, RowLine]] = []
                for lane_idx, (left_row, right_row) in enumerate(zip(self.model.rows[:-1], self.model.rows[1:])):
                    center = RowLine(
                        v_center=0.5 * (float(left_row.v_center) + float(right_row.v_center)),
                        u_min=max(float(left_row.u_min), float(right_row.u_min)),
                        u_max=min(float(left_row.u_max), float(right_row.u_max)),
                        z=0.5 * (float(left_row.z) + float(right_row.z)),
                    )
                    if float(center.u_max) <= float(center.u_min):
                        continue
                    centers.append((abs(float(center.v_center) - float(v_robot)), int(lane_idx), center))

                if self.publish_nearest_centerlines > 0:
                    centers.sort(key=lambda item: item[0])
                    centers = centers[: self.publish_nearest_centerlines]
                centers.sort(key=lambda item: float(item[2].v_center))

                current_lane_idx: Optional[int] = None
                if 0 < idx < len(self.model.rows):
                    current_lane_idx = int(idx - 1)
                else:
                    lane_centers_v = 0.5 * (v_centers[:-1] + v_centers[1:])
                    if lane_centers_v.size:
                        current_lane_idx = int(np.argmin(np.abs(lane_centers_v - float(v_robot))))

                for _, lane_idx, center in centers:
                    seg = self._row_segment_near_robot(center, u_robot)
                    if seg is None:
                        continue
                    p0, p1, z = seg
                    markers.markers.append(
                        self._make_line_marker(header, lane_idx, "row_center_all", p0, p1, z, (0.2, 0.7, 1.0))
                    )
                    if self.publish_centerline_labels:
                        if self.centerline_label_mode == "relative" and current_lane_idx is not None:
                            rel = int(lane_idx - int(current_lane_idx))
                            label = "C0" if rel == 0 else f"C{rel:+d}"
                        else:
                            label = f"C{int(lane_idx) + int(self.centerline_label_start_index)}"
                        mid = 0.5 * (p0 + p1)
                        label_z = float(self.marker_z) + float(self.centerline_label_z_offset)
                        markers.markers.append(
                            self._make_text_marker(
                                header,
                                lane_idx,
                                "row_center_all_label",
                                mid,
                                label_z,
                                label,
                                (1.0, 1.0, 1.0),
                            )
                        )

        if markers.markers:
            self.marker_pub.publish(markers)
        else:
            clear = Marker()
            clear.header = header
            clear.action = Marker.DELETEALL
            clear.pose.orientation.w = 1.0
            self.marker_pub.publish(MarkerArray(markers=[clear]))


def main() -> None:
    rospy.init_node("orchard_row_prior_node")
    try:
        OrchardRowPriorNode()
    except rospy.ROSInterruptException:
        return
    rospy.spin()


if __name__ == "__main__":
    main()
