#!/usr/bin/env python3
"""Export clustering results for tree-point frames into separate Chinese folders.

Input is typically produced by `segment_bag_to_tree_pcd.py`:
  output/识别树点帧_xxx/frames.csv
  output/识别树点帧_xxx/pcd/tree_000000.pcd ...

We export (per-frame) PCDs with fields:
  x y z rgb cluster

Algorithms:
  - cell_cc    : 2D grid connected-components on XY projection
  - dbscan     : sklearn DBSCAN on XY projection
  - euclidean  : DBSCAN(min_samples=1) + cluster size filter (similar to PCL Euclidean clustering)
  - kmeans_xy  : sklearn KMeans on XY projection (k auto per-frame or fixed)
  - kmeans_lr  : k=2 1D clustering on cluster-center y (left/right grouping)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


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
    phi = 0.618033988749895  # golden ratio
    h = (float(cluster_id) * phi) % 1.0
    return _hsv_to_rgb(h, 0.95, 1.0)


def _pack_rgb_float(r: int, g: int, b: int) -> float:
    # PCL packs BGRA into a float32 (same as our segmentation exporter).
    import struct

    rgb_int = struct.unpack("I", struct.pack("BBBB", int(b) & 255, int(g) & 255, int(r) & 255, 255))[0]
    return struct.unpack("f", struct.pack("I", rgb_int))[0]


def _read_pcd_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header_lines: List[str] = []
        data_mode: Optional[str] = None
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Invalid PCD header: {path}")
            decoded = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded.upper().startswith("DATA"):
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
        points_count = int(header.get("POINTS", header.get("WIDTH", "0"))) or 0
        if points_count <= 0:
            return np.empty((0, 3), dtype=np.float32)
        if data_mode is None:
            raise RuntimeError(f"Missing DATA line in PCD: {path}")

        if data_mode == "ascii":
            data = np.loadtxt(handle, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            name_to_index = {name: idx for idx, name in enumerate(fields)}
            for name in ("x", "y", "z"):
                if name not in name_to_index:
                    raise RuntimeError(f"PCD missing field '{name}': {path}")
            return data[:, [name_to_index["x"], name_to_index["y"], name_to_index["z"]]].astype(np.float32, copy=False)

        if data_mode != "binary":
            raise RuntimeError(f"Unsupported PCD DATA mode (expected binary/ascii): {data_mode} ({path})")

        if fields[:3] != ["x", "y", "z"]:
            # Our exporters always write xyz first; keep it simple to avoid a full generic PCD parser here.
            raise RuntimeError(f"Unsupported PCD field order (expected 'x y z ...'): {fields} ({path})")

        raw = handle.read(int(points_count) * 12)
        if len(raw) < int(points_count) * 12:
            raise RuntimeError(f"PCD data too short: {path}")
        return np.frombuffer(raw, dtype=np.float32).reshape((-1, 3)).astype(np.float32, copy=False)


def _write_pcd_xyz_rgb_cluster(path: Path, xyz: np.ndarray, rgb: np.ndarray, cluster: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = xyz.astype(np.float32, copy=False).reshape((-1, 3))
    rgb = rgb.astype(np.float32, copy=False).reshape((-1,))
    cluster = cluster.astype(np.float32, copy=False).reshape((-1,))
    if pts.shape[0] != rgb.shape[0] or pts.shape[0] != cluster.shape[0]:
        raise ValueError("xyz/rgb/cluster size mismatch")

    out = np.empty((pts.shape[0], 5), dtype=np.float32)
    out[:, 0:3] = pts
    out[:, 3] = rgb
    out[:, 4] = cluster

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb cluster\n"
        "SIZE 4 4 4 4 4\n"
        "TYPE F F F F F\n"
        "COUNT 1 1 1 1 1\n"
        f"WIDTH {out.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {out.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        if out.size:
            handle.write(out.tobytes())


def _cluster_cells(
    xy: np.ndarray,
    cell_size: float,
    neighbor_range: int,
    min_points: int,
    max_clusters: int,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape((-1, 2))
    if xy.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)
    cell_size = float(cell_size)
    if not (cell_size > 0.0):
        raise ValueError("cell_size must be > 0")

    neighbor_range = max(0, int(neighbor_range))
    min_points = max(1, int(min_points))
    max_clusters = int(max_clusters)

    grid = np.floor(xy / cell_size).astype(np.int32)
    cells: Dict[Tuple[int, int], List[int]] = {}
    for idx, (cx, cy) in enumerate(grid.tolist()):
        cells.setdefault((int(cx), int(cy)), []).append(int(idx))

    labels = np.full((xy.shape[0],), -1, dtype=np.int32)
    visited: set[Tuple[int, int]] = set()
    cluster_id = 0
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
                    if nb in visited or nb not in cells:
                        continue
                    visited.add(nb)
                    queue.append(nb)

        if len(member_indices) < min_points:
            continue
        labels[np.asarray(member_indices, dtype=np.int32)] = int(cluster_id)
        cluster_id += 1
        if max_clusters > 0 and cluster_id >= max_clusters:
            break
    return labels


def _reindex_labels(labels: np.ndarray, *, keep_noise: bool = True) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32).reshape((-1,))
    if labels.size == 0:
        return labels
    uniq = sorted(int(v) for v in set(labels.tolist()) if int(v) >= 0)
    mapping = {old: new for new, old in enumerate(uniq)}
    out = np.full_like(labels, -1)
    for old, new in mapping.items():
        out[labels == int(old)] = int(new)
    if keep_noise:
        out[labels < 0] = -1
    return out


def _filter_small_clusters(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32).reshape((-1,))
    if labels.size == 0:
        return labels
    min_cluster_size = int(min_cluster_size)
    if min_cluster_size <= 1:
        return labels
    keep = labels.copy()
    uniq, counts = np.unique(keep[keep >= 0], return_counts=True)
    for cid, cnt in zip(uniq.tolist(), counts.tolist()):
        if int(cnt) < min_cluster_size:
            keep[keep == int(cid)] = -1
    return keep


def _labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32).reshape((-1,))
    out = np.empty((labels.shape[0],), dtype=np.float32)
    noise_rgb = _pack_rgb_float(200, 200, 200)
    out[:] = float(noise_rgb)
    if labels.size == 0:
        return out
    max_id = int(labels.max()) if np.any(labels >= 0) else -1
    if max_id < 0:
        return out
    colors = [_pack_rgb_float(*_cluster_id_to_rgb(i)) for i in range(max_id + 1)]
    for cid in range(max_id + 1):
        out[labels == cid] = float(colors[cid])
    return out


def _select_points_mask_for_clustering(
    xyz: np.ndarray,
    *,
    mode: str,
    fixed_z_min: float,
    fixed_z_max: float,
    q_min: float,
    q_max: float,
    bottom_frac: float,
    ground_q: float,
    ground_offset_min: float,
    ground_offset_max: float,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape((-1, 3))
    if int(xyz.shape[0]) == 0:
        return np.zeros((0,), dtype=bool)

    mode = str(mode or "").strip().lower()
    z_all = xyz[:, 2].astype(np.float64, copy=False)
    finite = np.isfinite(z_all)
    if not bool(np.any(finite)):
        return np.zeros((int(xyz.shape[0]),), dtype=bool)

    if mode in ("", "none", "all", "full"):
        return finite

    if mode in ("fixed", "abs", "absolute"):
        z0 = float(fixed_z_min)
        z1 = float(fixed_z_max)
        if z1 < z0:
            z0, z1 = z1, z0
        return finite & (z_all >= float(z0)) & (z_all <= float(z1))

    z = z_all[finite]

    if mode in ("frame-quantile", "quantile", "band-quantile"):
        q0 = float(max(0.0, min(float(q_min), 1.0)))
        q1 = float(max(0.0, min(float(q_max), 1.0)))
        if q1 < q0:
            q0, q1 = q1, q0
        z0 = float(np.quantile(z, q0))
        z1 = float(np.quantile(z, q1))
        if z1 < z0:
            z0, z1 = z1, z0
        return finite & (z_all >= float(z0)) & (z_all <= float(z1))

    if mode in ("bottom", "frame-bottom"):
        frac = float(max(0.0, min(float(bottom_frac), 1.0)))
        z_th = float(np.quantile(z, frac))
        return finite & (z_all <= float(z_th))

    if mode in ("ground-offset", "ground"):
        gq = float(max(0.0, min(float(ground_q), 1.0)))
        ground_z = float(np.quantile(z, gq))
        z0 = float(ground_z) + float(ground_offset_min)
        z1 = float(ground_z) + float(ground_offset_max)
        if z1 < z0:
            z0, z1 = z1, z0
        return finite & (z_all >= float(z0)) & (z_all <= float(z1))

    raise ValueError(f"Unsupported cluster-z-mode: {mode}")


def _kmeans_1d_two_clusters(values: np.ndarray, *, iters: int = 10) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(values, dtype=np.float32).reshape((-1,))
    if int(x.shape[0]) == 0:
        return np.zeros((0,), dtype=bool), 0.0, 0.0
    c0 = float(np.min(x))
    c1 = float(np.max(x))
    if not (math.isfinite(c0) and math.isfinite(c1)) or abs(c1 - c0) < 1e-12:
        labels = np.zeros((int(x.shape[0]),), dtype=bool)
        return labels, float(c0), float(c1)

    labels = np.zeros((int(x.shape[0]),), dtype=bool)
    for _ in range(int(max(1, iters))):
        d0 = np.abs(x - float(c0))
        d1 = np.abs(x - float(c1))
        new_labels = d0 <= d1
        if bool(np.array_equal(new_labels, labels)):
            break
        labels = new_labels
        if not bool(np.any(labels)) or bool(np.all(labels)):
            break
        c0_new = float(np.mean(x[labels]))
        c1_new = float(np.mean(x[~labels]))
        if not (math.isfinite(c0_new) and math.isfinite(c1_new)):
            break
        if abs(c0_new - c0) < 1e-6 and abs(c1_new - c1) < 1e-6:
            c0, c1 = c0_new, c1_new
            break
        c0, c1 = c0_new, c1_new

    return labels, float(c0), float(c1)


def _cluster_kmeans_xy(
    xy: np.ndarray,
    *,
    k: int,
    min_cluster_size: int,
    sample_points: int,
    random_state: int,
    max_iter: int,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape((-1, 2))
    n = int(xy.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    k = int(k)
    if k <= 0:
        return np.full((n,), -1, dtype=np.int32)
    if k > n:
        k = n

    from sklearn.cluster import KMeans

    fit_xy = xy
    sample_points = int(sample_points)
    if sample_points > 0 and n > sample_points:
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(n, size=int(sample_points), replace=False)
        fit_xy = xy[idx]

    model = KMeans(
        n_clusters=int(k),
        random_state=int(random_state),
        n_init="auto",
        max_iter=int(max_iter),
    )
    model.fit(fit_xy)
    labels = model.predict(xy).astype(np.int32, copy=False)
    labels = _filter_small_clusters(labels, int(min_cluster_size))
    labels = _reindex_labels(labels)
    return labels


@dataclass
class FrameInfo:
    index: int
    t_sec: float
    pcd_path: Path


def _load_frames_csv(frames_csv: Path) -> List[FrameInfo]:
    out: List[FrameInfo] = []
    with frames_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                idx = int(str(row.get("index", "")).strip())
            except Exception:
                continue
            try:
                t_sec = float(str(row.get("t_sec", "")).strip() or "nan")
            except Exception:
                t_sec = float("nan")
            p = str(row.get("pcd_path", "")).strip()
            if not p:
                continue
            out.append(FrameInfo(index=idx, t_sec=t_sec, pcd_path=Path(p)))
    out.sort(key=lambda x: x.index)
    return out


def _ensure_empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, type=str, help="Input dir (e.g. output/识别树点帧_xxx)")
    parser.add_argument("--out-root", default="output", type=str, help="Parent output directory")
    parser.add_argument("--tag", default="", type=str, help="Optional suffix appended to folder names")
    parser.add_argument(
        "--stamp",
        default="",
        type=str,
        help="Optional output timestamp (e.g. 20260121_164528) to reuse existing output folders.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip frames whose output PCDs already exist (use with --stamp/--tag to continue an interrupted run).",
    )

    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument("--cell-size", type=float, default=0.10)
    parser.add_argument("--neighbor-range", type=int, default=1)
    parser.add_argument("--min-points", type=int, default=40)

    parser.add_argument(
        "--cluster-z-mode",
        choices=["none", "fixed", "frame-quantile", "bottom", "ground-offset"],
        default="none",
        help=(
            "Select a Z slice used for clustering only (avoids crown/ground linking trees). "
            "Output PCD still keeps all input points; points outside the slice are marked cluster=-1."
        ),
    )
    parser.add_argument("--cluster-z-min", type=float, default=0.7, help="Used when --cluster-z-mode=fixed.")
    parser.add_argument("--cluster-z-max", type=float, default=1.3, help="Used when --cluster-z-mode=fixed.")
    parser.add_argument("--cluster-z-q-min", type=float, default=0.2, help="Used when --cluster-z-mode=frame-quantile.")
    parser.add_argument("--cluster-z-q-max", type=float, default=0.6, help="Used when --cluster-z-mode=frame-quantile.")
    parser.add_argument("--cluster-z-bottom-frac", type=float, default=0.3, help="Used when --cluster-z-mode=bottom.")
    parser.add_argument("--cluster-ground-q", type=float, default=0.05, help="Used when --cluster-z-mode=ground-offset.")
    parser.add_argument(
        "--cluster-ground-offset-min",
        type=float,
        default=0.7,
        help="Used when --cluster-z-mode=ground-offset (meters).",
    )
    parser.add_argument(
        "--cluster-ground-offset-max",
        type=float,
        default=1.3,
        help="Used when --cluster-z-mode=ground-offset (meters).",
    )

    parser.add_argument("--dbscan-eps", type=float, default=0.25)
    parser.add_argument("--dbscan-min-samples", type=int, default=10)
    parser.add_argument("--dbscan-min-cluster-size", type=int, default=40)

    parser.add_argument("--euclid-eps", type=float, default=0.25)
    parser.add_argument("--euclid-min-cluster-size", type=int, default=40)

    parser.add_argument("--disable-cell", action="store_true")
    parser.add_argument("--disable-dbscan", action="store_true")
    parser.add_argument("--disable-euclid", action="store_true")
    parser.add_argument("--disable-kmeans", action="store_true")
    parser.add_argument("--enable-kmeans-xy", action="store_true", help="Enable point-cloud KMeans clustering on XY projection.")
    parser.add_argument("--kmeans-xy-k", type=int, default=0, help="Fixed K for KMeans XY (0=auto per-frame).")
    parser.add_argument(
        "--kmeans-xy-k-source",
        choices=["cell_cc", "dbscan", "euclidean"],
        default="cell_cc",
        help="When --kmeans-xy-k=0, choose K from another algorithm's cluster count (default: cell_cc).",
    )
    parser.add_argument("--kmeans-xy-max-k", type=int, default=80, help="Clamp auto K to this max value (0=disable clamp).")
    parser.add_argument("--kmeans-xy-min-cluster-size", type=int, default=40)
    parser.add_argument("--kmeans-xy-sample-points", type=int, default=20000, help="Subsample points for fitting centers (0=use all).")
    parser.add_argument("--kmeans-xy-random-state", type=int, default=0)
    parser.add_argument("--kmeans-xy-max-iter", type=int, default=100)
    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    frames_csv = in_dir / "frames.csv"
    if not frames_csv.is_file():
        raise FileNotFoundError(f"frames.csv not found: {frames_csv}")

    frames = _load_frames_csv(frames_csv)
    if not frames:
        raise RuntimeError(f"No frames found in {frames_csv}")

    stamp = str(args.stamp).strip() or time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag.strip()}" if str(args.tag).strip() else ""

    out_root = Path(args.out_root).expanduser().resolve()
    out_cell = out_root / f"聚类_栅格连通域_{stamp}{suffix}"
    out_db = out_root / f"聚类_DBSCAN_{stamp}{suffix}"
    out_eu = out_root / f"聚类_欧式聚类_{stamp}{suffix}"
    out_km = out_root / f"聚类_KMeans左右_{stamp}{suffix}"
    out_kmxy = out_root / f"聚类_KMeans_{stamp}{suffix}"

    writers: Dict[str, Tuple[Path, csv.writer]] = {}
    csv_files: Dict[str, object] = {}
    try:
        if not args.disable_cell:
            (out_cell / "pcd").mkdir(parents=True, exist_ok=True)
            frames_path = out_cell / "frames.csv"
            mode = "a" if bool(args.resume) and frames_path.is_file() else "w"
            f = frames_path.open(mode, newline="")
            csv_files["cell"] = f
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["index", "t_sec", "points", "clusters", "pcd_path"])
            writers["cell"] = (out_cell, w)
        if not args.disable_dbscan:
            (out_db / "pcd").mkdir(parents=True, exist_ok=True)
            frames_path = out_db / "frames.csv"
            mode = "a" if bool(args.resume) and frames_path.is_file() else "w"
            f = frames_path.open(mode, newline="")
            csv_files["dbscan"] = f
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["index", "t_sec", "points", "clusters", "pcd_path"])
            writers["dbscan"] = (out_db, w)
        if not args.disable_euclid:
            (out_eu / "pcd").mkdir(parents=True, exist_ok=True)
            frames_path = out_eu / "frames.csv"
            mode = "a" if bool(args.resume) and frames_path.is_file() else "w"
            f = frames_path.open(mode, newline="")
            csv_files["euclid"] = f
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["index", "t_sec", "points", "clusters", "pcd_path"])
            writers["euclid"] = (out_eu, w)
        if not args.disable_kmeans:
            (out_km / "pcd").mkdir(parents=True, exist_ok=True)
            frames_path = out_km / "frames.csv"
            mode = "a" if bool(args.resume) and frames_path.is_file() else "w"
            f = frames_path.open(mode, newline="")
            csv_files["kmeans"] = f
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["index", "t_sec", "centers", "left", "right", "pcd_path"])
            writers["kmeans"] = (out_km, w)
        if bool(args.enable_kmeans_xy):
            (out_kmxy / "pcd").mkdir(parents=True, exist_ok=True)
            frames_path = out_kmxy / "frames.csv"
            mode = "a" if bool(args.resume) and frames_path.is_file() else "w"
            f = frames_path.open(mode, newline="")
            csv_files["kmeans_xy"] = f
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["index", "t_sec", "points", "k", "clusters", "pcd_path"])
            writers["kmeans_xy"] = (out_kmxy, w)

        run_meta = {
            "input_dir": str(in_dir),
            "frames_csv": str(frames_csv),
            "every": int(args.every),
            "max_frames": int(args.max_frames),
            "resume": bool(args.resume),
            "stamp": str(stamp),
            "cluster_z": {
                "mode": str(args.cluster_z_mode).strip(),
                "fixed": {"z_min": float(args.cluster_z_min), "z_max": float(args.cluster_z_max)},
                "frame_quantile": {"q_min": float(args.cluster_z_q_min), "q_max": float(args.cluster_z_q_max)},
                "bottom": {"frac": float(args.cluster_z_bottom_frac)},
                "ground_offset": {
                    "ground_q": float(args.cluster_ground_q),
                    "offset_min": float(args.cluster_ground_offset_min),
                    "offset_max": float(args.cluster_ground_offset_max),
                },
                "note": "Only affects which points are used to compute clusters; output PCD keeps all points.",
            },
            "cell_cc": {
                "enabled": bool(not args.disable_cell),
                "cell_size": float(args.cell_size),
                "neighbor_range": int(args.neighbor_range),
                "min_points": int(args.min_points),
            },
            "dbscan": {
                "enabled": bool(not args.disable_dbscan),
                "eps": float(args.dbscan_eps),
                "min_samples": int(args.dbscan_min_samples),
                "min_cluster_size": int(args.dbscan_min_cluster_size),
            },
            "euclidean": {
                "enabled": bool(not args.disable_euclid),
                "eps": float(args.euclid_eps),
                "min_cluster_size": int(args.euclid_min_cluster_size),
                "note": "Implemented via sklearn DBSCAN(min_samples=1) + cluster size filter.",
            },
            "kmeans_lr": {
                "enabled": bool(not args.disable_kmeans),
                "note": "Runs k=2 clustering on cluster-center y values from cell_cc clusters.",
            },
            "kmeans_xy": {
                "enabled": bool(args.enable_kmeans_xy),
                "k_fixed": int(args.kmeans_xy_k),
                "k_source": str(args.kmeans_xy_k_source),
                "max_k": int(args.kmeans_xy_max_k),
                "min_cluster_size": int(args.kmeans_xy_min_cluster_size),
                "sample_points": int(args.kmeans_xy_sample_points),
                "random_state": int(args.kmeans_xy_random_state),
                "max_iter": int(args.kmeans_xy_max_iter),
            },
        }
        import json

        for out_dir in (out_cell, out_db, out_eu, out_km, out_kmxy):
            if out_dir.exists():
                (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        from sklearn.cluster import DBSCAN

        processed = 0
        for frame in frames:
            if int(args.every) > 1 and (int(frame.index) % int(args.every)) != 0:
                continue
            if args.resume:
                required: List[Path] = []
                if not args.disable_cell:
                    required.append((out_cell / "pcd" / f"cell_{frame.index:06d}.pcd").resolve())
                if not args.disable_dbscan:
                    required.append((out_db / "pcd" / f"dbscan_{frame.index:06d}.pcd").resolve())
                if not args.disable_euclid:
                    required.append((out_eu / "pcd" / f"euclid_{frame.index:06d}.pcd").resolve())
                if not args.disable_kmeans:
                    required.append((out_km / "pcd" / f"centers_{frame.index:06d}.pcd").resolve())
                if bool(args.enable_kmeans_xy):
                    required.append((out_kmxy / "pcd" / f"kmeans_{frame.index:06d}.pcd").resolve())
                if required and all(p.is_file() for p in required):
                    continue
            xyz = _read_pcd_xyz(frame.pcd_path)
            n_pts = int(xyz.shape[0])
            xy = xyz[:, :2] if n_pts else np.empty((0, 2), dtype=np.float32)

            mask_cluster = (
                _select_points_mask_for_clustering(
                    xyz,
                    mode=str(args.cluster_z_mode),
                    fixed_z_min=float(args.cluster_z_min),
                    fixed_z_max=float(args.cluster_z_max),
                    q_min=float(args.cluster_z_q_min),
                    q_max=float(args.cluster_z_q_max),
                    bottom_frac=float(args.cluster_z_bottom_frac),
                    ground_q=float(args.cluster_ground_q),
                    ground_offset_min=float(args.cluster_ground_offset_min),
                    ground_offset_max=float(args.cluster_ground_offset_max),
                )
                if n_pts
                else np.zeros((0,), dtype=bool)
            )
            if mask_cluster.size and not bool(np.any(mask_cluster)):
                # Avoid producing completely empty clusters if the selected slice happens to be empty.
                mask_cluster[:] = True
            idx_cluster = np.nonzero(mask_cluster)[0] if n_pts else np.zeros((0,), dtype=np.int32)
            xy_cluster = xyz[idx_cluster, :2] if idx_cluster.size else np.empty((0, 2), dtype=np.float32)

            labels_cell: Optional[np.ndarray] = None
            labels_db: Optional[np.ndarray] = None
            labels_eu: Optional[np.ndarray] = None
            if not args.disable_cell:
                labels_cell = _cluster_cells(
                    xy_cluster,
                    cell_size=float(args.cell_size),
                    neighbor_range=int(args.neighbor_range),
                    min_points=int(args.min_points),
                    max_clusters=0,
                )
                labels_cell = _reindex_labels(labels_cell)
                labels_full = np.full((n_pts,), -1, dtype=np.int32)
                if idx_cluster.size:
                    labels_full[idx_cluster] = labels_cell.astype(np.int32, copy=False)
                labels_cell = labels_full
                rgb = _labels_to_rgb(labels_cell)
                out_pcd = (out_cell / "pcd" / f"cell_{frame.index:06d}.pcd").resolve()
                _write_pcd_xyz_rgb_cluster(out_pcd, xyz, rgb, labels_cell)
                n_clusters = int(labels_cell.max()) + 1 if np.any(labels_cell >= 0) else 0
                writers["cell"][1].writerow([frame.index, f"{frame.t_sec:.6f}", n_pts, n_clusters, str(out_pcd)])

            if not args.disable_dbscan:
                if int(xy_cluster.shape[0]) > 0:
                    labels_db_local = DBSCAN(eps=float(args.dbscan_eps), min_samples=int(args.dbscan_min_samples), n_jobs=-1).fit_predict(xy_cluster)
                    labels_db_local = _filter_small_clusters(labels_db_local, int(args.dbscan_min_cluster_size))
                    labels_db_local = _reindex_labels(labels_db_local)
                else:
                    labels_db_local = np.empty((0,), dtype=np.int32)
                labels_full = np.full((n_pts,), -1, dtype=np.int32)
                if idx_cluster.size and int(labels_db_local.shape[0]) == int(idx_cluster.shape[0]):
                    labels_full[idx_cluster] = labels_db_local.astype(np.int32, copy=False)
                labels_db = labels_full
                rgb = _labels_to_rgb(labels_db)
                out_pcd = (out_db / "pcd" / f"dbscan_{frame.index:06d}.pcd").resolve()
                _write_pcd_xyz_rgb_cluster(out_pcd, xyz, rgb, labels_db)
                n_clusters = int(labels_db.max()) + 1 if np.any(labels_db >= 0) else 0
                writers["dbscan"][1].writerow([frame.index, f"{frame.t_sec:.6f}", n_pts, n_clusters, str(out_pcd)])

            if not args.disable_euclid:
                if int(xy_cluster.shape[0]) > 0:
                    labels_eu_local = DBSCAN(eps=float(args.euclid_eps), min_samples=1, n_jobs=-1).fit_predict(xy_cluster)
                    labels_eu_local = _filter_small_clusters(labels_eu_local, int(args.euclid_min_cluster_size))
                    labels_eu_local = _reindex_labels(labels_eu_local)
                else:
                    labels_eu_local = np.empty((0,), dtype=np.int32)
                labels_full = np.full((n_pts,), -1, dtype=np.int32)
                if idx_cluster.size and int(labels_eu_local.shape[0]) == int(idx_cluster.shape[0]):
                    labels_full[idx_cluster] = labels_eu_local.astype(np.int32, copy=False)
                labels_eu = labels_full
                rgb = _labels_to_rgb(labels_eu)
                out_pcd = (out_eu / "pcd" / f"euclid_{frame.index:06d}.pcd").resolve()
                _write_pcd_xyz_rgb_cluster(out_pcd, xyz, rgb, labels_eu)
                n_clusters = int(labels_eu.max()) + 1 if np.any(labels_eu >= 0) else 0
                writers["euclid"][1].writerow([frame.index, f"{frame.t_sec:.6f}", n_pts, n_clusters, str(out_pcd)])

            if not args.disable_kmeans:
                if labels_cell is None:
                    # KMeans output is defined on cell_cc clusters; skip if cell_cc disabled.
                    pass
                else:
                    uniq = [int(v) for v in sorted(set(labels_cell.tolist())) if int(v) >= 0]
                    centers = []
                    for cid in uniq:
                        pts = xyz[labels_cell == cid]
                        if pts.size == 0:
                            continue
                        centers.append(np.median(pts, axis=0).astype(np.float32))
                    centers_xyz = np.asarray(centers, dtype=np.float32).reshape((-1, 3))
                    n_centers = int(centers_xyz.shape[0])
                    left_count = 0
                    right_count = 0
                    if n_centers >= 2:
                        mask_a, ca, cb = _kmeans_1d_two_clusters(centers_xyz[:, 1], iters=10)
                        ya = float(np.median(centers_xyz[mask_a, 1])) if np.any(mask_a) else float("nan")
                        yb = float(np.median(centers_xyz[~mask_a, 1])) if np.any(~mask_a) else float("nan")
                        left_is_a = ya >= yb
                        groups = np.zeros((n_centers,), dtype=np.int32)
                        groups[mask_a] = 0 if left_is_a else 1
                        groups[~mask_a] = 1 if left_is_a else 0
                        left_count = int(np.sum(groups == 0))
                        right_count = int(np.sum(groups == 1))
                    else:
                        groups = np.zeros((n_centers,), dtype=np.int32)
                        left_count = int(n_centers)
                        right_count = 0

                    rgb = np.empty((n_centers,), dtype=np.float32)
                    rgb[:] = float(_pack_rgb_float(255, 255, 255))
                    if n_centers:
                        rgb[groups == 0] = float(_pack_rgb_float(56, 188, 75))   # left: green
                        rgb[groups == 1] = float(_pack_rgb_float(255, 160, 40))  # right: orange
                    out_pcd = (out_km / "pcd" / f"centers_{frame.index:06d}.pcd").resolve()
                    _write_pcd_xyz_rgb_cluster(out_pcd, centers_xyz, rgb, groups.astype(np.int32))
                    writers["kmeans"][1].writerow([frame.index, f"{frame.t_sec:.6f}", n_centers, left_count, right_count, str(out_pcd)])

            if bool(args.enable_kmeans_xy):
                k = int(args.kmeans_xy_k)
                if k <= 0:
                    src = str(args.kmeans_xy_k_source).strip().lower()
                    if src == "cell_cc":
                        if labels_cell is not None:
                            k = (int(labels_cell.max()) + 1) if np.any(labels_cell >= 0) else 0
                        else:
                            labels_tmp = _cluster_cells(
                                xy_cluster,
                                cell_size=float(args.cell_size),
                                neighbor_range=int(args.neighbor_range),
                                min_points=int(args.min_points),
                                max_clusters=0,
                            )
                            labels_tmp = _reindex_labels(labels_tmp)
                            k = (int(labels_tmp.max()) + 1) if np.any(labels_tmp >= 0) else 0
                    elif src == "dbscan":
                        k = (int(labels_db.max()) + 1) if (labels_db is not None and np.any(labels_db >= 0)) else 0
                    else:  # euclidean
                        k = (int(labels_eu.max()) + 1) if (labels_eu is not None and np.any(labels_eu >= 0)) else 0

                k = max(0, int(k))
                max_k = int(args.kmeans_xy_max_k)
                if max_k > 0:
                    k = min(int(k), int(max_k))

                if int(xy_cluster.shape[0]) > 0 and k > 0:
                    labels_km_local = _cluster_kmeans_xy(
                        xy_cluster,
                        k=int(k),
                        min_cluster_size=int(args.kmeans_xy_min_cluster_size),
                        sample_points=int(args.kmeans_xy_sample_points),
                        random_state=int(args.kmeans_xy_random_state),
                        max_iter=int(args.kmeans_xy_max_iter),
                    )
                else:
                    labels_km_local = np.full((int(xy_cluster.shape[0]),), -1, dtype=np.int32)

                labels_full = np.full((n_pts,), -1, dtype=np.int32)
                if idx_cluster.size and int(labels_km_local.shape[0]) == int(idx_cluster.shape[0]):
                    labels_full[idx_cluster] = labels_km_local.astype(np.int32, copy=False)
                labels_km = labels_full
                rgb = _labels_to_rgb(labels_km)
                out_pcd = (out_kmxy / "pcd" / f"kmeans_{frame.index:06d}.pcd").resolve()
                _write_pcd_xyz_rgb_cluster(out_pcd, xyz, rgb, labels_km)
                n_clusters = int(labels_km.max()) + 1 if np.any(labels_km >= 0) else 0
                writers["kmeans_xy"][1].writerow([frame.index, f"{frame.t_sec:.6f}", n_pts, int(k), n_clusters, str(out_pcd)])

            processed += 1
            if int(args.max_frames) > 0 and processed >= int(args.max_frames):
                break
            if processed % 50 == 0:
                print(f"[OK] processed {processed} frames")

    finally:
        for f in csv_files.values():
            try:
                f.close()
            except Exception:
                pass

    print("[OK] Done.")
    if not args.disable_cell:
        print(f"[OUT] {out_cell}")
    if not args.disable_dbscan:
        print(f"[OUT] {out_db}")
    if not args.disable_euclid:
        print(f"[OUT] {out_eu}")
    if not args.disable_kmeans:
        print(f"[OUT] {out_km}")
    if bool(args.enable_kmeans_xy):
        print(f"[OUT] {out_kmxy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
