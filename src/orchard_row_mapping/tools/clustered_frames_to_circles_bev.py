#!/usr/bin/env python3
"""Per-frame clustered PCD -> circles + BEV PNGs.

Input: a directory produced by `cluster_tree_frames_variants.py`:
  <in-dir>/frames.csv
  <in-dir>/pcd/{cell,dbscan,euclid}_000000.pcd  (FIELDS x y z rgb cluster)

Output:
  <out-dir>/circles/circles_000000.json
  <out-dir>/centers_pcd/centers_000000.pcd      (FIELDS x y z radius)
  <out-dir>/png/bev_000000.png
  <out-dir>/frames.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TreeCircle:
    x: float
    y: float
    z: float
    radius: float
    points: int = 0


def _merge_close_circles(
    circles: Sequence[TreeCircle],
    *,
    max_center_dist: float,
) -> List[TreeCircle]:
    circles = list(circles)
    n = int(len(circles))
    if n <= 1:
        return circles
    max_center_dist = float(max_center_dist)
    if not (max_center_dist > 0.0):
        return circles

    centers = np.asarray([[c.x, c.y] for c in circles], dtype=np.float64).reshape((-1, 2))
    radii = np.asarray([c.radius for c in circles], dtype=np.float64).reshape((-1,))
    weights = np.asarray([max(1, int(getattr(c, "points", 0) or 0)) for c in circles], dtype=np.float64).reshape((-1,))

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    thr2 = float(max_center_dist) ** 2
    for i in range(n):
        xi, yi = float(centers[i, 0]), float(centers[i, 1])
        for j in range(i + 1, n):
            dx = xi - float(centers[j, 0])
            dy = yi - float(centers[j, 1])
            if float(dx * dx + dy * dy) <= thr2:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(int(r), []).append(int(i))

    merged: List[TreeCircle] = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(circles[int(idxs[0])])
            continue

        w = weights[idxs]
        wsum = float(np.sum(w)) if w.size else float(len(idxs))
        if not (wsum > 0.0):
            wsum = float(len(idxs))
            w = np.ones((len(idxs),), dtype=np.float64)

        cx = float(np.sum(centers[idxs, 0] * w) / wsum)
        cy = float(np.sum(centers[idxs, 1] * w) / wsum)
        zs = [circles[i].z for i in idxs]
        z = float(np.median(np.asarray(zs, dtype=np.float64))) if zs else 0.0

        # Radius: cover all original circles (enclosing union).
        d = centers[idxs, :] - np.asarray([[cx, cy]], dtype=np.float64)
        dist = np.sqrt(np.sum(d**2, axis=1))
        r = float(np.max(dist + radii[idxs])) if dist.size else float(np.max(radii[idxs]))
        pts = int(sum(int(getattr(circles[i], "points", 0) or 0) for i in idxs))
        merged.append(TreeCircle(x=cx, y=cy, z=z, radius=r, points=pts))

    merged.sort(key=lambda c: (c.y, c.x))
    return merged


def _dtype_from_type_size(type_char: str, size: int) -> np.dtype:
    type_char = str(type_char).upper()
    size = int(size)
    if type_char == "F":
        if size == 4:
            return np.dtype("<f4")
        if size == 8:
            return np.dtype("<f8")
    if type_char == "U":
        if size == 1:
            return np.dtype("<u1")
        if size == 2:
            return np.dtype("<u2")
        if size == 4:
            return np.dtype("<u4")
        if size == 8:
            return np.dtype("<u8")
    if type_char == "I":
        if size == 1:
            return np.dtype("<i1")
        if size == 2:
            return np.dtype("<i2")
        if size == 4:
            return np.dtype("<i4")
        if size == 8:
            return np.dtype("<i8")
    raise ValueError(f"Unsupported PCD dtype: TYPE={type_char} SIZE={size}")


def _read_pcd_xyz_cluster(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        header: Dict[str, str] = {}
        data_mode: Optional[str] = None
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Invalid PCD header: {path}")
            decoded = line.decode("utf-8", errors="ignore").strip()
            if decoded and not decoded.startswith("#"):
                parts = decoded.split(maxsplit=1)
                if len(parts) == 2:
                    header[parts[0].upper()] = parts[1].strip()
            if decoded.upper().startswith("DATA"):
                parts = decoded.split()
                data_mode = parts[1].lower() if len(parts) >= 2 else "ascii"
                break

        fields = header.get("FIELDS", "").split()
        sizes = [int(x) for x in header.get("SIZE", "").split()] if header.get("SIZE") else []
        types = header.get("TYPE", "").split()
        counts = [int(x) for x in header.get("COUNT", "").split()] if header.get("COUNT") else [1] * len(fields)

        if not fields:
            raise RuntimeError(f"PCD missing FIELDS: {path}")
        if len(sizes) != len(fields) or len(types) != len(fields) or len(counts) != len(fields):
            raise RuntimeError(f"PCD header mismatch (FIELDS/SIZE/TYPE/COUNT): {path}")

        points = int(header.get("POINTS", "0") or "0")
        if points <= 0:
            width = int(header.get("WIDTH", "0") or "0")
            height = int(header.get("HEIGHT", "1") or "1")
            points = int(width) * int(height)
        points = max(0, int(points))
        if points == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

        if data_mode is None:
            raise RuntimeError(f"Missing DATA line in PCD: {path}")

        if data_mode == "ascii":
            mat = np.loadtxt(handle, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            name_to_index = {name: idx for idx, name in enumerate(fields)}
            for name in ("x", "y", "z", "cluster"):
                if name not in name_to_index:
                    raise RuntimeError(f"PCD missing field '{name}': {path} (fields={fields})")
            xyz = mat[:, [name_to_index["x"], name_to_index["y"], name_to_index["z"]]].astype(np.float32, copy=False)
            cluster_f = mat[:, name_to_index["cluster"]].astype(np.float32, copy=False)
            cluster = np.rint(cluster_f).astype(np.int32, copy=False)
            mask = np.isfinite(xyz.astype(np.float64)).all(axis=1) & np.isfinite(cluster_f.astype(np.float64))
            return xyz[mask], cluster[mask]

        if data_mode != "binary":
            raise RuntimeError(f"Unsupported PCD DATA mode: {data_mode} ({path})")

        offsets: List[int] = []
        names: List[str] = []
        formats: List[np.dtype] = []
        offset = 0
        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            cnt = int(cnt)
            if cnt != 1:
                raise ValueError(f"Unsupported PCD COUNT={cnt} for field={name} (only COUNT=1 supported)")
            offsets.append(int(offset))
            names.append(str(name))
            formats.append(_dtype_from_type_size(typ, size))
            offset += int(size) * cnt
        point_step = int(offset)

        dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step})
        raw = handle.read(int(points) * int(point_step))
        if len(raw) < int(points) * int(point_step):
            raise RuntimeError(f"PCD data too short: {path}")
        struct_arr = np.frombuffer(raw, dtype=dtype, count=int(points))

        for name in ("x", "y", "z", "cluster"):
            if name not in struct_arr.dtype.names:
                raise RuntimeError(f"PCD missing field '{name}': {path} (fields={struct_arr.dtype.names})")

        xyz = np.vstack([struct_arr["x"], struct_arr["y"], struct_arr["z"]]).T.astype(np.float32, copy=False)
        cluster_f = np.asarray(struct_arr["cluster"], dtype=np.float32).reshape((-1,))
        cluster = np.rint(cluster_f).astype(np.int32, copy=False)
        mask = np.isfinite(xyz.astype(np.float64)).all(axis=1) & np.isfinite(cluster_f.astype(np.float64))
        return xyz[mask], cluster[mask]


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


def _bounds_center_window(
    *,
    center_xy: Tuple[float, float],
    window_x: float,
    window_y: float,
) -> Tuple[float, float, float, float]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    window_x = float(window_x)
    window_y = float(window_y)
    if not (window_x > 0.0 and window_y > 0.0):
        raise ValueError("--window-x/--window-y must be > 0 for bounds-mode=center")
    hx = 0.5 * window_x
    hy = 0.5 * window_y
    return float(cx - hx), float(cx + hx), float(cy - hy), float(cy + hy)


def _finite_center_or_none(xy: np.ndarray) -> Optional[Tuple[float, float]]:
    xy = np.asarray(xy, dtype=np.float32).reshape((-1, 2))
    if xy.size == 0:
        return None
    mask = np.isfinite(xy.astype(np.float64)).all(axis=1)
    if not bool(np.any(mask)):
        return None
    med = np.median(xy[mask], axis=0)
    return float(med[0]), float(med[1])


def _filter_points(
    xyz: np.ndarray,
    *,
    labels: np.ndarray,
    z_min: float,
    z_max: float,
    x_min: float,
    x_max: float,
    y_abs_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32).reshape((-1, 3))
    labels = np.asarray(labels, dtype=np.int32).reshape((-1,))
    if xyz.shape[0] != labels.shape[0]:
        raise ValueError("xyz/labels size mismatch")
    mask = np.isfinite(xyz.astype(np.float64)).all(axis=1)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    mask &= z >= float(z_min)
    mask &= z <= float(z_max)
    mask &= x >= float(x_min)
    mask &= x <= float(x_max)
    mask &= np.abs(y) <= float(y_abs_max)
    return xyz[mask], labels[mask]


def _filter_points_xy(
    xyz: np.ndarray,
    *,
    labels: np.ndarray,
    x_min: float,
    x_max: float,
    y_abs_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32).reshape((-1, 3))
    labels = np.asarray(labels, dtype=np.int32).reshape((-1,))
    if xyz.shape[0] != labels.shape[0]:
        raise ValueError("xyz/labels size mismatch")
    if xyz.size == 0:
        return xyz, labels
    mask = np.isfinite(xyz.astype(np.float64)).all(axis=1)
    x = xyz[:, 0]
    y = xyz[:, 1]
    mask &= x >= float(x_min)
    mask &= x <= float(x_max)
    mask &= np.abs(y) <= float(y_abs_max)
    return xyz[mask], labels[mask]


def _clamp01(v: float) -> float:
    v = float(v)
    if v != v:  # nan
        return 0.0
    return float(max(0.0, min(v, 1.0)))


def _select_points_for_circle(
    pts: np.ndarray,
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
    pts = np.asarray(pts, dtype=np.float32).reshape((-1, 3))
    if pts.size == 0:
        return pts
    mode = (mode or "fixed").strip().lower()
    z = pts[:, 2].astype(np.float64, copy=False)
    mask = np.isfinite(z)
    if not bool(np.any(mask)):
        return pts[:0]
    z_valid = z[mask]

    if mode in ("none", "all", "full"):
        return pts

    if mode in ("fixed", "abs", "absolute"):
        keep = (z >= float(fixed_z_min)) & (z <= float(fixed_z_max))
        out = pts[keep]
        return out if out.size else pts

    if mode in ("frame-quantile", "quantile", "band-quantile"):
        q0 = _clamp01(float(q_min))
        q1 = _clamp01(float(q_max))
        if q1 < q0:
            q0, q1 = q1, q0
        z0 = float(np.quantile(z_valid, q0))
        z1 = float(np.quantile(z_valid, q1))
        keep = (z >= z0) & (z <= z1)
        out = pts[keep]
        return out if out.size else pts

    if mode in ("cluster-quantile", "cluster-band-quantile"):
        q0 = _clamp01(float(q_min))
        q1 = _clamp01(float(q_max))
        if q1 < q0:
            q0, q1 = q1, q0
        z0 = float(np.quantile(z_valid, q0))
        z1 = float(np.quantile(z_valid, q1))
        keep = (z >= z0) & (z <= z1)
        out = pts[keep]
        return out if out.size else pts

    if mode in ("cluster-bottom", "bottom", "low"):
        frac = _clamp01(float(bottom_frac))
        z1 = float(np.quantile(z_valid, frac))
        keep = z <= z1
        out = pts[keep]
        return out if out.size else pts

    if mode in ("cluster-ground-offset", "ground-offset", "ground"):
        qg = _clamp01(float(ground_q))
        gz = float(np.quantile(z_valid, qg))
        z0 = float(gz + float(ground_offset_min))
        z1 = float(gz + float(ground_offset_max))
        if z1 < z0:
            z0, z1 = z1, z0
        keep = (z >= z0) & (z <= z1)
        out = pts[keep]
        return out if out.size else pts

    raise ValueError(f"Unknown circle z mode: {mode}")


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


def _resolve_pcd_path(in_dir: Path, pcd_path_str: str) -> Optional[Path]:
    pcd_path_str = (pcd_path_str or "").strip()
    if not pcd_path_str:
        return None
    candidate = Path(pcd_path_str).expanduser()
    pcd_path = candidate.resolve() if candidate.is_absolute() else (in_dir / candidate).resolve()
    if pcd_path.is_file():
        return pcd_path
    fallback = (in_dir / "pcd" / Path(pcd_path_str).name).resolve()
    if fallback.is_file():
        return fallback
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, type=str, help="Folder with frames.csv + pcd/*.pcd (cluster frames).")
    parser.add_argument("--out-dir", default="", type=str, help="Output directory (can be Chinese).")
    parser.add_argument("--algo-name", default="", type=str, help="Optional algorithm name for metadata.")
    parser.add_argument("--resume", action="store_true", help="Skip frames whose outputs already exist.")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument("--z-min", type=float, default=-0.7)
    parser.add_argument("--z-max", type=float, default=0.7)
    parser.add_argument("--x-min", type=float, default=-2.0)
    parser.add_argument("--x-max", type=float, default=25.0)
    parser.add_argument("--y-abs-max", type=float, default=10.0)
    parser.add_argument(
        "--render-z-min",
        type=float,
        default=None,
        help="Optional Z filter for the gray point rendering only (keeps circle computation Z filter unchanged).",
    )
    parser.add_argument(
        "--render-z-max",
        type=float,
        default=None,
        help="Optional Z filter for the gray point rendering only (keeps circle computation Z filter unchanged).",
    )
    parser.add_argument(
        "--circle-z-mode",
        choices=["fixed", "none", "frame-quantile", "cluster-quantile", "cluster-bottom", "cluster-ground-offset"],
        default="fixed",
        help="How to choose points in each cluster for circle fitting (Z selection).",
    )
    parser.add_argument("--circle-z-q-min", type=float, default=0.2, help="Quantile low bound for *-quantile modes.")
    parser.add_argument("--circle-z-q-max", type=float, default=0.6, help="Quantile high bound for *-quantile modes.")
    parser.add_argument("--circle-z-bottom-frac", type=float, default=0.3, help="Bottom fraction for cluster-bottom mode.")
    parser.add_argument("--circle-ground-q", type=float, default=0.05, help="Ground quantile for cluster-ground-offset mode.")
    parser.add_argument("--circle-ground-offset-min", type=float, default=0.7, help="Ground-relative z_min offset (meters).")
    parser.add_argument("--circle-ground-offset-max", type=float, default=1.3, help="Ground-relative z_max offset (meters).")

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
    parser.add_argument(
        "--bounds-mode",
        choices=["fixed", "center"],
        default="fixed",
        help="Render bounds mode: fixed=use --bounds (or x/y defaults); center=per-frame centered window.",
    )
    parser.add_argument("--window-x", type=float, default=30.0, help="Window size in X (meters) when bounds-mode=center.")
    parser.add_argument("--window-y", type=float, default=20.0, help="Window size in Y (meters) when bounds-mode=center.")
    parser.add_argument("--bg", type=str, default="#ffffff")
    parser.add_argument("--point-color", type=str, default="#b3b3b3")
    parser.add_argument("--circle-color", type=str, default="#2ca02c")
    parser.add_argument("--draw-grid", type=int, default=1)
    parser.add_argument("--draw-axes", type=int, default=1)
    parser.add_argument("--draw-title", type=int, default=0)
    parser.add_argument("--draw-radius", type=int, default=1)

    parser.add_argument(
        "--merge-close-circles",
        action="store_true",
        help="Post-process circles: merge circles whose centers are closer than --merge-max-center-dist (helps fix KMeans over-splitting).",
    )
    parser.add_argument("--merge-max-center-dist", type=float, default=0.4, help="Merge threshold in meters (only used with --merge-close-circles).")
    args = parser.parse_args()

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

    fixed_bounds = (
        _parse_bounds(str(args.bounds))
        if str(args.bounds).strip()
        else (float(args.x_min), float(args.x_max), -float(args.y_abs_max), float(args.y_abs_max))
    )

    run_meta = {
        "algo_name": str(args.algo_name).strip(),
        "in_dir": str(in_dir),
        "frames_csv": str(frames_path),
        "filter": {"z_min": float(args.z_min), "z_max": float(args.z_max), "x_min": float(args.x_min), "x_max": float(args.x_max), "y_abs_max": float(args.y_abs_max)},
        "render_filter": {
            "z_min": float(args.render_z_min) if args.render_z_min is not None else float(args.z_min),
            "z_max": float(args.render_z_max) if args.render_z_max is not None else float(args.z_max),
            "note": "Only affects gray point rendering in PNG; circle computation still uses 'filter'.",
        },
        "circle_z": {
            "mode": str(args.circle_z_mode),
            "fixed": {"z_min": float(args.z_min), "z_max": float(args.z_max)},
            "frame_quantile": {"q_min": float(args.circle_z_q_min), "q_max": float(args.circle_z_q_max)},
            "cluster_quantile": {"q_min": float(args.circle_z_q_min), "q_max": float(args.circle_z_q_max)},
            "cluster_bottom": {"frac": float(args.circle_z_bottom_frac)},
            "cluster_ground_offset": {
                "ground_q": float(args.circle_ground_q),
                "offset_min": float(args.circle_ground_offset_min),
                "offset_max": float(args.circle_ground_offset_max),
            },
        },
        "radius": {
            "marker_z": float(args.marker_z),
            "radius_mode": str(args.radius_mode),
            "radius_constant": float(args.radius_constant),
            "radius_quantile": float(args.radius_quantile),
            "radius_min": float(args.radius_min),
            "radius_max": float(args.radius_max),
        },
        "render": {
            "save_png": bool(save_png),
            "bounds_mode": str(args.bounds_mode),
            "bounds": [float(v) for v in fixed_bounds],
            "window_x": float(args.window_x),
            "window_y": float(args.window_y),
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
        "postprocess": {
            "merge_close_circles": {
                "enabled": bool(args.merge_close_circles),
                "max_center_dist": float(args.merge_max_center_dist),
                "note": "Applied after per-cluster circle fitting; intended to merge duplicate circles from over-segmentation.",
            }
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_out = out_dir / "frames.csv"
    with csv_out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "t_sec", "points_in", "points_xy", "points_circle", "points_render", "clusters", "circles_json", "centers_pcd", "png"])

        every = max(1, int(args.every))
        max_frames = int(args.max_frames)
        max_frames = max_frames if max_frames > 0 else 0

        processed = 0
        for i, row in enumerate(frames):
            if i % every != 0:
                continue
            idx_str = (row.get("index") or row.get("frame") or row.get("idx") or "").strip()
            try:
                frame_idx = int(idx_str) if idx_str else int(i)
            except Exception:
                frame_idx = int(i)

            t_sec = (row.get("t_sec") or row.get("t") or "").strip()
            out_circles = circles_dir / f"circles_{frame_idx:06d}.json"
            out_centers = centers_dir / f"centers_{frame_idx:06d}.pcd"
            out_png = png_dir / f"bev_{frame_idx:06d}.png" if save_png else Path("")

            if args.resume and out_circles.is_file() and out_centers.is_file() and (not save_png or out_png.is_file()):
                try:
                    obj = json.loads(out_circles.read_text(encoding="utf-8"))
                    points_in = int(obj.get("points_in", 0))
                    points_xy = int(obj.get("points_xy", points_in))
                    points_circle = int(obj.get("points_circle", obj.get("points_used", 0)))
                    points_render = int(obj.get("points_render", 0))
                    clusters = int(len(obj.get("circles", []) or []))
                except Exception:
                    points_in = 0
                    points_xy = 0
                    points_circle = 0
                    points_render = 0
                    clusters = 0
                writer.writerow(
                    [
                        int(frame_idx),
                        t_sec,
                        int(points_in),
                        int(points_xy),
                        int(points_circle),
                        int(points_render),
                        int(clusters),
                        str(out_circles.relative_to(out_dir)),
                        str(out_centers.relative_to(out_dir)),
                        str(out_png.relative_to(out_dir)) if save_png else "",
                    ]
                )
                processed += 1
                if processed % 200 == 0:
                    print(f"[OK] recorded {processed} existing frames")
                if max_frames > 0 and processed >= max_frames:
                    break
                continue

            pcd_path = _resolve_pcd_path(in_dir, (row.get("pcd_path") or row.get("pcd") or "").strip())
            if pcd_path is None or not pcd_path.is_file():
                continue

            xyz, labels = _read_pcd_xyz_cluster(pcd_path)
            points_in = int(xyz.shape[0])
            xyz_xy, labels_xy = _filter_points_xy(
                xyz,
                labels=labels,
                x_min=float(args.x_min),
                x_max=float(args.x_max),
                y_abs_max=float(args.y_abs_max),
            )
            points_xy = int(xyz_xy.shape[0])

            render_z_min = float(args.render_z_min) if args.render_z_min is not None else float(args.z_min)
            render_z_max = float(args.render_z_max) if args.render_z_max is not None else float(args.z_max)
            if xyz_xy.size:
                z_all = xyz_xy[:, 2].astype(np.float64, copy=False)
                render_keep = (z_all >= float(render_z_min)) & (z_all <= float(render_z_max))
                xyz_render = xyz_xy[render_keep]
            else:
                xyz_render = xyz_xy
            points_render = int(xyz_render.shape[0])

            circle_mode = str(args.circle_z_mode).strip().lower()
            frame_z0: Optional[float] = None
            frame_z1: Optional[float] = None
            if circle_mode in ("fixed", "abs", "absolute"):
                frame_z0 = float(args.z_min)
                frame_z1 = float(args.z_max)
            elif circle_mode in ("frame-quantile", "quantile", "band-quantile"):
                if xyz_xy.size:
                    z_valid = xyz_xy[:, 2].astype(np.float64, copy=False)
                    z_valid = z_valid[np.isfinite(z_valid)]
                    if z_valid.size:
                        q0 = _clamp01(float(args.circle_z_q_min))
                        q1 = _clamp01(float(args.circle_z_q_max))
                        if q1 < q0:
                            q0, q1 = q1, q0
                        frame_z0 = float(np.quantile(z_valid, q0))
                        frame_z1 = float(np.quantile(z_valid, q1))

            uniq = sorted(int(v) for v in set(labels_xy.tolist()) if int(v) >= 0)
            circles: List[TreeCircle] = []
            points_circle_total = 0
            for cid in uniq:
                pts_all = xyz_xy[labels_xy == int(cid)]
                if pts_all.size == 0:
                    continue

                if circle_mode in ("fixed", "abs", "absolute", "frame-quantile", "quantile", "band-quantile"):
                    if frame_z0 is None or frame_z1 is None:
                        pts = pts_all
                    else:
                        zc = pts_all[:, 2].astype(np.float64, copy=False)
                        keep = (zc >= float(frame_z0)) & (zc <= float(frame_z1))
                        pts = pts_all[keep]
                        if pts.size == 0:
                            pts = pts_all
                elif circle_mode in ("none", "all", "full"):
                    pts = pts_all
                else:
                    pts = _select_points_for_circle(
                        pts_all,
                        mode=circle_mode,
                        fixed_z_min=float(args.z_min),
                        fixed_z_max=float(args.z_max),
                        q_min=float(args.circle_z_q_min),
                        q_max=float(args.circle_z_q_max),
                        bottom_frac=float(args.circle_z_bottom_frac),
                        ground_q=float(args.circle_ground_q),
                        ground_offset_min=float(args.circle_ground_offset_min),
                        ground_offset_max=float(args.circle_ground_offset_max),
                    )
                if pts.size == 0:
                    continue

                points_circle_total += int(pts.shape[0])
                center_xy = np.median(pts[:, :2], axis=0)
                z = float(args.marker_z) if float(args.marker_z) != 0.0 else float(np.median(pts[:, 2]))
                radius = _compute_radius(
                    pts_xy=pts[:, :2],
                    center_xy=center_xy,
                    mode=str(args.radius_mode),
                    radius_constant=float(args.radius_constant),
                    radius_quantile=float(args.radius_quantile),
                    radius_min=float(args.radius_min),
                    radius_max=float(args.radius_max),
                )
                circles.append(TreeCircle(x=float(center_xy[0]), y=float(center_xy[1]), z=float(z), radius=float(radius), points=int(pts.shape[0])))

            circles.sort(key=lambda c: (c.y, c.x))
            circles_before = int(len(circles))
            if bool(args.merge_close_circles) and circles:
                circles = _merge_close_circles(circles, max_center_dist=float(args.merge_max_center_dist))
            clusters = int(len(circles))

            out_obj = {
                "frame_index": int(frame_idx),
                "t_sec": float(t_sec) if t_sec else None,
                "pcd": str(pcd_path),
                "points_in": int(points_in),
                "points_xy": int(points_xy),
                "points_circle": int(points_circle_total),
                "points_render": int(points_render),
                "algo_name": str(args.algo_name).strip(),
                "circle_z_mode": str(args.circle_z_mode).strip(),
                "postprocess": {
                    "merge_close_circles": {
                        "enabled": bool(args.merge_close_circles),
                        "max_center_dist": float(args.merge_max_center_dist),
                        "circles_before": int(circles_before),
                        "circles_after": int(clusters),
                    }
                },
                "circles": [asdict(c) for c in circles],
            }
            out_circles.parent.mkdir(parents=True, exist_ok=True)
            out_circles.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            if circles:
                centers = np.array([[c.x, c.y, c.z, c.radius] for c in circles], dtype=np.float32)
            else:
                centers = np.zeros((0, 4), dtype=np.float32)
            _write_pcd_xyzr(out_centers, centers)

            if save_png:
                pts_xy = xyz_render[:, :2].astype(np.float32) if xyz_render.size else np.zeros((0, 2), dtype=np.float32)
                circles_xyr = centers[:, :3].copy() if centers.size else np.zeros((0, 3), dtype=np.float32)
                if circles_xyr.size:
                    circles_xyr[:, 2] = centers[:, 3]
                if str(args.bounds_mode).strip().lower() == "center":
                    center_xy = _finite_center_or_none(pts_xy)
                    if center_xy is None:
                        center_xy = _finite_center_or_none(circles_xyr[:, :2]) if circles_xyr.size else None
                    if center_xy is None:
                        xmin, xmax, ymin, ymax = fixed_bounds
                        center_xy = (0.5 * (float(xmin) + float(xmax)), 0.5 * (float(ymin) + float(ymax)))
                    frame_bounds = _bounds_center_window(center_xy=center_xy, window_x=float(args.window_x), window_y=float(args.window_y))
                else:
                    frame_bounds = fixed_bounds
                _render_bev_cv2(
                    out_png,
                    pts_xy,
                    circles_xyr,
                    bounds=frame_bounds,
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
                    t_sec,
                    int(points_in),
                    int(points_xy),
                    int(points_circle_total),
                    int(points_render),
                    int(clusters),
                    str(out_circles.relative_to(out_dir)),
                    str(out_centers.relative_to(out_dir)),
                    str(out_png.relative_to(out_dir)) if save_png else "",
                ]
            )

            processed += 1
            if processed % 50 == 0:
                print(f"[OK] processed {processed} frames")
            if max_frames > 0 and processed >= max_frames:
                break

    print(f"[OK] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
