#!/usr/bin/env python3
"""Fit orchard row direction (and a simple multi-row model) from a BEV circles run.

Input: a BEV run directory produced by `clustered_frames_to_circles_bev.py`:
  <run>/circles/circles_000123.json
  <run>/png/bev_000123.png (optional; used for overlay preview)
  <run>/run_meta.json      (optional; used to reconstruct per-frame bounds for overlay)

Output: a directory with:
  - row_model_auto.json         (direction_xy / perp_xy / rows_uv)
  - row_direction_per_frame.csv (per-frame PCA direction stats)
  - png_overlay/overlay_*.png   (optional, for quick visual check)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FrameCircles:
    index: int
    centers_xy: np.ndarray  # shape (N,2)


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        x = float(v)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return x if math.isfinite(x) else float(default)


def _iter_circle_json_paths(circles_dir: Path) -> List[Path]:
    if not circles_dir.is_dir():
        return []
    paths = sorted(circles_dir.glob("circles_*.json"))
    return [p for p in paths if p.is_file()]


def _parse_index_from_filename(path: Path) -> Optional[int]:
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return None


def _load_centers_xy(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    circles = obj.get("circles") or []
    pts: List[Tuple[float, float]] = []
    for c in circles:
        if not c:
            continue
        x = _safe_float((c or {}).get("x"), float("nan"))
        y = _safe_float((c or {}).get("y"), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            pts.append((float(x), float(y)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32).reshape((-1, 2))


def _pca_direction(xy: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Return (unit direction_xy, pca_ratio), or (None, nan) if not enough points."""
    xy = np.asarray(xy, dtype=np.float32).reshape((-1, 2))
    if xy.shape[0] < 2:
        return None, float("nan")
    mean = np.mean(xy, axis=0)
    centered = xy - mean
    cov = (centered.T @ centered) / float(max(1, xy.shape[0] - 1))
    if not np.all(np.isfinite(cov.astype(np.float64))):
        return None, float("nan")
    vals, vecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(vals)
    v0 = float(vals[order[0]])
    v1 = float(vals[order[-1]])
    direction = vecs[:, int(order[-1])].astype(np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        return None, float("nan")
    direction = direction / norm
    denom = max(1.0e-9, float(v0 + v1))
    ratio = float(v1 / denom)
    return direction.astype(np.float32), float(ratio)


def _normalize_direction_sign(direction_xy: np.ndarray) -> np.ndarray:
    direction_xy = np.asarray(direction_xy, dtype=np.float32).reshape((2,))
    if direction_xy[0] < 0:
        return -direction_xy
    if abs(float(direction_xy[0])) < 1.0e-6 and direction_xy[1] < 0:
        return -direction_xy
    return direction_xy


def _angle_deg_from_direction(direction_xy: np.ndarray) -> float:
    direction_xy = np.asarray(direction_xy, dtype=np.float64).reshape((2,))
    return float(math.degrees(math.atan2(float(direction_xy[1]), float(direction_xy[0]))))


def _perp_from_direction(direction_xy: np.ndarray) -> np.ndarray:
    d = np.asarray(direction_xy, dtype=np.float32).reshape((2,))
    perp = np.asarray([-float(d[1]), float(d[0])], dtype=np.float32)
    n = float(np.linalg.norm(perp))
    if n <= 1.0e-9:
        perp = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        perp = perp / n
    return perp.astype(np.float32)


def _cluster_rows_by_v_gap(v: np.ndarray, thr: float) -> List[np.ndarray]:
    v = np.asarray(v, dtype=np.float64).reshape((-1,))
    if v.size == 0:
        return []
    order = np.argsort(v)
    v_sorted = v[order]
    thr = float(thr)
    if not (thr > 0.0):
        return [order.astype(np.int32)]

    groups: List[List[int]] = []
    current: List[int] = [int(order[0])]
    for i in range(1, int(order.size)):
        gap = float(v_sorted[i] - v_sorted[i - 1])
        if gap > thr:
            groups.append(current)
            current = [int(order[i])]
        else:
            current.append(int(order[i]))
    if current:
        groups.append(current)
    return [np.asarray(g, dtype=np.int32) for g in groups if g]


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape((-1,))
    window = int(window)
    if x.size == 0 or window <= 1:
        return x
    window = min(window, int(x.size))
    kernel = np.ones((int(window),), dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def _pick_v_peaks(
    v: np.ndarray,
    *,
    bin_size: float,
    smooth_window: int,
    peak_min_fraction: float,
    min_separation: float,
    max_peaks: int,
    clip_percentile: float,
) -> List[float]:
    """Pick peak centers from the 1D v distribution (histogram + smoothing).

    Returns peak centers in v (meters), sorted by peak strength (desc).
    """
    v = np.asarray(v, dtype=np.float64).reshape((-1,))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return []

    bin_size = float(bin_size)
    if not (bin_size > 0.0):
        raise ValueError("--v-bin must be > 0")

    p = float(max(0.0, min(float(clip_percentile), 49.0)))
    if p > 0.0 and v.size >= 20:
        vmin, vmax = np.percentile(v, [p, 100.0 - p]).tolist()
    else:
        vmin, vmax = float(np.min(v)), float(np.max(v))

    # Ensure a reasonable range.
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or abs(vmax - vmin) < 1.0e-6:
        return [float(np.median(v))]

    # Build bins.
    n_bins = int(max(10, min(400, math.ceil((float(vmax) - float(vmin)) / bin_size))))
    edges = np.linspace(float(vmin), float(vmax), int(n_bins) + 1, dtype=np.float64)
    hist, _ = np.histogram(v, bins=edges)
    hist = hist.astype(np.float64, copy=False)

    sm = _moving_average(hist, int(smooth_window))
    if sm.size < 3:
        return []

    max_val = float(np.max(sm))
    if not (max_val > 0.0):
        return []
    thr = float(max_val) * float(max(0.0, min(float(peak_min_fraction), 1.0)))

    # Local maxima.
    candidates: List[Tuple[float, float]] = []  # (score, v_center)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for i in range(1, int(sm.size) - 1):
        if float(sm[i]) < thr:
            continue
        if float(sm[i]) >= float(sm[i - 1]) and float(sm[i]) >= float(sm[i + 1]):
            candidates.append((float(sm[i]), float(centers[i])))
    if not candidates:
        # Fallback: take the global max bin center.
        i = int(np.argmax(sm))
        return [float(centers[i])]

    candidates.sort(key=lambda s: float(s[0]), reverse=True)

    # Enforce separation.
    picked: List[float] = []
    min_sep = float(max(0.0, float(min_separation)))
    for _, c in candidates:
        if min_sep > 0.0 and any(abs(float(c) - float(p0)) < min_sep for p0 in picked):
            continue
        picked.append(float(c))
        if int(max_peaks) > 0 and len(picked) >= int(max_peaks):
            break
    return picked


def _quantile_span(values: np.ndarray, q0: float, q1: float) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape((-1,))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    q0 = float(max(0.0, min(q0, 1.0)))
    q1 = float(max(0.0, min(q1, 1.0)))
    if q1 < q0:
        q0, q1 = q1, q0
    return float(np.quantile(values, q0)), float(np.quantile(values, q1))


def _cluster_1d_by_gap(values: np.ndarray, gap_thr: float) -> List[np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape((-1,))
    if values.size == 0:
        return []
    gap_thr = float(gap_thr)
    if not (gap_thr > 0.0):
        return [np.arange(int(values.size), dtype=np.int32)]
    order = np.argsort(values)
    vs = values[order]
    groups: List[List[int]] = [[int(order[0])]]
    for i in range(1, int(order.size)):
        gap = float(vs[i] - vs[i - 1])
        if gap > gap_thr:
            groups.append([])
        groups[-1].append(int(order[i]))
    return [np.asarray(g, dtype=np.int32) for g in groups if g]


def _pick_two_lane_rows_from_v(
    v: np.ndarray,
    *,
    gap_thr: float,
    min_points_per_row: int,
    min_separation: float,
    max_separation: float,
) -> Optional[Tuple[float, float, int, int]]:
    """Pick two adjacent rows (lane boundaries) from per-frame v values.

    Strategy:
      1) cluster v by 1D gap threshold into multiple groups (rows)
      2) pick the pair with the smallest separation in [min_separation, max_separation]

    Returns (v0, v1, n0, n1) with v0 < v1, or None if not enough rows.
    """
    v = np.asarray(v, dtype=np.float64).reshape((-1,))
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None

    min_points_per_row = max(1, int(min_points_per_row))
    min_separation = float(max(0.0, float(min_separation)))
    max_separation = float(max(0.0, float(max_separation)))
    if max_separation > 0.0 and max_separation < min_separation:
        min_separation, max_separation = max_separation, min_separation

    groups = _cluster_1d_by_gap(v, float(gap_thr))
    rows: List[Tuple[float, int]] = []  # (v_center, n)
    for g in groups:
        if int(g.size) < int(min_points_per_row):
            continue
        vc = float(np.median(v[g]))
        rows.append((vc, int(g.size)))
    if len(rows) < 2:
        return None

    rows.sort(key=lambda x: float(x[0]))
    best: Optional[Tuple[float, float, int, int, float]] = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            v0, n0 = rows[i]
            v1, n1 = rows[j]
            sep = float(abs(float(v1) - float(v0)))
            if min_separation > 0.0 and sep < min_separation:
                continue
            if max_separation > 0.0 and sep > max_separation:
                continue
            if best is None or sep < float(best[4]):
                best = (float(v0), float(v1), int(n0), int(n1), float(sep))
    if best is None:
        # Fallback: just take the two closest centers.
        rows2 = rows[:]
        rows2.sort(key=lambda x: float(x[0]))
        best_pair = None
        for i in range(len(rows2) - 1):
            v0, n0 = rows2[i]
            v1, n1 = rows2[i + 1]
            sep = float(v1 - v0)
            if best_pair is None or sep < best_pair[4]:
                best_pair = (float(v0), float(v1), int(n0), int(n1), float(sep))
        if best_pair is None:
            return None
        v0, v1, n0, n1, _ = best_pair
        return float(v0), float(v1), int(n0), int(n1)
    v0, v1, n0, n1, _ = best
    return float(v0), float(v1), int(n0), int(n1)


def _load_run_meta(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _frame_bounds_center_window(center_xy: Tuple[float, float], window_x: float, window_y: float) -> Tuple[float, float, float, float]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    hx = 0.5 * float(window_x)
    hy = 0.5 * float(window_y)
    return float(cx - hx), float(cx + hx), float(cy - hy), float(cy + hy)


def _median_center_or_none(xy: np.ndarray) -> Optional[Tuple[float, float]]:
    xy = np.asarray(xy, dtype=np.float64).reshape((-1, 2))
    if xy.size == 0:
        return None
    mask = np.isfinite(xy).all(axis=1)
    if not bool(np.any(mask)):
        return None
    med = np.median(xy[mask], axis=0)
    return float(med[0]), float(med[1])


def _draw_overlay(
    *,
    in_png: Path,
    out_png: Path,
    direction_xy: np.ndarray,
    direction_frame_xy: Optional[np.ndarray],
    rows_uv: Sequence[Dict[str, float]],
    fill_drivable: bool,
    drivable_color_bgr: Tuple[int, int, int],
    drivable_alpha: float,
    drivable_margin_v: float,
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
    title: str,
) -> None:
    import cv2  # type: ignore

    img = cv2.imread(str(in_png), cv2.IMREAD_COLOR)
    if img is None:
        return

    xmin, xmax, ymin, ymax = bounds
    dx = max(1.0e-6, float(xmax) - float(xmin))
    dy = max(1.0e-6, float(ymax) - float(ymin))
    scale = float(min((float(width) - 2.0 * float(margin)) / dx, (float(height) - 2.0 * float(margin)) / dy))

    def map_to_px(x: float, y: float) -> Tuple[int, int]:
        px = float(margin) + (float(x) - float(xmin)) * scale
        py = float(margin) + (float(ymax) - float(y)) * scale
        return int(round(px)), int(round(py))

    # Direction arrows (draw at image center).
    cx_px, cy_px = int(width // 2), int(height // 2)

    def arrow_vec_px(vec_xy: np.ndarray, length_px: int) -> Tuple[int, int]:
        v = np.asarray(vec_xy, dtype=np.float64).reshape((2,))
        n = float(np.linalg.norm(v))
        if n <= 1.0e-9:
            return 0, -int(length_px)
        v = v / n
        # Map y-up coords to image y-down coords.
        vx = float(v[0])
        vy = -float(v[1])
        return int(round(vx * float(length_px))), int(round(vy * float(length_px)))

    # Global direction (green).
    dx_px, dy_px = arrow_vec_px(direction_xy, 180)
    cv2.arrowedLine(img, (cx_px, cy_px), (cx_px + dx_px, cy_px + dy_px), (44, 160, 44), 3, tipLength=0.15)

    # Per-frame direction (blue).
    if direction_frame_xy is not None:
        fx_px, fy_px = arrow_vec_px(direction_frame_xy, 140)
        cv2.arrowedLine(img, (cx_px, cy_px), (cx_px + fx_px, cy_px + fy_px), (255, 128, 0), 2, tipLength=0.18)

    # Row lines (red-ish).
    direction = np.asarray(direction_xy, dtype=np.float64).reshape((2,))
    perp = np.asarray([-float(direction[1]), float(direction[0])], dtype=np.float64)
    perp = perp / max(1.0e-9, float(np.linalg.norm(perp)))

    # Choose a u span that covers the current window.
    u_center = float(np.dot(np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=np.float64), direction))
    diag = float(math.hypot(dx, dy))
    u0 = float(u_center - diag)
    u1 = float(u_center + diag)

    if bool(fill_drivable) and len(rows_uv) == 2:
        v0 = float(rows_uv[0].get("v_center", 0.0))
        v1 = float(rows_uv[1].get("v_center", 0.0))
        if v1 < v0:
            v0, v1 = v1, v0
        margin_v = float(max(0.0, float(drivable_margin_v)))
        v0_fill = float(v0 + margin_v)
        v1_fill = float(v1 - margin_v)
        if v1_fill > v0_fill:
            p00 = direction * float(u0) + perp * float(v0_fill)
            p10 = direction * float(u1) + perp * float(v0_fill)
            p11 = direction * float(u1) + perp * float(v1_fill)
            p01 = direction * float(u0) + perp * float(v1_fill)
            poly = np.asarray(
                [
                    map_to_px(float(p00[0]), float(p00[1])),
                    map_to_px(float(p10[0]), float(p10[1])),
                    map_to_px(float(p11[0]), float(p11[1])),
                    map_to_px(float(p01[0]), float(p01[1])),
                ],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            overlay = img.copy()
            color = tuple(int(c) for c in drivable_color_bgr)
            cv2.fillPoly(overlay, [poly], color, lineType=cv2.LINE_AA)
            a = float(max(0.0, min(float(drivable_alpha), 0.95)))
            if a > 0.0:
                img = cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0)
    for row in rows_uv:
        v_center = float(row.get("v_center", 0.0))
        p0 = direction * u0 + perp * v_center
        p1 = direction * u1 + perp * v_center
        x0, y0 = map_to_px(float(p0[0]), float(p0[1]))
        x1, y1 = map_to_px(float(p1[0]), float(p1[1]))
        cv2.line(img, (x0, y0), (x1, y1), (30, 30, 220), 2, lineType=cv2.LINE_AA)

    # Title.
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, lineType=cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)


def _render_uv_global(
    *,
    out_png: Path,
    u: np.ndarray,
    v: np.ndarray,
    rows_uv: Sequence[Dict[str, float]],
    width: int,
    height: int,
    margin: int,
) -> None:
    import cv2  # type: ignore

    u = np.asarray(u, dtype=np.float64).reshape((-1,))
    v = np.asarray(v, dtype=np.float64).reshape((-1,))
    mask = np.isfinite(u) & np.isfinite(v)
    u = u[mask]
    v = v[mask]
    if u.size == 0:
        return

    u0, u1 = float(np.quantile(u, 0.01)), float(np.quantile(u, 0.99))
    if not (u1 > u0):
        u0, u1 = float(np.min(u)), float(np.max(u))
    if not (u1 > u0):
        u0, u1 = float(u0 - 1.0), float(u0 + 1.0)

    if rows_uv:
        v_centers = [float(r.get("v_center", 0.0)) for r in rows_uv]
        vmin = float(min(v_centers)) - 2.0
        vmax = float(max(v_centers)) + 2.0
    else:
        vmin, vmax = float(np.quantile(v, 0.01)), float(np.quantile(v, 0.99))
        if not (vmax > vmin):
            vmin, vmax = float(np.min(v)), float(np.max(v))
    if not (vmax > vmin):
        vmin, vmax = float(vmin - 1.0), float(vmin + 1.0)

    # Map (u,v) to pixels: u->right, v->up.
    dx = max(1.0e-6, u1 - u0)
    dy = max(1.0e-6, vmax - vmin)
    scale = float(min((float(width) - 2.0 * float(margin)) / dx, (float(height) - 2.0 * float(margin)) / dy))

    def uv_to_px(uu: np.ndarray, vv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        px = float(margin) + (uu.astype(np.float64, copy=False) - float(u0)) * scale
        py = float(margin) + (float(vmax) - vv.astype(np.float64, copy=False)) * scale
        return px.astype(np.int32), py.astype(np.int32)

    img = np.full((int(height), int(width), 3), (255, 255, 255), dtype=np.uint8)

    px, py = uv_to_px(u, v)
    inside = (px >= 0) & (px < int(width)) & (py >= 0) & (py < int(height))
    px = px[inside]
    py = py[inside]
    img[py, px] = (160, 160, 160)

    # Draw row center lines.
    for r in rows_uv:
        vc = float(r.get("v_center", 0.0))
        x0, y0 = uv_to_px(np.asarray([u0], dtype=np.float64), np.asarray([vc], dtype=np.float64))
        x1, y1 = uv_to_px(np.asarray([u1], dtype=np.float64), np.asarray([vc], dtype=np.float64))
        cv2.line(img, (int(x0[0]), int(y0[0])), (int(x1[0]), int(y1[0])), (30, 30, 220), 2, lineType=cv2.LINE_AA)

    title = f"UV aligned | rows={len(rows_uv)}"
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, lineType=cv2.LINE_AA)
    cv2.putText(img, f"u[{u0:.1f},{u1:.1f}]  v[{vmin:.1f},{vmax:.1f}]", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, lineType=cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str, help="Input BEV run dir (has circles/ and optionally png/).")
    parser.add_argument("--out-dir", required=True, type=str, help="Output directory (can be Chinese).")

    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-circles-per-frame", type=int, default=2)

    parser.add_argument(
        "--rows-mode",
        choices=["peaks", "gap"],
        default="peaks",
        help="How to infer multiple rows from v distribution: peaks=histogram peaks (recommended), gap=sorted gap clustering (sensitive to outliers).",
    )
    parser.add_argument("--row-split-thr", type=float, default=1.2, help="1D gap clustering threshold in v (meters).")
    parser.add_argument("--min-trees-per-row", type=int, default=5)
    parser.add_argument("--u-quantile-min", type=float, default=0.0)
    parser.add_argument("--u-quantile-max", type=float, default=1.0)

    parser.add_argument("--max-rows", type=int, default=2, help="Used in peaks mode: keep at most N strongest rows (default: 2).")
    parser.add_argument("--row-bandwidth", type=float, default=1.2, help="Used in peaks mode: row selection bandwidth in v (meters).")
    parser.add_argument("--v-bin", type=float, default=0.25, help="Used in peaks mode: histogram bin size in v (meters).")
    parser.add_argument("--smooth-window", type=int, default=5, help="Used in peaks mode: moving-average window (bins).")
    parser.add_argument("--peak-min-fraction", type=float, default=0.08, help="Used in peaks mode: keep peaks >= fraction of max peak.")
    parser.add_argument("--min-separation", type=float, default=1.2, help="Used in peaks mode: minimum separation between peaks (meters).")
    parser.add_argument("--v-clip-percentile", type=float, default=1.0, help="Used in peaks mode: clip v to [p,100-p] percentile before histogram.")

    parser.add_argument("--no-overlay", action="store_true", help="Do not write png_overlay/*.png")
    parser.add_argument("--overlay-every", type=int, default=1, help="Write overlay every N frames (default: 1).")
    parser.add_argument(
        "--overlay-indices",
        type=str,
        default="",
        help="Optional comma-separated frame indices; when set, only write overlays for these frames (overrides --overlay-every).",
    )
    parser.add_argument(
        "--overlay-row-lines",
        choices=["global", "frame_lane", "none"],
        default="global",
        help="Which row lines to draw in overlay: global=use rows_uv from row_model_auto; frame_lane=fit 2 lane rows per frame from v; none=draw no row lines.",
    )
    parser.add_argument(
        "--overlay-fill-drivable",
        action="store_true",
        help="Fill the drivable corridor between the two overlay row lines (only when exactly 2 rows are drawn).",
    )
    parser.add_argument("--drivable-color", type=str, default="#7fc97f", help="Drivable area fill color (#RRGGBB).")
    parser.add_argument("--drivable-alpha", type=float, default=0.18, help="Drivable area fill alpha (0~1).")
    parser.add_argument("--drivable-margin-v", type=float, default=0.0, help="Shrink corridor by this margin on both sides in v (meters).")
    parser.add_argument("--lane-gap-thr", type=float, default=0.8, help="frame_lane: 1D gap threshold to form row groups in v (meters).")
    parser.add_argument("--lane-min-points", type=int, default=3, help="frame_lane: minimum circles per row group.")
    parser.add_argument("--lane-min-sep", type=float, default=2.0, help="frame_lane: minimum lane width (meters).")
    parser.add_argument("--lane-max-sep", type=float, default=10.0, help="frame_lane: maximum lane width (meters; 0=disable).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    circles_dir = run_dir / "circles"
    png_dir = run_dir / "png"
    meta = _load_run_meta(run_dir / "run_meta.json")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    every = max(1, int(args.every))
    max_frames = max(0, int(args.max_frames))
    min_circles = max(0, int(args.min_circles_per_frame))

    frames: List[FrameCircles] = []
    all_xy: List[np.ndarray] = []

    paths = _iter_circle_json_paths(circles_dir)
    processed = 0
    for p in paths:
        idx = _parse_index_from_filename(p)
        if idx is None:
            continue
        if idx % every != 0:
            continue
        centers = _load_centers_xy(p)
        frames.append(FrameCircles(index=int(idx), centers_xy=centers))
        if centers.size:
            all_xy.append(centers)
        processed += 1
        if max_frames > 0 and processed >= max_frames:
            break

    if not frames:
        raise RuntimeError(f"No circles frames found under: {circles_dir}")

    xy_all = np.vstack(all_xy).astype(np.float32) if all_xy else np.zeros((0, 2), dtype=np.float32)
    direction_global, ratio_global = _pca_direction(xy_all)
    if direction_global is None:
        raise RuntimeError("Failed to fit global PCA direction (not enough circle centers).")
    direction_global = _normalize_direction_sign(direction_global)
    perp_global = _perp_from_direction(direction_global)

    theta_global_deg = _angle_deg_from_direction(direction_global)

    # Compute global rows_uv using all centers.
    rows_uv: List[Dict[str, float]] = []
    if xy_all.size:
        u_all = xy_all.dot(direction_global.astype(np.float32).reshape(2)).astype(np.float64)
        v_all = xy_all.dot(perp_global.astype(np.float32).reshape(2)).astype(np.float64)
        rows_mode = str(args.rows_mode).strip().lower()
        if rows_mode == "gap":
            groups = _cluster_rows_by_v_gap(v_all, float(args.row_split_thr))
            for g in groups:
                if int(g.size) < int(args.min_trees_per_row):
                    continue
                u0, u1 = _quantile_span(u_all[g], float(args.u_quantile_min), float(args.u_quantile_max))
                v_center = float(np.median(v_all[g])) if g.size else 0.0
                if u1 <= u0:
                    continue
                rows_uv.append({"v_center": float(v_center), "u_min": float(u0), "u_max": float(u1), "z": 0.0})
            rows_uv.sort(key=lambda r: float(r.get("v_center", 0.0)))
        else:
            peaks = _pick_v_peaks(
                v_all,
                bin_size=float(args.v_bin),
                smooth_window=int(args.smooth_window),
                peak_min_fraction=float(args.peak_min_fraction),
                min_separation=float(args.min_separation),
                max_peaks=int(args.max_rows),
                clip_percentile=float(args.v_clip_percentile),
            )
            half_bw = 0.5 * float(max(0.2, float(args.row_bandwidth)))
            for c in peaks:
                mask = np.abs(v_all - float(c)) <= float(half_bw)
                idxs = np.where(mask)[0]
                if int(idxs.size) < int(args.min_trees_per_row):
                    continue
                u0, u1 = _quantile_span(u_all[idxs], float(args.u_quantile_min), float(args.u_quantile_max))
                v_center = float(np.median(v_all[idxs])) if idxs.size else float(c)
                if u1 <= u0:
                    continue
                rows_uv.append({"v_center": float(v_center), "u_min": float(u0), "u_max": float(u1), "z": 0.0})
            rows_uv.sort(key=lambda r: float(r.get("v_center", 0.0)))

    row_model = {
        "direction_xy": [float(direction_global[0]), float(direction_global[1])],
        "perp_xy": [float(perp_global[0]), float(perp_global[1])],
        "rows_uv": rows_uv,
        "source": str(run_dir),
        "method": "global_pca + v_gap_clustering",
        "rows_mode": str(args.rows_mode).strip(),
        "row_split_thr": float(args.row_split_thr),
        "min_trees_per_row": int(args.min_trees_per_row),
        "peaks": {
            "max_rows": int(args.max_rows),
            "row_bandwidth": float(args.row_bandwidth),
            "v_bin": float(args.v_bin),
            "smooth_window": int(args.smooth_window),
            "peak_min_fraction": float(args.peak_min_fraction),
            "min_separation": float(args.min_separation),
            "v_clip_percentile": float(args.v_clip_percentile),
        },
        "u_quantile": {"min": float(args.u_quantile_min), "max": float(args.u_quantile_max)},
        "stats": {
            "frames_scanned": int(len(frames)),
            "circles_used": int(xy_all.shape[0]),
            "pca_ratio_global": float(ratio_global),
            "theta_global_deg": float(theta_global_deg),
        },
    }
    (out_dir / "row_model_auto.json").write_text(json.dumps(row_model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Per-frame stats.
    rows_csv: List[List[object]] = []
    prev_dir: Optional[np.ndarray] = None
    for fr in sorted(frames, key=lambda f: f.index):
        xy = fr.centers_xy
        n = int(xy.shape[0])
        dir_frame: Optional[np.ndarray] = None
        ratio_frame = float("nan")
        if n >= min_circles:
            dir_frame, ratio_frame = _pca_direction(xy)
            if dir_frame is not None:
                # Resolve the sign ambiguity to be consistent with global direction / previous.
                if float(np.dot(dir_frame, direction_global)) < 0.0:
                    dir_frame = -dir_frame
                if prev_dir is not None and float(np.dot(dir_frame, prev_dir)) < 0.0:
                    dir_frame = -dir_frame
                prev_dir = dir_frame

        theta_frame = _angle_deg_from_direction(dir_frame) if dir_frame is not None else float("nan")
        delta = float(theta_frame - theta_global_deg) if math.isfinite(theta_frame) else float("nan")
        rows_csv.append([int(fr.index), int(n), f"{theta_global_deg:.6f}", f"{theta_frame:.6f}" if math.isfinite(theta_frame) else "", f"{delta:.6f}" if math.isfinite(delta) else "", f"{ratio_frame:.6f}" if math.isfinite(ratio_frame) else ""])

    with (out_dir / "row_direction_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "circles", "theta_global_deg", "theta_frame_deg", "delta_deg", "pca_ratio_frame"])
        writer.writerows(rows_csv)

    # Per-frame lane rows (2 rows per frame) derived from v distribution.
    lane_rows_csv: List[List[object]] = []
    for fr in sorted(frames, key=lambda f: f.index):
        xy = fr.centers_xy
        if xy.size == 0:
            lane_rows_csv.append([int(fr.index), 0, "", "", "", "", "", ""])
            continue
        uv_u = xy.dot(direction_global.astype(np.float32).reshape(2)).astype(np.float64)
        uv_v = xy.dot(perp_global.astype(np.float32).reshape(2)).astype(np.float64)
        picked = _pick_two_lane_rows_from_v(
            uv_v,
            gap_thr=float(args.lane_gap_thr),
            min_points_per_row=int(args.lane_min_points),
            min_separation=float(args.lane_min_sep),
            max_separation=float(args.lane_max_sep),
        )
        if picked is None:
            lane_rows_csv.append([int(fr.index), int(xy.shape[0]), "", "", "", "", "", ""])
            continue
        v0, v1, n0, n1 = picked
        lane_rows_csv.append(
            [
                int(fr.index),
                int(xy.shape[0]),
                f"{float(v0):.6f}",
                f"{float(v1):.6f}",
                f"{float(v1 - v0):.6f}",
                int(n0),
                int(n1),
                f"{float(np.median(uv_u)):.6f}" if uv_u.size else "",
            ]
        )
    with (out_dir / "lane_rows_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "circles", "v0", "v1", "lane_width", "n0", "n1", "u_median"])
        writer.writerows(lane_rows_csv)

    # Global UV-aligned plot (paper-friendly).
    if xy_all.size:
        u_all = xy_all.dot(direction_global.astype(np.float32).reshape(2)).astype(np.float64)
        v_all = xy_all.dot(perp_global.astype(np.float32).reshape(2)).astype(np.float64)
        _render_uv_global(
            out_png=out_dir / "uv_global.png",
            u=u_all,
            v=v_all,
            rows_uv=rows_uv,
            width=1400,
            height=700,
            margin=60,
        )

    # Overlay previews.
    if not bool(args.no_overlay) and png_dir.is_dir():
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

        render = meta.get("render", {}) if isinstance(meta.get("render", {}), dict) else {}
        bounds_mode = str(render.get("bounds_mode", "center")).strip().lower()
        fixed_bounds = render.get("bounds", None)
        window_x = float(render.get("window_x", 30.0))
        window_y = float(render.get("window_y", 20.0))
        width = int(render.get("width", 1400))
        height = int(render.get("height", 1000))
        margin = int(render.get("margin_px", render.get("margin", 50)))

        overlay_dir = out_dir / "png_overlay"
        overlay_every = max(1, int(args.overlay_every))
        overlay_indices: Optional[set[int]] = None
        if str(args.overlay_indices).strip():
            overlay_indices = set()
            for part in str(args.overlay_indices).replace(" ", "").split(","):
                if not part:
                    continue
                try:
                    overlay_indices.add(int(part))
                except Exception:
                    continue

        prev_dir2: Optional[np.ndarray] = None
        for fr in sorted(frames, key=lambda f: f.index):
            if overlay_indices is not None:
                if int(fr.index) not in overlay_indices:
                    continue
            elif fr.index % overlay_every != 0:
                continue
            in_png = png_dir / f"bev_{fr.index:06d}.png"
            if not in_png.is_file():
                continue

            center = _median_center_or_none(fr.centers_xy)
            if bounds_mode == "fixed" and isinstance(fixed_bounds, list) and len(fixed_bounds) == 4:
                bounds = (float(fixed_bounds[0]), float(fixed_bounds[1]), float(fixed_bounds[2]), float(fixed_bounds[3]))
            elif center is not None:
                bounds = _frame_bounds_center_window(center, window_x, window_y)
            else:
                bounds = (-0.5 * window_x, 0.5 * window_x, -0.5 * window_y, 0.5 * window_y)

            # Per-frame direction for overlay (optional).
            dir_frame, _ = _pca_direction(fr.centers_xy) if int(fr.centers_xy.shape[0]) >= min_circles else (None, float("nan"))
            if dir_frame is not None:
                if float(np.dot(dir_frame, direction_global)) < 0.0:
                    dir_frame = -dir_frame
                if prev_dir2 is not None and float(np.dot(dir_frame, prev_dir2)) < 0.0:
                    dir_frame = -dir_frame
                prev_dir2 = dir_frame

            overlay_rows_mode = str(args.overlay_row_lines).strip().lower()
            if overlay_rows_mode == "none":
                rows_for_overlay: List[Dict[str, float]] = []
            elif overlay_rows_mode == "frame_lane":
                uv_v = fr.centers_xy.dot(perp_global.astype(np.float32).reshape(2)).astype(np.float64)
                picked = _pick_two_lane_rows_from_v(
                    uv_v,
                    gap_thr=float(args.lane_gap_thr),
                    min_points_per_row=int(args.lane_min_points),
                    min_separation=float(args.lane_min_sep),
                    max_separation=float(args.lane_max_sep),
                )
                if picked is None:
                    rows_for_overlay = []
                else:
                    v0, v1, _, _ = picked
                    rows_for_overlay = [{"v_center": float(v0)}, {"v_center": float(v1)}]
            else:
                rows_for_overlay = list(rows_uv)

            title = f"idx {fr.index:06d} | global {theta_global_deg:.1f}deg | overlay_rows={overlay_rows_mode}"
            _draw_overlay(
                in_png=in_png,
                out_png=overlay_dir / f"overlay_{fr.index:06d}.png",
                direction_xy=direction_global,
                direction_frame_xy=dir_frame,
                rows_uv=rows_for_overlay,
                fill_drivable=bool(args.overlay_fill_drivable),
                drivable_color_bgr=_hex_to_bgr(str(args.drivable_color), (120, 200, 120)),
                drivable_alpha=float(args.drivable_alpha),
                drivable_margin_v=float(args.drivable_margin_v),
                bounds=bounds,
                width=width,
                height=height,
                margin=margin,
                title=title,
            )

    # Run meta for reproducibility.
    run_meta = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "params": {
            "every": int(every),
            "max_frames": int(max_frames),
            "min_circles_per_frame": int(min_circles),
            "row_split_thr": float(args.row_split_thr),
            "min_trees_per_row": int(args.min_trees_per_row),
            "u_quantile_min": float(args.u_quantile_min),
            "u_quantile_max": float(args.u_quantile_max),
            "rows_mode": str(args.rows_mode).strip(),
            "max_rows": int(args.max_rows),
            "row_bandwidth": float(args.row_bandwidth),
            "v_bin": float(args.v_bin),
            "smooth_window": int(args.smooth_window),
            "peak_min_fraction": float(args.peak_min_fraction),
            "min_separation": float(args.min_separation),
            "v_clip_percentile": float(args.v_clip_percentile),
            "no_overlay": bool(args.no_overlay),
            "overlay_every": int(args.overlay_every),
            "overlay_indices": str(args.overlay_indices).strip(),
            "overlay_row_lines": str(args.overlay_row_lines).strip(),
            "overlay_fill_drivable": bool(args.overlay_fill_drivable),
            "drivable_color": str(args.drivable_color).strip(),
            "drivable_alpha": float(args.drivable_alpha),
            "drivable_margin_v": float(args.drivable_margin_v),
            "lane_gap_thr": float(args.lane_gap_thr),
            "lane_min_points": int(args.lane_min_points),
            "lane_min_sep": float(args.lane_min_sep),
            "lane_max_sep": float(args.lane_max_sep),
        },
        "result": {"theta_global_deg": float(theta_global_deg), "rows": int(len(rows_uv))},
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] direction_xy: {direction_global.tolist()} (theta={theta_global_deg:.2f} deg)")
    print(f"[OK] rows_uv: {len(rows_uv)}")
    print(f"[OK] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
