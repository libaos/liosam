#!/usr/bin/env python3
"""Render BEV (top-down) SVG from tree map + row model + circles CSV.

This is a dependency-free renderer (uses numpy only) to produce publication-
ready BEV figures with consistent view bounds across methods.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


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
    try:
        decoded = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON map: {text}") from exc
    out: Dict[int, float] = {}
    if isinstance(decoded, dict):
        for k, v in decoded.items():
            out[int(k)] = float(v)
        return out
    raise ValueError(f"Expected JSON object like {{\"4\": 0.43}}, got: {type(decoded)}")


def _load_circles_csv(path: Path) -> np.ndarray:
    data: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return np.zeros((0, 3), dtype=np.float32)
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                x = float(parts[1])
                y = float(parts[2])
                r = float(parts[4])
            except Exception:
                continue
            data.append((x, y, r))
    if not data:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def _row_lines(
    direction_xy: np.ndarray,
    perp_xy: np.ndarray,
    rows: Sequence[Dict[str, float]],
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


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(points.shape[0], int(max_points), replace=False)
    return points[idx]


def _compute_bounds(points_xy: np.ndarray, pad: float) -> Tuple[float, float, float, float]:
    if points_xy.size == 0:
        return -10.0, 10.0, -10.0, 10.0
    xmin = float(np.min(points_xy[:, 0]))
    xmax = float(np.max(points_xy[:, 0]))
    ymin = float(np.min(points_xy[:, 1]))
    ymax = float(np.max(points_xy[:, 1]))
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True, type=str)
    parser.add_argument("--row-model", required=True, type=str)
    parser.add_argument("--circles", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)

    parser.add_argument("--z-min", type=float, default=0.9)
    parser.add_argument("--z-max", type=float, default=1.1)
    parser.add_argument("--max-points", type=int, default=40000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--bounds", type=str, default="")
    parser.add_argument("--pad", type=float, default=2.0)

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

    parser.add_argument("--row-v-offsets", type=str, default='{"4":0.43}')
    parser.add_argument("--row-u-offsets", type=str, default="")
    parser.add_argument("--row-v-slopes", type=str, default="")
    parser.add_argument("--row-v-yaw-offsets-deg", type=str, default='{"4":2.36}')

    args = parser.parse_args()

    pcd_path = Path(args.pcd).expanduser().resolve()
    row_model_path = Path(args.row_model).expanduser().resolve()
    circles_path = Path(args.circles).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    impl = _load_tree_circles_impl()

    points = impl._load_pcd_xyz(pcd_path).astype(np.float32)
    points = impl._filter_points(
        points,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        x_min=-1.0e9,
        x_max=1.0e9,
        y_abs_max=1.0e9,
    )
    points_xy = points[:, :2].astype(np.float32)
    if args.bounds:
        parts = [float(p) for p in args.bounds.split(",")]
        if len(parts) != 4:
            raise ValueError("bounds must be xmin,xmax,ymin,ymax")
        xmin, xmax, ymin, ymax = parts
    else:
        xmin, xmax, ymin, ymax = _compute_bounds(points_xy, float(args.pad))

    points_xy = _filter_points_by_bounds(points_xy, xmin, xmax, ymin, ymax)
    points_xy = _sample_points(points_xy, int(args.max_points), int(args.sample_seed))

    direction_xy, perp_xy, rows = impl._load_row_model_file(row_model_path)
    rows = impl._apply_row_overrides(
        rows,
        row_v_offsets=_parse_indexed_float_map(args.row_v_offsets),
        row_u_offsets=_parse_indexed_float_map(args.row_u_offsets) if args.row_u_offsets.strip() else {},
        row_v_slopes=_parse_indexed_float_map(args.row_v_slopes) if args.row_v_slopes.strip() else {},
        row_v_yaw_offsets_deg=_parse_indexed_float_map(args.row_v_yaw_offsets_deg),
    )
    row_lines = _row_lines(direction_xy, perp_xy, rows)

    circles = _load_circles_csv(circles_path)
    circles = _filter_circles_by_bounds(circles, xmin, xmax, ymin, ymax)

    width = int(args.width)
    height = int(args.height)
    margin = int(args.margin)

    svg: List[str] = []
    svg.append(_svg_header(width, height, args.bg))

    # Points
    svg.append(f'  <g id="points" fill="{args.point_color}" fill-opacity="0.9" stroke="none">')
    r = max(0.1, float(args.point_size))
    for x, y in points_xy.tolist():
        px, py = _map_point(x, y, xmin, xmax, ymin, ymax, width, height, margin)
        svg.append(f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{r:.2f}" />')
    svg.append("  </g>")

    # Row lines
    svg.append(f'  <g id="rows" fill="none" stroke="{args.row_color}" stroke-width="{float(args.row_width):.2f}">')
    for p0, p1 in row_lines:
        if not _line_bbox_intersects(p0, p1, xmin, xmax, ymin, ymax):
            continue
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        px0, py0 = _map_point(x0, y0, xmin, xmax, ymin, ymax, width, height, margin)
        px1, py1 = _map_point(x1, y1, xmin, xmax, ymin, ymax, width, height, margin)
        svg.append(f'    <line x1="{px0:.2f}" y1="{py0:.2f}" x2="{px1:.2f}" y2="{py1:.2f}" />')
    svg.append("  </g>")

    # Circles
    svg.append(
        f'  <g id="circles" fill="none" stroke="{args.circle_color}" stroke-width="{float(args.circle_width):.2f}">'
    )
    for x, y, radius in circles.tolist():
        px, py = _map_point(float(x), float(y), xmin, xmax, ymin, ymax, width, height, margin)
        pr = float(radius) * float(args.circle_scale)
        # Map radius in meters to pixels (use x scale).
        px_r, _ = _map_point(float(x) + pr, float(y), xmin, xmax, ymin, ymax, width, height, margin)
        r_px = abs(px_r - px)
        svg.append(f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{r_px:.2f}" />')
    svg.append("  </g>")

    svg.append("</svg>\n")
    out_path.write_text("\n".join(svg), encoding="utf-8")
    print(f"[OK] Wrote BEV SVG: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
