#!/usr/bin/env python3
"""Interactive manual row-prior drawing on a BEV image using OpenCV.

This avoids needing an SVG editor. It renders a BEV background from a PCD
using the same bounds as `bev_meta*.json`, then lets you click 2 points per
row line and exports:
- an SVG with <line> primitives (pixel coordinates)
- a JSON prior in map coordinates (compatible with compare_row_fits_to_manual.py)

Controls:
  - Left click: add point (2 points = one line)
  - u: undo last point/line
  - r: reset all
  - s: save (writes outputs) and quit
  - q / ESC: quit without saving
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
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


def _parse_hex_rgb(text: str) -> Tuple[int, int, int]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty color")
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join([c * 2 for c in text])
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB, got: {text!r}")
    r = int(text[0:2], 16)
    g = int(text[2:4], 16)
    b = int(text[4:6], 16)
    return r, g, b


def _load_bev_meta(path: Path) -> Dict[str, Any]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    required = {"bounds", "width", "height", "margin"}
    missing = required - set(meta.keys())
    if missing:
        raise ValueError(f"BEV meta missing keys: {sorted(missing)}")
    bounds = meta["bounds"]
    if not (isinstance(bounds, list) and len(bounds) == 4):
        raise ValueError("BEV meta bounds must be [xmin,xmax,ymin,ymax]")
    return meta


def _compute_scale(bounds: Sequence[float], width: int, height: int, margin: int) -> float:
    xmin, xmax, ymin, ymax = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    dx = max(1.0e-6, xmax - xmin)
    dy = max(1.0e-6, ymax - ymin)
    return float(min((float(width) - 2.0 * float(margin)) / dx, (float(height) - 2.0 * float(margin)) / dy))


def _map_to_pixel(
    x: float,
    y: float,
    *,
    bounds: Sequence[float],
    width: int,
    height: int,
    margin: int,
    scale: float,
) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    px = float(margin) + (float(x) - float(xmin)) * float(scale)
    py = float(margin) + (float(ymax) - float(y)) * float(scale)
    return px, py


def _pixel_to_map(
    px: float,
    py: float,
    *,
    bounds: Sequence[float],
    width: int,
    height: int,
    margin: int,
    scale: float,
) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    x = (float(px) - float(margin)) / float(scale) + float(xmin)
    y = float(ymax) - (float(py) - float(margin)) / float(scale)
    return float(x), float(y)


def _segment_to_row_uv(p0: Sequence[float], p1: Sequence[float], direction_xy: np.ndarray, perp_xy: np.ndarray) -> Dict[str, float]:
    p0 = np.array(p0, dtype=np.float64).reshape(2)
    p1 = np.array(p1, dtype=np.float64).reshape(2)
    u0 = float(p0.dot(direction_xy))
    u1 = float(p1.dot(direction_xy))
    v0 = float(p0.dot(perp_xy))
    v1 = float(p1.dot(perp_xy))
    v_center = 0.5 * (v0 + v1)
    u_min = float(min(u0, u1))
    u_max = float(max(u0, u1))
    return {"v_center": float(v_center), "u_min": float(u_min), "u_max": float(u_max), "z": 0.0}


def _render_bev_points_bgr(
    *,
    pcd_path: Path,
    bounds: Sequence[float],
    width: int,
    height: int,
    margin: int,
    z_min: float,
    z_max: float,
    max_points: int,
    sample_seed: int,
    point_size_px: int,
    bg_rgb: Tuple[int, int, int],
    point_rgb: Tuple[int, int, int],
) -> np.ndarray:
    impl = _load_tree_circles_impl()

    points = impl._load_pcd_xyz(pcd_path).astype(np.float32)
    points = impl._filter_points(points, z_min=float(z_min), z_max=float(z_max), x_min=-1.0e9, x_max=1.0e9, y_abs_max=1.0e9)
    if points.size == 0:
        raise RuntimeError("No points after z filtering; adjust z_min/z_max.")

    xy = points[:, :2].astype(np.float32)
    xmin, xmax, ymin, ymax = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    keep = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
    xy = xy[keep]
    if xy.shape[0] == 0:
        raise RuntimeError("No points inside bounds; check bev_meta bounds.")

    if int(max_points) > 0 and xy.shape[0] > int(max_points):
        rng = np.random.default_rng(int(sample_seed))
        idx = rng.choice(xy.shape[0], int(max_points), replace=False)
        xy = xy[idx]

    bg_bgr = (int(bg_rgb[2]), int(bg_rgb[1]), int(bg_rgb[0]))
    pt_bgr = (int(point_rgb[2]), int(point_rgb[1]), int(point_rgb[0]))
    img = np.full((int(height), int(width), 3), bg_bgr, dtype=np.uint8)

    scale = _compute_scale(bounds, width, height, margin)
    px = (float(margin) + (xy[:, 0].astype(np.float64) - float(xmin)) * float(scale)).astype(np.int32)
    py = (float(margin) + (float(ymax) - xy[:, 1].astype(np.float64)) * float(scale)).astype(np.int32)

    # Clip to image bounds.
    mask = (px >= 0) & (px < int(width)) & (py >= 0) & (py < int(height))
    px = px[mask]
    py = py[mask]

    r = max(0, int(point_size_px) // 2)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            x = px + int(dx)
            y = py + int(dy)
            m = (x >= 0) & (x < int(width)) & (y >= 0) & (y < int(height))
            img[y[m], x[m]] = pt_bgr
    return img


def _write_svg(out_path: Path, width: int, height: int, stroke: str, stroke_width: float, lines_px: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    svg: List[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" '
        f'viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="xMidYMid meet">'
    )
    svg.append(f'  <g id="manual_lines" fill="none" stroke="{stroke}" stroke-width="{float(stroke_width):.2f}">')
    for (x0, y0), (x1, y1) in lines_px:
        svg.append(f'    <line x1="{float(x0):.2f}" y1="{float(y0):.2f}" x2="{float(x1):.2f}" y2="{float(y1):.2f}" />')
    svg.append("  </g>")
    svg.append("</svg>\n")
    out_path.write_text("\n".join(svg), encoding="utf-8")


@dataclass
class _State:
    current: List[Tuple[int, int]]
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    dirty: bool


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_meta = ws_dir / "maps" / "ablation_bev_smoke2" / "B_row_peaks_ransac" / "bev_meta_dense.json"
    default_row_model = ws_dir / "maps" / "row_model_from_map4.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--bev-meta", type=str, default=str(default_meta))
    parser.add_argument("--pcd", type=str, default="")
    parser.add_argument("--row-model", type=str, default=str(default_row_model))
    parser.add_argument("--out-svg", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--out-png", type=str, default="", help="Optional: save the rendered BEV background PNG.")

    parser.add_argument("--z-min", type=float, default=float("nan"))
    parser.add_argument("--z-max", type=float, default=float("nan"))
    parser.add_argument("--max-points", type=int, default=-1)
    parser.add_argument("--sample-seed", type=int, default=0)

    parser.add_argument("--bg", type=str, default="#ffffff")
    parser.add_argument("--point-color", type=str, default="#b4b4b4")
    parser.add_argument("--point-size-px", type=int, default=2)
    parser.add_argument("--line-color", type=str, default="#ff0000")
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--stroke-width", type=float, default=3.0)

    args = parser.parse_args()

    meta_path = Path(args.bev_meta).expanduser().resolve()
    meta = _load_bev_meta(meta_path)
    bounds = meta["bounds"]
    width = int(meta["width"])
    height = int(meta["height"])
    margin = int(meta["margin"])

    pcd_path = Path(str(args.pcd).strip() or str(meta.get("pcd", ""))).expanduser().resolve()
    if not pcd_path.is_file():
        raise FileNotFoundError(f"PCD not found: {pcd_path}")

    z_min = float(args.z_min) if math.isfinite(float(args.z_min)) else float(meta.get("z_min", 0.9))
    z_max = float(args.z_max) if math.isfinite(float(args.z_max)) else float(meta.get("z_max", 1.1))
    max_points = int(args.max_points) if int(args.max_points) >= 0 else int(meta.get("max_points", 120000))

    bg_rgb = _parse_hex_rgb(str(args.bg))
    point_rgb = _parse_hex_rgb(str(args.point_color))
    line_rgb = _parse_hex_rgb(str(args.line_color))
    line_bgr = (int(line_rgb[2]), int(line_rgb[1]), int(line_rgb[0]))

    base = _render_bev_points_bgr(
        pcd_path=pcd_path,
        bounds=bounds,
        width=width,
        height=height,
        margin=margin,
        z_min=z_min,
        z_max=z_max,
        max_points=max_points,
        sample_seed=int(args.sample_seed),
        point_size_px=int(args.point_size_px),
        bg_rgb=bg_rgb,
        point_rgb=point_rgb,
    )

    if str(args.out_png).strip():
        out_png = Path(args.out_png).expanduser().resolve()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), base)

    state = _State(current=[], lines=[], dirty=True)
    window = "Draw manual row lines (2 clicks per line)"

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state.current.append((int(x), int(y)))
        if len(state.current) >= 2:
            p0 = state.current[0]
            p1 = state.current[1]
            state.lines.append((p0, p1))
            state.current.clear()
        state.dirty = True

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    help_lines = [
        "Left click: 2 points = line",
        "u: undo   r: reset",
        "s: save+quit   q/ESC: quit",
    ]

    while True:
        if state.dirty:
            canvas = base.copy()
            # Draw committed lines.
            for idx, (p0, p1) in enumerate(state.lines):
                cv2.line(canvas, p0, p1, line_bgr, int(args.line_thickness), cv2.LINE_AA)
                mid = (int((p0[0] + p1[0]) * 0.5), int((p0[1] + p1[1]) * 0.5))
                cv2.putText(
                    canvas,
                    str(idx),
                    mid,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    line_bgr,
                    2,
                    cv2.LINE_AA,
                )
            # Draw current points.
            for p in state.current:
                cv2.circle(canvas, p, 4, line_bgr, -1, cv2.LINE_AA)

            y0 = 24
            cv2.putText(
                canvas,
                f"lines={len(state.lines)}  z=[{z_min:.2f},{z_max:.2f}]  pts<= {max_points if max_points>0 else 'all'}",
                (12, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            for i, t in enumerate(help_lines):
                cv2.putText(
                    canvas,
                    t,
                    (12, y0 + 26 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (40, 40, 40),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window, canvas)
            state.dirty = False

        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        if key in (27, ord("q")):  # ESC / q
            cv2.destroyAllWindows()
            return 0
        if key == ord("u"):
            if state.current:
                state.current.pop()
            elif state.lines:
                state.lines.pop()
            state.dirty = True
        if key == ord("r"):
            state.current.clear()
            state.lines.clear()
            state.dirty = True
        if key == ord("s"):
            if len(state.lines) < 2:
                print("[WARN] You drew <2 lines. For full orchard, draw all 4-5 rows.", file=sys.stderr)
            break

    cv2.destroyAllWindows()

    out_svg = Path(args.out_svg).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    row_model_path = Path(args.row_model).expanduser().resolve()
    if not row_model_path.is_file():
        raise FileNotFoundError(f"row-model not found: {row_model_path}")

    scale = _compute_scale(bounds, width, height, margin)
    lines_px_float: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    lines_map: List[Dict[str, Any]] = []

    for i, (p0, p1) in enumerate(state.lines):
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        lines_px_float.append(((x0, y0), (x1, y1)))
        mp0 = _pixel_to_map(x0, y0, bounds=bounds, width=width, height=height, margin=margin, scale=scale)
        mp1 = _pixel_to_map(x1, y1, bounds=bounds, width=width, height=height, margin=margin, scale=scale)
        lines_map.append(
            {
                "id": int(i),
                "p0": [float(mp0[0]), float(mp0[1])],
                "p1": [float(mp1[0]), float(mp1[1])],
                "stroke": str(args.line_color),
                "length_px": float(math.hypot(x1 - x0, y1 - y0)),
            }
        )

    _write_svg(
        out_path=out_svg,
        width=width,
        height=height,
        stroke=str(args.line_color),
        stroke_width=float(args.stroke_width),
        lines_px=lines_px_float,
    )

    row_model = json.loads(row_model_path.read_text(encoding="utf-8"))
    direction_xy = np.array(row_model["direction_xy"], dtype=np.float64).reshape(2)
    perp_xy = np.array(row_model["perp_xy"], dtype=np.float64).reshape(2)

    out: Dict[str, Any] = {
        "source_tool": "draw_manual_prior_cv2.py",
        "source_image": str(pcd_path),
        "bev_meta": str(meta_path),
        "bounds": [float(b) for b in bounds],
        "width": int(width),
        "height": int(height),
        "margin": int(margin),
        "lines": lines_map,
        "row_model": str(row_model_path),
        "direction_xy": [float(direction_xy[0]), float(direction_xy[1])],
        "perp_xy": [float(perp_xy[0]), float(perp_xy[1])],
        "rows_uv": [_segment_to_row_uv(l["p0"], l["p1"], direction_xy, perp_xy) for l in lines_map],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] Wrote SVG : {out_svg}")
    print(f"[OK] Wrote JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

