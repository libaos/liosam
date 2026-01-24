#!/usr/bin/env python3
"""Compare reference path (JSON) vs replay odometry (CSV), and optionally render SVG/PNG overlays.

Why not matplotlib?
- In many ROS/Gazebo workspaces people activate a python venv, which often breaks matplotlib/numpy ABI combos.
- By default this script uses only the Python standard library. If `--out-png` is used, it requires `Pillow`.

Inputs:
- Reference: JSON produced by `rosbag_path_to_json.py` or any JSON with a `points` list of {x,y[,yaw]}.
- Replay: CSV produced by `record_trajectory.py` (columns: timestamp,x,y,yaw).

Outputs:
- Console summary (length, closure, distance metrics).
- Optional: report JSON + SVG/PNG overlays for quick inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


XY = Tuple[float, float]


def _load_reference_xy(json_path: Path) -> Tuple[List[XY], Dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    points = data.get("points", data)
    if not isinstance(points, list) or not points:
        raise RuntimeError(f"Invalid reference JSON (missing points): {json_path}")
    xy: List[XY] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        if "x" not in p or "y" not in p:
            continue
        xy.append((float(p["x"]), float(p["y"])))
    if len(xy) < 2:
        raise RuntimeError(f"Too few reference points in: {json_path}")
    return xy, data


def _load_replay_rows(csv_path: Path) -> List[Tuple[float, float, float]]:
    rows: List[Tuple[float, float, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, r in enumerate(reader):
            if not r:
                continue
            if "x" not in r or "y" not in r:
                continue
            try:
                x = float(r["x"])
                y = float(r["y"])
            except Exception:
                continue
            try:
                t = float(r.get("timestamp", idx))
            except Exception:
                t = float(idx)
            rows.append((t, x, y))
    if len(rows) < 2:
        raise RuntimeError(f"Too few replay points in: {csv_path}")
    return rows


def _load_replay_xy(csv_path: Path) -> List[XY]:
    rows: List[XY] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for r in reader:
            if not r:
                continue
            if "x" not in r or "y" not in r:
                continue
            try:
                rows.append((float(r["x"]), float(r["y"])))
            except Exception:
                continue
    if len(rows) < 2:
        raise RuntimeError(f"Too few replay points in: {csv_path}")
    return rows


def _polyline_length(xy: Sequence[XY]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return float(total)


def _closure_dist(xy: Sequence[XY]) -> float:
    if len(xy) < 2:
        return float("nan")
    x0, y0 = xy[0]
    x1, y1 = xy[-1]
    return float(math.hypot(x1 - x0, y1 - y0))


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    q = float(q)
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    pos = q * (len(sorted_vals) - 1)
    i = int(math.floor(pos))
    j = int(math.ceil(pos))
    if i == j:
        return float(sorted_vals[i])
    t = pos - i
    return float(sorted_vals[i] * (1.0 - t) + sorted_vals[j] * t)


def _point_to_segment_dist2(p: XY, a: XY, b: XY) -> Tuple[float, float]:
    """Return (dist^2, t) where proj = a + t*(b-a), t in [0,1]."""
    px, py = p
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    vv = vx * vx + vy * vy
    if vv <= 1.0e-12:
        return (wx * wx + wy * wy, 0.0)
    t = (wx * vx + wy * vy) / vv
    if t <= 0.0:
        dx = px - ax
        dy = py - ay
        return (dx * dx + dy * dy, 0.0)
    if t >= 1.0:
        dx = px - bx
        dy = py - by
        return (dx * dx + dy * dy, 1.0)
    projx = ax + t * vx
    projy = ay + t * vy
    dx = px - projx
    dy = py - projy
    return (dx * dx + dy * dy, float(t))


def _nearest_on_polyline(p: XY, ref: Sequence[XY], ref_cum: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """Return (min_dist, s_along_ref_m).

    s_along_ref_m is approximate arc-length along `ref` for the closest projection point.
    """
    if len(ref) < 2:
        return (float("inf"), 0.0)
    best_d2 = float("inf")
    best_s = 0.0
    use_cum = ref_cum is not None and len(ref_cum) == len(ref)
    for i in range(len(ref) - 1):
        d2, t = _point_to_segment_dist2(p, ref[i], ref[i + 1])
        if d2 < best_d2:
            best_d2 = d2
            seg_len = math.hypot(ref[i + 1][0] - ref[i][0], ref[i + 1][1] - ref[i][1])
            base = float(ref_cum[i]) if use_cum else _polyline_length(ref[: i + 1])
            best_s = base + float(t) * float(seg_len)
    return (math.sqrt(best_d2), float(best_s))


def _nearest_on_polyline_detail(p: XY, ref: Sequence[XY], ref_cum: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return (min_dist, s_along_ref_m, proj_x, proj_y)."""
    if len(ref) < 2:
        return (float("inf"), 0.0, float("nan"), float("nan"))
    best_d2 = float("inf")
    best_s = 0.0
    best_proj = (float("nan"), float("nan"))
    for i in range(len(ref) - 1):
        d2, t = _point_to_segment_dist2(p, ref[i], ref[i + 1])
        if d2 < best_d2:
            best_d2 = d2
            ax, ay = ref[i]
            bx, by = ref[i + 1]
            projx = ax + float(t) * (bx - ax)
            projy = ay + float(t) * (by - ay)
            best_proj = (float(projx), float(projy))
            seg_len = math.hypot(bx - ax, by - ay)
            best_s = float(ref_cum[i]) + float(t) * float(seg_len)
    return (math.sqrt(best_d2), float(best_s), best_proj[0], best_proj[1])


def _downsample(xy: Sequence[XY], max_points: int) -> List[XY]:
    xy = list(xy)
    if max_points <= 0 or len(xy) <= max_points:
        return xy
    step = max(1, int(math.ceil(len(xy) / float(max_points))))
    out = xy[::step]
    if out[-1] != xy[-1]:
        out.append(xy[-1])
    return out


def _downsample_indices(n: int, max_points: int) -> List[int]:
    if max_points <= 0 or n <= max_points:
        return list(range(n))
    step = max(1, int(math.ceil(n / float(max_points))))
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs


def _svg_polyline(points: Sequence[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _hex_color(r: float, g: float, b: float) -> str:
    rr = max(0, min(255, int(round(r))))
    gg = max(0, min(255, int(round(g))))
    bb = max(0, min(255, int(round(b))))
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * float(t))


def _color_from_t(t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    # 4-stop gradient: blue -> green -> orange -> red
    stops = [
        (0.00, (31, 119, 180)),
        (0.33, (44, 160, 44)),
        (0.66, (255, 127, 14)),
        (1.00, (214, 39, 40)),
    ]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            return _hex_color(_lerp(c0[0], c1[0], u), _lerp(c0[1], c1[1], u), _lerp(c0[2], c1[2], u))
    return _hex_color(214, 39, 40)


def _write_svg(out_path: Path, ref_xy: Sequence[XY], rep_xy: Sequence[XY], title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 800
    margin = 60

    all_xy = list(ref_xy) + list(rep_xy)
    xs = [p[0] for p in all_xy]
    ys = [p[1] for p in all_xy]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    pad = 0.5
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad

    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    dx = max(1.0e-6, max_x - min_x)
    dy = max(1.0e-6, max_y - min_y)
    scale = min(plot_w / dx, plot_h / dy)

    def to_px(pt: XY) -> Tuple[float, float]:
        x, y = pt
        px = margin + (x - min_x) * scale
        py = height - margin - (y - min_y) * scale
        return (px, py)

    ref_px = [to_px(p) for p in ref_xy]
    rep_px = [to_px(p) for p in rep_xy]

    # Basic axes box
    box_x0 = margin
    box_y0 = margin
    box_w = plot_w
    box_h = plot_h

    # Start/end markers
    ref_start = to_px(ref_xy[0])
    ref_end = to_px(ref_xy[-1])
    rep_start = to_px(rep_xy[0])
    rep_end = to_px(rep_xy[-1])

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  <text x="{margin}" y="{margin-20}" font-family="monospace" font-size="16" fill="#111">{_xml_escape(title)}</text>
  <rect x="{box_x0}" y="{box_y0}" width="{box_w}" height="{box_h}" fill="none" stroke="#aaa" stroke-width="1"/>

  <polyline points="{_svg_polyline(ref_px)}" fill="none" stroke="#d62728" stroke-width="2" opacity="0.95"/>
  <polyline points="{_svg_polyline(rep_px)}" fill="none" stroke="#1f77b4" stroke-width="2" opacity="0.90"/>

  <circle cx="{ref_start[0]:.2f}" cy="{ref_start[1]:.2f}" r="5" fill="#2ca02c"/>
  <circle cx="{ref_end[0]:.2f}" cy="{ref_end[1]:.2f}" r="5" fill="#ff7f0e"/>
  <path d="M {rep_start[0]-5:.2f} {rep_start[1]-5:.2f} L {rep_start[0]+5:.2f} {rep_start[1]+5:.2f} M {rep_start[0]-5:.2f} {rep_start[1]+5:.2f} L {rep_start[0]+5:.2f} {rep_start[1]-5:.2f}" stroke="#2ca02c" stroke-width="2"/>
  <path d="M {rep_end[0]-5:.2f} {rep_end[1]-5:.2f} L {rep_end[0]+5:.2f} {rep_end[1]+5:.2f} M {rep_end[0]-5:.2f} {rep_end[1]+5:.2f} L {rep_end[0]+5:.2f} {rep_end[1]-5:.2f}" stroke="#ff7f0e" stroke-width="2"/>

  <rect x="{width-360}" y="{margin-10}" width="300" height="86" fill="white" stroke="#ddd"/>
  <line x1="{width-340}" y1="{margin+10}" x2="{width-300}" y2="{margin+10}" stroke="#d62728" stroke-width="3"/>
  <text x="{width-290}" y="{margin+15}" font-family="monospace" font-size="14" fill="#111">reference (json)</text>
  <line x1="{width-340}" y1="{margin+32}" x2="{width-300}" y2="{margin+32}" stroke="#1f77b4" stroke-width="3"/>
  <text x="{width-290}" y="{margin+37}" font-family="monospace" font-size="14" fill="#111">replay (odom)</text>
  <circle cx="{width-320}" cy="{margin+56}" r="5" fill="#2ca02c"/>
  <text x="{width-290}" y="{margin+61}" font-family="monospace" font-size="14" fill="#111">start</text>
  <circle cx="{width-320}" cy="{margin+76}" r="5" fill="#ff7f0e"/>
  <text x="{width-290}" y="{margin+81}" font-family="monospace" font-size="14" fill="#111">end</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def _write_png(out_path: Path, ref_xy: Sequence[XY], rep_xy: Sequence[XY], title: str) -> None:
    """Render a simple PNG (black=reference, red=replay) without matplotlib."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PIL (pillow) not available; cannot render PNG") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 800
    margin = 60

    all_xy = list(ref_xy) + list(rep_xy)
    xs = [p[0] for p in all_xy]
    ys = [p[1] for p in all_xy]
    min_x = min(xs) - 0.5
    max_x = max(xs) + 0.5
    min_y = min(ys) - 0.5
    max_y = max(ys) + 0.5

    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    dx = max(1.0e-6, max_x - min_x)
    dy = max(1.0e-6, max_y - min_y)
    scale = min(plot_w / dx, plot_h / dy)

    def to_px(pt: XY) -> Tuple[float, float]:
        x, y = pt
        px = margin + (x - min_x) * scale
        py = height - margin - (y - min_y) * scale
        return (px, py)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Grid (auto step).
    span = max(dx, dy)
    step_candidates = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    step = step_candidates[-1]
    for s in step_candidates:
        if span / s <= 16.0:
            step = float(s)
            break
    gx0 = math.floor(min_x / step) * step
    gx1 = math.ceil(max_x / step) * step
    gy0 = math.floor(min_y / step) * step
    gy1 = math.ceil(max_y / step) * step
    grid_color = (238, 238, 238)
    box_color = (170, 170, 170)

    x = gx0
    while x <= gx1 + 1.0e-9:
        px = margin + (x - min_x) * scale
        draw.line([(px, margin), (px, height - margin)], fill=grid_color, width=1)
        x += step
    y = gy0
    while y <= gy1 + 1.0e-9:
        py = height - margin - (y - min_y) * scale
        draw.line([(margin, py), (width - margin, py)], fill=grid_color, width=1)
        y += step

    draw.rectangle([margin, margin, width - margin, height - margin], outline=box_color, width=1)

    def draw_polyline(xy: Sequence[XY], color: Tuple[int, int, int], width_px: int) -> None:
        if len(xy) < 2:
            return
        pts = [to_px(p) for p in xy]
        for a, b in zip(pts, pts[1:]):
            draw.line([a, b], fill=color, width=width_px)

    # Draw reference first, then replay on top.
    draw_polyline(ref_xy, color=(0, 0, 0), width_px=3)
    draw_polyline(rep_xy, color=(220, 0, 0), width_px=3)

    def circle(center: Tuple[float, float], r: float, color: Tuple[int, int, int]) -> None:
        cx, cy = center
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, fill=color)

    def cross(center: Tuple[float, float], r: float, color: Tuple[int, int, int]) -> None:
        cx, cy = center
        draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=color, width=2)
        draw.line([(cx - r, cy + r), (cx + r, cy - r)], fill=color, width=2)

    ref_start = to_px(ref_xy[0])
    ref_end = to_px(ref_xy[-1])
    rep_start = to_px(rep_xy[0])
    rep_end = to_px(rep_xy[-1])

    circle(ref_start, 5, (0, 160, 0))
    circle(ref_end, 5, (0, 90, 220))
    cross(rep_start, 6, (0, 160, 0))
    cross(rep_end, 6, (0, 90, 220))

    font = ImageFont.load_default()
    y0 = 10
    for line in textwrap.wrap(title, width=120)[:2]:
        draw.text((margin, y0), line, fill=(20, 20, 20), font=font)
        y0 += 15
    draw.text((margin, y0 + 3), "Reference (black) / Replay (red)", fill=(60, 60, 60), font=font)

    img.save(str(out_path), format="PNG")


def _write_svg_colored(
    out_path: Path,
    ref_xy: Sequence[XY],
    rep_xy: Sequence[XY],
    rep_err_m: Sequence[float],
    title: str,
    err_cap_m: float,
    err_max_m: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(rep_xy) < 2:
        raise RuntimeError("replay too short for colored svg")
    if len(rep_xy) != len(rep_err_m):
        raise RuntimeError("replay points and error length mismatch")

    width = 1200
    height = 800
    margin = 60

    all_xy = list(ref_xy) + list(rep_xy)
    xs = [p[0] for p in all_xy]
    ys = [p[1] for p in all_xy]
    min_x = min(xs) - 0.5
    max_x = max(xs) + 0.5
    min_y = min(ys) - 0.5
    max_y = max(ys) + 0.5

    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    dx = max(1.0e-6, max_x - min_x)
    dy = max(1.0e-6, max_y - min_y)
    scale = min(plot_w / dx, plot_h / dy)

    def to_px(pt: XY) -> Tuple[float, float]:
        x, y = pt
        px = margin + (x - min_x) * scale
        py = height - margin - (y - min_y) * scale
        return (px, py)

    ref_px = [to_px(p) for p in ref_xy]
    rep_px = [to_px(p) for p in rep_xy]

    box_x0 = margin
    box_y0 = margin
    box_w = plot_w
    box_h = plot_h

    ref_start = to_px(ref_xy[0])
    ref_end = to_px(ref_xy[-1])
    rep_start = to_px(rep_xy[0])
    rep_end = to_px(rep_xy[-1])

    cap = max(1.0e-9, float(err_cap_m))
    segs: List[str] = []
    for i in range(len(rep_px) - 1):
        (x0, y0) = rep_px[i]
        (x1, y1) = rep_px[i + 1]
        e = 0.5 * (float(rep_err_m[i]) + float(rep_err_m[i + 1]))
        color = _color_from_t(min(max(e / cap, 0.0), 1.0))
        segs.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="2" opacity="0.95"/>'
        )
    segs_svg = "\n  ".join(segs)

    # Legend color bar (discrete steps).
    bar_parts = []
    bar_x = width - 360
    bar_y = margin - 10
    bar_w = 300
    bar_h = 12
    steps = 12
    for i in range(steps):
        t = i / max(1, steps - 1)
        c = _color_from_t(t)
        x = bar_x + int(round(t * (bar_w - bar_w / steps)))
        w = int(round(bar_w / steps)) + 1
        bar_parts.append(f'<rect x="{x}" y="{bar_y}" width="{w}" height="{bar_h}" fill="{c}" stroke="none"/>')
    bar_svg = "\n  ".join(bar_parts)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  <text x="{margin}" y="{margin-20}" font-family="monospace" font-size="16" fill="#111">{_xml_escape(title)}</text>
  <rect x="{box_x0}" y="{box_y0}" width="{box_w}" height="{box_h}" fill="none" stroke="#aaa" stroke-width="1"/>

  <polyline points="{_svg_polyline(ref_px)}" fill="none" stroke="#d62728" stroke-width="2" opacity="0.90"/>
  {segs_svg}

  <circle cx="{ref_start[0]:.2f}" cy="{ref_start[1]:.2f}" r="5" fill="#2ca02c"/>
  <circle cx="{ref_end[0]:.2f}" cy="{ref_end[1]:.2f}" r="5" fill="#ff7f0e"/>
  <path d="M {rep_start[0]-5:.2f} {rep_start[1]-5:.2f} L {rep_start[0]+5:.2f} {rep_start[1]+5:.2f} M {rep_start[0]-5:.2f} {rep_start[1]+5:.2f} L {rep_start[0]+5:.2f} {rep_start[1]-5:.2f}" stroke="#2ca02c" stroke-width="2"/>
  <path d="M {rep_end[0]-5:.2f} {rep_end[1]-5:.2f} L {rep_end[0]+5:.2f} {rep_end[1]+5:.2f} M {rep_end[0]-5:.2f} {rep_end[1]+5:.2f} L {rep_end[0]+5:.2f} {rep_end[1]-5:.2f}" stroke="#ff7f0e" stroke-width="2"/>

  <rect x="{width-380}" y="{margin-30}" width="350" height="110" fill="white" stroke="#ddd"/>
  <text x="{width-370}" y="{margin-10}" font-family="monospace" font-size="13" fill="#111">reference (red) + replay colored by error</text>
  {bar_svg}
  <text x="{width-370}" y="{margin+24}" font-family="monospace" font-size="13" fill="#111">0m</text>
  <text x="{width-250}" y="{margin+24}" font-family="monospace" font-size="13" fill="#111">cap=p95={err_cap_m:.3f}m</text>
  <text x="{width-120}" y="{margin+24}" font-family="monospace" font-size="13" fill="#111">max={err_max_m:.3f}m</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def _xml_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_ref = ws_dir / "src" / "pcd_gazebo_world" / "maps" / "runs" / "rosbag_path.json"
    default_replay = ws_dir / "trajectory_data" / "teb_odom.csv"
    default_out = ws_dir / "src" / "pcd_gazebo_world" / "maps" / "runs" / "reference_vs_replay.svg"
    default_report = ws_dir / "src" / "pcd_gazebo_world" / "maps" / "runs" / "reference_vs_replay_report.json"

    parser = argparse.ArgumentParser(description="Compare reference path JSON vs replay odometry CSV (and render SVG)")
    parser.add_argument("--reference", type=str, default=str(default_ref), help="参考路径 JSON")
    parser.add_argument("--replay", type=str, default=str(default_replay), help="回放轨迹 CSV (odom)")
    parser.add_argument("--out", type=str, default=str(default_out), help="输出 SVG（空=不输出）")
    parser.add_argument("--out-colored", type=str, default="", help="输出彩色 SVG（replay 按误差上色；空=不输出）")
    parser.add_argument("--out-png", type=str, default="", help="输出 PNG（黑=参考，红=回放；空=不输出）")
    parser.add_argument("--report", type=str, default=str(default_report), help="输出 JSON 报告（空=不输出）")
    parser.add_argument("--details-csv", type=str, default="", help="输出每帧误差 CSV（空=不输出）")
    parser.add_argument("--title", type=str, default="", help="可选标题（空则自动生成）")
    parser.add_argument("--max-plot-points", type=int, default=2500, help="输出 SVG 时每条线最多点数（降采样）")
    args = parser.parse_args()

    ref_path = Path(args.reference).expanduser().resolve()
    replay_path = Path(args.replay).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if str(args.out).strip() else None
    out_colored_path = Path(args.out_colored).expanduser().resolve() if str(args.out_colored).strip() else None
    out_png_path = Path(args.out_png).expanduser().resolve() if str(args.out_png).strip() else None
    report_path = Path(args.report).expanduser().resolve() if str(args.report).strip() else None
    details_csv_path = Path(args.details_csv).expanduser().resolve() if str(args.details_csv).strip() else None

    if not ref_path.is_file():
        raise SystemExit(f"Reference JSON not found: {ref_path}")
    if not replay_path.is_file():
        raise SystemExit(f"Replay CSV not found: {replay_path}")

    ref_xy, ref_meta = _load_reference_xy(ref_path)
    rep_rows = _load_replay_rows(replay_path)
    rep_xy = [(x, y) for (_t, x, y) in rep_rows]

    ref_len = _polyline_length(ref_xy)
    rep_len = _polyline_length(rep_xy)
    ref_close = _closure_dist(ref_xy)
    rep_close = _closure_dist(rep_xy)

    # Precompute arc-length along reference.
    ref_cum = [0.0]
    for (x0, y0), (x1, y1) in zip(ref_xy, ref_xy[1:]):
        ref_cum.append(ref_cum[-1] + math.hypot(x1 - x0, y1 - y0))

    dists: List[float] = []
    details_rows: List[Tuple[int, float, float, float, float, float, float, float]] = []
    max_s = 0.0
    for idx, (t, x, y) in enumerate(rep_rows):
        p = (float(x), float(y))
        d, s, projx, projy = _nearest_on_polyline_detail(p, ref_xy, ref_cum=ref_cum)
        dists.append(float(d))
        details_rows.append((idx, float(t), float(x), float(y), float(d), float(s), float(projx), float(projy)))
        if s > max_s:
            max_s = float(s)

    dists_sorted = sorted(dists)
    mean_d = float(sum(dists_sorted) / len(dists_sorted)) if dists_sorted else float("nan")
    med_d = _percentile(dists_sorted, 0.5)
    p95_d = _percentile(dists_sorted, 0.95)
    max_d = float(dists_sorted[-1]) if dists_sorted else float("nan")

    coverage_ratio = float(max_s / ref_len) if ref_len > 1.0e-9 else float("nan")

    title = str(args.title).strip()
    if not title:
        topic = str(ref_meta.get("topic", ""))
        title = (
            f"Reference vs replay | ref_close={ref_close:.2f}m replay_close={rep_close:.2f}m "
            f"mean={mean_d:.2f}m p95={p95_d:.2f}m max={max_d:.2f}m cover={coverage_ratio*100:.1f}%"
        )
        if topic:
            title = f"{title} | {Path(ref_meta.get('bag','')).name} {topic}"

    report: Dict[str, Any] = {
        "reference": str(ref_path),
        "replay": str(replay_path),
        "reference_points": int(len(ref_xy)),
        "replay_points": int(len(rep_xy)),
        "reference_length_m": float(ref_len),
        "replay_length_m": float(rep_len),
        "reference_closure_m": float(ref_close),
        "replay_closure_m": float(rep_close),
        "replay_to_reference": {
            "mean_m": float(mean_d),
            "median_m": float(med_d),
            "p95_m": float(p95_d),
            "max_m": float(max_d),
        },
        "coverage_ratio": float(coverage_ratio),
        "title": title,
    }

    # Keep stdout machine-readable for piping/redirection.
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote report: {report_path}", file=sys.stderr)

    if details_csv_path is not None:
        details_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with details_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "timestamp", "x", "y", "dist_to_ref_m", "s_ref_m", "proj_x", "proj_y"])
            for row in details_rows:
                writer.writerow(row)
        print(f"[OK] wrote details: {details_csv_path}", file=sys.stderr)

    if out_path is not None:
        ref_plot = _downsample(ref_xy, int(args.max_plot_points))
        rep_plot = _downsample(rep_xy, int(args.max_plot_points))
        _write_svg(out_path, ref_plot, rep_plot, title=title)
        print(f"[OK] wrote svg: {out_path}", file=sys.stderr)

    if out_colored_path is not None:
        idxs = _downsample_indices(len(rep_xy), int(args.max_plot_points))
        rep_plot = [rep_xy[i] for i in idxs]
        rep_err = [dists[i] for i in idxs]
        _write_svg_colored(
            out_colored_path,
            _downsample(ref_xy, int(args.max_plot_points)),
            rep_plot,
            rep_err,
            title=title,
            err_cap_m=float(p95_d),
            err_max_m=float(max_d),
        )
        print(f"[OK] wrote colored svg: {out_colored_path}", file=sys.stderr)

    if out_png_path is not None:
        ref_plot = _downsample(ref_xy, int(args.max_plot_points))
        rep_plot = _downsample(rep_xy, int(args.max_plot_points))
        _write_png(out_png_path, ref_plot, rep_plot, title=title)
        print(f"[OK] wrote png: {out_png_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
