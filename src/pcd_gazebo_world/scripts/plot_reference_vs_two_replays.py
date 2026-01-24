#!/usr/bin/env python3
"""Compare reference path (JSON) vs two replay odometry CSVs, and render an SVG overlay.

This is a lightweight helper to quickly answer: "baseline vs via-points, which matches better?"
It uses only the Python standard library (same motivation as plot_reference_vs_replay.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


XY = Tuple[float, float]
Circle = Tuple[float, float, float]


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


def _load_tree_circles(json_path: Path, default_radius: float) -> Tuple[Optional[str], List[Circle]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    frame_id = None
    if isinstance(data, dict):
        frame_id = str(data.get("frame_id") or "").strip() or None
        circles = data.get("circles", [])
    elif isinstance(data, list):
        circles = data
    else:
        circles = []
    if not isinstance(circles, list) or not circles:
        raise RuntimeError(f"Invalid circles JSON (missing circles): {json_path}")

    out: List[Circle] = []
    for c in circles:
        if not isinstance(c, dict):
            continue
        if "x" not in c or "y" not in c:
            continue
        r = float(c.get("radius", default_radius))
        out.append((float(c["x"]), float(c["y"]), r))
    if not out:
        raise RuntimeError(f"Invalid circles JSON (no usable circles): {json_path}")
    return frame_id, out


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


def _build_cum_s(xy: Sequence[XY]) -> List[float]:
    out = [0.0]
    total = 0.0
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
        out.append(float(total))
    return out


def _point_to_segment_dist2(p: XY, a: XY, b: XY) -> Tuple[float, float]:
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


def _nearest_on_polyline(p: XY, ref: Sequence[XY], ref_cum: Sequence[float]) -> Tuple[float, float]:
    best_d2 = float("inf")
    best_s = 0.0
    for i in range(len(ref) - 1):
        d2, t = _point_to_segment_dist2(p, ref[i], ref[i + 1])
        if d2 < best_d2:
            best_d2 = d2
            seg_len = math.hypot(ref[i + 1][0] - ref[i][0], ref[i + 1][1] - ref[i][1])
            best_s = float(ref_cum[i]) + float(t) * float(seg_len)
    return (math.sqrt(best_d2), float(best_s))


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


def _metrics(reference_xy: Sequence[XY], replay_xy: Sequence[XY]) -> Dict[str, Any]:
    ref_len = _polyline_length(reference_xy)
    ref_cum = _build_cum_s(reference_xy)
    dists: List[float] = []
    max_s = 0.0
    for p in replay_xy:
        d, s = _nearest_on_polyline(p, reference_xy, ref_cum)
        dists.append(float(d))
        if s > max_s:
            max_s = float(s)
    dists.sort()
    mean = float(sum(dists) / len(dists)) if dists else float("nan")
    return {
        "coverage_ratio": float(max_s / ref_len) if ref_len > 1.0e-9 else 0.0,
        "replay_to_reference": {
            "mean_m": mean,
            "p50_m": _percentile(dists, 0.50),
            "p90_m": _percentile(dists, 0.90),
            "p95_m": _percentile(dists, 0.95),
            "max_m": float(dists[-1]) if dists else float("nan"),
        },
        "reference_length_m": float(ref_len),
        "replay_length_m": float(_polyline_length(replay_xy)),
        "reference_points": int(len(reference_xy)),
        "replay_points": int(len(replay_xy)),
    }


def _downsample(xy: Sequence[XY], max_points: int) -> List[XY]:
    xy = list(xy)
    if max_points <= 0 or len(xy) <= max_points:
        return xy
    step = max(1, int(math.ceil(len(xy) / float(max_points))))
    out = xy[::step]
    if out[-1] != xy[-1]:
        out.append(xy[-1])
    return out


def _svg_polyline(points: Sequence[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _xml_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

def _write_svg(
    out_path: Path,
    ref_xy: Sequence[XY],
    a_xy: Sequence[XY],
    b_xy: Sequence[XY],
    title: str,
    label_a: str,
    label_b: str,
    circles: Sequence[Circle],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 800
    margin = 60

    all_xy = list(ref_xy) + list(a_xy) + list(b_xy) + [(cx, cy) for (cx, cy, _r) in circles]
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
    a_px = [to_px(p) for p in a_xy]
    b_px = [to_px(p) for p in b_xy]
    circles_px = [(to_px((cx, cy)), float(r)) for (cx, cy, r) in circles]

    box_x0 = margin
    box_y0 = margin
    box_w = plot_w
    box_h = plot_h

    circles_svg = ""
    if circles_px:
        parts = []
        for (px, py), r_m in circles_px:
            r_px = max(1.0, float(r_m) * float(scale))
            parts.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{r_px:.2f}" fill="none" stroke="#2ca02c" stroke-width="1" opacity="0.9"/>'
            )
        circles_svg = "\n  ".join(parts)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  <text x="{margin}" y="{margin-20}" font-family="monospace" font-size="16" fill="#111">{_xml_escape(title)}</text>
  <rect x="{box_x0}" y="{box_y0}" width="{box_w}" height="{box_h}" fill="none" stroke="#aaa" stroke-width="1"/>

  {circles_svg}

  <polyline points="{_svg_polyline(ref_px)}" fill="none" stroke="#d62728" stroke-width="2" opacity="0.95"/>
  <polyline points="{_svg_polyline(a_px)}" fill="none" stroke="#1f77b4" stroke-width="2" opacity="0.90"/>
  <polyline points="{_svg_polyline(b_px)}" fill="none" stroke="#2ca02c" stroke-width="2" opacity="0.90"/>

  <rect x="{width-420}" y="{margin-10}" width="360" height="110" fill="white" stroke="#ddd"/>
  <line x1="{width-400}" y1="{margin+12}" x2="{width-360}" y2="{margin+12}" stroke="#d62728" stroke-width="3"/>
  <text x="{width-350}" y="{margin+17}" font-family="monospace" font-size="14" fill="#111">reference</text>
  <line x1="{width-400}" y1="{margin+34}" x2="{width-360}" y2="{margin+34}" stroke="#1f77b4" stroke-width="3"/>
  <text x="{width-350}" y="{margin+39}" font-family="monospace" font-size="14" fill="#111">{_xml_escape(label_a)}</text>
  <line x1="{width-400}" y1="{margin+56}" x2="{width-360}" y2="{margin+56}" stroke="#2ca02c" stroke-width="3"/>
  <text x="{width-350}" y="{margin+61}" font-family="monospace" font-size="14" fill="#111">{_xml_escape(label_b)}</text>
  <circle cx="{width-380}" cy="{margin+82}" r="5" fill="none" stroke="#2ca02c" stroke-width="2"/>
  <text x="{width-350}" y="{margin+87}" font-family="monospace" font-size="14" fill="#111">trees</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_ref = ws_dir / "src" / "pcd_gazebo_world" / "maps" / "runs" / "rosbag_path.json"

    p = argparse.ArgumentParser(description="Compare reference JSON vs two replay CSVs (SVG + JSON report).")
    p.add_argument("--reference", type=str, default=str(default_ref))
    p.add_argument("--replay-a", type=str, required=True)
    p.add_argument("--replay-b", type=str, required=True)
    p.add_argument("--label-a", type=str, default="replay_a")
    p.add_argument("--label-b", type=str, default="replay_b")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--report", type=str, default="")
    p.add_argument("--max-points", type=int, default=3000)
    p.add_argument("--title", type=str, default="")
    p.add_argument("--circles", type=str, default="", help="Optional trees circles JSON (for plotting)")
    p.add_argument("--tree-default-radius", type=float, default=0.15)
    args = p.parse_args(argv)

    ref_path = Path(args.reference).expanduser().resolve()
    a_path = Path(args.replay_a).expanduser().resolve()
    b_path = Path(args.replay_b).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if str(args.report).strip() else None

    ref_xy, _ = _load_reference_xy(ref_path)
    a_xy = _load_replay_xy(a_path)
    b_xy = _load_replay_xy(b_path)

    circles_path = str(args.circles or "").strip()
    circles_frame = None
    circles: List[Circle] = []
    if circles_path:
        cp = Path(circles_path).expanduser().resolve()
        circles_frame, circles = _load_tree_circles(cp, default_radius=float(args.tree_default_radius))

    report = {
        "reference": str(ref_path),
        "replay_a": str(a_path),
        "replay_b": str(b_path),
        "label_a": str(args.label_a),
        "label_b": str(args.label_b),
        "circles": str(Path(circles_path).expanduser().resolve()) if circles_path else "",
        "circles_frame_id": circles_frame or "",
        "circles_count": int(len(circles)),
        "a": _metrics(ref_xy, a_xy),
        "b": _metrics(ref_xy, b_xy),
    }

    title = str(args.title).strip() or f"ref vs {args.label_a} vs {args.label_b}"

    ref_px = _downsample(ref_xy, int(args.max_points))
    a_px = _downsample(a_xy, int(args.max_points))
    b_px = _downsample(b_xy, int(args.max_points))
    _write_svg(out_path, ref_px, a_px, b_px, title=title, label_a=str(args.label_a), label_b=str(args.label_b), circles=circles)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
