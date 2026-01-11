#!/usr/bin/env python3
"""Extract user-drawn row lines from a BEV SVG into map coordinates.

Workflow (recommended):
1) Use `render_bev_svg.py` to render a BEV SVG from the tree-only map.
2) Make a copy and draw 2 (or more) straight lines on top of the points
   (do NOT draw using the algorithm output).
3) Run this script to convert drawn lines into a JSON prior that can be used
   for evaluation (per-frame fitting vs. manual prior).

Supported drawn primitives:
- <line x1 y1 x2 y2>
- <path d="M x y L x y"> (Inkscape often uses <path> for lines)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SegmentPx:
    x0: float
    y0: float
    x1: float
    y1: float
    stroke: Optional[str]

    def length_px(self) -> float:
        return float(math.hypot(self.x1 - self.x0, self.y1 - self.y0))


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_style(style: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in (style or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _get_stroke(elem: ET.Element) -> Optional[str]:
    stroke = elem.get("stroke")
    if stroke:
        return stroke.strip()
    style = elem.get("style")
    if style:
        return _parse_style(style).get("stroke")
    return None


def _iter_line_segments(svg_root: ET.Element) -> Iterable[SegmentPx]:
    for elem in svg_root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "line":
            try:
                x0 = float(elem.get("x1", "nan"))
                y0 = float(elem.get("y1", "nan"))
                x1 = float(elem.get("x2", "nan"))
                y1 = float(elem.get("y2", "nan"))
            except Exception:
                continue
            if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
                continue
            yield SegmentPx(x0=x0, y0=y0, x1=x1, y1=y1, stroke=_get_stroke(elem))
            continue

        if tag != "path":
            continue
        d = elem.get("d", "").strip()
        if not d:
            continue
        seg = _parse_simple_path_line(d)
        if seg is None:
            continue
        (x0, y0), (x1, y1) = seg
        yield SegmentPx(x0=x0, y0=y0, x1=x1, y1=y1, stroke=_get_stroke(elem))


_NUM_RE = re.compile(r"[-+]?(?:\\d+\\.\\d+|\\d+\\.|\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?")


def _parse_simple_path_line(d: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Parse a straight line path with minimal commands.

    Supports:
    - M x y L x y
    - m x y l dx dy
    - M x,y L x,y (comma-separated)
    """
    d = d.strip()
    if not d:
        return None
    # Only accept simple 2-point paths.
    # Extract command letters and numbers.
    cmds = re.findall(r"[MmLlHhVvZz]", d)
    nums = [float(x) for x in _NUM_RE.findall(d)]
    if not nums:
        return None

    # Very common case in Inkscape: "M x,y L x,y"
    # Sometimes: "m x,y l dx,dy"
    # We only handle M/m followed by L/l.
    # NOTE: we ignore other commands by refusing.
    d_lower = re.sub(r"\\s+", " ", d).strip().lower()
    if not (d_lower.startswith("m") or d_lower.startswith("m ")):
        return None
    if "l" not in d_lower:
        return None
    if any(c in d_lower for c in ("h", "v", "c", "q", "s", "a")):
        return None

    # Try to read 4 numbers: x0,y0,x1,y1 (absolute)
    if len(nums) >= 4 and (" m" in (" " + d_lower) or d_lower.startswith("m")):
        # Determine if relative (lowercase m/l) or absolute (uppercase).
        is_relative = "m" in cmds and "M" not in cmds
        has_lower_l = "l" in cmds and "L" not in cmds
        if is_relative and has_lower_l and len(nums) >= 4:
            x0, y0, dx, dy = nums[0], nums[1], nums[2], nums[3]
            return (x0, y0), (x0 + dx, y0 + dy)
        if len(nums) >= 4:
            x0, y0, x1, y1 = nums[0], nums[1], nums[2], nums[3]
            return (x0, y0), (x1, y1)
    return None


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


def _pixel_to_map(
    px: float,
    py: float,
    *,
    bounds: Sequence[float],
    width: int,
    height: int,
    margin: int,
) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    dx = max(1.0e-6, xmax - xmin)
    dy = max(1.0e-6, ymax - ymin)
    scale = min((float(width) - 2.0 * float(margin)) / dx, (float(height) - 2.0 * float(margin)) / dy)
    x = (float(px) - float(margin)) / scale + xmin
    y = ymax - (float(py) - float(margin)) / scale
    return float(x), float(y)


def _load_row_model(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    direction = np.array(data["direction_xy"], dtype=np.float64).reshape(2)
    perp = np.array(data["perp_xy"], dtype=np.float64).reshape(2)
    return direction, perp


def _segment_to_row_uv(seg_map: Dict[str, Any], direction_xy: np.ndarray, perp_xy: np.ndarray) -> Dict[str, float]:
    p0 = np.array(seg_map["p0"], dtype=np.float64).reshape(2)
    p1 = np.array(seg_map["p1"], dtype=np.float64).reshape(2)
    u0 = float(p0.dot(direction_xy))
    u1 = float(p1.dot(direction_xy))
    v0 = float(p0.dot(perp_xy))
    v1 = float(p1.dot(perp_xy))
    v_center = 0.5 * (v0 + v1)
    u_min = float(min(u0, u1))
    u_max = float(max(u0, u1))
    return {"v_center": float(v_center), "u_min": float(u_min), "u_max": float(u_max), "z": 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", required=True, type=str, help="SVG with user-drawn lines.")
    parser.add_argument("--bev-meta", required=True, type=str, help="bev_meta.json from the base BEV.")
    parser.add_argument("--out", required=True, type=str, help="Output JSON path.")
    parser.add_argument("--max-lines", type=int, default=2, help="Keep the longest N segments (default: 2).")
    parser.add_argument("--min-length-px", type=float, default=60.0, help="Ignore very short segments.")
    parser.add_argument(
        "--stroke",
        type=str,
        default="",
        help="Optional stroke filter (e.g. '#ff0000'); empty means accept any stroke.",
    )
    parser.add_argument(
        "--row-model",
        type=str,
        default="",
        help="Optional row_model JSON; when set, also export rows in (u,v) based on that direction/perp.",
    )

    args = parser.parse_args()

    svg_path = Path(args.svg).expanduser().resolve()
    meta_path = Path(args.bev_meta).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not svg_path.is_file():
        raise FileNotFoundError(svg_path)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    meta = _load_bev_meta(meta_path)
    bounds = meta["bounds"]
    width = int(meta["width"])
    height = int(meta["height"])
    margin = int(meta["margin"])

    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    stroke_filter = str(args.stroke).strip().lower()
    segments = []
    for seg in _iter_line_segments(root):
        if seg.length_px() < float(args.min_length_px):
            continue
        if stroke_filter:
            if (seg.stroke or "").strip().lower() != stroke_filter:
                continue
        segments.append(seg)

    if not segments:
        raise RuntimeError(
            "No usable segments found. Draw straight lines (line tool) and save, "
            "or set --min-length-px smaller."
        )

    segments.sort(key=lambda s: s.length_px(), reverse=True)
    keep_n = max(1, int(args.max_lines))
    kept = segments[:keep_n]

    lines_map: List[Dict[str, Any]] = []
    for i, seg in enumerate(kept):
        p0 = _pixel_to_map(seg.x0, seg.y0, bounds=bounds, width=width, height=height, margin=margin)
        p1 = _pixel_to_map(seg.x1, seg.y1, bounds=bounds, width=width, height=height, margin=margin)
        lines_map.append(
            {
                "id": int(i),
                "p0": [float(p0[0]), float(p0[1])],
                "p1": [float(p1[0]), float(p1[1])],
                "stroke": (seg.stroke or None),
                "length_px": float(seg.length_px()),
            }
        )

    out: Dict[str, Any] = {
        "source_svg": str(svg_path),
        "bev_meta": str(meta_path),
        "bounds": [float(b) for b in bounds],
        "width": int(width),
        "height": int(height),
        "margin": int(margin),
        "lines": lines_map,
    }

    if str(args.row_model).strip():
        row_model_path = Path(args.row_model).expanduser().resolve()
        direction_xy, perp_xy = _load_row_model(row_model_path)
        out["row_model"] = str(row_model_path)
        out["direction_xy"] = [float(direction_xy[0]), float(direction_xy[1])]
        out["perp_xy"] = [float(perp_xy[0]), float(perp_xy[1])]
        out["rows_uv"] = [_segment_to_row_uv(l, direction_xy, perp_xy) for l in lines_map]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] Extracted {len(lines_map)} line(s) -> {out_path}")
    for l in lines_map:
        print(f"  id={l['id']} length_px={l['length_px']:.1f} p0={l['p0']} p1={l['p1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

