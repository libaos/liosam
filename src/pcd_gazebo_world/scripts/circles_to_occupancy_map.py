#!/usr/bin/env python3
"""Generate a 2D occupancy grid (PGM + YAML) from tree circles.

This is useful for move_base global planning: a static map that contains the orchard trees
reduces large detours and "trajectory is not feasible" issues that happen when the global
planner assumes an empty world.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_circles(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    circles = data.get("circles", data)
    if not isinstance(circles, list) or not circles:
        raise RuntimeError(f"Invalid circles JSON (missing circles): {path}")
    out: List[Dict[str, Any]] = []
    for c in circles:
        if not isinstance(c, dict):
            continue
        if "x" not in c or "y" not in c:
            continue
        out.append(c)
    if not out:
        raise RuntimeError(f"No valid circles found in: {path}")
    return out


def _world_to_pixel(x: float, y: float, origin_xy: Tuple[float, float], res: float, width: int, height: int) -> Tuple[int, int]:
    ox, oy = origin_xy
    col_f = (float(x) - float(ox)) / float(res)
    row_from_bottom_f = (float(y) - float(oy)) / float(res)
    col = int(math.floor(col_f))
    row_from_bottom = int(math.floor(row_from_bottom_f))
    row = int(height - 1 - row_from_bottom)
    return col, row


def _draw_filled_circle(grid: np.ndarray, cx: int, cy: int, r_px: int, value: int) -> None:
    h, w = grid.shape
    r_px = int(max(0, r_px))
    if r_px <= 0:
        if 0 <= cx < w and 0 <= cy < h:
            grid[cy, cx] = value
        return
    x0 = max(0, cx - r_px)
    x1 = min(w - 1, cx + r_px)
    y0 = max(0, cy - r_px)
    y1 = min(h - 1, cy + r_px)
    rr = float(r_px * r_px)
    for yy in range(y0, y1 + 1):
        dy = float(yy - cy)
        dy2 = dy * dy
        for xx in range(x0, x1 + 1):
            dx = float(xx - cx)
            if dx * dx + dy2 <= rr:
                grid[yy, xx] = value


def _write_pgm(path: Path, grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = grid.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(grid.astype(np.uint8).tobytes())


def _write_yaml(path: Path, image_filename: str, resolution: float, origin_xyz: Tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ox, oy, oz = origin_xyz
    path.write_text(
        "\n".join(
            [
                f"image: {image_filename}",
                f"resolution: {float(resolution):.6f}",
                f"origin: [{float(ox):.6f}, {float(oy):.6f}, {float(oz):.6f}]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_circles = ws_dir / "maps" / "map4_bin_tree_label0_circles_validated_by_bag.json"
    default_out_dir = ws_dir / "src" / "pcd_gazebo_world" / "maps" / "orchard_from_pcd_validated_by_bag"

    parser = argparse.ArgumentParser(description="Generate PGM/YAML occupancy map from circles json")
    parser.add_argument("--circles", type=str, default=str(default_circles), help="circles json（树中心+半径）")
    parser.add_argument("--out-dir", type=str, default=str(default_out_dir), help="输出目录（生成 map.pgm + map.yaml）")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--origin-x", type=float, default=-5.0)
    parser.add_argument("--origin-y", type=float, default=-25.0)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--inflate", type=float, default=0.05, help="额外膨胀半径（m）")
    args = parser.parse_args()

    circles_path = Path(args.circles).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not circles_path.is_file():
        raise SystemExit(f"circles not found: {circles_path}")

    circles = _load_circles(circles_path)

    res = float(args.resolution)
    origin_xy = (float(args.origin_x), float(args.origin_y))
    width = int(args.width)
    height = int(args.height)
    inflate = float(args.inflate)

    grid = np.full((height, width), 254, dtype=np.uint8)

    for c in circles:
        x = float(c["x"])
        y = float(c["y"])
        r = float(c.get("radius", 0.15))
        col, row = _world_to_pixel(x, y, origin_xy, res, width, height)
        r_px = int(math.ceil(max(0.0, r + inflate) / res))
        _draw_filled_circle(grid, col, row, r_px, value=0)

    out_pgm = out_dir / "map.pgm"
    out_yaml = out_dir / "map.yaml"
    _write_pgm(out_pgm, grid)
    _write_yaml(out_yaml, image_filename="map.pgm", resolution=res, origin_xyz=(origin_xy[0], origin_xy[1], 0.0))
    print(out_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

