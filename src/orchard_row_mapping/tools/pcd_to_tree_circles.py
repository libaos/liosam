#!/usr/bin/env python3
"""Cluster a tree-point PCD into per-tree circles and export a circles.json.

This is the "聚类成圆圈" step for map PCDs like `maps/TreeMap_auto.pcd`.
It uses the same grid clustering implementation as `orchard_tree_circles_node.py`
but runs fully offline (no roscore required).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    # Dataclasses expect the module to be present in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="从 PCD 聚类提取树中心（圆圈）并导出 circles.json")
    parser.add_argument("--pcd", required=True, type=str)
    parser.add_argument("--out", default="", type=str, help="输出 circles.json（默认：output/树中心圆圈_时间戳/<pcd>_circles.json）")

    parser.add_argument("--z-min", type=float, default=0.7)
    parser.add_argument("--z-max", type=float, default=1.3)
    parser.add_argument("--x-min", type=float, default=-1.0e9)
    parser.add_argument("--x-max", type=float, default=1.0e9)
    parser.add_argument("--y-abs-max", type=float, default=1.0e9)

    parser.add_argument("--cell-size", type=float, default=0.12)
    parser.add_argument("--neighbor-range", type=int, default=1)
    parser.add_argument("--min-points", type=int, default=60)
    parser.add_argument("--max-clusters", type=int, default=0, help="0=不限制")

    parser.add_argument("--marker-z", type=float, default=0.0, help="0=用点云中值 z；否则固定 z")
    parser.add_argument("--radius-mode", choices=["constant", "median", "quantile"], default="quantile")
    parser.add_argument("--radius-constant", type=float, default=0.15)
    parser.add_argument("--radius-quantile", type=float, default=0.7)
    parser.add_argument("--radius-min", type=float, default=0.08)
    parser.add_argument("--radius-max", type=float, default=1.2)

    parser.add_argument("--max-points", type=int, default=0, help="可选随机采样点数（0=不采样）")
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args()

    pcd_path = Path(args.pcd).expanduser().resolve()
    if not pcd_path.is_file():
        raise FileNotFoundError(f"PCD not found: {pcd_path}")

    out_path: Path
    if str(args.out).strip():
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_dir = ws_dir / "output" / f"树中心圆圈_{time.strftime('%Y%m%d_%H%M%S')}"
        out_path = out_dir / f"{pcd_path.stem}_circles.json"

    impl = _load_tree_circles_impl()

    points = impl._load_pcd_xyz(pcd_path).astype(np.float32)
    points = impl._filter_points(
        points,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        y_abs_max=float(args.y_abs_max),
    )
    if int(args.max_points) > 0:
        points = impl._sample_points(points, int(args.max_points), seed=int(args.sample_seed))

    circles, labels = impl._tree_circles_and_labels_from_cell_clusters(
        points_xyz=points,
        cell_size=float(args.cell_size),
        neighbor_range=int(args.neighbor_range),
        min_points=int(args.min_points),
        max_clusters=int(args.max_clusters),
        marker_z=float(args.marker_z),
        radius_mode=str(args.radius_mode),
        radius_constant=float(args.radius_constant),
        radius_quantile=float(args.radius_quantile),
        radius_min=float(args.radius_min),
        radius_max=float(args.radius_max),
    )

    out = {
        "mode": "cell_clusters",
        "pcd": str(pcd_path),
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "x_min": float(args.x_min),
        "x_max": float(args.x_max),
        "y_abs_max": float(args.y_abs_max),
        "cell_size": float(args.cell_size),
        "neighbor_range": int(args.neighbor_range),
        "min_points": int(args.min_points),
        "max_clusters": int(args.max_clusters),
        "marker_z": float(args.marker_z),
        "radius_mode": str(args.radius_mode),
        "radius_constant": float(args.radius_constant),
        "radius_quantile": float(args.radius_quantile),
        "radius_min": float(args.radius_min),
        "radius_max": float(args.radius_max),
        "points_used": int(points.shape[0]),
        "clusters_found": int(len(circles)),
        "circles": [asdict(c) for c in circles],
    }
    _write_json(out_path, out)

    labeled = int(np.count_nonzero(np.asarray(labels).reshape(-1) >= 0)) if hasattr(labels, "shape") else 0
    print(f"[OK] circles: {len(circles)} (labeled_points={labeled}/{int(points.shape[0])})")
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

