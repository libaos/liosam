#!/usr/bin/env python3
"""Offline ablation runner for tree circles methods.

This script compares different tree-instance extraction / center refinement
methods under the SAME priors (tree-only map + row model), and exports:
  - per-method circle CSV (id,x,y,z,radius)
  - per-method cluster stats CSV
  - a summary CSV across methods

It does not require ROS running (it reuses the internal implementation of
orchard_tree_circles_node.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
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
    # Dataclasses expect the module to be present in sys.modules.
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


def _export_circles_csv(path: Path, circles: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "z", "radius"])
        for i, c in enumerate(circles):
            writer.writerow([int(i), f"{float(c.x):.6f}", f"{float(c.y):.6f}", f"{float(c.z):.6f}", f"{float(c.radius):.6f}"])


def _export_cluster_stats_csv(path: Path, circles: Sequence[Any], labels: np.ndarray, u: np.ndarray) -> Dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = labels.reshape(-1).astype(np.int32)
    mask = labels >= 0
    if not np.any(mask):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "points", "u_span"])
        return {
            "clusters": 0.0,
            "points_median": 0.0,
            "points_min": 0.0,
            "points_max": 0.0,
            "u_span_median": 0.0,
            "u_span_p90": 0.0,
            "u_span_gt_1p4": 0.0,
        }

    n_clusters = int(labels[mask].max()) + 1
    counts = np.bincount(labels[mask], minlength=n_clusters).astype(np.int32)

    u_spans: List[float] = []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "points", "u_span"])
        for cid in range(n_clusters):
            idxs = np.flatnonzero(labels == cid)
            if idxs.size == 0:
                continue
            uu = u[idxs]
            span = float(uu.max() - uu.min()) if uu.size else 0.0
            u_spans.append(span)
            writer.writerow([int(cid), int(counts[cid]), f"{span:.6f}"])

    u_spans_arr = np.asarray(u_spans, dtype=np.float32) if u_spans else np.zeros((0,), dtype=np.float32)
    return {
        "clusters": float(n_clusters),
        "points_median": float(np.median(counts)) if counts.size else 0.0,
        "points_min": float(counts.min()) if counts.size else 0.0,
        "points_max": float(counts.max()) if counts.size else 0.0,
        "u_span_median": float(np.median(u_spans_arr)) if u_spans_arr.size else 0.0,
        "u_span_p90": float(np.quantile(u_spans_arr, 0.9)) if u_spans_arr.size else 0.0,
        "u_span_gt_1p4": float(np.sum(u_spans_arr > 1.4)) if u_spans_arr.size else 0.0,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_out = ws_dir / "maps" / f"ablation_map4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", type=str, default=str(ws_dir / "maps" / "map4_label0.pcd"))
    parser.add_argument("--row-model", type=str, default=str(ws_dir / "maps" / "row_model_from_map4.json"))
    parser.add_argument("--out-dir", type=str, default=str(default_out))

    parser.add_argument("--z-min", type=float, default=0.9)
    parser.add_argument("--z-max", type=float, default=1.1)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)

    parser.add_argument("--row-bandwidth", type=float, default=0.9)
    parser.add_argument("--row-v-offsets", type=str, default='{"4":0.43}')
    parser.add_argument("--row-u-offsets", type=str, default="")
    parser.add_argument("--row-v-slopes", type=str, default="")
    parser.add_argument("--row-v-yaw-offsets-deg", type=str, default='{"4":2.36}')

    # Peaks params (row_model_peaks).
    parser.add_argument("--u-bin", type=float, default=0.05)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--peak-min-fraction", type=float, default=0.05)
    parser.add_argument("--min-separation", type=float, default=1.1)
    parser.add_argument("--refine-u-half-width", type=float, default=0.45)

    # Cell clusters params (row_model_cell_clusters).
    parser.add_argument("--cluster-cell-size", type=float, default=0.12)
    parser.add_argument("--cluster-neighbor-range", type=int, default=1)
    parser.add_argument("--min-points-per-tree", type=int, default=60)

    # RANSAC params.
    parser.add_argument("--ransac-iters", type=int, default=250)
    parser.add_argument("--ransac-inlier-threshold", type=float, default=0.08)
    parser.add_argument("--ransac-min-inliers", type=int, default=40)
    parser.add_argument("--ransac-min-points", type=int, default=60)

    args = parser.parse_args()

    pcd_path = Path(args.pcd).expanduser().resolve()
    row_model_path = Path(args.row_model).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    impl = _load_tree_circles_impl()

    row_v_offsets = _parse_indexed_float_map(args.row_v_offsets)
    row_u_offsets = _parse_indexed_float_map(args.row_u_offsets) if args.row_u_offsets.strip() else {}
    row_v_slopes = _parse_indexed_float_map(args.row_v_slopes) if args.row_v_slopes.strip() else {}
    row_v_yaw_offsets_deg = _parse_indexed_float_map(args.row_v_yaw_offsets_deg)

    points = impl._load_pcd_xyz(pcd_path).astype(np.float32)
    points = impl._filter_points(
        points,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        x_min=-1.0e9,
        x_max=1.0e9,
        y_abs_max=1.0e9,
    )
    points = impl._sample_points(points, int(args.max_points), seed=int(args.sample_seed))

    direction_xy, perp_xy, rows = impl._load_row_model_file(row_model_path)
    rows = impl._apply_row_overrides(
        rows,
        row_v_offsets=row_v_offsets,
        row_u_offsets=row_u_offsets,
        row_v_slopes=row_v_slopes,
        row_v_yaw_offsets_deg=row_v_yaw_offsets_deg,
    )

    xy = points[:, :2].astype(np.float32)
    u = xy.dot(direction_xy.astype(np.float32).reshape(2))

    base_config = {
        "pcd": str(pcd_path),
        "row_model": str(row_model_path),
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "max_points": int(args.max_points),
        "sample_seed": int(args.sample_seed),
        "row_bandwidth": float(args.row_bandwidth),
        "row_v_offsets": row_v_offsets,
        "row_u_offsets": row_u_offsets,
        "row_v_slopes": row_v_slopes,
        "row_v_yaw_offsets_deg": row_v_yaw_offsets_deg,
        "peaks": {
            "u_bin": float(args.u_bin),
            "smooth_window": int(args.smooth_window),
            "peak_min_fraction": float(args.peak_min_fraction),
            "min_separation": float(args.min_separation),
            "refine_u_half_width": float(args.refine_u_half_width),
        },
        "cell_clusters": {
            "cell_size": float(args.cluster_cell_size),
            "neighbor_range": int(args.cluster_neighbor_range),
            "min_points": int(args.min_points_per_tree),
        },
        "ransac": {
            "iters": int(args.ransac_iters),
            "inlier_threshold": float(args.ransac_inlier_threshold),
            "min_inliers": int(args.ransac_min_inliers),
            "min_points": int(args.ransac_min_points),
        },
    }
    _write_json(out_dir / "config.json", base_config)

    methods: List[Tuple[str, str, str]] = [
        ("A_row_peaks_median", "row_model_peaks", "median"),
        ("B_row_peaks_ransac", "row_model_peaks", "circle_ransac"),
        ("C_row_cell_median", "row_model_cell_clusters", "median"),
        ("D_row_cell_ransac", "row_model_cell_clusters", "circle_ransac"),
    ]

    summary_rows: List[Dict[str, Any]] = []
    for name, detection_mode, refine_mode in methods:
        method_dir = out_dir / name
        method_dir.mkdir(parents=True, exist_ok=True)

        ransac_cfg = impl.CircleRansacConfig(
            enabled=refine_mode == "circle_ransac",
            max_iterations=int(args.ransac_iters),
            inlier_threshold=float(args.ransac_inlier_threshold),
            min_inliers=int(args.ransac_min_inliers),
            min_points=int(args.ransac_min_points),
            use_inliers_for_radius=True,
            set_radius=False,
            seed=int(args.sample_seed),
        )

        if detection_mode == "row_model_peaks":
            circles, labels = impl._tree_circles_and_labels_from_row_model(
                points_xyz=points,
                direction_xy=direction_xy,
                perp_xy=perp_xy,
                rows=rows,
                row_bandwidth=float(args.row_bandwidth),
                u_bin_size=float(args.u_bin),
                smooth_window=int(args.smooth_window),
                peak_min_fraction=float(args.peak_min_fraction),
                min_separation=float(args.min_separation),
                u_padding=0.0,
                refine_u_half_width=float(args.refine_u_half_width),
                max_trees_per_row=0,
                max_trees=0,
                snap_to_row=False,
                circle_ransac=ransac_cfg,
                marker_z=0.0,
                radius_mode="constant",
                radius_constant=0.35,
                radius_quantile=0.8,
                radius_min=0.15,
                radius_max=1.5,
            )
        else:
            circles, labels = impl._tree_circles_and_labels_from_row_model_cell_clusters(
                points_xyz=points,
                direction_xy=direction_xy,
                perp_xy=perp_xy,
                rows=rows,
                row_bandwidth=float(args.row_bandwidth),
                u_padding=0.0,
                cell_size=float(args.cluster_cell_size),
                neighbor_range=int(args.cluster_neighbor_range),
                min_points=int(args.min_points_per_tree),
                max_trees_per_row=0,
                max_trees=0,
                snap_to_row=False,
                circle_ransac=ransac_cfg,
                marker_z=0.0,
                radius_mode="constant",
                radius_constant=0.35,
                radius_quantile=0.8,
                radius_min=0.15,
                radius_max=1.5,
            )

        circles_csv = method_dir / "tree_circles.csv"
        _export_circles_csv(circles_csv, circles)

        cluster_stats_csv = method_dir / "cluster_stats.csv"
        stats = _export_cluster_stats_csv(cluster_stats_csv, circles, labels, u=u)

        summary = {
            "method": name,
            "detection_mode": detection_mode,
            "center_refine_mode": refine_mode,
            "points_used": int(points.shape[0]),
            "circles": int(len(circles)),
            **stats,
            "ransac": asdict(ransac_cfg),
        }
        _write_json(method_dir / "summary.json", summary)
        summary_rows.append(summary)

    # Write summary CSV.
    summary_csv = out_dir / "summary.csv"
    keys = [
        "method",
        "detection_mode",
        "center_refine_mode",
        "points_used",
        "circles",
        "points_median",
        "points_min",
        "points_max",
        "u_span_median",
        "u_span_p90",
        "u_span_gt_1p4",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    print(f"[OK] Wrote ablation results to: {out_dir}")
    print(f"     Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

