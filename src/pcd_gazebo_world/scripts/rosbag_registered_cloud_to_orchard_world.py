#!/usr/bin/env python3
"""Extract tree centers from a rosbag registered point cloud and generate a Gazebo orchard world.

This script is used for "PCD <-> rosbag cross validation":
- PCD side: `pcd_to_orchard_world.py` extracts tree centers from a saved map PCD.
- Rosbag side: this script samples points from `/liorl/mapping/cloud_registered` (frame_id=map)
  and clusters trunk-height points into per-tree centers.

The output circles/world can be compared with the PCD-derived circles (see `plot_tree_centers_vs_rosbag.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_tree_circles_impl() -> Any:
    impl_path = Path(__file__).resolve().parents[2] / "orchard_row_mapping" / "scripts" / "orchard_tree_circles_node.py"
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


def _fmt(x: float) -> str:
    return f"{float(x):.6f}"


def _write_world(
    out_path: Path,
    world_name: str,
    tree_model_uri: str,
    tree_poses_xy: Sequence[Tuple[float, float]],
    ground_mode: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree_poses_xy = list(tree_poses_xy)

    lines: List[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<sdf version="1.6">')
    lines.append(f'  <world name="{world_name}">')
    lines.append("    <include>")
    lines.append("      <uri>model://sun</uri>")
    lines.append("    </include>")
    lines.append("")

    ground_mode = (ground_mode or "plane").strip().lower()
    if ground_mode == "terrain":
        lines.append("    <include>")
        lines.append("      <uri>model://pcd_terrain</uri>")
        lines.append("      <pose>0 0 0 0 0 0</pose>")
        lines.append("    </include>")
    else:
        lines.append("    <include>")
        lines.append("      <uri>model://ground_plane</uri>")
        lines.append("    </include>")

    lines.append("")
    lines.append(f"    <!-- Trees: {len(tree_poses_xy)} includes of {tree_model_uri} -->")

    for i, (x, y) in enumerate(tree_poses_xy):
        lines.append("    <include>")
        lines.append(f"      <uri>{tree_model_uri}</uri>")
        lines.append(f"      <name>tree_{i:04d}</name>")
        lines.append(f"      <pose>{_fmt(x)} {_fmt(y)} 0 0 0 0</pose>")
        lines.append("    </include>")

    lines.append("  </world>")
    lines.append("</sdf>")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


_DTYPE_TO_NP = {
    7: np.float32,  # sensor_msgs/PointField.FLOAT32
    8: np.float64,  # sensor_msgs/PointField.FLOAT64
}


def _pointcloud2_sample_xyz(msg: Any, rng: np.random.Generator, sample_points: int) -> np.ndarray:
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 0) or 0)
    n = int(width) * int(height)
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)

    point_step = int(getattr(msg, "point_step", 0) or 0)
    if point_step <= 0:
        return np.empty((0, 3), dtype=np.float32)

    field_by_name = {str(f.name): f for f in getattr(msg, "fields", [])}
    required = ("x", "y", "z")
    if any(name not in field_by_name for name in required):
        raise RuntimeError(f"PointCloud2 is missing xyz fields: {sorted(field_by_name.keys())}")

    fields = []
    for name in required:
        f = field_by_name[name]
        dt = _DTYPE_TO_NP.get(int(f.datatype), None)
        if dt is None:
            raise RuntimeError(f"Unsupported PointField datatype for {name}: {int(f.datatype)}")
        fields.append((name, int(f.offset), dt))

    endian = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
    formats = []
    offsets = []
    for _name, off, dt in fields:
        fmt = np.dtype(dt).newbyteorder(endian)
        formats.append(fmt)
        offsets.append(int(off))

    dtype = np.dtype({"names": [f[0] for f in fields], "formats": formats, "offsets": offsets, "itemsize": int(point_step)})
    arr = np.frombuffer(getattr(msg, "data"), dtype=dtype, count=n)

    if int(sample_points) <= 0 or int(sample_points) >= n:
        idx = slice(None)
    else:
        # For speed, allow duplicates (replace=True).
        idx = rng.integers(0, n, size=int(sample_points), endpoint=False)

    x = np.asarray(arr["x"][idx], dtype=np.float64).reshape(-1)
    y = np.asarray(arr["y"][idx], dtype=np.float64).reshape(-1)
    z = np.asarray(arr["z"][idx], dtype=np.float64).reshape(-1)
    xyz = np.stack([x, y, z], axis=1)
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    return xyz.astype(np.float32)


def _load_centers(circles_json: Path) -> np.ndarray:
    data = json.loads(circles_json.read_text(encoding="utf-8"))
    circles = data.get("circles", [])
    centers = [(float(c["x"]), float(c["y"])) for c in circles if "x" in c and "y" in c]
    if not centers:
        raise RuntimeError(f"No circles found in: {circles_json}")
    return np.asarray(centers, dtype=np.float32)


def _roi_from_centers(centers_xy: np.ndarray, margin: float) -> Tuple[float, float, float, float]:
    centers_xy = np.asarray(centers_xy, dtype=np.float32).reshape(-1, 2)
    x_min = float(np.min(centers_xy[:, 0])) - float(margin)
    x_max = float(np.max(centers_xy[:, 0])) + float(margin)
    y_min = float(np.min(centers_xy[:, 1])) - float(margin)
    y_max = float(np.max(centers_xy[:, 1])) + float(margin)
    return x_min, x_max, y_min, y_max


def _accumulate_points_from_registered_cloud(
    bag_path: Path,
    cloud_topic: str,
    sample_per_msg: int,
    every_n: int,
    max_points: int,
    seed: int,
    roi: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    import rosbag

    rng = np.random.default_rng(int(seed))
    pts_all: List[np.ndarray] = []
    total = 0

    msg_idx = 0
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, _t in bag.read_messages(topics=[cloud_topic]):
            msg_idx += 1
            if int(every_n) > 1 and (msg_idx % int(every_n)) != 0:
                continue

            xyz = _pointcloud2_sample_xyz(msg, rng=rng, sample_points=int(sample_per_msg))
            if xyz.size == 0:
                continue

            if roi is not None:
                x_min, x_max, y_min, y_max = roi
                mask = (
                    (xyz[:, 0] >= float(x_min))
                    & (xyz[:, 0] <= float(x_max))
                    & (xyz[:, 1] >= float(y_min))
                    & (xyz[:, 1] <= float(y_max))
                )
                xyz = xyz[mask]
                if xyz.size == 0:
                    continue

            pts_all.append(xyz)
            total += int(xyz.shape[0])
            if int(max_points) > 0 and int(total) >= int(max_points):
                break

    if not pts_all:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.concatenate(pts_all, axis=0).astype(np.float32)
    if int(max_points) > 0 and pts.shape[0] > int(max_points):
        idx = rng.choice(pts.shape[0], int(max_points), replace=False)
        pts = pts[idx]
    return pts


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_bag = ws_dir / "rosbags" / "2025-10-29-16-05-00.bag"
    default_out_circles = ws_dir / "rosbags" / "runs" / "tree_centers_from_cloud_registered.json"
    default_out_world = ws_dir / "src" / "pcd_gazebo_world" / "worlds" / "orchard_from_cloud_registered.world"

    parser = argparse.ArgumentParser(description="从 rosbag 的 cloud_registered (map frame) 抽样提取树中心并生成 Gazebo world")
    parser.add_argument("--bag", type=str, default=str(default_bag), help="输入 rosbag")
    parser.add_argument("--cloud-topic", type=str, default="/liorl/mapping/cloud_registered", help="registered cloud 话题（frame=map）")

    parser.add_argument("--roi-from-circles", type=str, default=str(ws_dir / "maps" / "map4_bin_tree_label0_circles.json"))
    parser.add_argument("--roi-margin", type=float, default=5.0, help="ROI 边距（m）")

    parser.add_argument("--sample-per-msg", type=int, default=500, help="每帧随机抽样点数（0=全取，不建议）")
    parser.add_argument("--every-n", type=int, default=1, help="每 N 帧取 1 帧（加速）")
    parser.add_argument("--max-points", type=int, default=180000, help="总点数上限（0=不限制）")
    parser.add_argument("--sample-seed", type=int, default=0)

    parser.add_argument("--z-min", type=float, default=0.7, help="树干高度切片下界（map frame）")
    parser.add_argument("--z-max", type=float, default=1.3, help="树干高度切片上界（map frame）")

    parser.add_argument("--cell-size", type=float, default=0.12, help="网格聚类 cell 大小（m）")
    parser.add_argument("--neighbor-range", type=int, default=1, help="聚类相邻 cell 范围")
    parser.add_argument("--min-points", type=int, default=30, help="每棵树最少点数（抽样后建议偏小）")

    parser.add_argument("--ransac", type=int, default=0, help="1=对每棵树做圆 RANSAC 精修中心")
    parser.add_argument("--ransac-iters", type=int, default=250)
    parser.add_argument("--ransac-inlier-threshold", type=float, default=0.08)
    parser.add_argument("--ransac-min-inliers", type=int, default=40)
    parser.add_argument("--ransac-min-points", type=int, default=60)

    parser.add_argument("--out-circles", type=str, default=str(default_out_circles), help="输出 circles json")
    parser.add_argument("--out-world", type=str, default=str(default_out_world), help="输出 Gazebo .world")

    parser.add_argument("--world-name", type=str, default="orchard_world")
    parser.add_argument("--ground", choices=["plane", "terrain"], default="plane")
    parser.add_argument("--tree-model-uri", type=str, default="model://tree_trunk")

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    out_circles = Path(args.out_circles).expanduser().resolve()
    out_world = Path(args.out_world).expanduser().resolve()

    if not bag_path.is_file():
        raise SystemExit(f"rosbag not found: {bag_path}")

    roi: Optional[Tuple[float, float, float, float]] = None
    roi_circles = Path(args.roi_from_circles).expanduser().resolve()
    if roi_circles.is_file():
        centers = _load_centers(roi_circles)
        roi = _roi_from_centers(centers, margin=float(args.roi_margin))
    elif str(args.roi_from_circles).strip():
        raise SystemExit(f"roi circles json not found: {roi_circles}")

    impl = _load_tree_circles_impl()

    points_xyz = _accumulate_points_from_registered_cloud(
        bag_path=bag_path,
        cloud_topic=str(args.cloud_topic),
        sample_per_msg=int(args.sample_per_msg),
        every_n=int(args.every_n),
        max_points=int(args.max_points),
        seed=int(args.sample_seed),
        roi=roi,
    )
    if points_xyz.size == 0:
        raise SystemExit("No points sampled from rosbag (check topic/roi).")

    points_xyz = impl._filter_points(
        points_xyz.astype(np.float32),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        x_min=-1.0e9,
        x_max=1.0e9,
        y_abs_max=1.0e9,
    )
    if points_xyz.size == 0:
        raise SystemExit("No points left after z filter; adjust --z-min/--z-max.")

    circle_ransac = impl.CircleRansacConfig(
        enabled=bool(int(args.ransac)),
        max_iterations=int(args.ransac_iters),
        inlier_threshold=float(args.ransac_inlier_threshold),
        min_inliers=int(args.ransac_min_inliers),
        min_points=int(args.ransac_min_points),
        use_inliers_for_radius=True,
        set_radius=False,
        seed=int(args.sample_seed),
    )

    circles, _labels = impl._tree_circles_and_labels_from_cell_clusters(
        points_xyz=points_xyz.astype(np.float32),
        cell_size=float(args.cell_size),
        neighbor_range=int(args.neighbor_range),
        min_points=int(args.min_points),
        max_clusters=0,
        circle_ransac=circle_ransac,
        marker_z=0.0,
        radius_mode="constant",
        radius_constant=0.15,
        radius_quantile=0.7,
        radius_min=0.08,
        radius_max=0.35,
    )

    out_circles.parent.mkdir(parents=True, exist_ok=True)
    out_circles.write_text(
        json.dumps(
            {
                "mode": "rosbag_cloud_registered_cell_clusters",
                "bag": str(bag_path),
                "cloud_topic": str(args.cloud_topic),
                "roi_from_circles": str(roi_circles) if roi_circles.is_file() else "",
                "roi": list(roi) if roi is not None else [],
                "sample_per_msg": int(args.sample_per_msg),
                "every_n": int(args.every_n),
                "max_points": int(args.max_points),
                "z_min": float(args.z_min),
                "z_max": float(args.z_max),
                "cell_size": float(args.cell_size),
                "neighbor_range": int(args.neighbor_range),
                "min_points": int(args.min_points),
                "ransac": bool(int(args.ransac)),
                "circles": [asdict(c) for c in circles],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    poses_xy = [(float(c.x), float(c.y)) for c in circles]
    _write_world(
        out_path=out_world,
        world_name=str(args.world_name),
        tree_model_uri=str(args.tree_model_uri),
        tree_poses_xy=poses_xy,
        ground_mode=str(args.ground),
    )

    print(f"[OK] sampled pts: {int(points_xyz.shape[0])}")
    print(f"[OK] circles:     {len(circles)} -> {out_circles}")
    print(f"[OK] world:       {out_world}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

