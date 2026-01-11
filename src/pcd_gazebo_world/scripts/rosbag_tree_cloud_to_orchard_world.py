#!/usr/bin/env python3
"""Extract tree centers from a rosbag topic and generate a Gazebo orchard world.

This is the rosbag counterpart of `pcd_to_orchard_world.py`:
- Input: a rosbag containing `/orchard_segmentation/tree_cloud` (tree-only PointCloud2) and `/tf`.
- Output: a `circles.json` with per-tree centers and a Gazebo `.world` that spawns one tree model per center.

It is useful for cross-validating whether the PCD-derived tree centers are aligned with what the rosbag sees.
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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if n <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _transform_to_T(transform: Any) -> np.ndarray:
    t = transform.translation
    q = transform.rotation
    R = _quat_to_R(float(q.x), float(q.y), float(q.z), float(q.w))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [float(t.x), float(t.y), float(t.z)]
    return T


def _read_points_xyz(msg: Any) -> np.ndarray:
    from sensor_msgs import point_cloud2

    pts = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def _accumulate_tree_points_in_map(
    bag_path: Path,
    tree_topic: str,
    tf_topic: str,
    map_frame: str,
    odom_frame: str,
    base_frame: str,
    max_points: int,
    seed: int,
) -> np.ndarray:
    import rosbag

    T_map_odom: Optional[np.ndarray] = None
    T_odom_base: Optional[np.ndarray] = None

    pts_all: List[np.ndarray] = []
    total = 0

    rng = np.random.default_rng(int(seed))

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, _t in bag.read_messages(topics=[tf_topic, tree_topic]):
            if topic == tf_topic:
                for tr in msg.transforms:
                    if tr.header.frame_id == map_frame and tr.child_frame_id == odom_frame:
                        T_map_odom = _transform_to_T(tr.transform)
                    elif tr.header.frame_id == odom_frame and tr.child_frame_id == base_frame:
                        T_odom_base = _transform_to_T(tr.transform)
                continue

            if T_map_odom is None or T_odom_base is None:
                continue

            xyz_base = _read_points_xyz(msg).astype(np.float64)
            if xyz_base.size == 0:
                continue

            T = T_map_odom @ T_odom_base
            R = T[:3, :3]
            t = T[:3, 3].reshape(1, 3)
            xyz_map = (xyz_base @ R.T) + t
            xyz_map = xyz_map.astype(np.float32)

            if int(max_points) > 0:
                remaining = int(max_points) - int(total)
                if remaining <= 0:
                    break
                if int(xyz_map.shape[0]) > remaining:
                    idx = rng.choice(int(xyz_map.shape[0]), size=int(remaining), replace=False)
                    xyz_map = xyz_map[idx]

            pts_all.append(xyz_map)
            total += int(xyz_map.shape[0])

    if not pts_all:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(pts_all, axis=0).astype(np.float32)


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


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_bag = ws_dir / "rosbags" / "tree_cloud_full_fast_20251228_172644.bag"
    if not default_bag.is_file():
        default_bag = ws_dir / "rosbags" / "tree_cloud_full_20251228_163736.bag"

    default_out_circles = ws_dir / "rosbags" / "runs" / "tree_cloud_bag_circles.json"
    default_out_world = ws_dir / "src" / "pcd_gazebo_world" / "worlds" / "orchard_from_tree_cloud_bag.world"

    parser = argparse.ArgumentParser(description="从 rosbag 的 tree_cloud + tf 反算树中心并生成 Gazebo world")
    parser.add_argument("--bag", type=str, default=str(default_bag), help="输入 rosbag")
    parser.add_argument("--tree-topic", type=str, default="/orchard_segmentation/tree_cloud", help="tree cloud 话题")
    parser.add_argument("--tf-topic", type=str, default="/tf", help="tf 话题")
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument("--odom-frame", type=str, default="odom_est")
    parser.add_argument("--base-frame", type=str, default="base_link_est")

    parser.add_argument("--out-circles", type=str, default=str(default_out_circles), help="输出 circles json")
    parser.add_argument("--out-world", type=str, default=str(default_out_world), help="输出 Gazebo .world")

    parser.add_argument("--z-min", type=float, default=0.2, help="树干高度切片下界（rosbag tree_cloud 常较低）")
    parser.add_argument("--z-max", type=float, default=0.6, help="树干高度切片上界")
    parser.add_argument("--cell-size", type=float, default=0.12, help="网格聚类 cell 大小（m）")
    parser.add_argument("--neighbor-range", type=int, default=1, help="聚类相邻 cell 范围")
    parser.add_argument("--min-points", type=int, default=60, help="每棵树最少点数")

    parser.add_argument("--max-points", type=int, default=0, help="可选随机采样总点数（0=不采样）")
    parser.add_argument("--sample-seed", type=int, default=0, help="采样随机种子")

    parser.add_argument("--ransac", type=int, default=0, help="1=对每棵树做圆 RANSAC 精修中心")
    parser.add_argument("--ransac-iters", type=int, default=250)
    parser.add_argument("--ransac-inlier-threshold", type=float, default=0.08)
    parser.add_argument("--ransac-min-inliers", type=int, default=40)
    parser.add_argument("--ransac-min-points", type=int, default=60)

    parser.add_argument("--world-name", type=str, default="orchard_world")
    parser.add_argument("--ground", choices=["plane", "terrain"], default="plane")
    parser.add_argument("--tree-model-uri", type=str, default="model://tree_trunk")

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    out_circles = Path(args.out_circles).expanduser().resolve()
    out_world = Path(args.out_world).expanduser().resolve()

    if not bag_path.is_file():
        raise SystemExit(f"rosbag not found: {bag_path}")

    impl = _load_tree_circles_impl()

    points_xyz = _accumulate_tree_points_in_map(
        bag_path=bag_path,
        tree_topic=str(args.tree_topic),
        tf_topic=str(args.tf_topic),
        map_frame=str(args.map_frame),
        odom_frame=str(args.odom_frame),
        base_frame=str(args.base_frame),
        max_points=int(args.max_points),
        seed=int(args.sample_seed),
    )
    if points_xyz.size == 0:
        raise SystemExit("No points accumulated (check topics/frames).")

    points_xyz = impl._filter_points(
        points_xyz.astype(np.float32),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        x_min=-1.0e9,
        x_max=1.0e9,
        y_abs_max=1.0e9,
    )

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
                "source": "rosbag_tree_cloud",
                "bag": str(bag_path),
                "tree_topic": str(args.tree_topic),
                "tf_topic": str(args.tf_topic),
                "map_frame": str(args.map_frame),
                "odom_frame": str(args.odom_frame),
                "base_frame": str(args.base_frame),
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

    print(f"[OK] circles: {len(circles)} -> {out_circles}")
    print(f"[OK] world:   {out_world}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

