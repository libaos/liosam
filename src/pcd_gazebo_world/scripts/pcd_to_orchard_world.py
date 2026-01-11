#!/usr/bin/env python3
"""从 PCD（树干高度切片）提取每棵树中心，并生成 Gazebo 果园 world。

和把整张点云 Poisson 重建成 “terrain mesh” 不同，这个脚本更适合果园场景：
- 用圆柱体表示树干（碰撞体），机器人可以在行间真实避障/导航；
- 树的位置从点云里自动提取（可选用行先验 row_model 限制/分行聚类）。

默认会复用 orchard_row_mapping 里的实现（同一套 row model 坐标定义）。
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


def _unit2(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32).reshape(2)
    n = float(np.linalg.norm(vec))
    if n <= 1.0e-8:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return (vec / n).astype(np.float32)


def _load_row_model(path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    direction_xy = _unit2(np.asarray(data.get("direction_xy", [1.0, 0.0]), dtype=np.float32))
    perp_xy = np.asarray(data.get("perp_xy", [-direction_xy[1], direction_xy[0]]), dtype=np.float32).reshape(2)
    perp_xy = _unit2(perp_xy)

    # Re-orthogonalize to be safe.
    perp_xy = perp_xy - direction_xy * float(np.dot(perp_xy, direction_xy))
    perp_xy = _unit2(perp_xy)

    rows_in = data.get("rows", None)
    if rows_in is None:
        rows_in = data.get("rows_uv", [])

    rows: List[Dict[str, float]] = []
    for row in rows_in:
        try:
            rows.append(
                {
                    "v_center": float(row["v_center"]),
                    "u_min": float(row["u_min"]),
                    "u_max": float(row["u_max"]),
                    "z": float(row.get("z", 0.0)),
                }
            )
        except Exception:
            continue

    if not rows:
        raise RuntimeError(f"row_model has no valid rows: {path}")
    rows.sort(key=lambda r: float(r["v_center"]))
    return direction_xy, perp_xy, rows


def _fmt(x: float) -> str:
    return f"{float(x):.6f}"


def _write_world(
    out_path: Path,
    world_name: str,
    tree_model_uri: str,
    trees_xyr: Sequence[Tuple[float, float, float]],
    ground_mode: str,
    tree_height_m: float,
    tree_radius_scale: float,
    world_tree_mode: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trees_xyr = list(trees_xyr)
    tree_height_m = float(tree_height_m)
    tree_radius_scale = float(tree_radius_scale)
    world_tree_mode = (world_tree_mode or "include").strip().lower()

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
    if world_tree_mode == "cylinder":
        if tree_height_m <= 0.0:
            raise ValueError("--tree-height must be > 0 when --world-tree-mode=cylinder")
        lines.append(f"    <!-- Trees: {len(trees_xyr)} cylinders (height={tree_height_m:.3f}m) -->")
        for i, (x, y, r) in enumerate(trees_xyr):
            r = float(r) * float(tree_radius_scale)
            r = max(r, 1.0e-3)
            lines.extend(
                [
                    f'    <model name="tree_{i:04d}">',
                    "      <static>true</static>",
                    f"      <pose>{_fmt(x)} {_fmt(y)} 0 0 0 0</pose>",
                    '      <link name="link">',
                    '        <collision name="trunk_collision">',
                    f"          <pose>0 0 {_fmt(tree_height_m/2.0)} 0 0 0</pose>",
                    "          <geometry>",
                    "            <cylinder>",
                    f"              <radius>{_fmt(r)}</radius>",
                    f"              <length>{_fmt(tree_height_m)}</length>",
                    "            </cylinder>",
                    "          </geometry>",
                    "        </collision>",
                    '        <visual name="trunk_visual">',
                    f"          <pose>0 0 {_fmt(tree_height_m/2.0)} 0 0 0</pose>",
                    "          <geometry>",
                    "            <cylinder>",
                    f"              <radius>{_fmt(r)}</radius>",
                    f"              <length>{_fmt(tree_height_m)}</length>",
                    "            </cylinder>",
                    "          </geometry>",
                    "          <material>",
                    "            <ambient>0.25 0.18 0.12 1</ambient>",
                    "            <diffuse>0.35 0.25 0.18 1</diffuse>",
                    "          </material>",
                    "        </visual>",
                    "      </link>",
                    "    </model>",
                ]
            )
    else:
        lines.append(f"    <!-- Trees: {len(trees_xyr)} includes of {tree_model_uri} -->")
        for i, (x, y, _r) in enumerate(trees_xyr):
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
    tree_only_pcd = ws_dir / "maps" / "map4_bin_tree_label0.pcd"
    dense_map_pcd = ws_dir / "maps" / "manual_priors" / "map4_liosam_deskew_map_dense.pcd"
    default_pcd = tree_only_pcd if tree_only_pcd.is_file() else dense_map_pcd
    default_row_model = ws_dir / "maps" / "manual_priors" / "map4_manual.json"
    default_out_world = ws_dir / "src" / "pcd_gazebo_world" / "worlds" / "orchard_from_pcd.world"
    default_mode = "cell_clusters" if default_pcd == tree_only_pcd else "row_model_cell_clusters"
    default_out_circles = (
        ws_dir / "maps" / "map4_bin_tree_label0_circles.json"
        if default_pcd == tree_only_pcd
        else ws_dir / "maps" / "manual_priors" / "map4_tree_circles.json"
    )
    default_radius_mode = "quantile" if default_pcd == tree_only_pcd else "constant"
    default_radius_max = 1.2 if default_pcd == tree_only_pcd else 0.35
    default_world_tree_mode = "cylinder" if default_pcd == tree_only_pcd else "include"

    parser = argparse.ArgumentParser(description="从 PCD 提取果树中心并生成 Gazebo world")
    parser.add_argument("--pcd", type=str, default=str(default_pcd), help="输入 PCD（建议用稠密地图）")
    parser.add_argument(
        "--mode",
        choices=["row_model_cell_clusters", "cell_clusters"],
        default=default_mode,
        help="row_model_cell_clusters=按行先验分行聚类；cell_clusters=仅按网格聚类（更贴近原始点云中心）",
    )
    parser.add_argument("--row-model", type=str, default=str(default_row_model), help="row_model json（仅 row_model* 模式需要）")
    parser.add_argument("--out-world", type=str, default=str(default_out_world), help="输出 Gazebo .world 路径")
    parser.add_argument("--out-circles", type=str, default=str(default_out_circles), help="输出树中心 circles json 路径")

    parser.add_argument("--z-min", type=float, default=0.7, help="树干高度切片下界")
    parser.add_argument("--z-max", type=float, default=1.3, help="树干高度切片上界")
    parser.add_argument("--max-points", type=int, default=0, help="可选随机采样点数（0=不采样）")
    parser.add_argument("--sample-seed", type=int, default=0, help="采样随机种子")

    parser.add_argument("--row-bandwidth", type=float, default=1.2, help="分配到行的带宽（越大越宽松）")
    parser.add_argument("--u-padding", type=float, default=5.0, help="每行 u 范围两侧 padding")
    parser.add_argument("--cell-size", type=float, default=0.08, help="网格聚类 cell 大小（m）")
    parser.add_argument("--neighbor-range", type=int, default=1, help="聚类相邻 cell 范围")
    parser.add_argument("--min-points", type=int, default=36, help="每棵树最少点数")
    parser.add_argument("--snap-to-row", type=int, default=1, help="1=树中心投影到先验行上（更像果园行）")

    parser.add_argument("--ransac", type=int, default=0, help="1=对每棵树做圆 RANSAC 精修中心")
    parser.add_argument("--ransac-iters", type=int, default=250)
    parser.add_argument("--ransac-inlier-threshold", type=float, default=0.08)
    parser.add_argument("--ransac-min-inliers", type=int, default=40)
    parser.add_argument("--ransac-min-points", type=int, default=60)

    parser.add_argument("--radius-mode", choices=["constant", "median", "quantile"], default=default_radius_mode, help="每棵树半径估计方式")
    parser.add_argument("--radius-constant", type=float, default=0.15, help="radius_mode=constant 时的半径（m）")
    parser.add_argument("--radius-quantile", type=float, default=0.7, help="radius_mode=quantile 时的分位数（0~1）")
    parser.add_argument("--radius-min", type=float, default=0.08, help="半径下限（m，0=不限制）")
    parser.add_argument("--radius-max", type=float, default=float(default_radius_max), help="半径上限（m，0=不限制）")

    parser.add_argument("--world-name", type=str, default="orchard_world")
    parser.add_argument("--ground", choices=["plane", "terrain"], default="plane", help="地面：plane 或 pcd_terrain mesh")
    parser.add_argument("--tree-model-uri", type=str, default="model://tree_trunk")
    parser.add_argument("--world-tree-mode", choices=["include", "cylinder"], default=default_world_tree_mode, help="include=复用模型；cylinder=直接写圆柱体（支持每棵树不同半径）")
    parser.add_argument("--tree-height", type=float, default=1.0, help="world-tree-mode=cylinder 时的树高度（m）")
    parser.add_argument("--tree-radius-scale", type=float, default=1.0, help="world-tree-mode=cylinder 时对半径整体缩放（m）")

    args = parser.parse_args()

    pcd_path = Path(args.pcd).expanduser().resolve()
    row_model_path = Path(args.row_model).expanduser().resolve()
    out_world = Path(args.out_world).expanduser().resolve()
    out_circles = Path(args.out_circles).expanduser().resolve()

    if not pcd_path.is_file():
        raise SystemExit(f"PCD not found: {pcd_path}")

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
    points = impl._sample_points(points, int(args.max_points), seed=int(args.sample_seed))

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

    mode = str(args.mode).strip().lower()
    if mode == "row_model_cell_clusters":
        if not row_model_path.is_file():
            raise SystemExit(f"row_model not found: {row_model_path}")
        direction_xy, perp_xy, rows = _load_row_model(row_model_path)
        circles, _labels = impl._tree_circles_and_labels_from_row_model_cell_clusters(
            points_xyz=points,
            direction_xy=direction_xy,
            perp_xy=perp_xy,
            rows=rows,
            row_bandwidth=float(args.row_bandwidth),
            u_padding=float(args.u_padding),
            cell_size=float(args.cell_size),
            neighbor_range=int(args.neighbor_range),
            min_points=int(args.min_points),
            max_trees_per_row=0,
            max_trees=0,
            snap_to_row=bool(int(args.snap_to_row)),
            circle_ransac=circle_ransac,
            marker_z=0.0,
            radius_mode=str(args.radius_mode),
            radius_constant=float(args.radius_constant),
            radius_quantile=float(args.radius_quantile),
            radius_min=float(args.radius_min),
            radius_max=float(args.radius_max),
        )
    else:
        circles, _labels = impl._tree_circles_and_labels_from_cell_clusters(
            points_xyz=points,
            cell_size=float(args.cell_size),
            neighbor_range=int(args.neighbor_range),
            min_points=int(args.min_points),
            max_clusters=0,
            circle_ransac=circle_ransac,
            marker_z=0.0,
            radius_mode=str(args.radius_mode),
            radius_constant=float(args.radius_constant),
            radius_quantile=float(args.radius_quantile),
            radius_min=float(args.radius_min),
            radius_max=float(args.radius_max),
        )

    out_circles.parent.mkdir(parents=True, exist_ok=True)
    out_circles.write_text(
        json.dumps(
            {
                "mode": mode,
                "pcd": str(pcd_path),
                "row_model": str(row_model_path) if mode == "row_model_cell_clusters" else "",
                "z_min": float(args.z_min),
                "z_max": float(args.z_max),
                "row_bandwidth": float(args.row_bandwidth),
                "u_padding": float(args.u_padding),
                "cell_size": float(args.cell_size),
                "neighbor_range": int(args.neighbor_range),
                "min_points": int(args.min_points),
                "snap_to_row": bool(int(args.snap_to_row)) if mode == "row_model_cell_clusters" else False,
                "ransac": bool(int(args.ransac)),
                "radius_mode": str(args.radius_mode),
                "radius_constant": float(args.radius_constant),
                "radius_quantile": float(args.radius_quantile),
                "radius_min": float(args.radius_min),
                "radius_max": float(args.radius_max),
                "circles": [asdict(c) for c in circles],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    trees_xyr = [(float(c.x), float(c.y), float(getattr(c, "radius", 0.15))) for c in circles]
    _write_world(
        out_path=out_world,
        world_name=str(args.world_name),
        tree_model_uri=str(args.tree_model_uri),
        trees_xyr=trees_xyr,
        ground_mode=str(args.ground),
        tree_height_m=float(args.tree_height),
        tree_radius_scale=float(args.tree_radius_scale),
        world_tree_mode=str(args.world_tree_mode),
    )

    print(f"[OK] circles: {len(circles)} -> {out_circles}")
    print(f"[OK] world:   {out_world}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
