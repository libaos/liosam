#!/usr/bin/env python3
"""从“果树行先验”(row model json) 生成一个简单的 Gazebo 果园世界。

用途：rosbag 回放只能“重放传感器”，无法真实交互控制；用 Gazebo 可以生成可控机器人 + 传感器，
并把果园抽象成一排排树干（圆柱碰撞体），用于导航/避障/SLAM 仿真。

输入 row model json 需要包含：
- direction_xy: [dx, dy]
- rows 或 rows_uv: [{"v_center","u_min","u_max", ...}, ...]

坐标转换（与本工作区里的可视化/导出工具一致）：
  [x, y] = direction_xy * u + perp_xy * v
其中 perp_xy 若 json 没提供则由 direction_xy 旋转 90° 得到。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RowLine:
    v_center: float
    u_min: float
    u_max: float


def _unit2(vec: Tuple[float, float]) -> Tuple[float, float]:
    norm = math.hypot(vec[0], vec[1])
    if norm < 1.0e-9:
        return (1.0, 0.0)
    return (vec[0] / norm, vec[1] / norm)


def _rotate90(vec: Tuple[float, float]) -> Tuple[float, float]:
    return (-vec[1], vec[0])


def _parse_indices(spec: Optional[str], max_count: int) -> Optional[List[int]]:
    if not spec:
        return None
    indices: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            if a <= b:
                indices.extend(range(a, b + 1))
            else:
                indices.extend(range(a, b - 1, -1))
        else:
            indices.append(int(part))
    indices = sorted(set(i for i in indices if 0 <= i < max_count))
    return indices


def _guess_default_row_model(workspace_root: Path) -> Optional[Path]:
    candidates = [
        workspace_root / "maps/manual_priors/map4_manual.json",
        workspace_root / "src/orchard_row_mapping/config/row_model_pca_major.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_row_model(path: Path) -> Tuple[Tuple[float, float], Tuple[float, float], List[RowLine]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if "direction_xy" not in raw:
        raise ValueError("row model json 缺少 direction_xy")
    direction_xy = _unit2((float(raw["direction_xy"][0]), float(raw["direction_xy"][1])))

    if "perp_xy" in raw:
        perp_xy = _unit2((float(raw["perp_xy"][0]), float(raw["perp_xy"][1])))
    else:
        perp_xy = _rotate90(direction_xy)

    rows_raw = raw.get("rows", None)
    if rows_raw is None:
        rows_raw = raw.get("rows_uv", None)
    if rows_raw is None:
        raise ValueError("row model json 缺少 rows 或 rows_uv")

    rows: List[RowLine] = []
    for r in rows_raw:
        rows.append(
            RowLine(
                v_center=float(r["v_center"]),
                u_min=float(r["u_min"]),
                u_max=float(r["u_max"]),
            )
        )
    rows.sort(key=lambda rr: rr.v_center)
    return direction_xy, perp_xy, rows


def _uv_to_xy(direction_xy: Tuple[float, float], perp_xy: Tuple[float, float], u: float, v: float) -> Tuple[float, float]:
    x = direction_xy[0] * float(u) + perp_xy[0] * float(v)
    y = direction_xy[1] * float(u) + perp_xy[1] * float(v)
    return (x, y)


def _iter_u(u_min: float, u_max: float, spacing_m: float, start_offset_m: float = 0.0) -> Sequence[float]:
    spacing_m = float(spacing_m)
    if spacing_m <= 0:
        raise ValueError("spacing_m 必须 > 0")
    u_min = float(u_min)
    u_max = float(u_max)
    if u_max < u_min:
        u_min, u_max = u_max, u_min
    u = u_min + float(start_offset_m)
    out: List[float] = []
    eps = 1.0e-6
    while u <= u_max + eps:
        out.append(u)
        u += spacing_m
    return out


def _fmt(x: float) -> str:
    # Gazebo/SDF 读 float 很宽松，但世界文件可读性更重要
    return f"{x:.6f}"


def generate_orchard_world(
    row_model_path: Path,
    out_world_path: Path,
    spacing_m: float,
    trunk_radius_m: float,
    trunk_height_m: float,
    canopy_radius_m: float,
    row_indices: Optional[List[int]],
    start_offset_m: float,
    world_name: str,
) -> int:
    direction_xy, perp_xy, rows = _load_row_model(row_model_path)
    if not rows:
        raise ValueError("row model rows 为空")

    selected_rows = rows
    if row_indices is not None:
        selected_rows = [rows[i] for i in row_indices]

    trunk_radius_m = float(trunk_radius_m)
    trunk_height_m = float(trunk_height_m)
    canopy_radius_m = float(canopy_radius_m)
    if trunk_radius_m <= 0:
        raise ValueError("trunk_radius 必须 > 0")
    if trunk_height_m <= 0:
        raise ValueError("trunk_height 必须 > 0")
    if canopy_radius_m < 0:
        raise ValueError("canopy_radius 必须 >= 0")

    models: List[str] = []
    tree_count = 0
    for row_idx, row in enumerate(selected_rows):
        u_values = _iter_u(row.u_min, row.u_max, spacing_m=spacing_m, start_offset_m=start_offset_m)
        for tree_idx, u in enumerate(u_values):
            x, y = _uv_to_xy(direction_xy, perp_xy, u=u, v=row.v_center)
            name = f"tree_r{row_idx:02d}_t{tree_idx:03d}"
            models.append(
                "\n".join(
                    [
                        f'    <model name="{name}">',
                        "      <static>true</static>",
                        f"      <pose>{_fmt(x)} {_fmt(y)} 0 0 0 0</pose>",
                        '      <link name="link">',
                        '        <collision name="trunk_collision">',
                        f"          <pose>0 0 {_fmt(trunk_height_m/2)} 0 0 0</pose>",
                        "          <geometry>",
                        "            <cylinder>",
                        f"              <radius>{_fmt(trunk_radius_m)}</radius>",
                        f"              <length>{_fmt(trunk_height_m)}</length>",
                        "            </cylinder>",
                        "          </geometry>",
                        "        </collision>",
                        '        <visual name="trunk_visual">',
                        f"          <pose>0 0 {_fmt(trunk_height_m/2)} 0 0 0</pose>",
                        "          <geometry>",
                        "            <cylinder>",
                        f"              <radius>{_fmt(trunk_radius_m)}</radius>",
                        f"              <length>{_fmt(trunk_height_m)}</length>",
                        "            </cylinder>",
                        "          </geometry>",
                        "          <material>",
                        "            <ambient>0.25 0.18 0.12 1</ambient>",
                        "            <diffuse>0.35 0.25 0.18 1</diffuse>",
                        "          </material>",
                        "        </visual>",
                    ]
                )
            )
            if canopy_radius_m > 0:
                models.append(
                    "\n".join(
                        [
                            '        <visual name="canopy_visual">',
                            f"          <pose>0 0 {_fmt(trunk_height_m + canopy_radius_m)} 0 0 0</pose>",
                            "          <geometry>",
                            "            <sphere>",
                            f"              <radius>{_fmt(canopy_radius_m)}</radius>",
                            "            </sphere>",
                            "          </geometry>",
                            "          <material>",
                            "            <ambient>0.10 0.25 0.10 1</ambient>",
                            "            <diffuse>0.15 0.35 0.15 1</diffuse>",
                            "          </material>",
                            "        </visual>",
                        ]
                    )
                )
            models.append("      </link>")
            models.append("    </model>")
            tree_count += 1

    out_world_path.parent.mkdir(parents=True, exist_ok=True)
    out_world_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<sdf version="1.6">',
                f'  <world name="{world_name}">',
                "    <include>",
                "      <uri>model://sun</uri>",
                "    </include>",
                "",
                "    <include>",
                "      <uri>model://ground_plane</uri>",
                "    </include>",
                "",
                *models,
                "  </world>",
                "</sdf>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tree_count


def main() -> int:
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parents[3]

    default_row_model = _guess_default_row_model(workspace_root)
    default_out = workspace_root / "src/pcd_gazebo_world/worlds/orchard_rows.world"

    parser = argparse.ArgumentParser(description="根据 row model 生成 Gazebo 果园 world")
    parser.add_argument(
        "--row-model",
        type=Path,
        default=default_row_model,
        help="row model json（默认会尝试 maps/manual_priors/map4_manual.json 或 src/orchard_row_mapping/config/row_model_pca_major.json）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"输出 world 路径 (默认: {default_out})",
    )
    parser.add_argument("--world-name", type=str, default="orchard_world", help="Gazebo world 名称")
    parser.add_argument("--spacing", type=float, default=4.0, help="同一行相邻树的间距 (m)")
    parser.add_argument("--start-offset", type=float, default=0.0, help="沿 u 方向的起始偏移 (m)")
    parser.add_argument("--trunk-radius", type=float, default=0.15, help="树干半径 (m)")
    parser.add_argument("--trunk-height", type=float, default=2.0, help="树干高度 (m)")
    parser.add_argument("--canopy-radius", type=float, default=0.0, help="树冠球半径 (m)，0 表示不生成")
    parser.add_argument("--rows", type=str, default=None, help="只生成指定行，例如 0-4 或 0,2,3")

    args = parser.parse_args()
    if args.row_model is None:
        parser.error("--row-model 未指定，且未找到默认 row model 文件")
    if not args.row_model.exists():
        parser.error(f"row model 不存在: {args.row_model}")

    direction_xy, _perp_xy, rows = _load_row_model(args.row_model)
    row_indices = _parse_indices(args.rows, max_count=len(rows))

    tree_count = generate_orchard_world(
        row_model_path=args.row_model,
        out_world_path=args.out,
        spacing_m=args.spacing,
        trunk_radius_m=args.trunk_radius,
        trunk_height_m=args.trunk_height,
        canopy_radius_m=args.canopy_radius,
        row_indices=row_indices,
        start_offset_m=args.start_offset,
        world_name=args.world_name,
    )
    print(f"[OK] 写入: {args.out}")
    print(f"[OK] direction_xy={direction_xy} rows={len(rows)} trees={tree_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

