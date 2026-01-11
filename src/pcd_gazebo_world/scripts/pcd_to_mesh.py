#!/usr/bin/env python3
"""将 PCD 点云转换为 Gazebo 可用的 Mesh。

默认会输出多种格式（`stl,obj,dae`），便于 Gazebo/SDF 引用。

示例：
  python3 pcd_to_mesh.py input.pcd /path/to/terrain --formats stl,obj
  python3 pcd_to_mesh.py input.pcd /path/to/terrain.stl  # 等价，会自动去掉扩展名当作 base
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import open3d as o3d

DEFAULT_FORMATS = ("stl", "obj", "dae")


def _parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(v.strip().lstrip(".").lower() for v in value.split(",") if v.strip())
    if not formats:
        raise argparse.ArgumentTypeError("--formats 不能为空")
    return formats


def _output_base(output: Path) -> Path:
    return output.with_suffix("") if output.suffix else output


def pcd_to_mesh(
    pcd_file: Union[str, Path],
    output: Union[str, Path],
    voxel_size: float = 0.2,
    depth: int = 8,
    formats: Tuple[str, ...] = DEFAULT_FORMATS,
) -> Union[o3d.geometry.TriangleMesh, None]:
    """将 PCD 点云转换为 Mesh，并按 formats 保存到 output(base)。

    Args:
        pcd_file: 输入 PCD 路径。
        output: 输出 base 路径（可带扩展名；带扩展名时会自动去掉）。
        voxel_size: 下采样体素大小（米）。
        depth: Poisson 重建深度（越大越精细但更慢/更占内存）。
        formats: 输出格式列表，例如 ("stl","obj")。
    """

    pcd_file = Path(pcd_file)
    output = Path(output)
    out_base = _output_base(output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在读取点云: {pcd_file}")
    pcd = o3d.io.read_point_cloud(str(pcd_file))
    print(f"点云包含 {len(pcd.points)} 个点")

    if len(pcd.points) == 0:
        print("错误: 点云为空!")
        return None

    points = np.asarray(pcd.points)
    print("点云范围:")
    print(f"  X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"  Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"  Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    print(f"正在下采样 (voxel_size={voxel_size})...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"下采样后: {len(pcd_down.points)} 个点")

    print("正在移除离群点...")
    pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"移除离群点后: {len(pcd_down.points)} 个点")

    print("正在估计点云法线...")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
    )
    pcd_down.orient_normals_consistent_tangent_plane(k=15)

    print(f"正在进行 Poisson 重建 (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_down, depth=depth
    )
    print(f"生成的 Mesh: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个面")

    # Poisson 会生成“外壳”，先按点云 bbox 裁剪，减少远处噪声面。
    bbox = pcd_down.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    mesh = mesh.crop(bbox)

    densities = np.asarray(densities)
    if densities.size == len(mesh.vertices):
        density_threshold = np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(densities < density_threshold)

    print(f"清理后: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个面")

    if len(mesh.triangles) > 200000:
        print("正在简化 Mesh...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=200000)
        print(f"简化后: {len(mesh.triangles)} 个面")

    print("正在计算 Mesh 法线...")
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    print("正在保存...")
    written: List[Path] = []
    for fmt in formats:
        out_path = out_base.with_suffix("." + fmt)
        try:
            ok = o3d.io.write_triangle_mesh(str(out_path), mesh)
        except Exception as e:
            print(f"{fmt.upper()} 保存失败: {e}")
            continue
        if not ok:
            print(f"{fmt.upper()} 保存失败: open3d 返回 false")
            continue
        print(f"✓ {fmt.upper()} 已保存到: {out_path}")
        written.append(out_path)

    if written:
        print("\n生成的文件:")
        for path in written:
            size = path.stat().st_size
            print(f"  {path}: {size/1024/1024:.2f} MB")
    else:
        print("未生成任何文件（请检查输出路径/格式是否受支持）。")

    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCD 转 Mesh 工具")
    parser.add_argument("input", help="输入 PCD 文件路径")
    parser.add_argument("output", help="输出 base 路径（可带扩展名）")
    parser.add_argument("--voxel", type=float, default=0.2, help="体素大小 (默认: 0.2)")
    parser.add_argument("--depth", type=int, default=8, help="Poisson 重建深度 (默认: 8)")
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=DEFAULT_FORMATS,
        help="输出格式，用逗号分隔 (默认: stl,obj,dae)",
    )

    args = parser.parse_args()
    pcd_to_mesh(args.input, args.output, args.voxel, args.depth, args.formats)
