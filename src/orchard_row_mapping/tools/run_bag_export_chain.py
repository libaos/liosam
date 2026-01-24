#!/usr/bin/env python3
"""一键跑通离线导出链路（rosbag -> PCD -> chunk -> 聚类 -> BEV 圆圈 -> 对比拼图）。

目标：先把“整条链路”跑通，后续再慢慢调参/做消融。

输出结构（全部中文目录，便于直接给论文/汇报用）：

  <out-root>/
    00_导航/                       # 软链接入口（不占空间）
    01_原始点云帧/                 # rosbag -> raw_*.pcd
    02_识别树点帧/                 # rosbag -> tree_*.pcd（RandLA-Net）
    03_原始点云上色/               # rosbag -> colored_*.pcd（全点 + rgb/label）
    07_每5帧合成地图PCD/
      02_原始点云_map_TF对齐_每5帧_不裁剪/
      03_树点_map_TF对齐_每5帧_不裁剪/
      04_原始点云上色_map_TF对齐_每5帧_不裁剪/
      06_原始点云_map_TF对齐_每10帧_不裁剪/
      07_树点_map_TF对齐_每10帧_不裁剪/
      08_原始点云上色_map_TF对齐_每10帧_不裁剪/
    09_聚类_每10帧合成树点map/      # chunk(tree) -> clustered PCD（多算法）
    05_BEV预览/                    # clustered -> circles/json + PNG
    11_对比图/                     # 拼图：差异最大帧

依赖：
  - ROS1 python: rosbag
  - PyTorch + RandLA-Net 权重（用于分割）
  - sklearn + cv2（用于聚类/渲染/拼图）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def _quote_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _run(cmd: List[str]) -> None:
    print(f"[RUN] {_quote_cmd(cmd)}")
    subprocess.run(cmd, check=True)


def _count_rows(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        _header = next(reader, None)
        n = 0
        for row in reader:
            if row:
                n += 1
        return int(n)


def _symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=str, help="rosbag 路径")
    parser.add_argument("--points-topic", default="/liorl/deskew/cloud_deskewed", type=str)
    parser.add_argument("--tf-topic", default="/tf", type=str)
    parser.add_argument("--map-frame", default="map", type=str)
    parser.add_argument("--base-frame", default="base_link_est", type=str)
    parser.add_argument("--missing-tf-policy", choices=["skip", "hold", "first"], default="first")

    parser.add_argument("--out-root", default="", type=str, help="输出根目录（默认 output/点云导出_<时间戳>_链路）")
    parser.add_argument("--stamp", default="", type=str, help="用于聚类输出目录名（默认当前时间戳）")

    parser.add_argument("--every", type=int, default=1, help="每 N 帧处理一帧")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧（0=全部）")
    parser.add_argument("--start-offset", type=float, default=0.0, help="从 bag 开始偏移多少秒")
    parser.add_argument("--duration", type=float, default=0.0, help="只处理多少秒（0=到结束）")

    parser.add_argument("--use-gpu", action="store_true", help="分割/上色推理用 GPU（若可用）")

    parser.add_argument("--skip-export-raw", action="store_true")
    parser.add_argument("--skip-seg-tree", action="store_true")
    parser.add_argument("--skip-seg-colored", action="store_true")
    parser.add_argument("--skip-chunks", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true")
    parser.add_argument("--skip-bev", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")

    parser.add_argument(
        "--enable-kmeans-merge",
        action="store_true",
        help="额外导出：KMeans 不切片 + 合并近邻圈（用于修正“一树两圈”）",
    )
    parser.add_argument("--kmeans-merge-dist", type=float, default=0.6, help="合并阈值（米）")
    args = parser.parse_args()

    bag = Path(args.bag).expanduser().resolve()
    if not bag.is_file():
        raise FileNotFoundError(f"bag not found: {bag}")

    ws_dir = Path(__file__).resolve().parents[3]
    stamp = str(args.stamp).strip() or time.strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if str(args.out_root).strip()
        else (ws_dir / "output" / f"点云导出_{stamp}_链路")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    tools_dir = Path(__file__).resolve().parent
    py = sys.executable

    raw_dir = out_root / "01_原始点云帧"
    tree_dir = out_root / "02_识别树点帧"
    colored_dir = out_root / "03_原始点云上色"

    chunks_root = out_root / "07_每5帧合成地图PCD"
    raw_chunk5 = chunks_root / "02_原始点云_map_TF对齐_每5帧_不裁剪"
    tree_chunk5 = chunks_root / "03_树点_map_TF对齐_每5帧_不裁剪"
    colored_chunk5 = chunks_root / "04_原始点云上色_map_TF对齐_每5帧_不裁剪"
    raw_chunk10 = chunks_root / "06_原始点云_map_TF对齐_每10帧_不裁剪"
    tree_chunk10 = chunks_root / "07_树点_map_TF对齐_每10帧_不裁剪"
    colored_chunk10 = chunks_root / "08_原始点云上色_map_TF对齐_每10帧_不裁剪"

    cluster_root = out_root / "09_聚类_每10帧合成树点map"
    bev_root = out_root / "05_BEV预览"
    compare_root = out_root / "11_对比图"

    # Record pipeline meta
    meta = {
        "generated_at": stamp,
        "bag": str(bag),
        "points_topic": str(args.points_topic),
        "tf_topic": str(args.tf_topic),
        "map_frame": str(args.map_frame),
        "base_frame": str(args.base_frame),
        "missing_tf_policy": str(args.missing_tf_policy),
        "every": int(args.every),
        "max_frames": int(args.max_frames),
        "start_offset": float(args.start_offset),
        "duration": float(args.duration),
        "use_gpu": bool(args.use_gpu),
        "out_root": str(out_root),
    }
    (out_root / "pipeline_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not bool(args.skip_export_raw):
        _run(
            [
                py,
                str(tools_dir / "export_bag_to_pcd_frames.py"),
                "--bag",
                str(bag),
                "--points-topic",
                str(args.points_topic),
                "--out-dir",
                str(raw_dir),
                "--prefix",
                "raw",
                "--skip-nans",
                "--every",
                str(int(args.every)),
                "--max-frames",
                str(int(args.max_frames)),
                "--start-offset",
                str(float(args.start_offset)),
                "--duration",
                str(float(args.duration)),
            ]
        )

    if not bool(args.skip_seg_tree):
        cmd = [
            py,
            str(tools_dir / "segment_bag_to_tree_pcd.py"),
            "--bag",
            str(bag),
            "--points-topic",
            str(args.points_topic),
            "--out-dir",
            str(tree_dir),
            "--every",
            str(int(args.every)),
            "--max-frames",
            str(int(args.max_frames)),
            "--start-offset",
            str(float(args.start_offset)),
            "--duration",
            str(float(args.duration)),
        ]
        if bool(args.use_gpu):
            cmd.append("--use-gpu")
        _run(cmd)

    if not bool(args.skip_seg_colored):
        cmd = [
            py,
            str(tools_dir / "segment_bag_to_colored_pcd.py"),
            "--bag",
            str(bag),
            "--points-topic",
            str(args.points_topic),
            "--out-dir",
            str(colored_dir),
            "--skip-nans",
            "--label-colors",
            "0:56,188,75;1:180,180,180",
            "--every",
            str(int(args.every)),
            "--max-frames",
            str(int(args.max_frames)),
            "--start-offset",
            str(float(args.start_offset)),
            "--duration",
            str(float(args.duration)),
        ]
        if bool(args.use_gpu):
            cmd.append("--use-gpu")
        _run(cmd)

    if not bool(args.skip_chunks):
        chunks_root.mkdir(parents=True, exist_ok=True)
        # 5 帧一张
        _run(
            [
                py,
                str(tools_dir / "raw_frames_to_map_pcd_chunks.py"),
                "--in-dir",
                str(raw_dir),
                "--out-dir",
                str(raw_chunk5),
                "--chunk-frames",
                "5",
                "--stride",
                "5",
                "--bag",
                str(bag),
                "--tf-topic",
                str(args.tf_topic),
                "--map-frame",
                str(args.map_frame),
                "--base-frame",
                str(args.base_frame),
                "--missing-tf-policy",
                str(args.missing_tf_policy),
            ]
        )
        _run(
            [
                py,
                str(tools_dir / "tree_frames_to_map_pcd_chunks.py"),
                "--in-dir",
                str(tree_dir),
                "--out-dir",
                str(tree_chunk5),
                "--chunk-frames",
                "5",
                "--stride",
                "5",
                "--bag",
                str(bag),
                "--tf-topic",
                str(args.tf_topic),
                "--map-frame",
                str(args.map_frame),
                "--base-frame",
                str(args.base_frame),
                "--missing-tf-policy",
                str(args.missing_tf_policy),
                "--z-min",
                "-1000000000",
                "--z-max",
                "1000000000",
                "--x-min",
                "-1000000000",
                "--x-max",
                "1000000000",
                "--y-abs-max",
                "1000000000",
            ]
        )
        _run(
            [
                py,
                str(tools_dir / "raw_frames_to_map_pcd_chunks.py"),
                "--in-dir",
                str(colored_dir),
                "--out-dir",
                str(colored_chunk5),
                "--chunk-frames",
                "5",
                "--stride",
                "5",
                "--bag",
                str(bag),
                "--tf-topic",
                str(args.tf_topic),
                "--map-frame",
                str(args.map_frame),
                "--base-frame",
                str(args.base_frame),
                "--missing-tf-policy",
                str(args.missing_tf_policy),
            ]
        )

        # 10 帧一张
        _run(
            [
                py,
                str(tools_dir / "raw_frames_to_map_pcd_chunks.py"),
                "--in-dir",
                str(raw_dir),
                "--out-dir",
                str(raw_chunk10),
                "--chunk-frames",
                "10",
                "--stride",
                "10",
                "--bag",
                str(bag),
                "--tf-topic",
                str(args.tf_topic),
                "--map-frame",
                str(args.map_frame),
                "--base-frame",
                str(args.base_frame),
                "--missing-tf-policy",
                str(args.missing_tf_policy),
            ]
        )
        _run(
            [
                py,
                str(tools_dir / "tree_frames_to_map_pcd_chunks.py"),
                "--in-dir",
                str(tree_dir),
                "--out-dir",
                str(tree_chunk10),
                "--chunk-frames",
                "10",
                "--stride",
                "10",
                "--bag",
                str(bag),
                "--tf-topic",
                str(args.tf_topic),
                "--map-frame",
                str(args.map_frame),
                "--base-frame",
                str(args.base_frame),
                "--missing-tf-policy",
                str(args.missing_tf_policy),
                "--z-min",
                "-1000000000",
                "--z-max",
                "1000000000",
                "--x-min",
                "-1000000000",
                "--x-max",
                "1000000000",
                "--y-abs-max",
                "1000000000",
            ]
        )
        _run(
            [
                py,
                str(tools_dir / "raw_frames_to_map_pcd_chunks.py"),
                "--in-dir",
                str(colored_dir),
                "--out-dir",
                str(colored_chunk10),
                "--chunk-frames",
                "10",
                "--stride",
                "10",
                "--bag",
                str(bag),
                "--tf-topic",
                str(args.tf_topic),
                "--map-frame",
                str(args.map_frame),
                "--base-frame",
                str(args.base_frame),
                "--missing-tf-policy",
                str(args.missing_tf_policy),
            ]
        )

    # Clustering (chunk10 tree map)
    tag = "树点map_每10帧_不裁剪"
    suffix = f"_{tag}"
    cell_dir = cluster_root / f"聚类_栅格连通域_{stamp}{suffix}"
    dbscan_dir = cluster_root / f"聚类_DBSCAN_{stamp}{suffix}"
    euclid_dir = cluster_root / f"聚类_欧式聚类_{stamp}{suffix}"
    kmeans_xy_dir = cluster_root / f"聚类_KMeans_{stamp}{suffix}"

    if not bool(args.skip_cluster):
        cluster_root.mkdir(parents=True, exist_ok=True)
        _run(
            [
                py,
                str(tools_dir / "cluster_tree_frames_variants.py"),
                "--in-dir",
                str(tree_chunk10),
                "--out-root",
                str(cluster_root),
                "--tag",
                tag,
                "--stamp",
                stamp,
                "--disable-kmeans",
                "--enable-kmeans-xy",
            ]
        )

    # BEV runs (4 algorithms, 不切片)
    chunk10_count = _count_rows(tree_chunk10 / "frames.csv")
    group_name = f"50_每10帧合成树点map({chunk10_count}张)" if chunk10_count > 0 else "50_每10帧合成树点map"
    group_dir = bev_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    bev_cell = group_dir / "01_圆圈_每10帧合成树点map_栅格连通域_不切片"
    bev_db = group_dir / "02_圆圈_每10帧合成树点map_DBSCAN_不切片"
    bev_eu = group_dir / "03_圆圈_每10帧合成树点map_欧式聚类_不切片"
    bev_km = group_dir / "04_圆圈_每10帧合成树点map_KMeans_不切片"
    bev_km_merge = group_dir / f"05_圆圈_每10帧合成树点map_KMeans_不切片_合并近邻圈{str(float(args.kmeans_merge_dist)).replace('.', 'p')}m"

    if not bool(args.skip_bev):
        bev_root.mkdir(parents=True, exist_ok=True)
        common_args = [
            "--x-min",
            "-20",
            "--x-max",
            "60",
            "--y-abs-max",
            "30",
            "--z-min",
            "0.7",
            "--z-max",
            "1.3",
            "--render-z-min",
            "-1000000000",
            "--render-z-max",
            "1000000000",
            "--circle-z-mode",
            "none",
            "--bounds-mode",
            "center",
            "--window-x",
            "30",
            "--window-y",
            "20",
            "--width",
            "1400",
            "--height",
            "1000",
            "--margin-px",
            "50",
            "--draw-grid",
            "1",
            "--draw-axes",
            "1",
            "--draw-radius",
            "1",
            "--draw-title",
            "0",
            "--radius-mode",
            "quantile",
            "--radius-quantile",
            "0.7",
            "--radius-min",
            "0.08",
            "--radius-max",
            "1.2",
        ]

        _run(
            [
                py,
                str(tools_dir / "clustered_frames_to_circles_bev.py"),
                "--in-dir",
                str(cell_dir),
                "--out-dir",
                str(bev_cell),
                "--algo-name",
                "栅格连通域_每10帧合成树点map_不切片",
                *common_args,
            ]
        )
        _run(
            [
                py,
                str(tools_dir / "clustered_frames_to_circles_bev.py"),
                "--in-dir",
                str(dbscan_dir),
                "--out-dir",
                str(bev_db),
                "--algo-name",
                "DBSCAN_每10帧合成树点map_不切片",
                *common_args,
            ]
        )
        _run(
            [
                py,
                str(tools_dir / "clustered_frames_to_circles_bev.py"),
                "--in-dir",
                str(euclid_dir),
                "--out-dir",
                str(bev_eu),
                "--algo-name",
                "欧式聚类_每10帧合成树点map_不切片",
                *common_args,
            ]
        )
        _run(
            [
                py,
                str(tools_dir / "clustered_frames_to_circles_bev.py"),
                "--in-dir",
                str(kmeans_xy_dir),
                "--out-dir",
                str(bev_km),
                "--algo-name",
                "KMeans_每10帧合成树点map_不切片",
                *common_args,
            ]
        )

        if bool(args.enable_kmeans_merge):
            _run(
                [
                    py,
                    str(tools_dir / "clustered_frames_to_circles_bev.py"),
                    "--in-dir",
                    str(kmeans_xy_dir),
                    "--out-dir",
                    str(bev_km_merge),
                    "--algo-name",
                    f"KMeans_每10帧合成树点map_不切片_合并近邻圈{args.kmeans_merge_dist}m",
                    *common_args,
                    "--merge-close-circles",
                    "--merge-max-center-dist",
                    str(float(args.kmeans_merge_dist)),
                ]
            )

    if not bool(args.skip_compare):
        compare_root.mkdir(parents=True, exist_ok=True)
        compare_group = compare_root / f"01_每10帧合成树点map({chunk10_count}张)" if chunk10_count > 0 else (compare_root / "01_每10帧合成树点map")
        compare_group.mkdir(parents=True, exist_ok=True)

        _run(
            [
                py,
                str(tools_dir / "compare_bev_runs_montage.py"),
                "--run",
                str(bev_cell),
                "--name",
                "栅格连通域",
                "--run",
                str(bev_db),
                "--name",
                "DBSCAN",
                "--run",
                str(bev_eu),
                "--name",
                "欧式聚类",
                "--run",
                str(bev_km),
                "--name",
                "KMeans",
                "--out-dir",
                str(compare_group / "01_不切片_四算法_偏差大帧对比"),
                "--top-k",
                "60",
            ]
        )

        if bool(args.enable_kmeans_merge) and bev_km_merge.is_dir():
            _run(
                [
                    py,
                    str(tools_dir / "compare_bev_runs_montage.py"),
                    "--run",
                    str(bev_km),
                    "--name",
                    "KMeans_不切片",
                    "--run",
                    str(bev_km_merge),
                    "--name",
                    f"KMeans_合并近邻{str(float(args.kmeans_merge_dist)).replace('.', 'p')}m",
                    "--out-dir",
                    str(compare_group / f"02_KMeans_不切片_合并近邻圈{str(float(args.kmeans_merge_dist)).replace('.', 'p')}m_前后对比"),
                    "--top-k",
                    "60",
                ]
            )

    # 00_导航（软链接入口）
    nav_dir = out_root / "00_导航"
    nav_dir.mkdir(parents=True, exist_ok=True)
    _symlink_force(raw_dir, nav_dir / "01_原始点云帧")
    _symlink_force(tree_dir, nav_dir / "02_识别树点帧")
    _symlink_force(colored_dir, nav_dir / "03_原始点云上色")
    _symlink_force(chunks_root, nav_dir / "07_每5帧合成地图PCD")
    _symlink_force(cluster_root, nav_dir / "09_聚类_每10帧合成树点map")
    _symlink_force(bev_root, nav_dir / "05_BEV预览")
    _symlink_force(compare_root, nav_dir / "11_对比图")

    (nav_dir / "README.md").write_text(
        "\n".join(
            [
                "# 00_导航（离线导出链路）",
                "",
                "这里都是软链接（快捷入口），不占空间。",
                "",
                "- `01_原始点云帧/`：rosbag 导出的 raw 每帧 PCD",
                "- `02_识别树点帧/`：RandLA-Net 分割后的 tree 每帧 PCD",
                "- `03_原始点云上色/`：全点上色 PCD（含 rgb/label）",
                "- `07_每5帧合成地图PCD/`：5/10 帧合成 map PCD（TF 对齐）",
                "- `09_聚类_每10帧合成树点map/`：chunk tree 点云聚类（多算法）",
                "- `05_BEV预览/`：灰点+圆圈+网格（每 chunk 一张 PNG）",
                "- `11_对比图/`：差异最大帧拼图对比",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# 点云离线导出链路（总入口）",
                "",
                "推荐从 `00_导航/` 进入。",
                "",
                f"- 输出根目录：`{out_root}`",
                f"- rosbag：`{bag}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] Done. Output: {out_root}")
    print(f"[OK] Entry: {nav_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

