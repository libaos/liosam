#!/usr/bin/env python3
"""Overlay tree centers (from circles JSON) with rosbag map/path for sanity-checking.

This is a lightweight "PCD <-> rosbag cross validation" helper:
- Tree centers are extracted offline from a PCD map (see pcd_to_orchard_world.py).
- Rosbag provides a global map topic (PointCloud2) and a trajectory topic (nav_msgs/Path).

The script plots them in the same XY plane (zoomed to the tree ROI).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _roi_from_centers(centers_xy: np.ndarray, margin: float) -> Tuple[float, float, float, float]:
    centers_xy = np.asarray(centers_xy, dtype=np.float32).reshape(-1, 2)
    x_min = float(np.min(centers_xy[:, 0])) - float(margin)
    x_max = float(np.max(centers_xy[:, 0])) + float(margin)
    y_min = float(np.min(centers_xy[:, 1])) - float(margin)
    y_max = float(np.max(centers_xy[:, 1])) + float(margin)
    return x_min, x_max, y_min, y_max


def _roi_union(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return min(ax0, bx0), max(ax1, bx1), min(ay0, by0), max(ay1, by1)


def _load_centers(circles_json: Path) -> np.ndarray:
    data = json.loads(circles_json.read_text(encoding="utf-8"))
    circles = data.get("circles", [])
    centers = [(float(c["x"]), float(c["y"])) for c in circles if "x" in c and "y" in c]
    if not centers:
        raise RuntimeError(f"No circles found in: {circles_json}")
    return np.asarray(centers, dtype=np.float32)


def _load_last_rosbag_msg(bag_path: Path, topic: str):
    import rosbag

    last = None
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, _t in bag.read_messages(topics=[topic]):
            last = msg
    if last is None:
        raise RuntimeError(f"No messages found for topic={topic} in bag={bag_path}")
    return last


def _load_map_points_xy(
    bag_path: Path,
    map_topic: str,
    roi: Tuple[float, float, float, float],
    max_points: int,
) -> np.ndarray:
    from sensor_msgs import point_cloud2

    msg = _load_last_rosbag_msg(bag_path, map_topic)
    x_min, x_max, y_min, y_max = roi

    pts: list[tuple[float, float]] = []
    for x, y, _z in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        if float(x) < x_min or float(x) > x_max or float(y) < y_min or float(y) > y_max:
            continue
        pts.append((float(x), float(y)))
        if int(max_points) > 0 and len(pts) >= int(max_points):
            break

    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _nearest_neighbor_stats(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[float, float, float]:
    src_xy = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst_xy = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    if src_xy.size == 0 or dst_xy.size == 0:
        return float("nan"), float("nan"), float("nan")
    dists = []
    for p in src_xy:
        dists.append(float(np.sqrt(np.sum((dst_xy - p) ** 2, axis=1)).min()))
    d = np.asarray(dists, dtype=np.float64)
    return float(np.mean(d)), float(np.median(d)), float(np.max(d))


def _load_path_xy(
    bag_path: Path,
    path_topic: str,
    roi: Tuple[float, float, float, float],
) -> np.ndarray:
    msg = _load_last_rosbag_msg(bag_path, path_topic)
    poses = getattr(msg, "poses", None) or []
    if not poses:
        return np.empty((0, 2), dtype=np.float32)

    x_min, x_max, y_min, y_max = roi
    pts: list[tuple[float, float]] = []
    for stamped in poses:
        p = stamped.pose.position
        x = float(p.x)
        y = float(p.y)
        if x < x_min or x > x_max or y < y_min or y > y_max:
            continue
        pts.append((x, y))
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_circles = ws_dir / "maps" / "map4_bin_tree_label0_circles.json"
    default_bag = ws_dir / "rosbags" / "2025-10-29-16-05-00.bag"
    default_out = ws_dir / "rosbags" / "runs" / "tree_centers_vs_rosbag.png"

    parser = argparse.ArgumentParser(description="Overlay PCD tree centers with rosbag map/path (XY plane)")
    parser.add_argument("--circles", type=str, default=str(default_circles), help="circles json（含 circles[x,y]）")
    parser.add_argument("--circles2", type=str, default="", help="可选第二份 circles json（例如 rosbag tree_cloud 提取结果）")
    parser.add_argument("--bag", type=str, default=str(default_bag), help="rosbag 路径")
    parser.add_argument("--map-topic", type=str, default="/liorl/mapping/map_global", help="PointCloud2 地图话题")
    parser.add_argument("--path-topic", type=str, default="/liorl/mapping/path", help="Path 轨迹话题")
    parser.add_argument("--margin", type=float, default=5.0, help="绘图时在树中心 bbox 周围加的边距（m）")
    parser.add_argument("--max-map-points", type=int, default=30000, help="绘制地图点最大数量（0=不限制）")
    parser.add_argument("--out", type=str, default=str(default_out), help="输出 PNG 路径")

    args = parser.parse_args()

    circles_path = Path(args.circles).expanduser().resolve()
    circles2_path = Path(args.circles2).expanduser().resolve() if str(args.circles2).strip() else None
    bag_path = Path(args.bag).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not circles_path.is_file():
        raise SystemExit(f"circles json not found: {circles_path}")
    if circles2_path is not None and not circles2_path.is_file():
        raise SystemExit(f"circles2 json not found: {circles2_path}")
    if not bag_path.is_file():
        raise SystemExit(f"rosbag not found: {bag_path}")

    centers_xy = _load_centers(circles_path)
    roi = _roi_from_centers(centers_xy, margin=float(args.margin))
    centers2_xy: Optional[np.ndarray] = None
    if circles2_path is not None:
        centers2_xy = _load_centers(circles2_path)
        roi = _roi_union(roi, _roi_from_centers(centers2_xy, margin=float(args.margin)))

    map_xy: Optional[np.ndarray] = None
    try:
        map_xy = _load_map_points_xy(
            bag_path=bag_path,
            map_topic=str(args.map_topic),
            roi=roi,
            max_points=int(args.max_map_points),
        )
    except Exception as exc:
        print(f"[WARN] Failed to load map points from rosbag ({args.map_topic}): {exc}")
        map_xy = np.empty((0, 2), dtype=np.float32)

    path_xy: Optional[np.ndarray] = None
    try:
        path_xy = _load_path_xy(bag_path=bag_path, path_topic=str(args.path_topic), roi=roi)
    except Exception as exc:
        print(f"[WARN] Failed to load path from rosbag ({args.path_topic}): {exc}")
        path_xy = np.empty((0, 2), dtype=np.float32)

    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    if map_xy is not None and map_xy.size:
        ax.scatter(map_xy[:, 0], map_xy[:, 1], s=2, c="0.7", alpha=0.6, linewidths=0, label="rosbag map")
    ax.scatter(
        centers_xy[:, 0],
        centers_xy[:, 1],
        s=28,
        facecolors="none",
        edgecolors="tab:green",
        linewidths=1.2,
        label=f"centers #1 ({circles_path.name})",
    )
    if centers2_xy is not None and centers2_xy.size:
        ax.scatter(
            centers2_xy[:, 0],
            centers2_xy[:, 1],
            s=26,
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=1.2,
            label=f"centers #2 ({circles2_path.name})",
        )
    if path_xy is not None and path_xy.size:
        ax.plot(path_xy[:, 0], path_xy[:, 1], "-", c="tab:red", linewidth=1.2, label="rosbag path")
        ax.scatter([path_xy[0, 0]], [path_xy[0, 1]], s=25, c="tab:red", marker="o")
        ax.scatter([path_xy[-1, 0]], [path_xy[-1, 1]], s=25, c="tab:red", marker="x")

    x_min, x_max, y_min, y_max = roi
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    ax.set_title(f"Tree centers vs rosbag ({bag_path.name})")

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)

    if centers2_xy is not None and centers2_xy.size:
        a_to_b = _nearest_neighbor_stats(centers_xy, centers2_xy)
        b_to_a = _nearest_neighbor_stats(centers2_xy, centers_xy)
        print(
            "[OK] NN distance #1->#2 (mean/median/max): "
            f"{a_to_b[0]:.3f} / {a_to_b[1]:.3f} / {a_to_b[2]:.3f}"
        )
        print(
            "[OK] NN distance #2->#1 (mean/median/max): "
            f"{b_to_a[0]:.3f} / {b_to_a[1]:.3f} / {b_to_a[2]:.3f}"
        )
    print(f"[OK] wrote: {out_path}")
    print(f"[OK] centers #1: {int(centers_xy.shape[0])}")
    if centers2_xy is not None:
        print(f"[OK] centers #2: {int(centers2_xy.shape[0])}")
    print(f"[OK] map pts (ROI): {int(map_xy.shape[0]) if map_xy is not None else 0}")
    print(f"[OK] path pts (ROI): {int(path_xy.shape[0]) if path_xy is not None else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
