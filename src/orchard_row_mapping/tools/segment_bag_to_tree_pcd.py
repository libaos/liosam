#!/usr/bin/env python3
"""Offline RandLA-Net segmentation for a rosbag; export per-frame tree PCD + PNG."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rosbag
import torch
from sensor_msgs import point_cloud2 as pc2

try:
    from orchard_row_mapping.segmentation import load_model, preprocess_points, run_inference
except ImportError:
    pkg_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(pkg_root))
    from orchard_row_mapping.segmentation import load_model, preprocess_points, run_inference


def _resolve_checkpoint(override: str) -> Optional[Path]:
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.append(Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pth")
    candidates.append(Path("/mysda/w/w/RandLA-Net-pytorch/noslam/checkpoints/best_model.pth"))
    candidates.append(Path("/mysda/w/w/RandLA-Net-pytorch/best_model.pth"))
    for path in candidates:
        if path and path.exists():
            return path
    return None


def _cloud_to_xyz(msg) -> np.ndarray:
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _write_pcd_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.astype(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {pts.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {pts.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(pts.tobytes())


def _filter_fit_bounds(
    points: np.ndarray,
    x_min: float,
    x_max: float,
    y_abs_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    if points.size == 0:
        return points
    mask = (
        (points[:, 0] >= float(x_min))
        & (points[:, 0] <= float(x_max))
        & (np.abs(points[:, 1]) <= float(y_abs_max))
        & (points[:, 2] >= float(z_min))
        & (points[:, 2] <= float(z_max))
    )
    return points[mask]


def _parse_hex_color(value: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    text = (value or "").strip().lstrip("#")
    if len(text) != 6:
        return default
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
        return (b, g, r)
    except Exception:
        return default


def _render_bev_png(
    out_path: Path,
    points_xy: np.ndarray,
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
    point_size: float,
    point_color: str,
    bg_color: str,
) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"cv2 is required for PNG rendering but not available: {exc}")

    xmin, xmax, ymin, ymax = bounds
    bg_bgr = _parse_hex_color(bg_color, (255, 255, 255))
    pt_bgr = _parse_hex_color(point_color, (200, 200, 200))

    img = np.full((int(height), int(width), 3), bg_bgr, dtype=np.uint8)
    if points_xy.size == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        return

    if xmax == xmin or ymax == ymin:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        return

    x = points_xy[:, 0].astype(np.float32)
    y = points_xy[:, 1].astype(np.float32)
    ix = ((x - float(xmin)) / (float(xmax) - float(xmin)) * float(width - 1)).astype(np.int32)
    iy = ((float(ymax) - y) / (float(ymax) - float(ymin)) * float(height - 1)).astype(np.int32)
    ix = np.clip(ix, 0, int(width) - 1)
    iy = np.clip(iy, 0, int(height) - 1)

    radius = max(0, int(round(float(point_size))))
    if radius <= 1:
        img[iy, ix] = pt_bgr
    else:
        for px, py in zip(ix.tolist(), iy.tolist()):
            cv2.circle(img, (int(px), int(py)), radius, pt_bgr, -1, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=str)
    parser.add_argument("--points-topic", default="/points_raw", type=str)
    parser.add_argument("--out-dir", default="", type=str)
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)

    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--sampling", type=str, default="random")
    parser.add_argument("--sampling-seed", type=int, default=-1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--tree-class", type=int, default=0)
    parser.add_argument("--tree-prob-threshold", type=float, default=0.7)

    parser.add_argument("--fit-x-min", type=float, default=0.0)
    parser.add_argument("--fit-x-max", type=float, default=20.0)
    parser.add_argument("--fit-y-abs-max", type=float, default=6.0)
    parser.add_argument("--fit-z-min", type=float, default=0.7)
    parser.add_argument("--fit-z-max", type=float, default=1.3)

    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-inter-op-threads", type=int, default=1)
    parser.add_argument("--enable-mkldnn", action="store_true")

    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--png-width", type=int, default=1600)
    parser.add_argument("--png-height", type=int, default=1200)
    parser.add_argument("--png-point-size", type=float, default=1.2)
    parser.add_argument("--png-point-color", type=str, default="#cfcfcf")
    parser.add_argument("--png-bg", type=str, default="#ffffff")
    parser.add_argument("--png-x-min", type=float, default=float("nan"))
    parser.add_argument("--png-x-max", type=float, default=float("nan"))
    parser.add_argument("--png-y-min", type=float, default=float("nan"))
    parser.add_argument("--png-y-max", type=float, default=float("nan"))

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"bag not found: {bag_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (
        Path(__file__).resolve().parents[3] / "maps" / f"seg_tree_frames_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    pcd_dir = out_dir / "pcd"
    png_dir = out_dir / "png"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    if args.save_png:
        png_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    if args.torch_inter_op_threads > 0:
        torch.set_num_interop_threads(int(args.torch_inter_op_threads))
    torch.backends.mkldnn.enabled = bool(args.enable_mkldnn)

    device = torch.device("cuda:0") if bool(args.use_gpu) and torch.cuda.is_available() else torch.device("cpu")
    checkpoint = _resolve_checkpoint(str(args.model_path))
    if checkpoint is None:
        raise RuntimeError("No valid checkpoint file found for segmentation model.")
    model = load_model(int(args.num_classes), device, checkpoint)

    with rosbag.Bag(str(bag_path)) as bag:
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())

    start_time = bag_start + float(args.start_offset)
    end_time = bag_end if float(args.duration) <= 0.0 else start_time + float(args.duration)

    csv_path = out_dir / "frames.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "t_sec", "points_tree", "pcd_path", "png_path"])

        processed = 0
        msg_idx = 0
        with rosbag.Bag(str(bag_path)) as bag:
            for _topic, msg, t in bag.read_messages(topics=[args.points_topic]):
                t_sec = float(t.to_sec())
                if t_sec < start_time:
                    continue
                if t_sec > end_time:
                    break
                msg_idx += 1
                if int(args.every) > 1 and (msg_idx - 1) % int(args.every) != 0:
                    continue

                points_xyz = _cloud_to_xyz(msg)
                if points_xyz.size == 0:
                    continue

                seed = None if int(args.sampling_seed) < 0 else int(args.sampling_seed) + processed
                points6 = preprocess_points(
                    points_xyz,
                    int(args.num_points),
                    sampling=str(args.sampling),
                    seed=seed,
                )
                labels, probs = run_inference(model, points6, device, int(args.num_classes))
                if int(args.tree_class) < 0 or int(args.tree_class) >= probs.shape[1]:
                    raise RuntimeError(f"tree_class out of range: {args.tree_class} (C={probs.shape[1]})")
                mask = labels == int(args.tree_class)
                if float(args.tree_prob_threshold) > 0.0:
                    mask &= probs[:, int(args.tree_class)] >= float(args.tree_prob_threshold)
                tree_points = points6[mask, :3]
                tree_points = _filter_fit_bounds(
                    tree_points,
                    x_min=float(args.fit_x_min),
                    x_max=float(args.fit_x_max),
                    y_abs_max=float(args.fit_y_abs_max),
                    z_min=float(args.fit_z_min),
                    z_max=float(args.fit_z_max),
                )

                out_pcd = pcd_dir / f"tree_{processed:06d}.pcd"
                _write_pcd_xyz(out_pcd, tree_points)

                out_png = ""
                if args.save_png:
                    xmin = float(args.png_x_min)
                    xmax = float(args.png_x_max)
                    ymin = float(args.png_y_min)
                    ymax = float(args.png_y_max)
                    if not np.isfinite(xmin) or not np.isfinite(xmax):
                        xmin, xmax = float(args.fit_x_min), float(args.fit_x_max)
                    if not np.isfinite(ymin) or not np.isfinite(ymax):
                        ymin, ymax = -float(args.fit_y_abs_max), float(args.fit_y_abs_max)
                    out_png_path = png_dir / f"tree_{processed:06d}.png"
                    _render_bev_png(
                        out_png_path,
                        tree_points[:, :2],
                        (xmin, xmax, ymin, ymax),
                        width=int(args.png_width),
                        height=int(args.png_height),
                        point_size=float(args.png_point_size),
                        point_color=str(args.png_point_color),
                        bg_color=str(args.png_bg),
                    )
                    out_png = str(out_png_path)

                writer.writerow([processed, f"{t_sec:.6f}", int(tree_points.shape[0]), str(out_pcd), out_png])

                processed += 1
                if int(args.max_frames) > 0 and processed >= int(args.max_frames):
                    break
                if processed % 50 == 0:
                    print(f"[OK] processed {processed} frames")

    print(f"[OK] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
