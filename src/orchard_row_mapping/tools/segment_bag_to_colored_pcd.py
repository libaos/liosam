#!/usr/bin/env python3
"""Offline RandLA-Net segmentation for a rosbag; export colored full-frame PCDs.

This script keeps the *raw frame points* (x,y,z[,intensity]) and adds:
- rgb: packed float32 compatible with PCL/CloudCompare
- label: float32 class id
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rosbag
import torch

try:
    from orchard_row_mapping.segmentation import load_model, run_inference
except ImportError:
    pkg_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(pkg_root))
    from orchard_row_mapping.segmentation import load_model, run_inference


_POINTFIELD_DATATYPE_TO_DTYPE: Dict[int, np.dtype] = {
    1: np.dtype("int8"),
    2: np.dtype("uint8"),
    3: np.dtype("int16"),
    4: np.dtype("uint16"),
    5: np.dtype("int32"),
    6: np.dtype("uint32"),
    7: np.dtype("float32"),
    8: np.dtype("float64"),
}


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


def _parse_label_colors(spec: str) -> Dict[int, Tuple[int, int, int]]:
    # Format: "0:56,188,75;1:180,180,180" (RGB)
    text = (spec or "").strip()
    if not text:
        return {0: (56, 188, 75), 1: (180, 180, 180)}
    out: Dict[int, Tuple[int, int, int]] = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid label-colors item (missing ':'): {item!r}")
        k_str, v_str = item.split(":", 1)
        k = int(k_str.strip())
        parts = [p.strip() for p in v_str.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid label-colors RGB triple for label={k}: {v_str!r}")
        r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
        out[k] = (r, g, b)
    return out


def _build_struct_dtype(msg, field_names: Sequence[str]) -> np.dtype:
    fields_by_name = {f.name: f for f in msg.fields}
    missing = [name for name in field_names if name not in fields_by_name]
    if missing:
        raise ValueError(f"PointCloud2 missing fields: {missing}. Available: {sorted(fields_by_name)}")

    endian = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
    names: List[str] = []
    formats: List[np.dtype] = []
    offsets: List[int] = []
    for name in field_names:
        field = fields_by_name[name]
        base = _POINTFIELD_DATATYPE_TO_DTYPE.get(int(field.datatype))
        if base is None:
            raise ValueError(f"Unsupported PointField datatype={field.datatype} for field '{name}'")
        if int(field.count) != 1:
            raise ValueError(f"Unsupported PointField count={field.count} for field '{name}' (only count=1 supported)")
        names.append(name)
        formats.append(base.newbyteorder(endian))
        offsets.append(int(field.offset))

    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(msg.point_step)})


def _cloud_to_xyz_intensity(msg, *, skip_nans: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n_points = int(msg.width) * int(msg.height)
    if n_points <= 0 or not msg.data:
        return np.empty((0, 3), dtype=np.float32), None

    available = {f.name for f in msg.fields}
    fields: List[str] = ["x", "y", "z"]
    has_intensity = "intensity" in available
    if has_intensity:
        fields.append("intensity")

    dtype = _build_struct_dtype(msg, fields)
    raw = np.frombuffer(msg.data, dtype=dtype, count=n_points)

    xyz = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float32, copy=False)
    intensity = raw["intensity"].astype(np.float32, copy=False) if has_intensity else None

    if skip_nans and xyz.size:
        mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[mask]
        if intensity is not None:
            intensity = intensity[mask]

    return xyz, intensity


def _pack_rgb_float(labels: np.ndarray, colors_rgb: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    if labels.size == 0:
        return labels.astype(np.float32)

    r = np.full(labels.shape, 255, dtype=np.uint32)
    g = np.full(labels.shape, 255, dtype=np.uint32)
    b = np.full(labels.shape, 255, dtype=np.uint32)

    for k, (rr, gg, bb) in colors_rgb.items():
        mask = labels == int(k)
        if not np.any(mask):
            continue
        r[mask] = np.uint32(rr) & np.uint32(255)
        g[mask] = np.uint32(gg) & np.uint32(255)
        b[mask] = np.uint32(bb) & np.uint32(255)

    a = np.uint32(255)
    rgb_uint = (a << np.uint32(24)) | (r << np.uint32(16)) | (g << np.uint32(8)) | b
    return rgb_uint.view(np.float32)


def _write_pcd_xyzi_rgb_label(path: Path, xyz: np.ndarray, intensity: Optional[np.ndarray], rgb: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    n = int(xyz.shape[0])
    if intensity is None:
        intensity = np.zeros((n,), dtype=np.float32)

    out = np.empty((n, 6), dtype=np.float32)
    out[:, 0:3] = xyz.astype(np.float32, copy=False)
    out[:, 3] = intensity.astype(np.float32, copy=False)
    out[:, 4] = rgb.astype(np.float32, copy=False)
    out[:, 5] = labels.astype(np.float32, copy=False)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity rgb label\n"
        "SIZE 4 4 4 4 4 4\n"
        "TYPE F F F F F F\n"
        "COUNT 1 1 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")

    with path.open("wb") as handle:
        handle.write(header)
        if out.size:
            handle.write(out.tobytes())


def _read_last_index(csv_path: Path) -> int:
    if not csv_path.is_file():
        return -1
    last = -1
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        _header = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                last = int(str(row[0]).strip())
            except Exception:
                continue
    return last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=str)
    parser.add_argument("--points-topic", default="/liorl/deskew/cloud_deskewed", type=str)
    parser.add_argument("--out-dir", default="", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-nans", action="store_true")
    parser.add_argument("--keep-nans", action="store_true")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)

    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-inter-op-threads", type=int, default=1)
    parser.add_argument("--enable-mkldnn", action="store_true")

    parser.add_argument("--label-colors", type=str, default="", help="e.g. '0:56,188,75;1:180,180,180' (RGB)")
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"bag not found: {bag_path}")

    ws_dir = Path(__file__).resolve().parents[3]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (ws_dir / "output" / f"原始点云上色_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    if args.resume and not out_dir.is_dir():
        raise FileNotFoundError(f"--resume requested but out-dir does not exist: {out_dir}")

    pcd_dir = out_dir / "pcd"
    pcd_dir.mkdir(parents=True, exist_ok=True)

    skip_nans = bool(args.skip_nans) and not bool(args.keep_nans)
    colors_rgb = _parse_label_colors(str(args.label_colors))

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
    resume_index = _read_last_index(csv_path) + 1 if args.resume else 0

    csv_mode = "a" if resume_index > 0 else "w"
    with csv_path.open(csv_mode, newline="") as handle:
        writer = csv.writer(handle)
        if csv_mode == "w":
            writer.writerow(["index", "t_sec", "points", "pcd_path"])

        processed = int(resume_index)
        skipped = 0
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

                xyz, intensity = _cloud_to_xyz_intensity(msg, skip_nans=skip_nans)
                if xyz.size == 0:
                    continue

                if resume_index > 0 and skipped < resume_index:
                    skipped += 1
                    continue

                points6 = np.hstack([xyz, np.zeros((xyz.shape[0], 3), dtype=np.float32)]).astype(np.float32, copy=False)
                labels, _probs = run_inference(model, points6, device, int(args.num_classes))
                rgb = _pack_rgb_float(labels.astype(np.int32, copy=False), colors_rgb)

                out_pcd = pcd_dir / f"colored_{processed:06d}.pcd"
                _write_pcd_xyzi_rgb_label(out_pcd, xyz, intensity, rgb, labels)
                writer.writerow([processed, f"{t_sec:.6f}", int(xyz.shape[0]), str(out_pcd)])

                processed += 1
                if int(args.max_frames) > 0 and processed >= int(args.max_frames):
                    break
                if processed % 50 == 0:
                    print(f"[OK] processed {processed} frames")

    print(f"[OK] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

