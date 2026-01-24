#!/usr/bin/env python3
"""Export raw PointCloud2 frames from a rosbag into per-frame PCD files."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rosbag


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


def _infer_default_fields(msg) -> List[str]:
    names = {f.name for f in msg.fields}
    fields = ["x", "y", "z"]
    if "intensity" in names:
        fields.append("intensity")
    return fields


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


def _cloud_to_numpy(msg, field_names: Sequence[str], *, skip_nans: bool) -> np.ndarray:
    n_points = int(msg.width) * int(msg.height)
    if n_points <= 0 or not msg.data:
        return np.empty((0, len(field_names)), dtype=np.float32)

    dtype = _build_struct_dtype(msg, field_names)
    raw = np.frombuffer(msg.data, dtype=dtype, count=n_points)
    cols = [raw[name].astype(np.float32, copy=False) for name in field_names]
    points = np.column_stack(cols)

    if skip_nans and points.size:
        xyz = points[:, :3]
        mask = np.isfinite(xyz).all(axis=1)
        points = points[mask]
    return points


def _write_pcd(path: Path, field_names: Sequence[str], points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.astype(np.float32, copy=False)

    sizes = " ".join(["4"] * len(field_names))
    types = " ".join(["F"] * len(field_names))
    counts = " ".join(["1"] * len(field_names))
    fields = " ".join(field_names)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {pts.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {pts.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")

    with path.open("wb") as handle:
        handle.write(header)
        if pts.size:
            handle.write(pts.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=str)
    parser.add_argument("--points-topic", default="/liorl/deskew/cloud_deskewed", type=str)
    parser.add_argument("--out-dir", default="", type=str, help="Output directory (can be Chinese).")
    parser.add_argument("--fields", default="", type=str, help="Comma-separated fields to export (default: x,y,z[,intensity]).")
    parser.add_argument("--prefix", default="raw", type=str, help="PCD filename prefix.")
    parser.add_argument("--skip-nans", action="store_true", help="Drop points with NaN x/y/z.")
    parser.add_argument("--keep-nans", action="store_true", help="Keep NaN points (overrides --skip-nans).")
    parser.add_argument("--every", type=int, default=1, help="Export every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"bag not found: {bag_path}")

    ws_dir = Path(__file__).resolve().parents[3]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (ws_dir / "output" / f"原始点云帧_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    pcd_dir = out_dir / "pcd"
    pcd_dir.mkdir(parents=True, exist_ok=True)

    skip_nans = bool(args.skip_nans) and not bool(args.keep_nans)

    with rosbag.Bag(str(bag_path)) as bag:
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())

        start_time = bag_start + float(args.start_offset)
        end_time = bag_end if float(args.duration) <= 0.0 else start_time + float(args.duration)

        csv_path = out_dir / "frames.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "t_sec", "points", "pcd_path"])

            requested_fields: Optional[List[str]] = None
            processed = 0
            msg_idx = 0
            for _topic, msg, t in bag.read_messages(topics=[str(args.points_topic)]):
                t_sec = float(t.to_sec())
                if t_sec < start_time:
                    continue
                if t_sec > end_time:
                    break
                msg_idx += 1
                if int(args.every) > 1 and (msg_idx - 1) % int(args.every) != 0:
                    continue

                if requested_fields is None:
                    fields_arg = str(args.fields).strip()
                    if fields_arg:
                        requested_fields = [s.strip() for s in fields_arg.split(",") if s.strip()]
                    else:
                        requested_fields = _infer_default_fields(msg)

                points = _cloud_to_numpy(msg, requested_fields, skip_nans=skip_nans)
                out_pcd = pcd_dir / f"{str(args.prefix)}_{processed:06d}.pcd"
                _write_pcd(out_pcd, requested_fields, points)
                writer.writerow([processed, f"{t_sec:.6f}", int(points.shape[0]), str(out_pcd)])

                processed += 1
                if int(args.max_frames) > 0 and processed >= int(args.max_frames):
                    break
                if processed % 50 == 0:
                    print(f"[OK] exported {processed} frames")

    print(f"[OK] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

