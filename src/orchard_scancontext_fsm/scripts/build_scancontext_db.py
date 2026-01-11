#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import rosbag
from sensor_msgs import point_cloud2 as pc2

try:
    from orchard_scancontext_fsm.scancontext import ScanContextParams, downsample_xyz, make_scancontext
except ModuleNotFoundError:  # allows running without catkin install
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from orchard_scancontext_fsm.scancontext import ScanContextParams, downsample_xyz, make_scancontext  # type: ignore


def _cloud_to_xyz(msg, max_points: int) -> np.ndarray:
    pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    xyz = np.asarray(pts, dtype=np.float32)
    return downsample_xyz(xyz, max_points=max_points)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build a simple ScanContext database (timestamps + descriptors) from a rosbag.")
    p.add_argument("--bag", required=True, help="Input bag")
    p.add_argument("--cloud-topic", required=True, help="sensor_msgs/PointCloud2 topic")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--sample-every", type=int, default=10, help="Use every Nth cloud (default: 10 ~ 1Hz for 10Hz lidar)")
    p.add_argument("--max-points", type=int, default=60000, help="Downsample cap per cloud")
    p.add_argument("--num-ring", type=int, default=20)
    p.add_argument("--num-sector", type=int, default=60)
    p.add_argument("--max-radius", type=float, default=80.0)
    p.add_argument("--lidar-height", type=float, default=2.0)
    p.add_argument("--search-ratio", type=float, default=0.1)
    args = p.parse_args(argv)

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise RuntimeError(f"Bag not found: {bag_path}")

    params = ScanContextParams(
        num_ring=int(args.num_ring),
        num_sector=int(args.num_sector),
        max_radius=float(args.max_radius),
        lidar_height=float(args.lidar_height),
        search_ratio=float(args.search_ratio),
    )

    stamps: List[float] = []
    descs: List[np.ndarray] = []

    with rosbag.Bag(str(bag_path)) as bag:
        idx = 0
        for _, msg, t in bag.read_messages(topics=[str(args.cloud_topic)]):
            if int(args.sample_every) > 1 and (idx % int(args.sample_every) != 0):
                idx += 1
                continue
            stamp_s = float(t.to_sec())
            xyz = _cloud_to_xyz(msg, max_points=int(args.max_points))
            desc = make_scancontext(xyz, params)
            stamps.append(stamp_s)
            descs.append(desc.astype(np.float16))
            idx += 1

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        stamps=np.asarray(stamps, dtype=np.float64),
        descs=np.stack(descs, axis=0).astype(np.float16),
        params=np.asarray([params.num_ring, params.num_sector, params.max_radius, params.lidar_height, params.search_ratio], dtype=np.float64),
    )
    print(f"[OK] Wrote: {out_path} (N={len(stamps)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
