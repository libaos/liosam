#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rosbag
from sensor_msgs import point_cloud2 as pc2

from orchard_nn_scancontext.scan_context import ScanContext


def _downsample_xyz(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points_xyz.shape[0] <= max_points:
        return points_xyz
    step = int(np.ceil(float(points_xyz.shape[0]) / float(max_points)))
    return points_xyz[::step]


def _cloud_to_xyz(msg) -> np.ndarray:
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _segment_bounds(total_msgs: int, num_classes: int) -> Tuple[int, List[Tuple[int, int]]]:
    per = int(total_msgs) // int(num_classes)
    if per <= 0:
        raise ValueError(f"invalid per={per} from total_msgs={total_msgs} num_classes={num_classes}")
    bounds: List[Tuple[int, int]] = []
    for seg in range(int(num_classes)):
        start = int(seg) * int(per)
        end = int(total_msgs) - 1 if seg == int(num_classes) - 1 else (int(seg) + 1) * int(per) - 1
        bounds.append((start, end))
    return per, bounds


def _write_md(
    path: Path,
    bag_path: Path,
    cloud_topic: str,
    bounds: List[Tuple[int, int]],
    times: List[Tuple[float, float]],
) -> None:
    if not bounds:
        return
    t0 = float(times[0][0]) if times[0][0] == times[0][0] else float("nan")
    lines = [
        "# route_id (0..K-1) ↔ rosbag 段号映射（按帧数均分）",
        "",
        f"- rosbag: `{bag_path}`",
        f"- topic: `{cloud_topic}`",
        "",
        "| route_id | idx_start | idx_end | t_start_sec | t_end_sec | t_rel_start | t_rel_end |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rid, ((i0, i1), (ts0, ts1)) in enumerate(zip(bounds, times)):
        rel0 = float(ts0) - t0 if t0 == t0 and ts0 == ts0 else float("nan")
        rel1 = float(ts1) - t0 if t0 == t0 and ts1 == ts1 else float("nan")
        lines.append(f"| {rid} | {i0} | {i1} | {ts0:.6f} | {ts1:.6f} | {rel0:.3f} | {rel1:.3f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build a ScanContext route database (per-segment prototypes) from a rosbag.")
    p.add_argument("--bag", required=True, help="Input .bag path")
    p.add_argument("--cloud-topic", required=True, help="sensor_msgs/PointCloud2 topic")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--num-classes", type=int, default=20)
    p.add_argument("--process-hz", type=float, default=2.0)
    p.add_argument("--max-points", type=int, default=60000)
    p.add_argument("--sample-every", type=int, default=1, help="Use every Nth message (before process-hz gating)")
    p.add_argument("--start-offset", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0)

    p.add_argument("--num-ring", type=int, default=20)
    p.add_argument("--num-sector", type=int, default=60)
    p.add_argument("--min-range", type=float, default=0.1)
    p.add_argument("--max-range", type=float, default=80.0)
    p.add_argument("--height-lower", type=float, default=-1.0)
    p.add_argument("--height-upper", type=float, default=9.0)

    p.add_argument("--write-md", action="store_true", help="Also write a markdown mapping table next to the npz")
    args = p.parse_args(argv)

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"bag not found: {bag_path}")

    cloud_topic = str(args.cloud_topic).strip()
    if not cloud_topic:
        raise ValueError("--cloud-topic is required")

    with rosbag.Bag(str(bag_path)) as bag:
        info = bag.get_type_and_topic_info()
        if cloud_topic not in info.topics:
            raise RuntimeError(f"topic not found in bag: {cloud_topic}")
        total_msgs = int(getattr(info.topics[cloud_topic], "message_count", 0))
        if total_msgs <= 0:
            raise RuntimeError(f"invalid message_count for topic {cloud_topic}: {total_msgs}")

        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())

        start_time = bag_start + float(args.start_offset)
        end_time = bag_end if float(args.duration) <= 0.0 else start_time + float(args.duration)

        per, bounds = _segment_bounds(total_msgs, int(args.num_classes))

        times: List[Tuple[float, float]] = [(float("nan"), float("nan")) for _ in range(int(args.num_classes))]
        sc = ScanContext(
            num_sectors=int(args.num_sector),
            num_rings=int(args.num_ring),
            min_range=float(args.min_range),
            max_range=float(args.max_range),
            height_lower_bound=float(args.height_lower),
            height_upper_bound=float(args.height_upper),
        )

        sums = np.zeros((int(args.num_classes), int(args.num_ring), int(args.num_sector)), dtype=np.float64)
        counts = np.zeros((int(args.num_classes),), dtype=np.int32)

        min_dt = 0.0 if float(args.process_hz) <= 0.0 else 1.0 / float(args.process_hz)
        last_proc_t: Optional[float] = None

        idx = -1
        msg_idx = 0
        for _topic, msg, t in bag.read_messages(topics=[cloud_topic]):
            idx += 1
            t_sec = float(t.to_sec())
            if t_sec < start_time:
                continue
            if t_sec > end_time:
                break

            seg = min(int(args.num_classes) - 1, int(idx // per))
            ts0, ts1 = times[seg]
            if ts0 != ts0:
                ts0 = t_sec
            ts1 = t_sec
            times[seg] = (ts0, ts1)

            msg_idx += 1
            if int(args.sample_every) > 1 and (msg_idx - 1) % int(args.sample_every) != 0:
                continue
            if last_proc_t is not None and min_dt > 0.0 and (t_sec - float(last_proc_t)) < float(min_dt):
                continue
            last_proc_t = t_sec

            xyz = _cloud_to_xyz(msg)
            if xyz.size == 0:
                continue
            xyz = _downsample_xyz(xyz, int(args.max_points))
            desc = sc.generate_scan_context(xyz)
            sums[seg] += desc.astype(np.float64)
            counts[seg] += 1

    prototypes = np.zeros_like(sums, dtype=np.float32)
    for seg in range(int(args.num_classes)):
        if int(counts[seg]) <= 0:
            continue
        prototypes[seg] = (sums[seg] / float(counts[seg])).astype(np.float32)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params = np.asarray(
        [
            int(args.num_ring),
            int(args.num_sector),
            float(args.min_range),
            float(args.max_range),
            float(args.height_lower),
            float(args.height_upper),
        ],
        dtype=np.float64,
    )
    segment_ranges = np.asarray(
        [[float(i0), float(i1), float(ts0), float(ts1)] for (i0, i1), (ts0, ts1) in zip(bounds, times)],
        dtype=np.float64,
    )
    np.savez_compressed(
        str(out_path),
        prototypes=prototypes.astype(np.float16),
        counts=counts,
        params=params,
        segment_ranges=segment_ranges,
        bag_path=np.asarray(str(bag_path), dtype=np.str_),
        cloud_topic=np.asarray(str(cloud_topic), dtype=np.str_),
        labeling=np.asarray(f"uniform_by_frame_count:{total_msgs}//{int(args.num_classes)}={per}", dtype=np.str_),
        process_hz=np.asarray(float(args.process_hz), dtype=np.float64),
        max_points=np.asarray(int(args.max_points), dtype=np.int64),
        total_msgs=np.asarray(int(total_msgs), dtype=np.int64),
        per=np.asarray(int(per), dtype=np.int64),
    )

    if bool(args.write_md):
        _write_md(out_path.with_suffix(".md"), bag_path, cloud_topic, bounds, times)

    print(f"[OK] wrote: {out_path} (K={int(args.num_classes)} total_msgs={total_msgs} per={per})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
