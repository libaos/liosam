#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import rosbag
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import String

try:
    from orchard_scancontext_fsm.scancontext import (
        ScanContextParams,
        distance_between_scancontexts,
        downsample_xyz,
        make_scancontext,
    )
except ModuleNotFoundError:  # allows running without catkin install
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from orchard_scancontext_fsm.scancontext import (  # type: ignore
        ScanContextParams,
        distance_between_scancontexts,
        downsample_xyz,
        make_scancontext,
    )


@dataclass
class _HistoryItem:
    stamp_s: float
    desc: np.ndarray


def _cloud_to_xyz(msg, max_points: int) -> np.ndarray:
    pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    xyz = np.asarray(pts, dtype=np.float32)
    return downsample_xyz(xyz, max_points=max_points)


def _parse_topics(text: str) -> List[str]:
    parts = [p.strip() for p in str(text or "").split(",")]
    return [p for p in parts if p]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Annotate a rosbag with /fsm/mode computed from ScanContext yaw-rate (offline).")
    p.add_argument("--in-bag", required=True, help="Input .bag")
    p.add_argument("--out-bag", required=True, help="Output .bag (will be overwritten)")
    p.add_argument("--cloud-topic", default="/liorl/deskew/cloud_deskewed", help="PointCloud2 topic")
    p.add_argument("--mode-topic", default="/fsm/mode", help="Output std_msgs/String topic")
    p.add_argument(
        "--copy-topics",
        default="/liorl/mapping/path",
        help="Comma-separated topics to copy into output bag (default: /liorl/mapping/path)",
    )
    p.add_argument("--copy-cloud", action="store_true", help="Also copy cloud messages into output bag (large)")
    p.add_argument("--start", type=float, default=None, help="Start time (sec, ROS time in bag)")
    p.add_argument("--duration", type=float, default=None, help="Duration (sec)")

    p.add_argument("--process-hz", type=float, default=2.0)
    p.add_argument("--baseline-s", type=float, default=1.0)
    p.add_argument("--history-s", type=float, default=4.0)
    p.add_argument("--max-points", type=int, default=60000)
    p.add_argument("--sc-dist-max", type=float, default=0.35)
    p.add_argument("--yaw-rate-threshold", type=float, default=0.25)
    p.add_argument("--consistency-n", type=int, default=3)

    p.add_argument("--num-ring", type=int, default=20)
    p.add_argument("--num-sector", type=int, default=60)
    p.add_argument("--max-radius", type=float, default=80.0)
    p.add_argument("--lidar-height", type=float, default=2.0)
    p.add_argument("--search-ratio", type=float, default=0.1)

    args = p.parse_args(argv)

    in_bag = Path(args.in_bag).expanduser().resolve()
    out_bag = Path(args.out_bag).expanduser().resolve()
    if not in_bag.is_file():
        raise RuntimeError(f"Input bag not found: {in_bag}")
    out_bag.parent.mkdir(parents=True, exist_ok=True)

    params = ScanContextParams(
        num_ring=int(args.num_ring),
        num_sector=int(args.num_sector),
        max_radius=float(args.max_radius),
        lidar_height=float(args.lidar_height),
        search_ratio=float(args.search_ratio),
    )

    cloud_topic = str(args.cloud_topic).strip()
    mode_topic = str(args.mode_topic).strip()
    copy_topics = _parse_topics(args.copy_topics)
    if cloud_topic not in copy_topics and args.copy_cloud:
        copy_topics.append(cloud_topic)

    read_topics = list(dict.fromkeys([cloud_topic] + copy_topics))

    start_time = None
    end_time = None
    if args.start is not None:
        start_time = float(args.start)
        if args.duration is not None and float(args.duration) > 0.0:
            end_time = float(start_time) + float(args.duration)

    history: Deque[_HistoryItem] = deque()
    last_processed_s: Optional[float] = None

    mode = "straight"
    candidate: Optional[str] = None
    candidate_count = 0

    def pick_ref(now_s: float) -> Optional[_HistoryItem]:
        target = float(now_s) - float(args.baseline_s)
        for item in reversed(history):
            if float(item.stamp_s) <= target:
                return item
        return None

    def prune(now_s: float) -> None:
        cutoff = float(now_s) - float(args.history_s)
        while history and float(history[0].stamp_s) < cutoff:
            history.popleft()

    def raw_mode_from_yaw_rate(yaw_rate: float) -> str:
        if abs(float(yaw_rate)) < float(args.yaw_rate_threshold):
            return "straight"
        return "left" if float(yaw_rate) > 0.0 else "right"

    def update_mode(raw: str) -> None:
        nonlocal mode, candidate, candidate_count
        raw = str(raw).strip().lower()
        if raw == mode:
            candidate = None
            candidate_count = 0
            return
        if candidate != raw:
            candidate = raw
            candidate_count = 1
            return
        candidate_count += 1
        if candidate_count >= int(args.consistency_n):
            mode = raw
            candidate = None
            candidate_count = 0

    with rosbag.Bag(str(in_bag)) as bag_in, rosbag.Bag(str(out_bag), "w") as bag_out:
        for topic, msg, t in bag_in.read_messages(topics=read_topics):
            now_s = float(t.to_sec())
            if start_time is not None and now_s < float(start_time):
                continue
            if end_time is not None and now_s > float(end_time):
                break

            if topic in copy_topics:
                bag_out.write(topic, msg, t)

            if topic != cloud_topic:
                continue
            if args.process_hz > 0.0 and last_processed_s is not None:
                if float(now_s) - float(last_processed_s) < 1.0 / float(args.process_hz):
                    continue
            last_processed_s = now_s

            xyz = _cloud_to_xyz(msg, max_points=int(args.max_points))
            desc = make_scancontext(xyz, params)

            history.append(_HistoryItem(stamp_s=now_s, desc=desc))
            prune(now_s)

            yaw_rate = 0.0
            ref = pick_ref(now_s)
            if ref is not None:
                dt = float(now_s) - float(ref.stamp_s)
                if dt > 1.0e-3:
                    sc_dist, yaw_diff = distance_between_scancontexts(ref.desc, desc, params)
                    if float(sc_dist) <= float(args.sc_dist_max):
                        yaw_rate = float(yaw_diff) / dt

            update_mode(raw_mode_from_yaw_rate(yaw_rate))
            bag_out.write(mode_topic, String(data=mode), t)

    print(f"[OK] Wrote: {out_bag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
