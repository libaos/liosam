#!/usr/bin/env python3

import argparse
import math
import os
from typing import List, Optional, Sequence, Tuple

import rosbag


def _distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pick_best_path_topic(bag_path: str, requested_topic: Optional[str]) -> str:
    if requested_topic:
        return requested_topic

    with rosbag.Bag(bag_path) as bag:
        topics = bag.get_type_and_topic_info().topics

    path_topics = [
        topic
        for topic, info in topics.items()
        if getattr(info, "msg_type", None) == "nav_msgs/Path"
    ]
    if not path_topics:
        raise RuntimeError("No nav_msgs/Path topics found in bag; pass --topic explicitly.")

    priority = [
        "/lio_sam/mapping/path",
        "/liorl/mapping/path",
        "/liorf/mapping/path",
    ]
    for candidate in priority:
        if candidate in path_topics:
            return candidate

    mapping_paths = [t for t in path_topics if t.endswith("/mapping/path")]
    if mapping_paths:
        return sorted(mapping_paths)[0]

    return sorted(path_topics)[0]


def _select_path_poses(bag_path: str, topic: Optional[str]):
    topic = _pick_best_path_topic(bag_path, topic)
    last_msg = None
    with rosbag.Bag(bag_path) as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            last_msg = msg

    if last_msg is None:
        raise RuntimeError(f"Topic not found in bag: {topic}")
    if not hasattr(last_msg, "poses"):
        raise RuntimeError(f"Topic is not nav_msgs/Path (missing poses[]): {topic}")
    if not last_msg.poses:
        raise RuntimeError(f"Path is empty on topic: {topic}")
    return topic, last_msg.poses


def _downsample_poses_by_distance(poses, min_dist_m: float):
    if min_dist_m <= 0:
        return list(poses)

    selected = [poses[0]]
    last_xy = (
        poses[0].pose.position.x,
        poses[0].pose.position.y,
    )

    for pose_stamped in poses[1:-1]:
        xy = (pose_stamped.pose.position.x, pose_stamped.pose.position.y)
        if _distance_xy(xy, last_xy) >= min_dist_m:
            selected.append(pose_stamped)
            last_xy = xy

    if poses[-1] is not selected[-1]:
        selected.append(poses[-1])
    return selected


def _cap_waypoints_uniform(poses, max_waypoints: Optional[int]):
    if not max_waypoints or max_waypoints <= 0:
        return list(poses)
    if max_waypoints < 2:
        raise ValueError("--max-waypoints must be >= 2")
    if len(poses) <= max_waypoints:
        return list(poses)

    # Uniform sampling, keep first/last.
    out = [poses[0]]
    inner = poses[1:-1]
    need_inner = max_waypoints - 2
    step = max(1, len(inner) // need_inner)
    out.extend(inner[::step][:need_inner])
    out.append(poses[-1])
    return out


def _yaw_from_direction(this_xy: Tuple[float, float], next_xy: Tuple[float, float]) -> float:
    return math.atan2(next_xy[1] - this_xy[1], next_xy[0] - this_xy[0])


def _format_yaml(poses, name_prefix: str, keep_z: bool) -> str:
    lines: List[str] = ["waypoints:"]
    cached_xy: List[Tuple[float, float]] = [
        (ps.pose.position.x, ps.pose.position.y) for ps in poses
    ]

    last_yaw = 0.0
    for i, pose_stamped in enumerate(poses):
        if len(poses) >= 2:
            if i < len(poses) - 1:
                yaw = _yaw_from_direction(cached_xy[i], cached_xy[i + 1])
            else:
                yaw = last_yaw
        else:
            yaw = 0.0
        last_yaw = yaw

        p = pose_stamped.pose.position
        lines.append(f'  - name: "{name_prefix}{i:03d}"')
        lines.append(f"    x: {p.x:.6f}")
        lines.append(f"    y: {p.y:.6f}")
        lines.append(f"    z: {p.z:.6f}" if keep_z else "    z: 0.000000")
        lines.append(f"    yaw: {yaw:.6f}")

    return "\n".join(lines) + "\n"


def _write_text(path: str, content: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a nav_msgs/Path from a rosbag and export it as continuous_navigation waypoints.yaml"
    )
    parser.add_argument("bag", help="Input .bag file")
    parser.add_argument(
        "--topic",
        default=None,
        help="nav_msgs/Path topic to extract (default: auto-detect)",
    )
    parser.add_argument(
        "--out",
        default="src/continuous_navigation/config/waypoints_from_bag.yaml",
        help="Output waypoints YAML path",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=1.0,
        help="Minimum XY distance (meters) between consecutive waypoints (default: 1.0)",
    )
    parser.add_argument(
        "--max-waypoints",
        type=int,
        default=0,
        help="Cap number of waypoints (0 = no cap). First/last always kept.",
    )
    parser.add_argument(
        "--name-prefix",
        default="wp_",
        help='Waypoint name prefix (default: "wp_")',
    )
    parser.add_argument(
        "--keep-z",
        action="store_true",
        help="Keep Z from bag poses (default: write z=0.0 for 2D navigation)",
    )

    args = parser.parse_args(argv)

    topic, poses = _select_path_poses(args.bag, args.topic)
    poses = _downsample_poses_by_distance(poses, args.min_dist)
    poses = _cap_waypoints_uniform(poses, args.max_waypoints)

    content = _format_yaml(poses, args.name_prefix, keep_z=args.keep_z)
    _write_text(args.out, content)

    print(
        f"[OK] Exported {len(poses)} waypoints to: {os.path.abspath(args.out)}\n"
        f"     Source: {os.path.abspath(args.bag)} ({topic})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
