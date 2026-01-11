#!/usr/bin/env python3
"""Publish a static TF from map->odom so the first /odom pose matches a path start pose.

Problem this solves:
- In Gazebo, the odom produced by drive plugins can be initialized in different ways.
- If the robot is teleported to a start pose (SetModelState) after startup, /odom may still start at (0,0,0).
- move_base/TEB uses TF (map->odom + odom->base_link). If map->odom is wrong, goals and costmaps are misaligned.

This node reads the start pose from a path JSON/CSV (same format as rosbag_path.json),
waits for the first Odometry message, then publishes a /tf_static transform so that:

  ^map T_base(start) == start_pose_from_path

It is deterministic and removes the timing/race dependence between Gazebo startup and SetModelState.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def _load_json(path_file: Path) -> List[Tuple[float, float, float, Optional[float]]]:
    data = json.loads(path_file.read_text(encoding="utf-8"))
    points = []
    for pt in data.get("points", []):
        x = float(pt.get("x", 0.0))
        y = float(pt.get("y", 0.0))
        z = float(pt.get("z", 0.0))
        yaw = pt.get("yaw", None)
        yaw = float(yaw) if yaw is not None else None
        points.append((x, y, z, yaw))
    if not points:
        raise RuntimeError(f"No points in JSON: {path_file}")
    return points


def _load_csv(path_file: Path) -> List[Tuple[float, float, float, Optional[float]]]:
    points = []
    with path_file.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].strip().startswith("#") or row[0].strip().lower() == "index":
                continue
            try:
                _idx, x, y, z, yaw = row[:5]
                points.append((float(x), float(y), float(z), float(yaw)))
            except Exception:
                continue
    if not points:
        raise RuntimeError(f"No points in CSV: {path_file}")
    return points


def _compute_yaw(points: List[Tuple[float, float, float, Optional[float]]], idx: int) -> float:
    if idx + 1 < len(points):
        x0, y0, _z0, _yaw0 = points[idx]
        x1, y1, _z1, _yaw1 = points[idx + 1]
        return math.atan2(y1 - y0, x1 - x0)
    if idx > 0:
        x0, y0, _z0, _yaw0 = points[idx - 1]
        x1, y1, _z1, _yaw1 = points[idx]
        return math.atan2(y1 - y0, x1 - x0)
    return 0.0


def _yaw_from_quat(q) -> float:
    return float(euler_from_quaternion([q.x, q.y, q.z, q.w])[2])


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a static map->odom transform aligned to path start")
    parser.add_argument("--path", type=str, default="maps/runs/rosbag_path.json", help="输入 JSON/CSV 路径文件")
    parser.add_argument("--index", type=int, default=0, help="使用的路径索引（-1=最后）")
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument("--odom-frame", type=str, default="odom")
    parser.add_argument("--odom-topic", type=str, default="/odom")
    parser.add_argument("--wait", type=float, default=10.0, help="等待 /odom 的超时 (s)")
    args = parser.parse_args(rospy.myargv()[1:])

    path_file = Path(args.path).expanduser().resolve()
    if not path_file.is_file():
        raise SystemExit(f"path file not found: {path_file}")

    if path_file.suffix.lower() == ".json":
        points = _load_json(path_file)
    else:
        points = _load_csv(path_file)

    idx = int(args.index)
    if idx < 0:
        idx = len(points) - 1
    idx = max(0, min(idx, len(points) - 1))

    x_ref, y_ref, _z_ref, yaw_ref = points[idx]
    if yaw_ref is None:
        yaw_ref = _compute_yaw(points, idx)

    rospy.init_node("set_map_to_odom_from_path", anonymous=True)
    wait_s = float(args.wait)
    try:
        odom_msg: Odometry = rospy.wait_for_message(str(args.odom_topic), Odometry, timeout=wait_s)
    except rospy.ROSException as exc:
        raise SystemExit(f"timeout waiting for {args.odom_topic}: {exc}")

    pos = odom_msg.pose.pose.position
    yaw_odom = _yaw_from_quat(odom_msg.pose.pose.orientation)

    theta = float(yaw_ref) - float(yaw_odom)
    c = math.cos(theta)
    s = math.sin(theta)
    x_odom = float(pos.x)
    y_odom = float(pos.y)

    # Solve: [x_ref,y_ref]^T = R(theta)*[x_odom,y_odom]^T + t
    tx = float(x_ref) - (c * x_odom - s * y_odom)
    ty = float(y_ref) - (s * x_odom + c * y_odom)

    tf_msg = TransformStamped()
    tf_msg.header.stamp = rospy.Time.now()
    tf_msg.header.frame_id = str(args.map_frame)
    tf_msg.child_frame_id = str(args.odom_frame)
    tf_msg.transform.translation.x = tx
    tf_msg.transform.translation.y = ty
    tf_msg.transform.translation.z = 0.0
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, theta)
    tf_msg.transform.rotation.x = float(qx)
    tf_msg.transform.rotation.y = float(qy)
    tf_msg.transform.rotation.z = float(qz)
    tf_msg.transform.rotation.w = float(qw)

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    broadcaster.sendTransform(tf_msg)

    rospy.loginfo(
        "[set_map_to_odom] map->odom: tx=%.3f ty=%.3f yaw=%.3f (odom start: x=%.3f y=%.3f yaw=%.3f | ref: x=%.3f y=%.3f yaw=%.3f)",
        tx,
        ty,
        theta,
        x_odom,
        y_odom,
        yaw_odom,
        float(x_ref),
        float(y_ref),
        float(yaw_ref),
    )
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

