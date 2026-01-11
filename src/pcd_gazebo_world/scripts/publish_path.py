#!/usr/bin/env python3
"""Publish a stored path as nav_msgs/Path and optionally a goal.

Note about TEB via-points:
- `teb_local_planner` accepts custom via-points on the `via_points` topic as `nav_msgs/Path`,
  but it does **not** transform them using TF (it just reads x/y).
- That means you must publish via-points in the same coordinate frame that TEB optimizes in
  (usually the local costmap's `global_frame`, e.g. `odom`/`odom_est`).
- Use `--target-frame` to publish the same stored path in another frame (via TF).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Tuple

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def _load_json(path_file: Path) -> Tuple[str, List[Tuple[float, float, float, float]]]:
    data = json.loads(path_file.read_text(encoding="utf-8"))
    frame_id = str(data.get("frame_id", "map"))
    points = []
    for pt in data.get("points", []):
        points.append((float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0)), float(pt.get("yaw", 0.0))))
    if not points:
        raise RuntimeError(f"No points in JSON: {path_file}")
    return frame_id, points


def _load_csv(path_file: Path) -> Tuple[str, List[Tuple[float, float, float, float]]]:
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
    return "map", points


def _compute_yaws(points: List[Tuple[float, float, float, float]]) -> List[float]:
    yaws: List[float] = []
    for i in range(len(points)):
        if i + 1 < len(points):
            x0, y0, _z0, _yaw0 = points[i]
            x1, y1, _z1, _yaw1 = points[i + 1]
            yaws.append(math.atan2(y1 - y0, x1 - x0))
        elif yaws:
            yaws.append(yaws[-1])
        else:
            yaws.append(0.0)
    return yaws


def _build_path(frame_id: str, points: List[Tuple[float, float, float, float]]) -> NavPath:
    msg = NavPath()
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    for x, y, z, yaw in points:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = msg.header.stamp
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        q = quaternion_from_euler(0.0, 0.0, float(yaw))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        msg.poses.append(pose)
    return msg


def _yaw_from_quat(q) -> float:
    return float(euler_from_quaternion([float(q.x), float(q.y), float(q.z), float(q.w)])[2])


def _transform_points_xy_yaw(
    points: List[Tuple[float, float, float, float]],
    tf_buffer: tf2_ros.Buffer,
    source_frame: str,
    target_frame: str,
    tf_timeout_s: float,
    stamp: rospy.Time,
) -> List[Tuple[float, float, float, float]]:
    tr = tf_buffer.lookup_transform(target_frame, source_frame, stamp, rospy.Duration(tf_timeout_s))
    t = tr.transform.translation
    q = tr.transform.rotation
    yaw_tf = _yaw_from_quat(q)
    c = float(math.cos(yaw_tf))
    s = float(math.sin(yaw_tf))

    out: List[Tuple[float, float, float, float]] = []
    for x, y, z, yaw in points:
        x_out = float(t.x) + c * float(x) - s * float(y)
        y_out = float(t.y) + s * float(x) + c * float(y)
        yaw_out = float(yaw_tf + float(yaw))
        out.append((x_out, y_out, float(z), yaw_out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a saved path as nav_msgs/Path")
    parser.add_argument("--path", type=str, default="maps/runs/rosbag_path.json", help="输入 JSON/CSV 路径文件")
    parser.add_argument("--topic", type=str, default="/reference_path", help="nav_msgs/Path 发布话题")
    parser.add_argument("--frame-id", type=str, default="", help="覆盖 frame_id")
    parser.add_argument(
        "--target-frame",
        type=str,
        default="",
        help="可选：通过 TF 把路径点从 frame_id 变换到该 frame（TEB via-points 常用 odom/odom_est）",
    )
    parser.add_argument("--tf-timeout", type=float, default=0.5, help="TF 查询超时 (s)")
    parser.add_argument("--wait-tf", type=float, default=10.0, help="等待 TF 的超时 (s)")
    parser.add_argument("--rate", type=float, default=1.0, help="发布频率 Hz（0=仅发布一次）")
    parser.add_argument("--publish-goal", type=int, default=1, help="1=发布最终 goal")
    parser.add_argument("--goal-topic", type=str, default="/move_base_simple/goal", help="goal 话题")
    parser.add_argument("--goal-index", type=int, default=-1, help="goal 点索引（-1=最后）")
    args = parser.parse_args(rospy.myargv()[1:])

    path_file = Path(args.path).expanduser().resolve()
    if not path_file.is_file():
        raise SystemExit(f"path file not found: {path_file}")

    if path_file.suffix.lower() == ".json":
        frame_id, points = _load_json(path_file)
    else:
        frame_id, points = _load_csv(path_file)

    if any(abs(p[3]) < 1.0e-9 for p in points):
        yaws = _compute_yaws(points)
        points = [(x, y, z, yaw) for (x, y, z, _old_yaw), yaw in zip(points, yaws)]

    rospy.init_node("publish_path", anonymous=True)

    source_frame = str(args.frame_id).strip() or frame_id
    target_frame = str(args.target_frame).strip()

    tf_buffer = None
    tf_listener = None
    if target_frame and target_frame != source_frame:
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

    path_pub = rospy.Publisher(str(args.topic), NavPath, queue_size=1, latch=True)
    goal_pub = rospy.Publisher(str(args.goal_topic), PoseStamped, queue_size=1, latch=True)

    rate_hz = float(args.rate)
    rate = rospy.Rate(rate_hz) if rate_hz > 0 else None

    goal_idx = int(args.goal_index)
    if goal_idx < 0:
        goal_idx = len(points) - 1
    goal_idx = max(0, min(goal_idx, len(points) - 1))

    while not rospy.is_shutdown():
        out_frame = source_frame
        out_points = points
        if target_frame and target_frame != source_frame:
            assert tf_buffer is not None and tf_listener is not None
            deadline = rospy.Time.now().to_sec() + float(args.wait_tf)
            last_err = None
            while not rospy.is_shutdown():
                try:
                    out_points = _transform_points_xy_yaw(
                        out_points,
                        tf_buffer,
                        source_frame=source_frame,
                        target_frame=target_frame,
                        tf_timeout_s=float(args.tf_timeout),
                        stamp=rospy.Time(0),
                    )
                    out_frame = target_frame
                    break
                except Exception as exc:
                    last_err = exc
                    if rospy.Time.now().to_sec() >= deadline:
                        rospy.logerr(
                            "publish_path: TF transform failed (%s -> %s) after %.1fs: %s",
                            source_frame,
                            target_frame,
                            float(args.wait_tf),
                            exc,
                        )
                        raise
                    rospy.sleep(0.05)

            if last_err is not None:
                rospy.logwarn_once(
                    "publish_path: waiting for TF (%s -> %s) ... (%s)",
                    source_frame,
                    target_frame,
                    last_err,
                )

        msg = _build_path(out_frame, out_points)
        path_pub.publish(msg)

        if bool(int(args.publish_goal)) and out_points:
            gx, gy, gz, gyaw = out_points[goal_idx]
            goal = PoseStamped()
            goal.header.frame_id = out_frame
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = float(gx)
            goal.pose.position.y = float(gy)
            goal.pose.position.z = float(gz)
            q = quaternion_from_euler(0.0, 0.0, float(gyaw))
            goal.pose.orientation.x = q[0]
            goal.pose.orientation.y = q[1]
            goal.pose.orientation.z = q[2]
            goal.pose.orientation.w = q[3]
            goal_pub.publish(goal)

        if rate is None:
            break
        rate.sleep()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
