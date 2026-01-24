#!/usr/bin/env python3
"""Follow a JSON path by publishing sequential move_base goals.

Why this exists:
- Many rosbags contain loop-like trajectories where the *final* pose is close to the start.
- If you only publish the final goal, move_base/TEB will just drive a short segment.

This node publishes intermediate goals along the reference path so the robot traverses the full route.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from pathlib import Path
from typing import List, Optional, Tuple

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def _load_path_xyz_yaw(json_path: Path) -> List[Tuple[float, float, float]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    points = data.get("points", data)
    if not isinstance(points, list) or not points:
        raise RuntimeError(f"Invalid path JSON (missing points): {json_path}")
    out: List[Tuple[float, float, float]] = []
    for p in points:
        if "x" not in p or "y" not in p:
            continue
        yaw = float(p.get("yaw", 0.0))
        out.append((float(p["x"]), float(p["y"]), yaw))
    if len(out) < 2:
        raise RuntimeError(f"Too few points in: {json_path}")
    return out


def _load_tree_circles(json_path: Path, default_radius: float) -> Tuple[Optional[str], List[Tuple[float, float, float]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    frame_id = None
    if isinstance(data, dict):
        frame_id = str(data.get("frame_id") or "").strip() or None
        circles = data.get("circles", [])
    elif isinstance(data, list):
        circles = data
    else:
        circles = []
    if not isinstance(circles, list) or not circles:
        raise RuntimeError(f"Invalid circles JSON (missing circles): {json_path}")

    out: List[Tuple[float, float, float]] = []
    for c in circles:
        if not isinstance(c, dict):
            continue
        if "x" not in c or "y" not in c:
            continue
        r = float(c.get("radius", default_radius))
        out.append((float(c["x"]), float(c["y"]), r))
    if not out:
        raise RuntimeError(f"Invalid circles JSON (no usable circles): {json_path}")
    return frame_id, out


def _downsample_min_dist(points: List[Tuple[float, float, float]], min_dist: float) -> List[Tuple[float, float, float]]:
    if min_dist <= 0:
        return list(points)
    keep = [points[0]]
    last_x, last_y, _ = keep[0]
    for x, y, yaw in points[1:]:
        if math.hypot(x - last_x, y - last_y) >= min_dist:
            keep.append((x, y, yaw))
            last_x, last_y = x, y
    if keep[-1] != points[-1]:
        keep.append(points[-1])
    return keep


def _filter_goals_by_circles(
    goals: List[Tuple[float, float, float]], circles: List[Tuple[float, float, float]], clearance: float
) -> Tuple[List[Tuple[float, float, float]], int, float]:
    if not circles or clearance < 0:
        return (list(goals), 0, float("inf"))

    removed = 0
    min_removed_dist = float("inf")
    kept: List[Tuple[float, float, float]] = []

    for x, y, yaw in goals:
        reject = False
        nearest = float("inf")
        for cx, cy, r in circles:
            d = float(math.hypot(float(x) - float(cx), float(y) - float(cy)))
            if d < nearest:
                nearest = d
            if d < float(r) + float(clearance):
                reject = True
                break
        if reject:
            removed += 1
            if nearest < min_removed_dist:
                min_removed_dist = nearest
            continue
        kept.append((x, y, yaw))

    return (kept, removed, min_removed_dist)


def _yaw_from_quat(q) -> float:
    return float(euler_from_quaternion([q.x, q.y, q.z, q.w])[2])


class GoalFollower:
    def __init__(self, path: List[Tuple[float, float, float]], frame_id: str, goal_topic: str, odom_topic: str):
        self.path = path
        self.frame_id = str(frame_id)
        self.goal_topic = str(goal_topic)
        self.odom_topic = str(odom_topic)
        self.odom: Optional[Odometry] = None

        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.35))
        self.final_goal_tolerance = float(rospy.get_param("~final_goal_tolerance", self.goal_tolerance))
        if self.final_goal_tolerance <= 0:
            self.final_goal_tolerance = float(self.goal_tolerance)
        self.goal_timeout_s = float(rospy.get_param("~goal_timeout_s", 0.0))
        self.sleep_rate = float(rospy.get_param("~rate", 5.0))
        self.wait_conn_s = float(rospy.get_param("~wait_connections_s", 10.0))
        self.progress_every = int(rospy.get_param("~progress_every", 10))
        self.status_topic = str(rospy.get_param("~status_topic", "/move_base/status"))
        self.republish_s = float(rospy.get_param("~republish_s", 2.0))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.2))

        self._warned_tf = False
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self.pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1, latch=True)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)
        self._status: Optional[GoalStatusArray] = None
        rospy.Subscriber(self.status_topic, GoalStatusArray, self._status_cb, queue_size=1)

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom = msg

    def _status_cb(self, msg: GoalStatusArray) -> None:
        self._status = msg

    def _wait_for_odom(self) -> None:
        start = rospy.Time.now().to_sec()
        while not rospy.is_shutdown() and self.odom is None:
            if rospy.Time.now().to_sec() - start > 10.0:
                rospy.logwarn("waiting for odom on %s ...", self.odom_topic)
                start = rospy.Time.now().to_sec()
            try:
                rospy.sleep(0.1)
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                start = rospy.Time.now().to_sec()
                continue

    def _wait_for_connections(self) -> None:
        if self.wait_conn_s <= 0:
            return
        deadline = rospy.Time.now().to_sec() + self.wait_conn_s
        while not rospy.is_shutdown() and self.pub.get_num_connections() <= 0:
            if rospy.Time.now().to_sec() >= deadline:
                rospy.logwarn("goal_topic has no subscribers: %s (continue anyway)", self.goal_topic)
                return
            rospy.sleep(0.1)

    def _odom_xy_in_goal_frame(self) -> Optional[Tuple[float, float]]:
        if self.odom is None:
            return None
        p = self.odom.pose.pose.position
        x = float(p.x)
        y = float(p.y)

        odom_frame = str(getattr(self.odom.header, "frame_id", "") or "").strip()
        goal_frame = str(self.frame_id or "").strip()
        if not goal_frame or not odom_frame or odom_frame == goal_frame:
            return (x, y)

        try:
            tr = self._tf_buffer.lookup_transform(
                goal_frame, odom_frame, self.odom.header.stamp, rospy.Duration(self.tf_timeout)
            )
        except Exception as exc:
            if not self._warned_tf:
                rospy.logwarn(
                    "goal_follower: TF lookup failed (%s -> %s): %s; falling back to raw odom coords",
                    odom_frame,
                    goal_frame,
                    exc,
                )
                self._warned_tf = True
            return (x, y)

        t = tr.transform.translation
        q = tr.transform.rotation
        yaw_tf = _yaw_from_quat(q)
        c = float(math.cos(yaw_tf))
        s = float(math.sin(yaw_tf))
        x_out = float(t.x) + c * float(x) - s * float(y)
        y_out = float(t.y) + s * float(x) + c * float(y)
        return (x_out, y_out)

    def _dist_to(self, gx: float, gy: float) -> float:
        xy = self._odom_xy_in_goal_frame()
        if xy is None:
            return float("inf")
        x, y = xy
        return float(math.hypot(gx - float(x), gy - float(y)))

    def _has_active_goal(self) -> bool:
        if self._status is None:
            return False
        for s in self._status.status_list:
            if int(getattr(s, "status", -1)) in (0, 1, 6, 7):
                return True
        return False

    def _publish_goal(self, x: float, y: float, yaw: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(yaw))
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        self.pub.publish(msg)

    def run(self) -> None:
        self._wait_for_connections()
        self._wait_for_odom()

        rate = rospy.Rate(max(0.5, self.sleep_rate))
        skipped = 0
        for i, (gx, gy, gyaw) in enumerate(self.path):
            if rospy.is_shutdown():
                return
            self._publish_goal(gx, gy, gyaw)
            rospy.loginfo("[goal_follower] goal %d/%d: x=%.2f y=%.2f", i + 1, len(self.path), gx, gy)
            goal_start = rospy.Time.now().to_sec()
            last_republish = rospy.Time.now().to_sec()
            tol = float(self.goal_tolerance)
            if i == (len(self.path) - 1):
                tol = float(self.final_goal_tolerance)

            while not rospy.is_shutdown():
                d = self._dist_to(gx, gy)
                if d <= tol:
                    break
                if self.goal_timeout_s > 0 and (rospy.Time.now().to_sec() - goal_start) >= self.goal_timeout_s:
                    skipped += 1
                    rospy.logwarn(
                        "[goal_follower] goal timeout idx=%d (%.1fs), skip to next; dist=%.2f",
                        i,
                        float(self.goal_timeout_s),
                        d,
                    )
                    break
                if self.republish_s > 0:
                    now = rospy.Time.now().to_sec()
                    if (now - last_republish) >= self.republish_s:
                        if not self._has_active_goal():
                            self._publish_goal(gx, gy, gyaw)
                            rospy.logwarn_throttle(
                                2.0, "[goal_follower] republish goal idx=%d (no active goal on %s)", i, self.status_topic
                            )
                        last_republish = now
                if self.progress_every > 0 and (i % self.progress_every) == 0:
                    rospy.loginfo_throttle(2.0, "[goal_follower] idx=%d dist=%.2f", i, d)
                try:
                    rate.sleep()
                except rospy.exceptions.ROSTimeMovedBackwardsException:
                    continue

        rospy.loginfo("[goal_follower] done (%d goals, skipped=%d)", len(self.path), skipped)


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_path = ws_dir / "src" / "pcd_gazebo_world" / "maps" / "runs" / "rosbag_path.json"

    parser = argparse.ArgumentParser(description="Publish sequential move_base goals from a JSON path")
    parser.add_argument("--path", type=str, default=str(default_path), help="路径 JSON")
    parser.add_argument("--frame-id", type=str, default="map")
    parser.add_argument("--goal-topic", type=str, default="/move_base_simple/goal")
    parser.add_argument("--odom-topic", type=str, default="/odom")
    parser.add_argument("--min-dist", type=float, default=0.8, help="目标点最小间距（m，0=不降采样）")
    parser.add_argument("--circles", type=str, default="", help="树中心 circles JSON（空=禁用过滤）")
    parser.add_argument("--tree-clearance", type=float, default=None, help="过滤阈值额外安全距离（m）")
    parser.add_argument("--tree-default-radius", type=float, default=None, help="circles 中缺失 radius 时的默认半径（m）")
    args = parser.parse_args(rospy.myargv()[1:])

    path_file = Path(args.path).expanduser().resolve()
    if not path_file.is_file():
        raise SystemExit(f"path file not found: {path_file}")

    rospy.init_node("follow_path_goals", anonymous=True)

    pts = _load_path_xyz_yaw(path_file)
    pts = _downsample_min_dist(pts, float(args.min_dist))

    circles_arg = str(args.circles or rospy.get_param("~circles", "")).strip()
    clearance = (
        float(args.tree_clearance)
        if args.tree_clearance is not None
        else float(rospy.get_param("~tree_clearance", 0.30))
    )
    default_radius = (
        float(args.tree_default_radius)
        if args.tree_default_radius is not None
        else float(rospy.get_param("~tree_default_radius", 0.15))
    )

    if circles_arg:
        circles_path = Path(circles_arg).expanduser().resolve()
        if not circles_path.is_file():
            raise SystemExit(f"circles file not found: {circles_path}")
        circles_frame, circles = _load_tree_circles(circles_path, default_radius=float(default_radius))
        if circles_frame and str(circles_frame) != str(args.frame_id):
            rospy.logwarn(
                "[goal_follower] circles frame_id=%s does not match goal frame_id=%s; filtering may be wrong",
                circles_frame,
                str(args.frame_id),
            )

        before = len(pts)
        pts, removed, min_removed_dist = _filter_goals_by_circles(pts, circles, clearance=float(clearance))
        if removed:
            rospy.logwarn(
                "[goal_follower] filtered %d/%d goals too close to trees (clearance=%.2f, min_removed_dist=%.2f)",
                removed,
                before,
                float(clearance),
                float(min_removed_dist) if min_removed_dist != float("inf") else -1.0,
            )
        else:
            rospy.loginfo("[goal_follower] circles filter enabled: removed 0/%d goals", before)

    if len(pts) < 2:
        raise SystemExit(f"too few goals after filtering: {len(pts)}")
    GoalFollower(pts, str(args.frame_id), str(args.goal_topic), str(args.odom_topic)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
