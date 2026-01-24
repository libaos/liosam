#!/usr/bin/env python3
"""Publish a sliding window of TEB custom via-points from a stored path.

Why:
- teb_local_planner subscribes to "via_points" as nav_msgs/Path, but it does NOT TF-transform the poses.
- Publishing a *full* long path as via-points over-constrains TEB's short-horizon optimization and can
  lead to infeasible trajectories / multi-meter deviations.

This node publishes only a short segment ahead of the robot (e.g. next 3m), transformed into the
local planner frame (usually odom/odom_est).
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
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from tf.transformations import euler_from_quaternion


XY = Tuple[float, float]


def _yaw_from_quat(q) -> float:
    return float(euler_from_quaternion([float(q.x), float(q.y), float(q.z), float(q.w)])[2])


def _load_path_xy(path_file: Path) -> Tuple[str, List[XY]]:
    data = json.loads(path_file.read_text(encoding="utf-8"))
    frame_id = str(data.get("frame_id", "map")).strip() or "map"
    points = data.get("points", data)
    if not isinstance(points, list) or not points:
        raise RuntimeError(f"Invalid path JSON (missing points): {path_file}")
    xy: List[XY] = []
    for p in points:
        if not isinstance(p, dict) or "x" not in p or "y" not in p:
            continue
        xy.append((float(p["x"]), float(p["y"])))
    if len(xy) < 2:
        raise RuntimeError(f"Too few points in: {path_file}")
    return frame_id, xy


def _transform_xy_points(
    xy: List[XY],
    tf_buffer: tf2_ros.Buffer,
    source_frame: str,
    target_frame: str,
    tf_timeout_s: float,
    stamp: rospy.Time,
) -> List[XY]:
    if source_frame == target_frame:
        return list(xy)
    tr = tf_buffer.lookup_transform(target_frame, source_frame, stamp, rospy.Duration(tf_timeout_s))
    t = tr.transform.translation
    q = tr.transform.rotation
    yaw_tf = _yaw_from_quat(q)
    c = float(math.cos(yaw_tf))
    s = float(math.sin(yaw_tf))
    out: List[XY] = []
    for x, y in xy:
        x_out = float(t.x) + c * float(x) - s * float(y)
        y_out = float(t.y) + s * float(x) + c * float(y)
        out.append((x_out, y_out))
    return out


def _nearest_index(xy: List[XY], p: XY, start: int, end: int) -> int:
    best_i = start
    best_d2 = float("inf")
    px, py = p
    for i in range(start, end):
        x, y = xy[i]
        d2 = (x - px) * (x - px) + (y - py) * (y - py)
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def _window_ahead(
    xy: List[XY],
    start_idx: int,
    window_dist_m: float,
    min_sep_m: float,
    max_points: int,
) -> List[XY]:
    if not xy:
        return []
    start_idx = max(0, min(start_idx, len(xy) - 1))
    window_dist_m = max(0.1, float(window_dist_m))
    min_sep_m = max(0.0, float(min_sep_m))
    max_points = int(max_points)
    if max_points <= 0:
        max_points = 12

    out = [xy[start_idx]]
    last_x, last_y = out[0]
    dist_acc = 0.0
    prev_x, prev_y = out[0]
    for x, y in xy[start_idx + 1 :]:
        seg = float(math.hypot(x - prev_x, y - prev_y))
        dist_acc += seg
        prev_x, prev_y = x, y
        if dist_acc > window_dist_m:
            break
        if min_sep_m > 0 and math.hypot(x - last_x, y - last_y) < min_sep_m:
            continue
        out.append((x, y))
        last_x, last_y = x, y
        if len(out) >= max_points:
            break
    return out


def _build_path(frame_id: str, xy: List[XY]) -> NavPath:
    msg = NavPath()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = str(frame_id)
    for x, y in xy:
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


class ViaPointsWindowPublisher:
    def __init__(
        self,
        path_xy: List[XY],
        path_frame: str,
        local_frame: str,
        odom_topic: str,
        out_topic: str,
        rate_hz: float,
        window_dist_m: float,
        min_sep_m: float,
        max_points: int,
        tf_timeout_s: float,
        wait_tf_s: float,
        search_back: int,
        search_fwd: int,
    ) -> None:
        self.path_xy = path_xy
        self.path_frame = str(path_frame).strip()
        self.local_frame = str(local_frame).strip()
        self.odom_topic = str(odom_topic)
        self.out_topic = str(out_topic)
        self.rate_hz = float(rate_hz)
        self.window_dist_m = float(window_dist_m)
        self.min_sep_m = float(min_sep_m)
        self.max_points = int(max_points)
        self.tf_timeout_s = float(tf_timeout_s)
        self.wait_tf_s = float(wait_tf_s)
        self.search_back = int(search_back)
        self.search_fwd = int(search_fwd)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._odom: Optional[Odometry] = None
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)

        self._pub = rospy.Publisher(self.out_topic, NavPath, queue_size=1)
        self._last_idx: Optional[int] = None

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom = msg

    def _wait_for_odom(self) -> None:
        start = rospy.Time.now().to_sec()
        while not rospy.is_shutdown() and self._odom is None:
            if rospy.Time.now().to_sec() - start > 10.0:
                rospy.logwarn("publish_via_points_window: waiting for odom on %s ...", self.odom_topic)
                start = rospy.Time.now().to_sec()
            try:
                rospy.sleep(0.05)
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                start = rospy.Time.now().to_sec()
                continue

    def _wait_for_tf(self) -> None:
        if self.path_frame == self.local_frame:
            return
        deadline = rospy.Time.now().to_sec() + self.wait_tf_s
        while not rospy.is_shutdown():
            try:
                self._tf_buffer.lookup_transform(self.local_frame, self.path_frame, rospy.Time(0), rospy.Duration(0.2))
                self._tf_buffer.lookup_transform(self.path_frame, self.local_frame, rospy.Time(0), rospy.Duration(0.2))
                return
            except Exception as exc:
                if rospy.Time.now().to_sec() >= deadline:
                    raise RuntimeError(f"TF not available between {self.path_frame} and {self.local_frame}: {exc}") from exc
                try:
                    rospy.sleep(0.05)
                except rospy.exceptions.ROSTimeMovedBackwardsException:
                    deadline = rospy.Time.now().to_sec() + self.wait_tf_s
                    continue

    def run(self) -> None:
        self._wait_for_odom()
        self._wait_for_tf()

        rate = rospy.Rate(max(0.2, self.rate_hz))
        while not rospy.is_shutdown():
            odom = self._odom
            if odom is None:
                rate.sleep()
                continue

            odom_frame = str(getattr(odom.header, "frame_id", "") or "").strip() or self.local_frame
            p = odom.pose.pose.position
            robot_local = (float(p.x), float(p.y))

            # If odom frame differs from local_frame, transform robot pose into local_frame.
            if odom_frame != self.local_frame:
                robot_local = _transform_xy_points(
                    [robot_local],
                    self._tf_buffer,
                    source_frame=odom_frame,
                    target_frame=self.local_frame,
                    tf_timeout_s=self.tf_timeout_s,
                    stamp=odom.header.stamp,
                )[0]

            # Find robot position in path frame to find the nearest path index.
            robot_in_path = robot_local
            if self.path_frame != self.local_frame:
                robot_in_path = _transform_xy_points(
                    [robot_local],
                    self._tf_buffer,
                    source_frame=self.local_frame,
                    target_frame=self.path_frame,
                    tf_timeout_s=self.tf_timeout_s,
                    stamp=odom.header.stamp,
                )[0]

            if self._last_idx is None:
                idx = _nearest_index(self.path_xy, robot_in_path, 0, len(self.path_xy))
            else:
                start = max(0, self._last_idx - max(0, self.search_back))
                end = min(len(self.path_xy), self._last_idx + max(1, self.search_fwd))
                idx = _nearest_index(self.path_xy, robot_in_path, start, end)
            self._last_idx = idx

            win = _window_ahead(self.path_xy, idx, self.window_dist_m, self.min_sep_m, self.max_points)
            if self.path_frame != self.local_frame:
                win = _transform_xy_points(
                    win,
                    self._tf_buffer,
                    source_frame=self.path_frame,
                    target_frame=self.local_frame,
                    tf_timeout_s=self.tf_timeout_s,
                    stamp=rospy.Time(0),
                )

            msg = _build_path(self.local_frame, win)
            self._pub.publish(msg)
            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                continue


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_path = ws_dir / "src" / "pcd_gazebo_world" / "maps" / "runs" / "rosbag_path.json"

    parser = argparse.ArgumentParser(description="Publish a short via-points window for teb_local_planner")
    parser.add_argument("--path", type=str, default=str(default_path), help="路径 JSON（points: {x,y}）")
    parser.add_argument("--topic", type=str, default="/move_base_benchmark/TebLocalPlannerROS/via_points")
    parser.add_argument("--odom-topic", type=str, default="/odom")
    parser.add_argument("--path-frame", type=str, default="", help="路径 frame（空=读 JSON 的 frame_id）")
    parser.add_argument("--local-frame", type=str, default="odom", help="TEB/local_costmap 的 global_frame（通常 odom/odom_est）")
    parser.add_argument("--rate", type=float, default=5.0, help="发布频率 Hz")
    parser.add_argument("--window-dist", type=float, default=3.0, help="向前窗口长度（m）")
    parser.add_argument("--min-sep", type=float, default=0.5, help="窗口内点最小间距（m）")
    parser.add_argument("--max-points", type=int, default=12, help="窗口最多点数")
    parser.add_argument("--tf-timeout", type=float, default=0.5, help="TF 查询超时 (s)")
    parser.add_argument("--wait-tf", type=float, default=10.0, help="启动等待 TF 超时 (s)")
    parser.add_argument("--search-back", type=int, default=10, help="最近点搜索：允许回退的点数")
    parser.add_argument("--search-fwd", type=int, default=60, help="最近点搜索：允许前进的点数")
    args = parser.parse_args(rospy.myargv()[1:])

    path_file = Path(args.path).expanduser().resolve()
    if not path_file.is_file():
        raise SystemExit(f"path file not found: {path_file}")

    json_frame, xy = _load_path_xy(path_file)
    path_frame = str(args.path_frame).strip() or json_frame
    local_frame = str(args.local_frame).strip() or "odom"

    rospy.init_node("publish_via_points_window", anonymous=True)
    ViaPointsWindowPublisher(
        path_xy=xy,
        path_frame=path_frame,
        local_frame=local_frame,
        odom_topic=str(args.odom_topic),
        out_topic=str(args.topic),
        rate_hz=float(args.rate),
        window_dist_m=float(args.window_dist),
        min_sep_m=float(args.min_sep),
        max_points=int(args.max_points),
        tf_timeout_s=float(args.tf_timeout),
        wait_tf_s=float(args.wait_tf),
        search_back=int(args.search_back),
        search_fwd=int(args.search_fwd),
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
