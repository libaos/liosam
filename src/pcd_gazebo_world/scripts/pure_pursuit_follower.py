#!/usr/bin/env python3
"""Pure pursuit path follower for differential drive robots."""

from __future__ import annotations

import math
import threading
from typing import List, Optional, Tuple

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion


def _yaw_from_quat(q) -> float:
    return float(euler_from_quaternion([q.x, q.y, q.z, q.w])[2])


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PurePursuitFollower:
    def __init__(self):
        self.path: List[Tuple[float, float]] = []
        self.goal: Optional[Tuple[float, float]] = None
        self.odom: Optional[Odometry] = None
        self._nearest_idx: Optional[int] = None

        self.v_max = float(rospy.get_param("~v_max", 0.45))
        self.v_min = float(rospy.get_param("~v_min", 0.05))
        self.max_omega = float(rospy.get_param("~max_omega", 0.8))

        self.lookahead = float(rospy.get_param("~lookahead", 0.8))
        self.k_dist = float(rospy.get_param("~k_dist", 0.8))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.4))
        self.stop_at_goal = bool(int(rospy.get_param("~stop_at_goal", 1)))

        # Keep the nearest search within a sliding window around the last matched index to
        # enforce forward progress on self-crossing paths (e.g. figure-8).
        self.search_back_points = int(rospy.get_param("~search_back_points", 10))
        self.search_ahead_points = int(rospy.get_param("~search_ahead_points", 25))

        self.path_topic = str(rospy.get_param("~path_topic", "/reference_path"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.cmd_vel_topic = str(rospy.get_param("~cmd_vel_topic", "/cmd_vel"))

        self.reached = False

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        rospy.Subscriber(self.path_topic, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.05), self._on_timer)

    def _path_cb(self, msg: Path) -> None:
        pts = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]
        if pts:
            self.path = pts
            self.goal = pts[-1]
            self.reached = False
            self._nearest_idx = None

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom = msg

    def _pick_target(self, x: float, y: float) -> Tuple[float, float]:
        if not self.path:
            return x, y

        search_back = max(0, int(self.search_back_points))
        search_ahead = max(1, int(self.search_ahead_points))
        start_idx = 0
        end_idx = len(self.path)
        if self._nearest_idx is not None:
            start_idx = max(0, int(self._nearest_idx) - search_back)
            end_idx = min(len(self.path), int(self._nearest_idx) + search_ahead + 1)

        nearest_idx = start_idx
        nearest_d = float("inf")
        for i in range(start_idx, end_idx):
            px, py = self.path[i]
            d = math.hypot(px - x, py - y)
            if d < nearest_d:
                nearest_d = d
                nearest_idx = i
        self._nearest_idx = nearest_idx

        target_idx = nearest_idx
        lookahead = max(0.05, float(self.lookahead))
        for j in range(nearest_idx, min(len(self.path), nearest_idx + search_ahead + 1)):
            if math.hypot(self.path[j][0] - x, self.path[j][1] - y) >= lookahead:
                target_idx = j
                break

        return self.path[target_idx]

    def _on_timer(self, _evt) -> None:
        if self.odom is None or not self.path or self.reached:
            return

        pose = self.odom.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        yaw = _yaw_from_quat(pose.orientation)

        tx, ty = self._pick_target(x, y)
        dx = float(tx - x)
        dy = float(ty - y)

        heading = math.atan2(dy, dx)
        heading_err = _normalize_angle(heading - yaw)

        gx, gy = self.goal if self.goal is not None else (tx, ty)
        dist_goal = float(math.hypot(float(gx) - x, float(gy) - y))

        v = min(float(self.v_max), float(self.k_dist) * dist_goal)
        v = max(float(self.v_min), v)
        v *= max(0.0, math.cos(heading_err))

        # Target point in robot frame.
        c = float(math.cos(yaw))
        s = float(math.sin(yaw))
        x_r = c * dx + s * dy
        y_r = -s * dx + c * dy
        L = float(math.hypot(x_r, y_r))

        omega = 0.0
        if L > 1.0e-3:
            curvature = 2.0 * y_r / (L * L)
            omega = float(v) * float(curvature)
        omega = max(min(omega, float(self.max_omega)), -float(self.max_omega))

        if dist_goal <= float(self.goal_tolerance):
            v = 0.0
            omega = 0.0
            if self.stop_at_goal:
                self.reached = True

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)


def main() -> int:
    rospy.init_node("pure_pursuit_follower")
    PurePursuitFollower()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

