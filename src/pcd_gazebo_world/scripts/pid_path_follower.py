#!/usr/bin/env python3
"""Simple PID path follower for differential drive robots."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

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


class PidFollower:
    def __init__(self):
        self.path: List[Tuple[float, float]] = []
        self.goal: Optional[Tuple[float, float]] = None
        self.odom: Optional[Odometry] = None

        self.kp = float(rospy.get_param("~kp", 1.6))
        self.ki = float(rospy.get_param("~ki", 0.0))
        self.kd = float(rospy.get_param("~kd", 0.15))

        self.v_max = float(rospy.get_param("~v_max", 0.45))
        self.v_min = float(rospy.get_param("~v_min", 0.05))
        self.max_omega = float(rospy.get_param("~max_omega", 0.8))

        self.lookahead = float(rospy.get_param("~lookahead", 0.8))
        self.k_dist = float(rospy.get_param("~k_dist", 0.8))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.4))
        self.stop_at_goal = bool(int(rospy.get_param("~stop_at_goal", 1)))

        self.prev_err = 0.0
        self.err_i = 0.0
        self.last_time: Optional[float] = None
        self.reached = False

        self.path_topic = str(rospy.get_param("~path_topic", "/reference_path"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.cmd_vel_topic = str(rospy.get_param("~cmd_vel_topic", "/cmd_vel"))

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

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom = msg

    def _pick_target(self, x: float, y: float) -> Tuple[float, float, float]:
        if not self.path:
            return x, y, 0.0
        # Find nearest point (small path, so full scan is OK)
        nearest_idx = 0
        nearest_d = float("inf")
        for i, (px, py) in enumerate(self.path):
            d = math.hypot(px - x, py - y)
            if d < nearest_d:
                nearest_d = d
                nearest_idx = i

        target_idx = nearest_idx
        for j in range(nearest_idx, len(self.path)):
            if math.hypot(self.path[j][0] - x, self.path[j][1] - y) >= self.lookahead:
                target_idx = j
                break

        tx, ty = self.path[target_idx]
        heading = math.atan2(ty - y, tx - x)
        return tx, ty, heading

    def _on_timer(self, _evt) -> None:
        if self.odom is None or not self.path or self.reached:
            return

        pose = self.odom.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        yaw = _yaw_from_quat(pose.orientation)

        tx, ty, heading = self._pick_target(x, y)
        heading_err = _normalize_angle(heading - yaw)

        now = rospy.Time.now().to_sec()
        if self.last_time is None:
            self.last_time = now
            return
        dt = max(1.0e-3, now - self.last_time)
        self.last_time = now

        self.err_i = max(min(self.err_i + heading_err * dt, 1.5), -1.5)
        d_err = (heading_err - self.prev_err) / dt
        self.prev_err = heading_err

        omega = self.kp * heading_err + self.ki * self.err_i + self.kd * d_err
        omega = max(min(omega, self.max_omega), -self.max_omega)

        gx, gy = self.goal if self.goal is not None else (tx, ty)
        dist_goal = math.hypot(gx - x, gy - y)
        v = min(self.v_max, self.k_dist * dist_goal)
        v = max(self.v_min, v)
        v *= max(0.0, math.cos(heading_err))

        if dist_goal <= self.goal_tolerance:
            v = 0.0
            omega = 0.0
            if self.stop_at_goal:
                self.reached = True

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)


def main() -> int:
    rospy.init_node("pid_path_follower")
    PidFollower()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
