#!/usr/bin/env python3
"""LQR-based path follower (yaw + lateral error) for differential drive robots."""

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


def _dlqr_2x2(
    a11: float,
    a12: float,
    a21: float,
    a22: float,
    b1: float,
    b2: float,
    q11: float,
    q22: float,
    r: float,
    *,
    iters: int = 25,
) -> Tuple[float, float]:
    # Discrete-time Riccati iteration for 2x2 A and 2x1 B.
    p11 = float(q11)
    p12 = 0.0
    p22 = float(q22)
    r = max(1.0e-9, float(r))

    for _ in range(max(1, int(iters))):
        # S = R + B^T P B (scalar)
        pb1 = p11 * b1 + p12 * b2
        pb2 = p12 * b1 + p22 * b2
        s = r + b1 * pb1 + b2 * pb2
        inv_s = 1.0 / max(1.0e-9, float(s))

        # K = inv(S) * B^T P A (1x2)
        # First compute P*A
        pa11 = p11 * a11 + p12 * a21
        pa12 = p11 * a12 + p12 * a22
        pa21 = p12 * a11 + p22 * a21
        pa22 = p12 * a12 + p22 * a22
        k1 = inv_s * (b1 * pa11 + b2 * pa21)
        k2 = inv_s * (b1 * pa12 + b2 * pa22)

        # P_next = A^T P A - A^T P B K + Q
        # Compute A^T * (P*A)
        atpa11 = a11 * pa11 + a21 * pa21
        atpa12 = a11 * pa12 + a21 * pa22
        atpa22 = a12 * pa12 + a22 * pa22

        # Compute A^T P B (2x1)
        apb1 = a11 * pb1 + a21 * pb2
        apb2 = a12 * pb1 + a22 * pb2

        # Subtract outer(APB, K)
        p11 = atpa11 - apb1 * k1 + q11
        p12 = atpa12 - apb1 * k2
        p22 = atpa22 - apb2 * k2 + q22

    # Final gain from converged P.
    pb1 = p11 * b1 + p12 * b2
    pb2 = p12 * b1 + p22 * b2
    s = r + b1 * pb1 + b2 * pb2
    inv_s = 1.0 / max(1.0e-9, float(s))

    pa11 = p11 * a11 + p12 * a21
    pa12 = p11 * a12 + p12 * a22
    pa21 = p12 * a11 + p22 * a21
    pa22 = p12 * a12 + p22 * a22

    k1 = inv_s * (b1 * pa11 + b2 * pa21)
    k2 = inv_s * (b1 * pa12 + b2 * pa22)
    return (float(k1), float(k2))


class LqrFollower:
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

        self.q_y = float(rospy.get_param("~q_y", 2.0))
        self.q_yaw = float(rospy.get_param("~q_yaw", 1.0))
        self.r_omega = float(rospy.get_param("~r_omega", 0.5))
        self.lqr_iters = int(rospy.get_param("~lqr_iters", 25))

        self.search_back_points = int(rospy.get_param("~search_back_points", 10))
        self.search_ahead_points = int(rospy.get_param("~search_ahead_points", 25))

        self.path_topic = str(rospy.get_param("~path_topic", "/reference_path"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.cmd_vel_topic = str(rospy.get_param("~cmd_vel_topic", "/cmd_vel"))

        self.last_time: Optional[float] = None
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

        desired_heading = math.atan2(dy, dx)
        # Yaw error is defined as "current - desired" so that positive omega reduces positive error.
        e_yaw = _normalize_angle(yaw - desired_heading)

        # Target point in robot frame: x_r forward, y_r left.
        c = float(math.cos(yaw))
        s = float(math.sin(yaw))
        y_r = -s * dx + c * dy

        # Define lateral error positive when robot is to the left of the reference (so turning right is negative).
        e_y = -float(y_r)

        gx, gy = self.goal if self.goal is not None else (tx, ty)
        dist_goal = float(math.hypot(float(gx) - x, float(gy) - y))

        v = min(float(self.v_max), float(self.k_dist) * dist_goal)
        v = max(float(self.v_min), v)
        v *= max(0.0, math.cos(e_yaw))

        now = rospy.Time.now().to_sec()
        if self.last_time is None:
            self.last_time = now
            return
        dt = max(1.0e-3, now - self.last_time)
        self.last_time = now

        v_nom = max(0.05, float(v))
        # Discrete-time error dynamics: [e_y, e_yaw]^T
        a11, a12 = 1.0, v_nom * dt
        a21, a22 = 0.0, 1.0
        b1, b2 = 0.0, dt
        k1, k2 = _dlqr_2x2(
            a11,
            a12,
            a21,
            a22,
            b1,
            b2,
            q11=float(self.q_y),
            q22=float(self.q_yaw),
            r=float(self.r_omega),
            iters=int(self.lqr_iters),
        )

        omega = -float(k1) * float(e_y) - float(k2) * float(e_yaw)
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
    rospy.init_node("lqr_path_follower")
    LqrFollower()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
