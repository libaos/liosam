#!/usr/bin/env python3
"""Hybrid cmd_vel mux: straight uses PID, turning uses TEB.

This node never plans; it only selects which upstream controller's Twist to forward.

Typical setup:
- PID follower publishes to /cmd_vel_pid
- move_base + TEB publishes to /cmd_vel_teb
- This mux publishes the selected output to /cmd_vel (Gazebo subscribes to it)

Switching rule (AUTO mode):
- Compute the heading change of the reference path over a lookahead distance.
- If abs(delta_yaw) >= turn_enter_yaw -> select TEB.
- If abs(delta_yaw) <= turn_exit_yaw  -> select PID.
- Otherwise keep previous selection (hysteresis) and enforce min_hold_s.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path


XY = Tuple[float, float]


def _normalize_angle(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return float(a)


class HybridCmdVelMux:
    def __init__(self) -> None:
        self._path_xy: List[XY] = []
        self._nearest_idx: Optional[int] = None
        self._odom_xy: Optional[XY] = None

        self._pid_latest: Optional[Twist] = None
        self._pid_latest_t: Optional[float] = None
        self._teb_latest: Optional[Twist] = None
        self._teb_latest_t: Optional[float] = None

        self.mode = str(rospy.get_param("~mode", "auto")).strip().lower()  # auto|pid|teb
        if self.mode not in ("auto", "pid", "teb"):
            raise ValueError(f"invalid ~mode: {self.mode} (expected auto|pid|teb)")

        self.cmd_vel_out_topic = str(rospy.get_param("~cmd_vel_out_topic", "/cmd_vel"))
        self.pid_cmd_vel_topic = str(rospy.get_param("~pid_cmd_vel_topic", "/cmd_vel_pid"))
        self.teb_cmd_vel_topic = str(rospy.get_param("~teb_cmd_vel_topic", "/cmd_vel_teb"))
        self.path_topic = str(rospy.get_param("~path_topic", "/reference_path_odom"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))

        self.search_back_points = int(rospy.get_param("~search_back_points", 10))
        self.search_ahead_points = int(rospy.get_param("~search_ahead_points", 40))

        self.lookahead_dist_m = float(rospy.get_param("~lookahead_dist_m", 3.0))
        self.turn_enter_yaw = float(rospy.get_param("~turn_enter_yaw", 0.35))
        self.turn_exit_yaw = float(rospy.get_param("~turn_exit_yaw", 0.25))
        if self.turn_exit_yaw > self.turn_enter_yaw:
            rospy.logwarn("turn_exit_yaw > turn_enter_yaw; switching hysteresis may oscillate")

        self.min_hold_s = float(rospy.get_param("~min_hold_s", 1.0))
        self.source_timeout_s = float(rospy.get_param("~source_timeout_s", 0.5))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))

        # Selection state.
        self._selected = "pid" if self.mode != "teb" else "teb"
        self._last_switch_t = rospy.Time.now().to_sec()

        self._pub = rospy.Publisher(self.cmd_vel_out_topic, Twist, queue_size=1)
        rospy.Subscriber(self.pid_cmd_vel_topic, Twist, self._pid_cb, queue_size=5)
        rospy.Subscriber(self.teb_cmd_vel_topic, Twist, self._teb_cb, queue_size=5)
        rospy.Subscriber(self.path_topic, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)

        period_s = 1.0 / max(1.0, self.publish_rate_hz)
        self._timer = rospy.Timer(rospy.Duration(period_s), self._on_timer)

        rospy.loginfo(
            "[hybrid_mux] mode=%s out=%s pid=%s teb=%s path=%s odom=%s",
            self.mode,
            self.cmd_vel_out_topic,
            self.pid_cmd_vel_topic,
            self.teb_cmd_vel_topic,
            self.path_topic,
            self.odom_topic,
        )

    def _pid_cb(self, msg: Twist) -> None:
        self._pid_latest = msg
        self._pid_latest_t = rospy.Time.now().to_sec()

    def _teb_cb(self, msg: Twist) -> None:
        self._teb_latest = msg
        self._teb_latest_t = rospy.Time.now().to_sec()

    def _path_cb(self, msg: Path) -> None:
        pts: List[XY] = []
        for p in msg.poses:
            pts.append((float(p.pose.position.x), float(p.pose.position.y)))
        if len(pts) >= 2:
            self._path_xy = pts
            self._nearest_idx = None
        else:
            self._path_xy = []
            self._nearest_idx = None

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._odom_xy = (float(p.x), float(p.y))

    def _pick_nearest_idx(self, x: float, y: float) -> int:
        if not self._path_xy:
            return 0
        n = len(self._path_xy)

        search_back = max(0, int(self.search_back_points))
        search_ahead = max(1, int(self.search_ahead_points))
        start_idx = 0
        end_idx = n
        if self._nearest_idx is not None:
            start_idx = max(0, int(self._nearest_idx) - search_back)
            end_idx = min(n, int(self._nearest_idx) + search_ahead + 1)

        best_i = start_idx
        best_d = float("inf")
        for i in range(start_idx, end_idx):
            px, py = self._path_xy[i]
            d = math.hypot(px - float(x), py - float(y))
            if d < best_d:
                best_d = d
                best_i = i

        self._nearest_idx = int(best_i)
        return int(best_i)

    def _turn_metric_abs_delta_yaw(self, x: float, y: float) -> float:
        if len(self._path_xy) < 2:
            return 0.0
        idx = self._pick_nearest_idx(x, y)
        if idx >= len(self._path_xy) - 1:
            return 0.0

        ax, ay = self._path_xy[idx]
        bx, by = self._path_xy[idx + 1]
        yaw_now = math.atan2(by - ay, bx - ax)

        lookahead = max(0.1, float(self.lookahead_dist_m))
        dist = 0.0
        j = idx + 1
        while j < (len(self._path_xy) - 1) and dist < lookahead:
            x0, y0 = self._path_xy[j - 1]
            x1, y1 = self._path_xy[j]
            dist += math.hypot(x1 - x0, y1 - y0)
            j += 1
        j = max(idx + 1, min(j, len(self._path_xy) - 1))
        jx, jy = self._path_xy[j]

        yaw_ahead = math.atan2(jy - ay, jx - ax)
        delta = _normalize_angle(yaw_ahead - yaw_now)
        return abs(float(delta))

    def _maybe_switch(self, now_s: float) -> None:
        if self.mode in ("pid", "teb"):
            desired = self.mode
        else:
            if self._odom_xy is None:
                desired = self._selected
            else:
                x, y = self._odom_xy
                abs_dyaw = self._turn_metric_abs_delta_yaw(x, y)
                if abs_dyaw >= self.turn_enter_yaw:
                    desired = "teb"
                elif abs_dyaw <= self.turn_exit_yaw:
                    desired = "pid"
                else:
                    desired = self._selected

        if desired != self._selected and (now_s - self._last_switch_t) >= self.min_hold_s:
            rospy.loginfo("[hybrid_mux] switch %s -> %s", self._selected, desired)
            self._selected = desired
            self._last_switch_t = float(now_s)

    def _latest_selected(self) -> Tuple[Optional[Twist], Optional[float]]:
        if self._selected == "teb":
            return self._teb_latest, self._teb_latest_t
        return self._pid_latest, self._pid_latest_t

    def _on_timer(self, _evt) -> None:
        now_s = rospy.Time.now().to_sec()
        self._maybe_switch(now_s)

        msg, t = self._latest_selected()
        out = Twist()
        if msg is not None and t is not None and (now_s - float(t)) <= self.source_timeout_s:
            out.linear.x = float(msg.linear.x)
            out.linear.y = float(msg.linear.y)
            out.linear.z = float(msg.linear.z)
            out.angular.x = float(msg.angular.x)
            out.angular.y = float(msg.angular.y)
            out.angular.z = float(msg.angular.z)
        else:
            # Safety stop if the selected source is stale/unavailable.
            out.linear.x = 0.0
            out.angular.z = 0.0
        self._pub.publish(out)


def main() -> int:
    rospy.init_node("hybrid_cmd_vel_mux")
    HybridCmdVelMux()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

