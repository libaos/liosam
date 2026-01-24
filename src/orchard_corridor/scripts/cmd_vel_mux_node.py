#!/usr/bin/env python3
from __future__ import annotations

import math
import threading
from typing import List, Optional, Tuple

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, String


XY = Tuple[float, float]


def _clamp(v: float, v_min: float, v_max: float) -> float:
    return max(v_min, min(v_max, v))


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    return False


def _point_to_segment_dist2(p: XY, a: XY, b: XY) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    vv = vx * vx + vy * vy
    if vv <= 1.0e-12:
        return wx * wx + wy * wy
    t = (wx * vx + wy * vy) / vv
    if t <= 0.0:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    if t >= 1.0:
        dx = px - bx
        dy = py - by
        return dx * dx + dy * dy
    projx = ax + t * vx
    projy = ay + t * vy
    dx = px - projx
    dy = py - projy
    return dx * dx + dy * dy


def _nearest_dist_to_path(p: XY, path_xy: List[XY]) -> float:
    if len(path_xy) < 2:
        return float("inf")
    best = float("inf")
    for a, b in zip(path_xy, path_xy[1:]):
        d2 = _point_to_segment_dist2(p, a, b)
        if d2 < best:
            best = d2
    return float(math.sqrt(best))


def _nearest_dist_to_path_window(p: XY, path_xy: List[XY], start_idx: int, end_idx: int) -> Tuple[float, int]:
    """Return (min_dist, best_segment_start_idx) within [start_idx, end_idx) points range."""
    n = len(path_xy)
    if n < 2:
        return (float("inf"), 0)
    start_idx = max(0, min(int(start_idx), n - 1))
    end_idx = max(start_idx + 1, min(int(end_idx), n))

    best_d2 = float("inf")
    best_i = start_idx
    for i in range(start_idx, end_idx - 1):
        d2 = _point_to_segment_dist2(p, path_xy[i], path_xy[i + 1])
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return (float(math.sqrt(best_d2)), int(best_i))


def _yaw_from_quat(q) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _normalize_angle(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return float(a)


class CmdVelMuxNode:
    def __init__(self) -> None:
        self.cmd_out_topic = str(rospy.get_param("~cmd_out_topic", "/cmd_vel"))
        self.cmd_corridor_topic = str(rospy.get_param("~cmd_corridor_topic", "/cmd_vel_corridor"))
        self.cmd_pid_topic = str(rospy.get_param("~cmd_pid_topic", "/cmd_vel_pid"))

        self.corridor_path_topic = str(rospy.get_param("~corridor_path_topic", "/corridor_centerline"))
        self.corridor_quality_topic = str(rospy.get_param("~corridor_quality_topic", "/corridor_quality"))
        self.corridor_clearance_topic = str(rospy.get_param("~corridor_clearance_topic", "/corridor_clearance_min"))

        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.ref_path_topic = str(rospy.get_param("~ref_path_topic", "/reference_path_odom"))

        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.cmd_timeout = float(rospy.get_param("~cmd_timeout", 0.5))
        self.corridor_path_timeout = float(rospy.get_param("~corridor_path_timeout", 0.6))
        self.corridor_quality_timeout = float(rospy.get_param("~corridor_quality_timeout", 1.0))
        self.min_mode_dwell_s = float(rospy.get_param("~min_mode_dwell_s", 3.0))
        # Split dwell times for mode switches. When `min_mode_dwell_s` is large, it can prevent
        # exiting corridor quickly enough once it starts diverging. Keep PID->corridor dwell large
        # (avoid flapping), but allow corridor->PID to be smaller (fast safety exit).
        self.min_pid_to_corridor_dwell_s = float(
            rospy.get_param("~min_pid_to_corridor_dwell_s", self.min_mode_dwell_s)
        )
        self.min_corridor_to_pid_dwell_s = float(
            rospy.get_param("~min_corridor_to_pid_dwell_s", self.min_mode_dwell_s)
        )
        # Smooth mode switches by blending previous and new commands for a short duration.
        # Set to 0 to disable.
        self.transition_s = float(rospy.get_param("~transition_s", 0.4))
        # Optional output smoothing / rate limiting (applied after mode selection).
        self.out_filter_tau = float(rospy.get_param("~out_filter_tau", 0.0))
        self.out_max_linear_accel = float(rospy.get_param("~out_max_linear_accel", 0.0))
        self.out_max_angular_accel = float(rospy.get_param("~out_max_angular_accel", 0.0))

        self.initial_mode = str(rospy.get_param("~initial_mode", "pid")).strip().lower()
        if self.initial_mode not in ("pid", "corridor"):
            self.initial_mode = "pid"

        # Defaults bias toward PID (reference follower). Corridor controller is best used as a
        # "local rescue" when the centerline is confident AND we are already close to the
        # reference, otherwise it can pull the robot away at self-intersections/forks.
        self.q_enter = float(rospy.get_param("~q_enter", 0.95))
        self.q_exit = float(rospy.get_param("~q_exit", 0.90))
        self.enter_hold_s = float(rospy.get_param("~enter_hold_s", 1.0))
        self.exit_hold_s = float(rospy.get_param("~exit_hold_s", 0.2))

        self.use_ref_guard = _to_bool(rospy.get_param("~use_ref_guard", True))
        self.ref_err_enter = float(rospy.get_param("~ref_err_enter", 0.5))
        self.ref_err_exit = float(rospy.get_param("~ref_err_exit", 0.9))

        self.ref_err_hard = float(rospy.get_param("~ref_err_hard", 3.0))
        # For self-intersecting reference paths (e.g. figure-8), a global nearest-distance can
        # incorrectly stay "small" when the robot jumps onto a different branch at the crossing.
        # Track a local nearest segment index and restrict distance computation to a sliding window
        # to enforce forward progress.
        self.ref_search_back_points = int(rospy.get_param("~ref_search_back_points", 10))
        self.ref_search_ahead_points = int(rospy.get_param("~ref_search_ahead_points", 60))
        self.ref_reacquire_dist = float(rospy.get_param("~ref_reacquire_dist", 5.0))

        self.use_heading_guard = _to_bool(rospy.get_param("~use_heading_guard", True))
        self.heading_enter = float(rospy.get_param("~heading_enter", 0.5))
        self.heading_exit = float(rospy.get_param("~heading_exit", 0.8))
        self.heading_hard = float(rospy.get_param("~heading_hard", 2.2))

        # Additional guard: compare corridor path direction (in base frame) against reference path
        # direction (also expressed in base frame). This helps avoid switching into corridor when the
        # centerline points toward a wrong branch at self-intersections/forks.
        self.use_corridor_path_heading_guard = _to_bool(rospy.get_param("~use_corridor_path_heading_guard", False))
        self.corridor_path_heading_lookahead_m = float(rospy.get_param("~corridor_path_heading_lookahead_m", 1.0))
        self.corridor_path_heading_enter = float(rospy.get_param("~corridor_path_heading_enter", 0.35))
        self.corridor_path_heading_exit = float(rospy.get_param("~corridor_path_heading_exit", 0.70))

        self._mode = self.initial_mode
        self._candidate_since: Optional[rospy.Time] = None
        self._bad_since: Optional[rospy.Time] = None
        self._last_switch_time = rospy.Time(0)

        self._last_out_cmd: Twist = Twist()
        self._last_out_time = rospy.Time(0)
        self._transition_start = rospy.Time(0)
        self._transition_from_cmd: Optional[Twist] = None

        self._last_cmd_corridor: Optional[Twist] = None
        self._last_cmd_corridor_time = rospy.Time(0)
        self._last_cmd_pid: Optional[Twist] = None
        self._last_cmd_pid_time = rospy.Time(0)

        self._last_quality: Optional[float] = None
        self._last_quality_time = rospy.Time(0)
        self._last_path: Optional[Path] = None
        self._last_path_time = rospy.Time(0)
        self._last_clearance: Optional[float] = None
        self._last_clearance_time = rospy.Time(0)

        self._last_odom: Optional[Odometry] = None
        self._last_odom_time = rospy.Time(0)
        self._ref_path_xy: List[XY] = []
        self._ref_nearest_seg_idx: Optional[int] = None

        self._pub_cmd = rospy.Publisher(self.cmd_out_topic, Twist, queue_size=1)
        self._pub_mode = rospy.Publisher("~mode", String, queue_size=1)

        rospy.Subscriber(self.cmd_corridor_topic, Twist, self._cmd_corridor_cb, queue_size=1)
        rospy.Subscriber(self.cmd_pid_topic, Twist, self._cmd_pid_cb, queue_size=1)

        rospy.Subscriber(self.corridor_path_topic, Path, self._corridor_path_cb, queue_size=1)
        rospy.Subscriber(self.corridor_quality_topic, Float32, self._corridor_quality_cb, queue_size=1)
        rospy.Subscriber(self.corridor_clearance_topic, Float32, self._corridor_clearance_cb, queue_size=1)

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber(self.ref_path_topic, Path, self._ref_path_cb, queue_size=1)

        period = 1.0 / max(self.publish_rate, 1e-3)
        rospy.Timer(rospy.Duration.from_sec(period), self._on_timer)

        rospy.loginfo(
            "[cmd_vel_mux] out=%s corridor=%s pid=%s initial=%s q=[%.2f->%.2f] ref_guard=%s heading_guard=%s dwell=[pid->corridor %.1fs, corridor->pid %.1fs] transition=%.2fs out_tau=%.2fs out_accel=[%.2f,%.2f]",
            self.cmd_out_topic,
            self.cmd_corridor_topic,
            self.cmd_pid_topic,
            self._mode,
            self.q_exit,
            self.q_enter,
            self.use_ref_guard,
            self.use_heading_guard,
            self.min_pid_to_corridor_dwell_s,
            self.min_corridor_to_pid_dwell_s,
            self.transition_s,
            self.out_filter_tau,
            self.out_max_linear_accel,
            self.out_max_angular_accel,
        )

    def _is_stop_cmd(self, cmd: Twist) -> bool:
        return (
            abs(float(cmd.linear.x)) < 1.0e-4
            and abs(float(cmd.linear.y)) < 1.0e-4
            and abs(float(cmd.linear.z)) < 1.0e-4
            and abs(float(cmd.angular.x)) < 1.0e-4
            and abs(float(cmd.angular.y)) < 1.0e-4
            and abs(float(cmd.angular.z)) < 1.0e-4
        )

    def _copy_twist(self, src: Twist) -> Twist:
        out = Twist()
        out.linear.x = float(src.linear.x)
        out.linear.y = float(src.linear.y)
        out.linear.z = float(src.linear.z)
        out.angular.x = float(src.angular.x)
        out.angular.y = float(src.angular.y)
        out.angular.z = float(src.angular.z)
        return out

    def _blend_twist(self, a: Twist, b: Twist, t: float) -> Twist:
        tt = _clamp(float(t), 0.0, 1.0)
        out = Twist()
        out.linear.x = float((1.0 - tt) * float(a.linear.x) + tt * float(b.linear.x))
        out.linear.y = float((1.0 - tt) * float(a.linear.y) + tt * float(b.linear.y))
        out.linear.z = float((1.0 - tt) * float(a.linear.z) + tt * float(b.linear.z))
        out.angular.x = float((1.0 - tt) * float(a.angular.x) + tt * float(b.angular.x))
        out.angular.y = float((1.0 - tt) * float(a.angular.y) + tt * float(b.angular.y))
        out.angular.z = float((1.0 - tt) * float(a.angular.z) + tt * float(b.angular.z))
        return out

    def _apply_output_filter(self, now: rospy.Time, desired: Twist, *, bypass: bool = False) -> Twist:
        if bypass or self._is_stop_cmd(desired):
            self._last_out_time = now
            self._last_out_cmd = self._copy_twist(desired)
            return desired

        if self._last_out_time == rospy.Time(0):
            self._last_out_time = now
            self._last_out_cmd = self._copy_twist(desired)
            return desired

        dt = float((now - self._last_out_time).to_sec())
        dt = max(1.0e-3, dt)

        v_des = float(desired.linear.x)
        w_des = float(desired.angular.z)
        v_prev = float(self._last_out_cmd.linear.x)
        w_prev = float(self._last_out_cmd.angular.z)

        if self.out_max_linear_accel > 0.0:
            dv_max = float(self.out_max_linear_accel) * dt
            v_des = float(_clamp(v_des, v_prev - dv_max, v_prev + dv_max))

        if self.out_max_angular_accel > 0.0:
            dw_max = float(self.out_max_angular_accel) * dt
            w_des = float(_clamp(w_des, w_prev - dw_max, w_prev + dw_max))

        if self.out_filter_tau > 0.0:
            alpha = dt / (float(self.out_filter_tau) + dt)
            v_des = float(v_prev + alpha * (v_des - v_prev))
            w_des = float(w_prev + alpha * (w_des - w_prev))

        out = self._copy_twist(desired)
        out.linear.x = float(v_des)
        out.angular.z = float(w_des)

        self._last_out_time = now
        self._last_out_cmd = self._copy_twist(out)
        return out

    def _cmd_corridor_cb(self, msg: Twist) -> None:
        self._last_cmd_corridor = msg
        self._last_cmd_corridor_time = rospy.Time.now()

    def _cmd_pid_cb(self, msg: Twist) -> None:
        self._last_cmd_pid = msg
        self._last_cmd_pid_time = rospy.Time.now()

    def _corridor_path_cb(self, msg: Path) -> None:
        self._last_path = msg
        self._last_path_time = rospy.Time.now()

    def _corridor_quality_cb(self, msg: Float32) -> None:
        self._last_quality = float(msg.data)
        self._last_quality_time = rospy.Time.now()

    def _corridor_clearance_cb(self, msg: Float32) -> None:
        self._last_clearance = float(msg.data)
        self._last_clearance_time = rospy.Time.now()

    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_time = rospy.Time.now()

    def _ref_path_cb(self, msg: Path) -> None:
        xy: List[XY] = []
        for pose in msg.poses:
            p = pose.pose.position
            xy.append((float(p.x), float(p.y)))
        self._ref_path_xy = xy
        self._ref_nearest_seg_idx = None

    def _corridor_quality(self, now: rospy.Time) -> Optional[float]:
        if self._last_quality is None or self._last_quality_time == rospy.Time(0):
            return None
        if self.corridor_quality_timeout > 0.0 and (now - self._last_quality_time).to_sec() > self.corridor_quality_timeout:
            return None
        return float(self._last_quality)

    def _corridor_path_ok(self, now: rospy.Time) -> bool:
        if self._last_path is None or self._last_path_time == rospy.Time(0):
            return False
        if self.corridor_path_timeout > 0.0 and (now - self._last_path_time).to_sec() > self.corridor_path_timeout:
            return False
        return len(self._last_path.poses) >= 3

    def _ref_error(self) -> Optional[float]:
        if not self._ref_path_xy:
            return None
        if self._last_odom is None:
            return None
        p = self._last_odom.pose.pose.position
        xy = (float(p.x), float(p.y))
        n = len(self._ref_path_xy)
        if n < 2:
            return None

        if self._ref_nearest_seg_idx is None:
            dist, seg_idx = _nearest_dist_to_path_window(xy, self._ref_path_xy, 0, n)
            self._ref_nearest_seg_idx = seg_idx
            return float(dist)

        back = max(0, int(self.ref_search_back_points))
        ahead = max(1, int(self.ref_search_ahead_points))
        start = max(0, int(self._ref_nearest_seg_idx) - back)
        end = min(n, int(self._ref_nearest_seg_idx) + ahead + 2)
        dist, seg_idx = _nearest_dist_to_path_window(xy, self._ref_path_xy, start, end)

        if float(dist) > float(self.ref_reacquire_dist):
            dist2, seg_idx2 = _nearest_dist_to_path_window(xy, self._ref_path_xy, 0, n)
            if float(dist2) < float(dist):
                dist, seg_idx = dist2, seg_idx2

        self._ref_nearest_seg_idx = seg_idx
        return float(dist)

    def _ref_heading(self) -> Optional[float]:
        if not self._ref_path_xy or len(self._ref_path_xy) < 2:
            return None
        if self._ref_nearest_seg_idx is None:
            return None
        i = int(self._ref_nearest_seg_idx)
        i = max(0, min(i, len(self._ref_path_xy) - 2))
        ax, ay = self._ref_path_xy[i]
        bx, by = self._ref_path_xy[i + 1]
        return float(math.atan2(float(by - ay), float(bx - ax)))

    def _ref_heading_in_base(self) -> Optional[float]:
        if self._last_odom is None:
            return None
        yaw_ref = self._ref_heading()
        if yaw_ref is None:
            return None
        yaw = _yaw_from_quat(self._last_odom.pose.pose.orientation)
        return float(_normalize_angle(float(yaw_ref) - float(yaw)))

    def _heading_error(self) -> Optional[float]:
        if self._last_odom is None:
            return None
        yaw_ref = self._ref_heading()
        if yaw_ref is None:
            return None
        yaw = _yaw_from_quat(self._last_odom.pose.pose.orientation)
        return abs(_normalize_angle(float(yaw_ref) - float(yaw)))

    def _corridor_path_heading(self, now: rospy.Time) -> Optional[float]:
        if not self._corridor_path_ok(now):
            return None
        if self._last_path is None or len(self._last_path.poses) < 2:
            return None
        lookahead = max(0.05, float(self.corridor_path_heading_lookahead_m))
        x0 = float(self._last_path.poses[0].pose.position.x)
        y0 = float(self._last_path.poses[0].pose.position.y)
        for pose in self._last_path.poses[1:]:
            x = float(pose.pose.position.x)
            y = float(pose.pose.position.y)
            dx = x - x0
            dy = y - y0
            if math.hypot(dx, dy) >= lookahead:
                return float(math.atan2(dy, dx))
        x1 = float(self._last_path.poses[-1].pose.position.x)
        y1 = float(self._last_path.poses[-1].pose.position.y)
        dx = x1 - x0
        dy = y1 - y0
        if math.hypot(dx, dy) < 1e-6:
            return None
        return float(math.atan2(dy, dx))

    def _corridor_path_heading_error(self, now: rospy.Time) -> Optional[float]:
        yaw_corr = self._corridor_path_heading(now)
        yaw_ref_base = self._ref_heading_in_base()
        if yaw_corr is None or yaw_ref_base is None:
            return None
        return abs(_normalize_angle(float(yaw_corr) - float(yaw_ref_base)))

    def _corridor_ok(self, now: rospy.Time, q_enter: float) -> bool:
        if not self._corridor_path_ok(now):
            return False
        q = self._corridor_quality(now)
        if q is None or float(q) < float(q_enter):
            return False
        if self.use_ref_guard:
            err = self._ref_error()
            if err is not None and float(err) > float(self.ref_err_enter):
                return False
        if self.use_heading_guard:
            yaw_err = self._heading_error()
            if yaw_err is not None and float(yaw_err) > float(self.heading_enter):
                return False
        if self.use_corridor_path_heading_guard:
            path_yaw_err = self._corridor_path_heading_error(now)
            if path_yaw_err is not None and float(path_yaw_err) > float(self.corridor_path_heading_enter):
                return False
        return True

    def _corridor_bad(self, now: rospy.Time, q_exit: float) -> bool:
        if not self._corridor_path_ok(now):
            return True
        q = self._corridor_quality(now)
        if q is None or float(q) <= float(q_exit):
            return True
        if self.use_ref_guard:
            err = self._ref_error()
            if err is not None and float(err) >= float(self.ref_err_exit):
                return True
        if self.use_heading_guard:
            yaw_err = self._heading_error()
            if yaw_err is not None and float(yaw_err) >= float(self.heading_exit):
                return True
        if self.use_corridor_path_heading_guard:
            path_yaw_err = self._corridor_path_heading_error(now)
            if path_yaw_err is not None and float(path_yaw_err) >= float(self.corridor_path_heading_exit):
                return True
        return False

    def _get_cmd(self, now: rospy.Time, mode: str) -> Optional[Twist]:
        if mode == "corridor":
            if self._last_cmd_corridor is None:
                return None
            if self.cmd_timeout > 0.0 and (now - self._last_cmd_corridor_time).to_sec() > self.cmd_timeout:
                return None
            return self._last_cmd_corridor
        if mode == "pid":
            if self._last_cmd_pid is None:
                return None
            if self.cmd_timeout > 0.0 and (now - self._last_cmd_pid_time).to_sec() > self.cmd_timeout:
                return None
            return self._last_cmd_pid
        return None

    def _switch_to(self, now: rospy.Time, mode: str, reason: str, *, force: bool = False) -> None:
        if mode == self._mode:
            return
        dwell_s = self.min_mode_dwell_s
        if self._mode == "pid" and mode == "corridor":
            dwell_s = self.min_pid_to_corridor_dwell_s
        elif self._mode == "corridor" and mode == "pid":
            dwell_s = self.min_corridor_to_pid_dwell_s
        if not force and float(dwell_s) > 0.0 and self._last_switch_time != rospy.Time(0):
            if (now - self._last_switch_time).to_sec() < float(dwell_s):
                return

        # Prepare transition blending from last published cmd to the new mode cmd.
        if not force and self.transition_s > 0.0:
            self._transition_start = now
            self._transition_from_cmd = self._copy_twist(self._last_out_cmd)
        else:
            self._transition_start = rospy.Time(0)
            self._transition_from_cmd = None

        self._mode = mode
        self._candidate_since = None
        self._bad_since = None
        self._last_switch_time = now
        q = self._corridor_quality(now)
        err = self._ref_error() if self.use_ref_guard else None
        yaw_err = self._heading_error() if self.use_heading_guard else None
        rospy.loginfo(
            "[cmd_vel_mux] switch -> %s (%s q=%s err=%s yaw=%s)",
            mode,
            reason,
            "na" if q is None else f"{q:.2f}",
            "na" if err is None else f"{err:.2f}",
            "na" if yaw_err is None else f"{yaw_err:.2f}",
        )
        self._pub_mode.publish(String(data=f"{mode}:{reason}"))

    def _maybe_update_mode(self, now: rospy.Time) -> None:
        err = self._ref_error()
        if self.use_ref_guard and err is not None and float(err) >= float(self.ref_err_hard):
            self._switch_to(now, "pid", f"ref_err_hard({err:.2f}m)", force=True)
            return
        yaw_err = self._heading_error()
        if self.use_heading_guard and yaw_err is not None and float(yaw_err) >= float(self.heading_hard):
            self._switch_to(now, "pid", f"yaw_hard({yaw_err:.2f}rad)", force=True)
            return

        if self._mode == "pid":
            if self._corridor_ok(now, self.q_enter):
                if self._candidate_since is None:
                    self._candidate_since = now
                if (now - self._candidate_since).to_sec() >= self.enter_hold_s:
                    self._switch_to(now, "corridor", "corridor_ok")
            else:
                self._candidate_since = None
            return

        if self._mode == "corridor":
            if self._corridor_bad(now, self.q_exit):
                if self._bad_since is None:
                    self._bad_since = now
                if (now - self._bad_since).to_sec() >= self.exit_hold_s:
                    self._switch_to(now, "pid", "corridor_bad")
            else:
                self._bad_since = None

    def _on_timer(self, _evt) -> None:
        now = rospy.Time.now()
        self._maybe_update_mode(now)

        used_fallback = False
        cmd = self._get_cmd(now, self._mode)
        if cmd is None:
            alt = "pid" if self._mode == "corridor" else "corridor"
            cmd = self._get_cmd(now, alt)
            if cmd is None:
                cmd = Twist()
                used_fallback = True
            else:
                rospy.logwarn_throttle(2.0, "[cmd_vel_mux] %s cmd stale; fallback to %s", self._mode, alt)
                used_fallback = True

        out_cmd = cmd
        if (
            self.transition_s > 0.0
            and self._transition_from_cmd is not None
            and self._transition_start != rospy.Time(0)
        ):
            t = float((now - self._transition_start).to_sec())
            if t < float(self.transition_s):
                out_cmd = self._blend_twist(self._transition_from_cmd, cmd, t / float(self.transition_s))
            else:
                self._transition_start = rospy.Time(0)
                self._transition_from_cmd = None

        if self.out_filter_tau > 0.0 or self.out_max_linear_accel > 0.0 or self.out_max_angular_accel > 0.0:
            out_cmd = self._apply_output_filter(now, out_cmd, bypass=used_fallback)
        else:
            self._last_out_time = now
            self._last_out_cmd = self._copy_twist(out_cmd)

        self._pub_cmd.publish(out_cmd)


def main() -> None:
    rospy.init_node("cmd_vel_mux")
    _ = CmdVelMuxNode()
    rospy.spin()


if __name__ == "__main__":
    main()
