#!/usr/bin/env python3
import math
import threading
from typing import List, Optional, Tuple

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Float32, Int32
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from visualization_msgs.msg import Marker, MarkerArray


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


def _scale_factor(value: float, stop: float, full: float, min_factor: float = 0.0, max_factor: float = 1.0) -> float:
    if math.isnan(value):
        return float(min_factor)
    if value == float("inf"):
        return float(max_factor)
    if value == float("-inf"):
        return float(min_factor)
    if full <= stop:
        return float(max_factor if value >= full else min_factor)
    if value <= stop:
        return float(min_factor)
    if value >= full:
        return float(max_factor)
    t = float((value - stop) / (full - stop))
    return float(min_factor + (max_factor - min_factor) * t)


class CorridorControllerNode:
    def __init__(self) -> None:
        self.path_topic = rospy.get_param("~path_topic", "/corridor_centerline")
        self.quality_topic = rospy.get_param("~quality_topic", "/corridor_quality")
        self.clearance_topic = rospy.get_param("~clearance_topic", "/corridor_clearance_min")
        self.route_id_topic = rospy.get_param("~route_id_topic", "/route_id_stable")
        self.route_valid_topic = rospy.get_param("~route_valid_topic", "/route_id_valid")

        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.markers_topic = rospy.get_param("~markers_topic", "/corridor_controller_markers")
        self.default_frame = str(rospy.get_param("~default_frame", "base_link_est"))

        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.max_path_age = float(rospy.get_param("~max_path_age", 0.5))
        self.require_quality = _to_bool(rospy.get_param("~require_quality", False))
        self.max_quality_age = float(rospy.get_param("~max_quality_age", 1.0))
        self.require_clearance = _to_bool(rospy.get_param("~require_clearance", False))
        self.max_clearance_age = float(rospy.get_param("~max_clearance_age", 1.0))
        self.require_route_valid = _to_bool(rospy.get_param("~require_route_valid", False))
        self.max_route_age = float(rospy.get_param("~max_route_age", 1.0))

        self.enabled = _to_bool(rospy.get_param("~enabled", False))
        self.enable_ramp_time = float(rospy.get_param("~enable_ramp_time", 0.5))
        self.disable_brake_pulses = int(rospy.get_param("~disable_brake_pulses", 5))

        self.lookahead = float(rospy.get_param("~lookahead", 0.8))
        self.min_target_x = float(rospy.get_param("~min_target_x", 0.15))
        self.min_points = int(rospy.get_param("~min_points", 3))

        self.k_pursuit = float(rospy.get_param("~k_pursuit", 1.0))
        self.v_max = float(rospy.get_param("~v_max", 0.35))
        self.max_omega = float(rospy.get_param("~max_omega", 0.8))

        # Command smoothing / rate limiting.
        # These are applied only in TRACK state (i.e. when we have a valid target).
        self.cmd_filter_tau = float(rospy.get_param("~cmd_filter_tau", 0.0))
        self.max_linear_accel = float(rospy.get_param("~max_linear_accel", 0.0))
        self.max_angular_accel = float(rospy.get_param("~max_angular_accel", 0.0))

        self.q_stop = float(rospy.get_param("~q_stop", 0.3))
        self.q_full = float(rospy.get_param("~q_full", 0.8))
        self.v_low_q = float(rospy.get_param("~v_low_q", 0.0))

        self.clearance_stop = float(rospy.get_param("~clearance_stop", 0.25))
        self.clearance_full = float(rospy.get_param("~clearance_full", 0.6))

        self.marker_z = float(rospy.get_param("~marker_z", 0.15))
        self.marker_lifetime = float(rospy.get_param("~marker_lifetime", 0.2))
        self.log_interval = float(rospy.get_param("~log_interval", 2.0))

        self._last_path: Optional[Path] = None
        self._last_path_stamp: Optional[rospy.Time] = None

        self._last_quality: Optional[float] = None
        self._last_quality_time: Optional[rospy.Time] = None

        self._last_clearance: Optional[float] = None
        self._last_clearance_time: Optional[rospy.Time] = None

        self._last_route_id: Optional[int] = None
        self._last_route_id_time: Optional[rospy.Time] = None
        self._last_route_valid: Optional[bool] = None
        self._last_route_valid_time: Optional[rospy.Time] = None

        self._enabled_since: Optional[rospy.Time] = rospy.Time.now() if self.enabled else None
        self._disable_brake_remaining = 0
        self._last_log_time = rospy.Time(0)

        self._cmd_prev_time: Optional[rospy.Time] = None
        self._cmd_prev_v = 0.0
        self._cmd_prev_omega = 0.0

        self._cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self._markers_pub = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=1)

        rospy.Subscriber(self.path_topic, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.quality_topic, Float32, self._quality_cb, queue_size=1)
        rospy.Subscriber(self.clearance_topic, Float32, self._clearance_cb, queue_size=1)
        rospy.Subscriber(self.route_id_topic, Int32, self._route_id_cb, queue_size=1)
        rospy.Subscriber(self.route_valid_topic, Bool, self._route_valid_cb, queue_size=1)

        rospy.Service("~set_enabled", SetBool, self._on_set_enabled)

        period = 1.0 / max(self.publish_rate, 1e-3)
        rospy.Timer(rospy.Duration(period), self._on_timer)

        rospy.loginfo(
            "[corridor_controller] path=%s cmd_vel=%s enabled=%s",
            self.path_topic,
            self.cmd_vel_topic,
            self.enabled,
        )

    def _reset_cmd_filter(self, now: Optional[rospy.Time] = None) -> None:
        if now is None:
            now = rospy.Time.now()
        self._cmd_prev_time = now
        self._cmd_prev_v = 0.0
        self._cmd_prev_omega = 0.0

    def _apply_cmd_filter(self, now: rospy.Time, v: float, omega: float) -> Tuple[float, float]:
        if self._cmd_prev_time is None:
            self._reset_cmd_filter(now)
            return (float(v), float(omega))

        dt = float((now - self._cmd_prev_time).to_sec())
        dt = max(1.0e-3, dt)

        v_out = float(v)
        omega_out = float(omega)

        if self.max_linear_accel > 0.0:
            dv_max = float(self.max_linear_accel) * dt
            dv = _clamp(v_out - self._cmd_prev_v, -dv_max, dv_max)
            v_out = float(self._cmd_prev_v + dv)

        if self.max_angular_accel > 0.0:
            dw_max = float(self.max_angular_accel) * dt
            dw = _clamp(omega_out - self._cmd_prev_omega, -dw_max, dw_max)
            omega_out = float(self._cmd_prev_omega + dw)

        if self.cmd_filter_tau > 0.0:
            alpha = dt / (float(self.cmd_filter_tau) + dt)
            v_out = float(self._cmd_prev_v + alpha * (v_out - self._cmd_prev_v))
            omega_out = float(self._cmd_prev_omega + alpha * (omega_out - self._cmd_prev_omega))

        self._cmd_prev_time = now
        self._cmd_prev_v = float(v_out)
        self._cmd_prev_omega = float(omega_out)
        return (float(v_out), float(omega_out))

    def _path_cb(self, msg: Path) -> None:
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        self._last_path = msg
        self._last_path_stamp = stamp

    def _quality_cb(self, msg: Float32) -> None:
        self._last_quality = float(msg.data)
        self._last_quality_time = rospy.Time.now()

    def _clearance_cb(self, msg: Float32) -> None:
        self._last_clearance = float(msg.data)
        self._last_clearance_time = rospy.Time.now()

    def _route_id_cb(self, msg: Int32) -> None:
        self._last_route_id = int(msg.data)
        self._last_route_id_time = rospy.Time.now()

    def _route_valid_cb(self, msg: Bool) -> None:
        self._last_route_valid = bool(msg.data)
        self._last_route_valid_time = rospy.Time.now()

    def _on_set_enabled(self, req: SetBoolRequest) -> SetBoolResponse:
        self.enabled = bool(req.data)
        self._reset_cmd_filter()
        if self.enabled:
            self._enabled_since = rospy.Time.now()
            self._disable_brake_remaining = 0
        else:
            self._enabled_since = None
            self._disable_brake_remaining = max(0, int(self.disable_brake_pulses))
        return SetBoolResponse(success=True, message=f"enabled={self.enabled}")

    def _on_timer(self, _evt) -> None:
        now = rospy.Time.now()
        if not self.enabled:
            state = "DISABLED"
            v = 0.0
            omega = 0.0
            target = None
            if self._disable_brake_remaining > 0:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self._cmd_pub.publish(cmd)
                self._disable_brake_remaining -= 1
            self._reset_cmd_filter(now)
            self._markers_pub.publish(self._build_markers(now, state, v, omega, target))
            self._maybe_log(now, state, v, omega)
            return

        v, omega, state, target = self._compute_cmd(now)
        if state == "TRACK":
            v, omega = self._apply_cmd_filter(now, v, omega)
        else:
            self._reset_cmd_filter(now)
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self._cmd_pub.publish(cmd)
        self._markers_pub.publish(self._build_markers(now, state, v, omega, target))
        self._maybe_log(now, state, v, omega)

    def _maybe_log(self, now: rospy.Time, state: str, v: float, omega: float) -> None:
        if (now - self._last_log_time).to_sec() < self.log_interval:
            return
        q = self._last_quality if self._last_quality is not None else float("nan")
        clr = self._last_clearance if self._last_clearance is not None else float("nan")
        rid = self._last_route_id if self._last_route_id is not None else -999
        rvalid = self._last_route_valid if self._last_route_valid is not None else False
        rospy.loginfo(
            "[corridor_controller] state=%s v=%.2f w=%.2f q=%.2f clr=%.2f rid=%d rvalid=%s",
            state,
            v,
            omega,
            q,
            clr,
            rid,
            rvalid,
        )
        self._last_log_time = now

    def _compute_cmd(self, now: rospy.Time) -> Tuple[float, float, str, Optional[Tuple[float, float]]]:
        if self._last_path is None or self._last_path_stamp is None:
            return (0.0, 0.0, "NO_PATH", None)

        if self.max_path_age > 0.0 and (now - self._last_path_stamp).to_sec() > self.max_path_age:
            return (0.0, 0.0, "PATH_STALE", None)

        pts = self._path_points_xy(self._last_path)
        if len(pts) < self.min_points:
            return (0.0, 0.0, "PATH_SHORT", None)

        q = self._get_quality(now)
        if q is None:
            return (0.0, 0.0, "NO_QUALITY", None)

        clearance = self._get_clearance(now)
        if clearance is None:
            return (0.0, 0.0, "NO_CLEARANCE", None)

        if self.require_route_valid and not self._get_route_valid(now):
            return (0.0, 0.0, "ROUTE_INVALID", None)

        target = self._select_target(pts, self.lookahead)
        if target is None:
            return (0.0, 0.0, "NO_TARGET", None)

        tx, ty, d = target
        if d <= 1e-3 or tx < self.min_target_x:
            return (0.0, 0.0, "TARGET_INVALID", None)

        curvature = 2.0 * float(ty) / float(d * d)
        v = float(self.v_max)

        if self.v_max > 1e-6:
            min_factor_q = float(_clamp(self.v_low_q / self.v_max, 0.0, 1.0))
        else:
            min_factor_q = 0.0
        q_factor = _scale_factor(q, self.q_stop, self.q_full, min_factor=min_factor_q, max_factor=1.0)
        clr_factor = _scale_factor(clearance, self.clearance_stop, self.clearance_full, min_factor=0.0, max_factor=1.0)
        v *= float(q_factor) * float(clr_factor)

        if abs(curvature) > 1e-6 and self.k_pursuit != 0.0:
            v_curve = float(self.max_omega) / max(abs(curvature) * abs(self.k_pursuit), 1e-6)
            v = min(v, v_curve)

        v = max(0.0, v)
        omega = float(v * curvature * self.k_pursuit)
        omega = _clamp(omega, -float(self.max_omega), float(self.max_omega))

        if self.enable_ramp_time > 0.0 and self._enabled_since is not None:
            t = (now - self._enabled_since).to_sec()
            ramp = _clamp(float(t / self.enable_ramp_time), 0.0, 1.0)
            v *= ramp
            omega *= ramp

        return (v, omega, "TRACK", (float(tx), float(ty)))

    def _path_points_xy(self, path: Path) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for pose in path.poses:
            p = pose.pose.position
            out.append((float(p.x), float(p.y)))
        return out

    def _select_target(self, pts: List[Tuple[float, float]], lookahead: float) -> Optional[Tuple[float, float, float]]:
        last: Optional[Tuple[float, float, float]] = None
        for x, y in pts:
            if x < self.min_target_x:
                continue
            d = float(math.hypot(float(x), float(y)))
            last = (float(x), float(y), d)
            if d >= lookahead:
                return last
        return last

    def _get_quality(self, now: rospy.Time) -> Optional[float]:
        if self._last_quality is None or self._last_quality_time is None:
            return None if self.require_quality else 1.0
        if self.max_quality_age > 0.0 and (now - self._last_quality_time).to_sec() > self.max_quality_age:
            return None if self.require_quality else float(self._last_quality)
        return float(self._last_quality)

    def _get_clearance(self, now: rospy.Time) -> Optional[float]:
        if self._last_clearance is None or self._last_clearance_time is None:
            return None if self.require_clearance else float("inf")
        if self.max_clearance_age > 0.0 and (now - self._last_clearance_time).to_sec() > self.max_clearance_age:
            return None if self.require_clearance else float(self._last_clearance)
        return float(self._last_clearance)

    def _get_route_valid(self, now: rospy.Time) -> bool:
        if self._last_route_valid is None or self._last_route_valid_time is None:
            return False
        if self.max_route_age > 0.0 and (now - self._last_route_valid_time).to_sec() > self.max_route_age:
            return False
        return bool(self._last_route_valid)

    def _build_markers(
        self, stamp: rospy.Time, state: str, v: float, omega: float, target: Optional[Tuple[float, float]]
    ) -> MarkerArray:
        frame_id = ""
        if self._last_path is not None:
            frame_id = str(self._last_path.header.frame_id or "")
        if not frame_id:
            frame_id = self.default_frame

        markers = MarkerArray()

        target_m = Marker()
        target_m.header.stamp = stamp
        target_m.header.frame_id = frame_id
        target_m.ns = "corridor_controller"
        target_m.id = 0
        target_m.type = Marker.SPHERE
        target_m.action = Marker.ADD
        target_m.scale.x = 0.18
        target_m.scale.y = 0.18
        target_m.scale.z = 0.18
        target_m.pose.orientation.w = 1.0
        target_m.lifetime = rospy.Duration(self.marker_lifetime)
        if state == "TRACK":
            target_m.color.r, target_m.color.g, target_m.color.b, target_m.color.a = (0.9, 0.9, 0.1, 1.0)
        else:
            target_m.color.r, target_m.color.g, target_m.color.b, target_m.color.a = (0.4, 0.4, 0.4, 0.8)
        if target is not None:
            tx, ty = target
            target_m.pose.position.x = float(tx)
            target_m.pose.position.y = float(ty)
            target_m.pose.position.z = float(self.marker_z)
        markers.markers.append(target_m)

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = frame_id
        text.ns = "corridor_controller"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.scale.z = 0.18
        text.pose.orientation.w = 1.0
        text.pose.position.x = 0.2
        text.pose.position.y = 0.0
        text.pose.position.z = float(self.marker_z + 0.35)
        text.lifetime = rospy.Duration(self.marker_lifetime)
        text.color.r, text.color.g, text.color.b, text.color.a = (0.9, 0.9, 0.9, 1.0)
        text.text = f"{state} v={v:.2f} w={omega:.2f}"
        markers.markers.append(text)

        return markers


if __name__ == "__main__":
    rospy.init_node("corridor_controller")
    CorridorControllerNode()
    rospy.spin()
