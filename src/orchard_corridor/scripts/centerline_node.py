#!/usr/bin/env python3
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool, Float32, Int32
from visualization_msgs.msg import Marker, MarkerArray

try:
    import cv2
except Exception:
    cv2 = None


class CenterlineNode:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/bev_occ")
        self.output_path_topic = rospy.get_param("~output_path_topic", "/corridor_centerline")
        self.quality_topic = rospy.get_param("~quality_topic", "/corridor_quality")
        self.clearance_topic = rospy.get_param("~clearance_topic", "/corridor_clearance_min")
        self.markers_topic = rospy.get_param("~markers_topic", "/corridor_debug_markers")

        self.r_req = float(rospy.get_param("~r_req", 0.38))
        self.r_dil = float(rospy.get_param("~r_dil", 0.10))
        self.r_gate = float(rospy.get_param("~r_gate", -1.0))
        if self.r_gate < 0.0:
            self.r_gate = max(0.0, self.r_req - self.r_dil)

        self.x_step = float(rospy.get_param("~x_step", 0.1))
        self.y_win = float(rospy.get_param("~y_win", 0.5))
        self.dy_max = float(rospy.get_param("~dy_max", 0.2))
        self.smooth_tau = float(rospy.get_param("~smooth_tau", 0.3))
        self.hold_time = float(rospy.get_param("~hold_time", 0.3))
        self.q_th = float(rospy.get_param("~q_th", 0.3))
        self.min_valid_cols = int(rospy.get_param("~min_valid_cols", 0))
        self.quality_mode = str(rospy.get_param("~quality_mode", "ratio")).strip().lower()
        self.reacquire_after_invalid = int(rospy.get_param("~reacquire_after_invalid", 0))
        self.center_y = float(rospy.get_param("~center_y", 0.0))
        self.center_bias = float(rospy.get_param("~center_bias", 0.0))
        self.min_occ_cells = int(rospy.get_param("~min_occ_cells", 0))
        self.use_route_center_y = bool(rospy.get_param("~use_route_center_y", False))
        self.route_id_topic = str(rospy.get_param("~route_id_topic", "/route_id_stable"))
        self.route_valid_topic = str(rospy.get_param("~route_valid_topic", "/route_id_valid"))
        self.require_route_valid = bool(rospy.get_param("~require_route_valid", False))
        self.max_route_age = float(rospy.get_param("~max_route_age", 1.0))
        self.route_center_y_default = float(rospy.get_param("~route_center_y_default", self.center_y))
        self.route_center_y_map = self._parse_route_center_y_map(rospy.get_param("~route_center_y_map", {}))
        self._route_center_y = float(self.route_center_y_default)
        self._route_center_y_time = rospy.Time(0)
        self._route_valid: Optional[bool] = None
        self._route_valid_time = rospy.Time(0)

        self.occ_threshold = int(rospy.get_param("~occ_threshold", 50))
        self.unknown_as_obstacle = bool(rospy.get_param("~unknown_as_obstacle", True))

        self.line_width = float(rospy.get_param("~line_width", 0.05))
        self.point_size = float(rospy.get_param("~point_size", 0.08))
        self.marker_z = float(rospy.get_param("~marker_z", 0.05))
        self.marker_lifetime = float(rospy.get_param("~marker_lifetime", 0.2))
        self.log_interval = float(rospy.get_param("~log_interval", 2.0))

        self._last_good_path: Optional[Path] = None
        self._last_good_stamp: Optional[rospy.Time] = None
        self._last_log_time = rospy.Time(0)

        self.path_pub = rospy.Publisher(self.output_path_topic, Path, queue_size=1)
        self.quality_pub = rospy.Publisher(self.quality_topic, Float32, queue_size=1)
        self.clearance_pub = rospy.Publisher(self.clearance_topic, Float32, queue_size=1)
        self.markers_pub = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, OccupancyGrid, self._callback, queue_size=1)
        if self.use_route_center_y:
            rospy.Subscriber(self.route_id_topic, Int32, self._route_id_cb, queue_size=1)
        if self.use_route_center_y and self.require_route_valid:
            rospy.Subscriber(self.route_valid_topic, Bool, self._route_valid_cb, queue_size=1)

        rospy.loginfo("[centerline] input=%s output=%s", self.input_topic, self.output_path_topic)

    @staticmethod
    def _parse_route_center_y_map(raw) -> dict:
        if not isinstance(raw, dict):
            return {}
        out = {}
        for k, v in raw.items():
            try:
                rid = int(k) if not isinstance(k, int) else int(k)
                out[int(rid)] = float(v)
            except Exception:
                continue
        return out

    def _route_id_cb(self, msg: Int32) -> None:
        rid = int(msg.data)
        y = self.route_center_y_map.get(rid, self.route_center_y_default)
        self._route_center_y = float(y)
        self._route_center_y_time = rospy.Time.now()

    def _route_valid_cb(self, msg: Bool) -> None:
        self._route_valid = bool(msg.data)
        self._route_valid_time = rospy.Time.now()

    def _get_center_y(self, now: rospy.Time) -> float:
        if not self.use_route_center_y:
            return float(self.center_y)
        if self.require_route_valid:
            if self._route_valid_time == rospy.Time(0) or self._route_valid is not True:
                return float(self.route_center_y_default)
            if self.max_route_age > 0.0 and (now - self._route_valid_time).to_sec() > self.max_route_age:
                return float(self.route_center_y_default)
        if self._route_center_y_time == rospy.Time(0):
            return float(self.route_center_y_default)
        if self.max_route_age > 0.0 and (now - self._route_center_y_time).to_sec() > self.max_route_age:
            return float(self.route_center_y_default)
        return float(self._route_center_y)

    def _callback(self, msg: OccupancyGrid) -> None:
        if cv2 is None:
            rospy.logwarn_throttle(2.0, "[centerline] cv2 not available; distance transform disabled")
            return

        start = time.time()
        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0:
            return

        data = np.array(msg.data, dtype=np.int16)
        if data.size != width * height:
            rospy.logwarn_throttle(2.0, "[centerline] occupancy size mismatch: %d vs %d", data.size, width * height)
            return

        grid = data.reshape((height, width))
        occ = grid >= self.occ_threshold
        if self.unknown_as_obstacle:
            occ |= grid < 0
        occ_cells = int(np.count_nonzero(occ))
        free = (~occ).astype(np.uint8)

        dist_cells = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_m = dist_cells * float(msg.info.resolution)

        res = float(msg.info.resolution)
        x_min = float(msg.info.origin.position.x)
        y_min = float(msg.info.origin.position.y)
        now = rospy.Time.now()
        center_y = self._get_center_y(now)
        iy_center = int(round((center_y - y_min) / max(res, 1e-9)))
        iy_center = max(0, min(height - 1, iy_center))
        y_bias_m = np.abs(np.arange(height, dtype=np.int32) - int(iy_center)).astype(np.float32) * float(res)

        x_step_cells = max(1, int(round(self.x_step / res)))
        y_win_cells = max(1, int(round(self.y_win / res)))
        dy_max_cells = max(0, int(round(self.dy_max / res)))

        points_xy: List[Tuple[float, float]] = []
        invalid_xy: List[Tuple[float, float]] = []
        valid_count = 0
        total_cols = 0
        min_clearance = None
        sum_clearance = 0.0
        y_prev = None
        y_prev_sm = None
        invalid_run = 0

        for ix in range(0, width, x_step_cells):
            total_cols += 1
            column = dist_m[:, ix]
            if y_prev is None:
                if self.center_bias > 0.0:
                    score = column - float(self.center_bias) * y_bias_m
                    iy = int(np.argmax(score))
                else:
                    iy = int(np.argmax(column))
                max_d = float(column[iy])
            else:
                y_low = max(0, y_prev - y_win_cells)
                y_high = min(height - 1, y_prev + y_win_cells)
                window = column[y_low : y_high + 1]
                if window.size == 0:
                    max_d = -1.0
                    iy = y_prev
                else:
                    if self.center_bias > 0.0:
                        score = window - float(self.center_bias) * y_bias_m[y_low : y_high + 1]
                        rel = int(np.argmax(score))
                    else:
                        rel = int(np.argmax(window))
                    iy = y_low + rel
                    max_d = float(column[iy])

                # Enforce continuity by selecting the best point within the step limit,
                # instead of invalidating the entire column (which can cascade into many invalid columns).
                if dy_max_cells > 0 and abs(iy - y_prev) > dy_max_cells:
                    y_low2 = max(0, y_prev - dy_max_cells)
                    y_high2 = min(height - 1, y_prev + dy_max_cells)
                    window2 = column[y_low2 : y_high2 + 1]
                    if window2.size == 0:
                        max_d = -1.0
                        iy = y_prev
                    else:
                        if self.center_bias > 0.0:
                            score2 = window2 - float(self.center_bias) * y_bias_m[y_low2 : y_high2 + 1]
                            rel2 = int(np.argmax(score2))
                        else:
                            rel2 = int(np.argmax(window2))
                        iy = y_low2 + rel2
                        max_d = float(column[iy])

            if max_d < self.r_gate:
                x_coord = x_min + (ix + 0.5) * res
                y_mark = float(y_prev) if y_prev is not None else float(height / 2)
                invalid_xy.append((x_coord, y_min + (y_mark + 0.5) * res))
                invalid_run += 1
                if self.reacquire_after_invalid > 0 and invalid_run >= self.reacquire_after_invalid:
                    y_prev = None
                    y_prev_sm = None
                    invalid_run = 0
                continue

            y_prev = iy
            invalid_run = 0
            valid_count += 1
            sum_clearance += max_d
            if min_clearance is None or max_d < min_clearance:
                min_clearance = max_d

            if self.smooth_tau > 0.0:
                alpha = min(1.0, max(0.0, self.smooth_tau))
                if y_prev_sm is None:
                    y_prev_sm = float(iy)
                else:
                    y_prev_sm = (1.0 - alpha) * y_prev_sm + alpha * float(iy)
                iy_out = int(round(y_prev_sm))
            else:
                iy_out = iy

            iy_out = max(0, min(height - 1, iy_out))
            x_coord = x_min + (ix + 0.5) * res
            y_coord = y_min + (iy_out + 0.5) * res
            points_xy.append((x_coord, y_coord))

        ratio = float(valid_count) / float(total_cols) if total_cols > 0 else 0.0
        if min_clearance is None:
            min_clearance = 0.0
        mean_clearance = sum_clearance / float(valid_count) if valid_count > 0 else 0.0

        if self.quality_mode == "ratio_clearance":
            denom = max(self.r_req, 1e-3)
            q = ratio * min(1.0, float(min_clearance) / denom)
        else:
            q = ratio
        if self.min_valid_cols > 0 and valid_count < self.min_valid_cols:
            q = 0.0
        if self.min_occ_cells > 0 and occ_cells < self.min_occ_cells:
            q = 0.0

        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else now
        hold_active = False
        path = self._build_path(msg.header.frame_id, stamp, points_xy)

        if q >= self.q_th and valid_count > 0:
            self._last_good_path = path
            self._last_good_stamp = stamp
        else:
            if self._last_good_path is not None and self._last_good_stamp is not None:
                if (stamp - self._last_good_stamp).to_sec() <= self.hold_time:
                    hold_active = True
                    path = self._clone_path(self._last_good_path, stamp)
                else:
                    path = self._build_path(msg.header.frame_id, stamp, [])
            else:
                path = self._build_path(msg.header.frame_id, stamp, [])

        self.path_pub.publish(path)
        self.quality_pub.publish(Float32(data=float(q)))
        self.clearance_pub.publish(Float32(data=float(min_clearance)))
        self.markers_pub.publish(self._build_markers(msg.header.frame_id, stamp, points_xy, invalid_xy, q, hold_active))

        if (now - self._last_log_time).to_sec() >= self.log_interval:
            duration_ms = (time.time() - start) * 1000.0
            rospy.loginfo(
                "[centerline] valid=%d/%d q=%.2f min=%.2f mean=%.2f cy=%.2f dt=%.1fms hold=%s",
                valid_count,
                total_cols,
                q,
                min_clearance,
                mean_clearance,
                center_y,
                duration_ms,
                hold_active,
            )
            self._last_log_time = now

    def _build_path(self, frame_id: str, stamp: rospy.Time, points_xy: List[Tuple[float, float]]) -> Path:
        path = Path()
        path.header.frame_id = frame_id
        path.header.stamp = stamp
        for x, y in points_xy:
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def _clone_path(self, path: Path, stamp: rospy.Time) -> Path:
        cloned = Path()
        cloned.header.frame_id = path.header.frame_id
        cloned.header.stamp = stamp
        for pose in path.poses:
            new_pose = PoseStamped()
            new_pose.header.frame_id = pose.header.frame_id
            new_pose.header.stamp = stamp
            new_pose.pose = pose.pose
            cloned.poses.append(new_pose)
        return cloned

    def _build_markers(
        self,
        frame_id: str,
        stamp: rospy.Time,
        points_xy: List[Tuple[float, float]],
        invalid_xy: List[Tuple[float, float]],
        q: float,
        hold_active: bool,
    ) -> MarkerArray:
        markers = MarkerArray()

        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = stamp
        line.ns = "centerline"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = float(self.line_width)
        line.pose.orientation.w = 1.0
        line.lifetime = rospy.Duration(self.marker_lifetime)
        if q >= self.q_th:
            line.color.r, line.color.g, line.color.b, line.color.a = (0.1, 0.9, 0.2, 1.0)
        elif hold_active:
            line.color.r, line.color.g, line.color.b, line.color.a = (0.9, 0.6, 0.1, 1.0)
        else:
            line.color.r, line.color.g, line.color.b, line.color.a = (0.9, 0.1, 0.1, 1.0)
        for x, y in points_xy:
            line.points.append(Point(x=float(x), y=float(y), z=float(self.marker_z)))
        markers.markers.append(line)

        pts = Marker()
        pts.header.frame_id = frame_id
        pts.header.stamp = stamp
        pts.ns = "centerline_points"
        pts.id = 1
        pts.type = Marker.POINTS
        pts.action = Marker.ADD
        pts.scale.x = float(self.point_size)
        pts.scale.y = float(self.point_size)
        pts.pose.orientation.w = 1.0
        pts.lifetime = rospy.Duration(self.marker_lifetime)
        pts.color.r, pts.color.g, pts.color.b, pts.color.a = (0.2, 0.6, 1.0, 1.0)
        for x, y in points_xy:
            pts.points.append(Point(x=float(x), y=float(y), z=float(self.marker_z)))
        markers.markers.append(pts)

        invalid = Marker()
        invalid.header.frame_id = frame_id
        invalid.header.stamp = stamp
        invalid.ns = "invalid_cols"
        invalid.id = 2
        invalid.type = Marker.POINTS
        invalid.action = Marker.ADD
        invalid.scale.x = float(self.point_size)
        invalid.scale.y = float(self.point_size)
        invalid.pose.orientation.w = 1.0
        invalid.lifetime = rospy.Duration(self.marker_lifetime)
        invalid.color.r, invalid.color.g, invalid.color.b, invalid.color.a = (1.0, 0.2, 0.2, 1.0)
        for x, y in invalid_xy:
            invalid.points.append(Point(x=float(x), y=float(y), z=float(self.marker_z)))
        markers.markers.append(invalid)

        return markers


if __name__ == "__main__":
    rospy.init_node("centerline")
    CenterlineNode()
    rospy.spin()
