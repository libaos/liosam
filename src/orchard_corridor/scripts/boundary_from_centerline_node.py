#!/usr/bin/env python3
import math
import threading
from typing import List, Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray


class BoundaryFromCenterlineNode:
    def __init__(self) -> None:
        self.input_occ_topic = rospy.get_param("~input_occ_topic", "/bev_occ")
        self.input_centerline_topic = rospy.get_param("~input_centerline_topic", "/corridor_centerline")
        self.markers_topic = rospy.get_param("~markers_topic", "/corridor_boundary_markers")
        self.width_topic = rospy.get_param("~width_topic", "/corridor_width")

        self.occ_threshold = int(rospy.get_param("~occ_threshold", 50))
        self.unknown_as_obstacle = bool(rospy.get_param("~unknown_as_obstacle", True))
        self.max_scan_distance = float(rospy.get_param("~max_scan_distance", 3.0))
        self.scan_step = float(rospy.get_param("~scan_step", 0.05))
        self.sample_stride = max(1, int(rospy.get_param("~sample_stride", 1)))
        self.width_mode = str(rospy.get_param("~width_mode", "min")).strip().lower()
        self.max_age = float(rospy.get_param("~max_age", 0.5))

        self.line_width = float(rospy.get_param("~line_width", 0.05))
        self.point_size = float(rospy.get_param("~point_size", 0.08))
        self.marker_z = float(rospy.get_param("~marker_z", 0.05))
        self.marker_lifetime = float(rospy.get_param("~marker_lifetime", 0.2))
        self.log_interval = float(rospy.get_param("~log_interval", 2.0))

        self._occ_msg: Optional[OccupancyGrid] = None
        self._path_msg: Optional[Path] = None
        self._last_processed: Tuple[Optional[rospy.Time], Optional[rospy.Time]] = (None, None)
        self._last_log_time = rospy.Time(0)
        self._warned_orientation = False

        self._markers_pub = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=1)
        self._width_pub = rospy.Publisher(self.width_topic, Float32, queue_size=1)

        self._occ_sub = rospy.Subscriber(self.input_occ_topic, OccupancyGrid, self._occ_callback, queue_size=1)
        self._path_sub = rospy.Subscriber(self.input_centerline_topic, Path, self._path_callback, queue_size=1)

        rospy.loginfo(
            "[boundary_from_centerline] occ=%s centerline=%s",
            self.input_occ_topic,
            self.input_centerline_topic,
        )

    def _occ_callback(self, msg: OccupancyGrid) -> None:
        self._occ_msg = msg
        self._try_compute()

    def _path_callback(self, msg: Path) -> None:
        self._path_msg = msg
        self._try_compute()

    def _try_compute(self) -> None:
        if self._occ_msg is None or self._path_msg is None:
            return
        now = rospy.Time.now()
        if self.max_age > 0.0:
            if (now - self._occ_msg.header.stamp).to_sec() > self.max_age:
                return
            if (now - self._path_msg.header.stamp).to_sec() > self.max_age:
                return

        occ_stamp = self._occ_msg.header.stamp
        path_stamp = self._path_msg.header.stamp
        if self._last_processed == (occ_stamp, path_stamp):
            return
        self._last_processed = (occ_stamp, path_stamp)

        self._compute_boundaries(self._occ_msg, self._path_msg)

    def _compute_boundaries(self, occ_msg: OccupancyGrid, path_msg: Path) -> None:
        width = int(occ_msg.info.width)
        height = int(occ_msg.info.height)
        if width <= 0 or height <= 0:
            return

        if not self._warned_orientation:
            ori = occ_msg.info.origin.orientation
            if abs(ori.x) > 1e-6 or abs(ori.y) > 1e-6 or abs(ori.z) > 1e-6 or abs(ori.w - 1.0) > 1e-6:
                rospy.logwarn_throttle(5.0, "[boundary_from_centerline] occupancy grid has non-identity orientation")
                self._warned_orientation = True

        data = np.array(occ_msg.data, dtype=np.int16)
        if data.size != width * height:
            rospy.logwarn_throttle(
                2.0,
                "[boundary_from_centerline] occupancy size mismatch: %d vs %d",
                data.size,
                width * height,
            )
            return

        grid = data.reshape((height, width))
        occ = grid >= self.occ_threshold
        if self.unknown_as_obstacle:
            occ |= grid < 0

        points_xy = [(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses]
        if len(points_xy) < 2:
            self._publish_markers(path_msg.header.frame_id, path_msg.header.stamp, [], [], [])
            self._width_pub.publish(Float32(data=0.0))
            return

        res = float(occ_msg.info.resolution)
        origin_x = float(occ_msg.info.origin.position.x)
        origin_y = float(occ_msg.info.origin.position.y)

        left_segments: List[List[Tuple[float, float]]] = []
        right_segments: List[List[Tuple[float, float]]] = []
        current_left: List[Tuple[float, float]] = []
        current_right: List[Tuple[float, float]] = []
        invalid_points: List[Tuple[float, float]] = []
        widths: List[float] = []

        step = self.scan_step if self.scan_step > 0.0 else res
        max_steps = int(self.max_scan_distance / step) if step > 0.0 else 0

        for idx in range(0, len(points_xy), self.sample_stride):
            x, y = points_xy[idx]
            if idx < len(points_xy) - 1:
                dx = points_xy[idx + 1][0] - x
                dy = points_xy[idx + 1][1] - y
            else:
                dx = x - points_xy[idx - 1][0]
                dy = y - points_xy[idx - 1][1]

            norm = math.hypot(dx, dy)
            if norm < 1e-6:
                continue
            dir_x = dx / norm
            dir_y = dy / norm
            left_x, left_y = -dir_y, dir_x
            right_x, right_y = dir_y, -dir_x

            left_point, left_dist = self._scan_to_obstacle(
                x,
                y,
                left_x,
                left_y,
                occ,
                origin_x,
                origin_y,
                res,
                width,
                height,
                step,
                max_steps,
            )
            right_point, right_dist = self._scan_to_obstacle(
                x,
                y,
                right_x,
                right_y,
                occ,
                origin_x,
                origin_y,
                res,
                width,
                height,
                step,
                max_steps,
            )

            if left_point is not None:
                current_left.append(left_point)
            elif current_left:
                left_segments.append(current_left)
                current_left = []

            if right_point is not None:
                current_right.append(right_point)
            elif current_right:
                right_segments.append(current_right)
                current_right = []

            if left_point is None or right_point is None:
                invalid_points.append((x, y))
            else:
                widths.append(left_dist + right_dist)

        if current_left:
            left_segments.append(current_left)
        if current_right:
            right_segments.append(current_right)

        width_value = 0.0
        if widths:
            if self.width_mode == "mean":
                width_value = float(sum(widths) / len(widths))
            else:
                width_value = float(min(widths))

        stamp = path_msg.header.stamp if path_msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        frame_id = path_msg.header.frame_id or occ_msg.header.frame_id

        self._publish_markers(frame_id, stamp, left_segments, right_segments, invalid_points)
        self._width_pub.publish(Float32(data=width_value))

        now = rospy.Time.now()
        if (now - self._last_log_time).to_sec() >= self.log_interval:
            rospy.loginfo(
                "[boundary_from_centerline] segments L=%d R=%d invalid=%d width=%.2f",
                len(left_segments),
                len(right_segments),
                len(invalid_points),
                width_value,
            )
            self._last_log_time = now

    def _scan_to_obstacle(
        self,
        x: float,
        y: float,
        dir_x: float,
        dir_y: float,
        occ: np.ndarray,
        origin_x: float,
        origin_y: float,
        res: float,
        width: int,
        height: int,
        step: float,
        max_steps: int,
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        if max_steps <= 0:
            return None, 0.0

        for step_idx in range(1, max_steps + 1):
            dist = step * step_idx
            px = x + dir_x * dist
            py = y + dir_y * dist
            ix = int((px - origin_x) / res)
            iy = int((py - origin_y) / res)
            if ix < 0 or ix >= width or iy < 0 or iy >= height:
                return None, dist
            if occ[iy, ix]:
                return (px, py), dist
        return None, step * max_steps

    def _publish_markers(
        self,
        frame_id: str,
        stamp: rospy.Time,
        left_segments: List[List[Tuple[float, float]]],
        right_segments: List[List[Tuple[float, float]]],
        invalid_points: List[Tuple[float, float]],
    ) -> None:
        markers = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        lifetime = rospy.Duration(self.marker_lifetime)

        for idx, segment in enumerate(left_segments):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "boundary_left"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = self.line_width
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = lifetime
            marker.points = [Point(x=p[0], y=p[1], z=self.marker_z) for p in segment]
            markers.markers.append(marker)

        for idx, segment in enumerate(right_segments):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "boundary_right"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = self.line_width
            marker.color.r = 0.0
            marker.color.g = 0.6
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = lifetime
            marker.points = [Point(x=p[0], y=p[1], z=self.marker_z) for p in segment]
            markers.markers.append(marker)

        if invalid_points:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "boundary_invalid"
            marker.id = 0
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = self.point_size
            marker.scale.y = self.point_size
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = lifetime
            marker.points = [Point(x=p[0], y=p[1], z=self.marker_z) for p in invalid_points]
            markers.markers.append(marker)

        self._markers_pub.publish(markers)


if __name__ == "__main__":
    rospy.init_node("boundary_from_centerline")
    BoundaryFromCenterlineNode()
    rospy.spin()
