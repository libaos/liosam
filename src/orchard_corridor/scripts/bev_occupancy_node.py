#!/usr/bin/env python3
import threading
import time
from typing import Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

try:
    import cv2
except Exception:
    cv2 = None

import tf2_ros


def _quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = float(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n

    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


class BevOccupancyNode:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/tree_points")
        self.output_topic = rospy.get_param("~output_topic", "/bev_occ")
        self.output_topic_raw = rospy.get_param("~output_topic_raw", "/bev_occ_raw")
        self.publish_raw = bool(rospy.get_param("~publish_raw", True))
        self.output_frame = rospy.get_param("~output_frame", "")
        self.use_tf = bool(rospy.get_param("~use_tf", False))

        self.grid_res = float(rospy.get_param("~grid_res", 0.05))
        self.grid_x_min = float(rospy.get_param("~grid_x_min", 0.0))
        self.grid_x_max = float(rospy.get_param("~grid_x_max", 12.0))
        self.grid_y_min = float(rospy.get_param("~grid_y_min", -4.0))
        self.grid_y_max = float(rospy.get_param("~grid_y_max", 4.0))
        self.dilation_radius = float(rospy.get_param("~dilation_radius", 0.10))
        self.unknown_mode = bool(rospy.get_param("~unknown_mode", False))
        self.log_interval = float(rospy.get_param("~log_interval", 2.0))

        self._tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0)) if self.use_tf else None
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer) if self.use_tf else None

        self._last_log_time = rospy.Time(0)
        self._last_msg_time = None

        self.pub = rospy.Publisher(self.output_topic, OccupancyGrid, queue_size=1)
        self.pub_raw = rospy.Publisher(self.output_topic_raw, OccupancyGrid, queue_size=1) if self.publish_raw else None
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self._callback, queue_size=1)

        rospy.loginfo("[bev_occupancy] input=%s output=%s", self.input_topic, self.output_topic)

    def _lookup_transform(self, source_frame: str, stamp: rospy.Time):
        if self._tf_buffer is None:
            return None
        try:
            query_stamp = stamp if stamp != rospy.Time() else rospy.Time(0)
            return self._tf_buffer.lookup_transform(self.output_frame, source_frame, query_stamp, rospy.Duration(0.2))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[bev_occupancy] TF transform failed: %s", exc)
            return None

    def _transform_points_inplace(self, points_xyz: np.ndarray, transform) -> None:
        if points_xyz.size == 0:
            return
        t = transform.transform.translation
        q = transform.transform.rotation
        r = _quat_to_rot_matrix(float(q.x), float(q.y), float(q.z), float(q.w))

        points_xyz[:] = points_xyz @ r.T
        points_xyz[:, 0] += float(t.x)
        points_xyz[:, 1] += float(t.y)
        points_xyz[:, 2] += float(t.z)

    def _grid_shape(self) -> Tuple[int, int]:
        width = int(np.ceil((self.grid_x_max - self.grid_x_min) / self.grid_res))
        height = int(np.ceil((self.grid_y_max - self.grid_y_min) / self.grid_res))
        return max(width, 1), max(height, 1)

    def _callback(self, msg: PointCloud2) -> None:
        start = time.time()
        out_frame = self.output_frame or msg.header.frame_id
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points_xyz = np.asarray(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)
        if self.use_tf and self.output_frame and msg.header.frame_id and msg.header.frame_id != self.output_frame:
            transform = self._lookup_transform(msg.header.frame_id, msg.header.stamp)
            if transform is None:
                out_frame = msg.header.frame_id
            else:
                self._transform_points_inplace(points_xyz, transform)
                out_frame = self.output_frame

        width, height = self._grid_shape()
        if self.unknown_mode:
            grid = np.full((height, width), -1, dtype=np.int8)
        else:
            grid = np.zeros((height, width), dtype=np.int8)

        occ_count = 0
        if points_xyz.size > 0:
            x = points_xyz[:, 0]
            y = points_xyz[:, 1]
            ix = np.floor((x - self.grid_x_min) / self.grid_res).astype(np.int32)
            iy = np.floor((y - self.grid_y_min) / self.grid_res).astype(np.int32)
            mask = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            ix = ix[mask]
            iy = iy[mask]
            if ix.size > 0:
                grid[iy, ix] = 100
                occ_count = int(np.count_nonzero(grid == 100))

        if self.publish_raw and self.pub_raw is not None:
            self.pub_raw.publish(self._build_occ_msg(msg, out_frame, grid))

        grid_out = grid.copy()
        if self.dilation_radius > 0.0:
            if cv2 is None:
                rospy.logwarn_throttle(2.0, "[bev_occupancy] cv2 not available; dilation disabled")
            else:
                radius_cells = int(round(self.dilation_radius / self.grid_res))
                if radius_cells >= 1:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (radius_cells * 2 + 1, radius_cells * 2 + 1),
                    )
                    occ_mask = (grid_out == 100).astype(np.uint8)
                    dilated = cv2.dilate(occ_mask, kernel)
                    grid_out[grid_out != 100] = 0 if not self.unknown_mode else grid_out[grid_out != 100]
                    grid_out[dilated > 0] = 100
                    occ_count = int(np.count_nonzero(grid_out == 100))

        self.pub.publish(self._build_occ_msg(msg, out_frame, grid_out))

        now = rospy.Time.now()
        hz = 0.0
        if self._last_msg_time is not None:
            dt = (now - self._last_msg_time).to_sec()
            if dt > 0:
                hz = 1.0 / dt
        self._last_msg_time = now

        if (now - self._last_log_time).to_sec() >= self.log_interval:
            duration_ms = (time.time() - start) * 1000.0
            rospy.loginfo(
                "[bev_occupancy] occ=%d grid=%dx%d hz=%.1f dt=%.1fms frame=%s",
                occ_count,
                width,
                height,
                hz,
                duration_ms,
                out_frame,
            )
            self._last_log_time = now

    def _build_occ_msg(self, msg: PointCloud2, out_frame: str, grid: np.ndarray) -> OccupancyGrid:
        occ = OccupancyGrid()
        occ.header.stamp = msg.header.stamp
        occ.header.frame_id = out_frame
        occ.info.resolution = float(self.grid_res)
        occ.info.width = grid.shape[1]
        occ.info.height = grid.shape[0]
        occ.info.origin.position.x = float(self.grid_x_min)
        occ.info.origin.position.y = float(self.grid_y_min)
        occ.info.origin.position.z = 0.0
        occ.info.origin.orientation.w = 1.0
        occ.data = grid.reshape(-1).tolist()
        return occ


if __name__ == "__main__":
    rospy.init_node("bev_occupancy")
    BevOccupancyNode()
    rospy.spin()
