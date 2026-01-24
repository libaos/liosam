#!/usr/bin/env python3
import threading
import time
from typing import List, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

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


class PointcloudPreprocessNode:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/points_raw")
        self.output_topic = rospy.get_param("~output_topic", "/pc_roi")
        self.output_frame = rospy.get_param("~output_frame", "")
        self.use_tf = bool(rospy.get_param("~use_tf", False))

        self.roi_x_min = float(rospy.get_param("~roi_x_min", 0.0))
        self.roi_x_max = float(rospy.get_param("~roi_x_max", 12.0))
        self.roi_y_min = float(rospy.get_param("~roi_y_min", -4.0))
        self.roi_y_max = float(rospy.get_param("~roi_y_max", 4.0))
        self.z_min = float(rospy.get_param("~z_min", 0.6))
        self.z_max = float(rospy.get_param("~z_max", 1.6))
        # Filter out very close points (often self-hits / unstable near-field returns in sim).
        self.min_range = float(rospy.get_param("~min_range", 0.0))
        self.voxel = float(rospy.get_param("~voxel", 0.05))
        self.log_interval = float(rospy.get_param("~log_interval", 2.0))

        self._tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0)) if self.use_tf else None
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer) if self.use_tf else None

        self._last_log_time = rospy.Time(0)
        self._last_msg_time = None

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self._callback, queue_size=1)

        rospy.loginfo("[pointcloud_preprocess] input=%s output=%s", self.input_topic, self.output_topic)

    def _lookup_transform(self, source_frame: str, stamp: rospy.Time):
        if self._tf_buffer is None:
            return None
        try:
            query_stamp = stamp if stamp != rospy.Time() else rospy.Time(0)
            return self._tf_buffer.lookup_transform(self.output_frame, source_frame, query_stamp, rospy.Duration(0.2))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[pointcloud_preprocess] TF transform failed: %s", exc)
            return None

    def _transform_points_inplace(self, points: np.ndarray, transform) -> None:
        if points.size == 0:
            return
        t = transform.transform.translation
        q = transform.transform.rotation
        r = _quat_to_rot_matrix(float(q.x), float(q.y), float(q.z), float(q.w))

        xyz = points[:, :3]
        xyz[:] = xyz @ r.T
        xyz[:, 0] += float(t.x)
        xyz[:, 1] += float(t.y)
        xyz[:, 2] += float(t.z)

    def _read_points(self, msg: PointCloud2) -> Tuple[np.ndarray, bool]:
        field_names = [f.name for f in msg.fields]
        has_intensity = "intensity" in field_names
        if has_intensity:
            fields = ("x", "y", "z", "intensity")
        else:
            fields = ("x", "y", "z")
        points = list(pc2.read_points(msg, field_names=fields, skip_nans=True))
        if not points:
            return np.empty((0, len(fields)), dtype=np.float32), has_intensity
        return np.asarray(points, dtype=np.float32), has_intensity

    def _downsample_voxel(self, points: np.ndarray) -> np.ndarray:
        if self.voxel <= 0.0 or points.size == 0:
            return points
        coords = np.floor(points[:, :3] / float(self.voxel)).astype(np.int32)
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        return points[np.sort(unique_idx)]

    def _callback(self, msg: PointCloud2) -> None:
        start = time.time()
        points, has_intensity = self._read_points(msg)
        out_frame = self.output_frame or msg.header.frame_id
        if self.use_tf and self.output_frame and msg.header.frame_id and msg.header.frame_id != self.output_frame:
            transform = self._lookup_transform(msg.header.frame_id, msg.header.stamp)
            if transform is None:
                out_frame = msg.header.frame_id
            else:
                self._transform_points_inplace(points, transform)
                out_frame = self.output_frame
        in_count = int(points.shape[0])
        if in_count == 0:
            self._publish_empty(msg, has_intensity, out_frame)
            return

        mask = (
            (points[:, 0] >= self.roi_x_min)
            & (points[:, 0] <= self.roi_x_max)
            & (points[:, 1] >= self.roi_y_min)
            & (points[:, 1] <= self.roi_y_max)
            & (points[:, 2] >= self.z_min)
            & (points[:, 2] <= self.z_max)
        )
        if self.min_range > 0.0:
            r2 = float(self.min_range) * float(self.min_range)
            mask &= (points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]) >= r2
        points = points[mask]
        points = self._downsample_voxel(points)
        out_count = int(points.shape[0])

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = out_frame
        fields = self._build_fields(has_intensity)
        cloud_points = points[:, : len(fields)].tolist() if out_count > 0 else []
        out_msg = pc2.create_cloud(header, fields, cloud_points)
        self.pub.publish(out_msg)

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
                "[pointcloud_preprocess] in=%d out=%d hz=%.1f dt=%.1fms frame=%s",
                in_count,
                out_count,
                hz,
                duration_ms,
                header.frame_id,
            )
            self._last_log_time = now

    def _publish_empty(self, msg: PointCloud2, has_intensity: bool, out_frame: str) -> None:
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = out_frame
        fields = self._build_fields(has_intensity)
        out_msg = pc2.create_cloud(header, fields, [])
        self.pub.publish(out_msg)

    @staticmethod
    def _build_fields(has_intensity: bool) -> List[PointField]:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        if has_intensity:
            fields.append(PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1))
        return fields


if __name__ == "__main__":
    rospy.init_node("pointcloud_preprocess")
    PointcloudPreprocessNode()
    rospy.spin()
