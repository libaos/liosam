#!/usr/bin/env python3
"""Accumulate segmented tree points into a global tree-only PCD map."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
import tf2_ros
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Empty
from std_msgs.msg import Header


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz

    xx = qx * x2
    xy = qx * y2
    xz = qx * z2
    yy = qy * y2
    yz = qy * z2
    zz = qz * z2
    wx = qw * x2
    wy = qw * y2
    wz = qw * z2

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


class OrchardTreeMapBuilder:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/orchard_segmentation/tree_cloud")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.2))
        self.tf_cache_time = float(rospy.get_param("~tf_cache_time", 300.0))
        if self.tf_cache_time <= 0.0:
            raise ValueError("~tf_cache_time must be > 0")

        self.voxel_size = float(rospy.get_param("~voxel_size", 0.10))
        if self.voxel_size <= 0.0:
            raise ValueError("~voxel_size must be > 0")
        self.max_voxels = int(rospy.get_param("~max_voxels", 800000))

        output_param = str(rospy.get_param("~output_pcd", "/mysda/w/w/lio_ws/maps/TreeMap_auto.pcd")).strip()
        self.output_pcd = Path(output_param).expanduser()
        self.save_on_shutdown = bool(rospy.get_param("~save_on_shutdown", True))

        self.publish_map = bool(rospy.get_param("~publish_map", True))
        self.publish_rate = float(rospy.get_param("~publish_rate", 0.5))
        self.publish_max_points = int(rospy.get_param("~publish_max_points", 50000))

        self._lock = threading.Lock()
        self._voxels: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._last_header: Optional[Header] = None

        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.tf_cache_time))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._sub = rospy.Subscriber(self.input_topic, PointCloud2, self._on_cloud, queue_size=1)
        self._save_sub = rospy.Subscriber("~save_now", Empty, self._on_save_now, queue_size=1)

        self._pub: Optional[rospy.Publisher] = None
        self._timer: Optional[rospy.Timer] = None
        if self.publish_map:
            self._pub = rospy.Publisher("~tree_map", PointCloud2, queue_size=1, latch=True)
            period = 1.0 / max(self.publish_rate, 0.01)
            self._timer = rospy.Timer(rospy.Duration(period), self._on_publish_timer)

        if self.save_on_shutdown:
            rospy.on_shutdown(self._save_on_shutdown)

        rospy.loginfo(
            "[orchard_row_mapping] TreeMapBuilder ready. input=%s voxel=%.3f output=%s",
            self.input_topic,
            self.voxel_size,
            self.output_pcd,
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        points = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )
        if points.size == 0:
            return

        src_frame = msg.header.frame_id or self.map_frame
        if src_frame != self.map_frame:
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self.map_frame, src_frame, msg.header.stamp, rospy.Duration(self.tf_timeout)
                )
            except Exception:
                try:
                    tf_msg = self._tf_buffer.lookup_transform(
                        self.map_frame, src_frame, rospy.Time(0), rospy.Duration(self.tf_timeout)
                    )
                except Exception:
                    rospy.logwarn_throttle(
                        2.0, "[orchard_row_mapping] TF lookup failed: %s -> %s", src_frame, self.map_frame
                    )
                    return

            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            rot = _quat_to_rot(float(q.x), float(q.y), float(q.z), float(q.w))
            trans = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float32)
            points = (rot @ points.T).T + trans

        voxel = np.floor(points / self.voxel_size).astype(np.int32)
        keys = [tuple(int(v) for v in row) for row in voxel]

        with self._lock:
            if self._last_header is None:
                self._last_header = msg.header
            else:
                self._last_header = msg.header

            for key, point in zip(keys, points):
                if self.max_voxels > 0 and len(self._voxels) >= self.max_voxels:
                    rospy.logwarn_throttle(5.0, "[orchard_row_mapping] Reached max_voxels=%d; ignoring new points", self.max_voxels)
                    break
                if key in self._voxels:
                    continue
                self._voxels[key] = point

    def _on_publish_timer(self, _: rospy.TimerEvent) -> None:
        if self._pub is None:
            return

        with self._lock:
            if not self._voxels:
                return
            points = np.vstack(list(self._voxels.values())).astype(np.float32)
            header = self._last_header

        if header is None:
            header = Header()
            header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        if self.publish_max_points > 0 and points.shape[0] > self.publish_max_points:
            stride = int(np.ceil(points.shape[0] / float(self.publish_max_points)))
            points = points[:: max(stride, 1)]

        msg = pc2.create_cloud_xyz32(header, points.tolist())
        self._pub.publish(msg)

    def _on_save_now(self, _: Empty) -> None:
        self._save()

    def _save_on_shutdown(self) -> None:
        try:
            self._save()
        except Exception as exc:
            rospy.logerr("[orchard_row_mapping] Failed to save tree map: %s", exc)

    def _save(self) -> None:
        with self._lock:
            points = np.vstack(list(self._voxels.values())).astype(np.float32) if self._voxels else np.empty((0, 3), dtype=np.float32)

        self.output_pcd.parent.mkdir(parents=True, exist_ok=True)
        with self.output_pcd.open("wb") as handle:
            handle.write(b"# .PCD v0.7 - Point Cloud Data file format\n")
            handle.write(b"VERSION 0.7\n")
            handle.write(b"FIELDS x y z\n")
            handle.write(b"SIZE 4 4 4\n")
            handle.write(b"TYPE F F F\n")
            handle.write(b"COUNT 1 1 1\n")
            handle.write(f"WIDTH {points.shape[0]}\n".encode("utf-8"))
            handle.write(b"HEIGHT 1\n")
            handle.write(b"VIEWPOINT 0 0 0 1 0 0 0\n")
            handle.write(f"POINTS {points.shape[0]}\n".encode("utf-8"))
            handle.write(b"DATA binary\n")
            if points.shape[0] > 0:
                points.astype(np.float32).tofile(handle)

        rospy.loginfo("[orchard_row_mapping] Saved tree map: %s (%d pts)", self.output_pcd, points.shape[0])


def main() -> None:
    rospy.init_node("orchard_tree_map_builder")
    OrchardTreeMapBuilder()
    rospy.spin()


if __name__ == "__main__":
    main()
