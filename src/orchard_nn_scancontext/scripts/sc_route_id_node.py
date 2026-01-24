#!/usr/bin/env python3
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, Int32

from orchard_nn_scancontext.scan_context import ScanContext
from orchard_nn_scancontext.sc_route_db import load_route_db, predict_route_id_cosine, predict_route_id_l2


def _downsample_xyz(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points_xyz.shape[0] <= max_points:
        return points_xyz
    step = int(np.ceil(float(points_xyz.shape[0]) / float(max_points)))
    return points_xyz[::step]


def _cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


class ScanContextRouteIdNode:
    def __init__(self) -> None:
        self.cloud_topic = rospy.get_param("~cloud_topic", "/points_raw")
        self.route_id_topic = rospy.get_param("~route_id_topic", "/route_id")
        self.route_conf_topic = rospy.get_param("~route_conf_topic", "/route_conf")

        self.db_path = Path(str(rospy.get_param("~db_path", ""))).expanduser()
        if not str(self.db_path).strip():
            raise ValueError("~db_path is required (npz produced by build_sc_route_db.py)")

        self.process_hz = float(rospy.get_param("~process_hz", 2.0))
        self.max_points = int(rospy.get_param("~max_points", 60000))

        self.metric = str(rospy.get_param("~metric", "cosine")).strip().lower()
        self.temperature = float(rospy.get_param("~temperature", 0.02))

        self.db = load_route_db(self.db_path)
        rospy.loginfo(
            "[sc_route_id] db=%s K=%d params=(R=%d,S=%d,min=%.2f,max=%.2f,z=[%.2f,%.2f])",
            self.db_path,
            int(self.db.prototypes.shape[0]),
            self.db.params.num_ring,
            self.db.params.num_sector,
            self.db.params.min_range,
            self.db.params.max_range,
            self.db.params.height_lower_bound,
            self.db.params.height_upper_bound,
        )

        self.sc = ScanContext(
            num_sectors=int(self.db.params.num_sector),
            num_rings=int(self.db.params.num_ring),
            min_range=float(self.db.params.min_range),
            max_range=float(self.db.params.max_range),
            height_lower_bound=float(self.db.params.height_lower_bound),
            height_upper_bound=float(self.db.params.height_upper_bound),
        )

        self._last_msg: Optional[PointCloud2] = None
        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)

        self.pub_id = rospy.Publisher(self.route_id_topic, Int32, queue_size=10)
        self.pub_conf = rospy.Publisher(self.route_conf_topic, Float32, queue_size=10)

        period = 1.0 / max(self.process_hz, 1e-6)
        self.timer = rospy.Timer(rospy.Duration.from_sec(period), self._on_timer)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        self._last_msg = msg

    def _infer(self, msg: PointCloud2) -> Optional[Tuple[int, float]]:
        points_xyz = _cloud_to_xyz(msg)
        if points_xyz.size == 0:
            return None
        points_xyz = _downsample_xyz(points_xyz, self.max_points)
        desc = self.sc.generate_scan_context(points_xyz)

        if self.metric == "cosine":
            return predict_route_id_cosine(desc, self.db, temperature=self.temperature)
        if self.metric in ("l2", "mse"):
            return predict_route_id_l2(desc, self.db, temperature=self.temperature)
        raise ValueError(f"unknown metric: {self.metric}")

    def _publish(self, route_id: int, route_conf: float) -> None:
        self.pub_id.publish(Int32(data=int(route_id)))
        self.pub_conf.publish(Float32(data=float(route_conf)))

    def _on_timer(self, _evt: rospy.TimerEvent) -> None:
        msg = self._last_msg
        if msg is None:
            return
        start = time.time()
        result = self._infer(msg)
        if result is None:
            self._publish(-1, 0.0)
            return
        route_id, conf = result
        self._publish(route_id, conf)
        dt_ms = (time.time() - start) * 1000.0
        rospy.loginfo_throttle(2.0, "[sc_route_id] id=%d conf=%.3f dt=%.1fms", route_id, conf, dt_ms)


def main() -> None:
    rospy.init_node("sc_route_id")
    _ = ScanContextRouteIdNode()
    rospy.spin()


if __name__ == "__main__":
    main()

