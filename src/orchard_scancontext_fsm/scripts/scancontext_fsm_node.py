#!/usr/bin/env python3

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

try:
    from orchard_scancontext_fsm.scancontext import (
        ScanContextParams,
        distance_between_scancontexts,
        downsample_xyz,
        make_scancontext,
    )
except ModuleNotFoundError:  # allows running without catkin install
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from orchard_scancontext_fsm.scancontext import (  # type: ignore
        ScanContextParams,
        distance_between_scancontexts,
        downsample_xyz,
        make_scancontext,
    )


@dataclass
class _HistoryItem:
    stamp_s: float
    desc: np.ndarray


class ScancontextFSMNode:
    def __init__(self) -> None:
        self.cloud_topic = str(rospy.get_param("~cloud_topic", "/liorl/deskew/cloud_deskewed"))
        self.mode_topic = str(rospy.get_param("~mode_topic", "/fsm/mode"))

        self.params = ScanContextParams(
            num_ring=int(rospy.get_param("~num_ring", 20)),
            num_sector=int(rospy.get_param("~num_sector", 60)),
            max_radius=float(rospy.get_param("~max_radius", 80.0)),
            lidar_height=float(rospy.get_param("~lidar_height", 2.0)),
            search_ratio=float(rospy.get_param("~search_ratio", 0.1)),
        )

        self.process_hz = float(rospy.get_param("~process_hz", 2.0))
        self.baseline_s = float(rospy.get_param("~baseline_s", 1.0))
        self.history_s = float(rospy.get_param("~history_s", max(4.0, 2.0 * self.baseline_s)))
        self.max_points = int(rospy.get_param("~max_points", 60000))

        self.sc_dist_max = float(rospy.get_param("~sc_dist_max", 0.35))
        self.yaw_rate_thresh = float(rospy.get_param("~yaw_rate_threshold", 0.25))
        self.consistency_n = int(rospy.get_param("~consistency_n", 3))

        self._history: Deque[_HistoryItem] = deque()
        self._last_processed_s: Optional[float] = None

        self._mode: str = "straight"
        self._candidate_mode: Optional[str] = None
        self._candidate_count: int = 0

        self._mode_pub = rospy.Publisher(self.mode_topic, String, queue_size=10)
        self._sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)

        rospy.loginfo("[orchard_scancontext_fsm] cloud_topic=%s mode_topic=%s", self.cloud_topic, self.mode_topic)

    def _cloud_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if not pts:
            return np.empty((0, 3), dtype=np.float32)
        xyz = np.asarray(pts, dtype=np.float32)
        return downsample_xyz(xyz, max_points=self.max_points)

    def _pick_reference(self, now_s: float) -> Optional[_HistoryItem]:
        target = float(now_s) - float(self.baseline_s)
        for item in reversed(self._history):
            if float(item.stamp_s) <= target:
                return item
        return None

    def _prune_history(self, now_s: float) -> None:
        cutoff = float(now_s) - float(self.history_s)
        while self._history and float(self._history[0].stamp_s) < cutoff:
            self._history.popleft()

    def _raw_mode_from_yaw_rate(self, yaw_rate: float) -> str:
        if abs(float(yaw_rate)) < float(self.yaw_rate_thresh):
            return "straight"
        return "left" if float(yaw_rate) > 0.0 else "right"

    def _update_mode_with_consistency(self, raw_mode: str) -> None:
        raw_mode = str(raw_mode).strip().lower()
        if raw_mode == self._mode:
            self._candidate_mode = None
            self._candidate_count = 0
            return

        if self._candidate_mode != raw_mode:
            self._candidate_mode = raw_mode
            self._candidate_count = 1
            return

        self._candidate_count += 1
        if self._candidate_count >= int(self.consistency_n):
            self._mode = raw_mode
            self._candidate_mode = None
            self._candidate_count = 0

    def _cloud_cb(self, msg: PointCloud2) -> None:
        stamp = msg.header.stamp
        now_s = float(stamp.to_sec()) if stamp is not None else float(rospy.Time.now().to_sec())

        if self.process_hz > 0.0 and self._last_processed_s is not None:
            if float(now_s) - float(self._last_processed_s) < 1.0 / float(self.process_hz):
                return
        self._last_processed_s = now_s

        xyz = self._cloud_to_xyz(msg)
        desc = make_scancontext(xyz, self.params)

        self._history.append(_HistoryItem(stamp_s=now_s, desc=desc))
        self._prune_history(now_s)

        ref = self._pick_reference(now_s)
        yaw_rate = 0.0
        if ref is not None:
            dt = float(now_s) - float(ref.stamp_s)
            if dt > 1.0e-3:
                sc_dist, yaw_diff = distance_between_scancontexts(ref.desc, desc, self.params)
                if float(sc_dist) <= float(self.sc_dist_max):
                    yaw_rate = float(yaw_diff) / dt

        raw_mode = self._raw_mode_from_yaw_rate(yaw_rate)
        self._update_mode_with_consistency(raw_mode)

        self._mode_pub.publish(String(data=self._mode))


def main() -> None:
    rospy.init_node("scancontext_fsm_node")
    _ = ScancontextFSMNode()
    rospy.spin()


if __name__ == "__main__":
    main()
