#!/usr/bin/env python3
"""Record odometry and optional planner path to CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

import rospy
import tf2_ros
from nav_msgs.msg import Odometry, Path as NavPath
from tf.transformations import euler_from_quaternion


def _yaw_from_quat(q) -> float:
    return float(euler_from_quaternion([q.x, q.y, q.z, q.w])[2])


class Recorder:
    def __init__(
        self,
        out_csv: Path,
        plan_csv: Optional[Path],
        min_dt: float,
        odom_topic: str,
        plan_topic: str,
        output_frame: str,
        tf_timeout: float,
    ):
        self.out_csv = out_csv
        self.plan_csv = plan_csv
        self.min_dt = float(min_dt)
        self.last_odom_time = 0.0
        self.odom_rows: List[Tuple[float, float, float, float]] = []
        self.plan_rows: List[Tuple[float, float, float, float]] = []

        self.output_frame = str(output_frame).strip()
        self.tf_timeout = float(tf_timeout)
        self._warned_tf = False
        self._tf_buffer: Optional[tf2_ros.Buffer] = None
        self._tf_listener: Optional[tf2_ros.TransformListener] = None
        if self.output_frame:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=1)
        if self.plan_csv is not None and plan_topic:
            rospy.Subscriber(plan_topic, NavPath, self._plan_cb, queue_size=1)

        rospy.on_shutdown(self._flush)

    def _transform_xy_yaw(self, stamp: rospy.Time, frame_id: str, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        if not self.output_frame:
            return (x, y, yaw)
        if self._tf_buffer is None:
            return (x, y, yaw)
        try:
            tr = self._tf_buffer.lookup_transform(self.output_frame, frame_id, stamp, rospy.Duration(self.tf_timeout))
        except Exception as exc:
            if not self._warned_tf:
                rospy.logwarn("record_trajectory: TF lookup failed (%s -> %s): %s", frame_id, self.output_frame, exc)
                self._warned_tf = True
            raise

        t = tr.transform.translation
        q = tr.transform.rotation
        yaw_tf = _yaw_from_quat(q)
        c = float(math.cos(yaw_tf))
        s = float(math.sin(yaw_tf))

        x_out = float(t.x) + c * float(x) - s * float(y)
        y_out = float(t.y) + s * float(x) + c * float(y)
        yaw_out = float(yaw_tf) + float(yaw)
        return (x_out, y_out, yaw_out)

    def _odom_cb(self, msg: Odometry) -> None:
        t = msg.header.stamp.to_sec()
        if self.min_dt > 0 and (t - self.last_odom_time) < self.min_dt:
            return
        self.last_odom_time = t
        pos = msg.pose.pose.position
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        x = float(pos.x)
        y = float(pos.y)
        try:
            x, y, yaw = self._transform_xy_yaw(msg.header.stamp, str(msg.header.frame_id), x, y, float(yaw))
        except Exception:
            return
        self.odom_rows.append((t, x, y, float(yaw)))

    def _plan_cb(self, msg: NavPath) -> None:
        if not msg.poses:
            return
        last = msg.poses[-1]
        t = last.header.stamp.to_sec()
        pos = last.pose.position
        yaw = _yaw_from_quat(last.pose.orientation)
        x = float(pos.x)
        y = float(pos.y)
        try:
            x, y, yaw = self._transform_xy_yaw(last.header.stamp, str(last.header.frame_id), x, y, float(yaw))
        except Exception:
            return
        self.plan_rows.append((t, x, y, float(yaw)))

    def _flush(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "x", "y", "yaw"])
            for row in self.odom_rows:
                writer.writerow(row)
        if self.plan_csv is not None:
            self.plan_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.plan_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "x", "y", "yaw"])
                for row in self.plan_rows:
                    writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record /odom and optional planner path to CSV")
    parser.add_argument("--out", type=str, default="trajectory_data/pid_or_teb_odom.csv", help="输出 CSV (odom)")
    parser.add_argument("--plan-out", type=str, default="", help="可选输出 CSV (plan)")
    parser.add_argument("--min-dt", type=float, default=0.05, help="最小采样间隔 (s)")
    parser.add_argument("--odom-topic", type=str, default="/odom")
    parser.add_argument("--plan-topic", type=str, default="", help="nav_msgs/Path 话题")
    parser.add_argument("--output-frame", type=str, default="", help="可选输出坐标系（例如 map），会用 TF 把 /odom 投到该坐标系")
    parser.add_argument("--tf-timeout", type=float, default=0.2, help="TF 查询超时 (s)")
    args = parser.parse_args(rospy.myargv()[1:])

    out_csv = Path(args.out).expanduser().resolve()
    plan_csv = Path(args.plan_out).expanduser().resolve() if str(args.plan_out).strip() else None

    rospy.init_node("record_trajectory", anonymous=True)
    Recorder(
        out_csv,
        plan_csv,
        float(args.min_dt),
        str(args.odom_topic),
        str(args.plan_topic),
        str(args.output_frame),
        float(args.tf_timeout),
    )
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
