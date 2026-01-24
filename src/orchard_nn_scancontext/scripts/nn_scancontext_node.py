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
import torch
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, Int32

from orchard_nn_scancontext.cnn_2d_models import Enhanced2DCNN, ResNet2D, Simple2DCNN
from orchard_nn_scancontext.scan_context import ScanContext


def _default_model_path() -> Path:
    pkg_root = Path(__file__).resolve().parents[1]
    return pkg_root / "models" / "trajectory_localizer_simple2dcnn_acc97.5.pth"


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


def _load_model(
    model_type: str,
    num_classes: int,
    device: torch.device,
    checkpoint_path: Path,
) -> torch.nn.Module:
    model_type = (model_type or "simple2dcnn").strip().lower()
    if model_type == "simple2dcnn":
        model = Simple2DCNN(num_classes=num_classes)
    elif model_type == "enhanced2dcnn":
        model = Enhanced2DCNN(num_classes=num_classes)
    elif model_type == "resnet2d":
        model = ResNet2D(num_classes=num_classes)
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    return model


class NNScancontextNode:
    def __init__(self) -> None:
        self.cloud_topic = rospy.get_param("~cloud_topic", "/points_raw")
        self.route_id_topic = rospy.get_param("~route_id_topic", "/route_id")
        self.route_conf_topic = rospy.get_param("~route_conf_topic", "/route_conf")

        self.process_hz = float(rospy.get_param("~process_hz", 2.0))
        self.max_points = int(rospy.get_param("~max_points", 60000))
        self.use_gpu = bool(rospy.get_param("~use_gpu", True))

        self.model_type = str(rospy.get_param("~model_type", "simple2dcnn"))
        self.num_classes = int(rospy.get_param("~num_classes", 20))
        self.model_path = Path(str(rospy.get_param("~model_path", str(_default_model_path())))).expanduser()

        self.sc = ScanContext(
            num_sectors=int(rospy.get_param("~num_sector", 60)),
            num_rings=int(rospy.get_param("~num_ring", 20)),
            min_range=float(rospy.get_param("~min_range", 0.1)),
            max_range=float(rospy.get_param("~max_range", 80.0)),
            height_lower_bound=float(rospy.get_param("~height_lower_bound", -1.0)),
            height_upper_bound=float(rospy.get_param("~height_upper_bound", 9.0)),
        )

        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        rospy.loginfo("[nn_scancontext] device=%s", self.device)
        rospy.loginfo("[nn_scancontext] cloud_topic=%s process_hz=%.2f max_points=%d", self.cloud_topic, self.process_hz, self.max_points)
        rospy.loginfo("[nn_scancontext] model_type=%s num_classes=%d model_path=%s", self.model_type, self.num_classes, self.model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"model_path not found: {self.model_path}")

        self.model = _load_model(self.model_type, self.num_classes, self.device, self.model_path)

        self._last_msg: Optional[PointCloud2] = None
        self._last_msg_time: Optional[rospy.Time] = None
        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)

        self.pub_id = rospy.Publisher(self.route_id_topic, Int32, queue_size=10)
        self.pub_conf = rospy.Publisher(self.route_conf_topic, Float32, queue_size=10)

        period = 1.0 / max(self.process_hz, 1e-6)
        self.timer = rospy.Timer(rospy.Duration.from_sec(period), self._on_timer)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        self._last_msg = msg
        self._last_msg_time = rospy.Time.now()

    def _infer(self, msg: PointCloud2) -> Optional[Tuple[int, float]]:
        points_xyz = _cloud_to_xyz(msg)
        if points_xyz.size == 0:
            return None
        points_xyz = _downsample_xyz(points_xyz, self.max_points)
        desc = self.sc.generate_scan_context(points_xyz)

        x = torch.from_numpy(desc).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,20,60)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
        return int(pred.item()), float(conf.item())

    def _publish(self, route_id: int, route_conf: float, stamp: rospy.Time) -> None:
        id_msg = Int32()
        id_msg.data = int(route_id)
        conf_msg = Float32()
        conf_msg.data = float(route_conf)
        self.pub_id.publish(id_msg)
        self.pub_conf.publish(conf_msg)

    def _on_timer(self, _evt: rospy.TimerEvent) -> None:
        msg = self._last_msg
        if msg is None:
            return
        start = time.time()
        result = self._infer(msg)
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        if result is None:
            self._publish(-1, 0.0, stamp)
            return
        route_id, conf = result
        self._publish(route_id, conf, stamp)
        dt_ms = (time.time() - start) * 1000.0
        rospy.loginfo_throttle(2.0, "[nn_scancontext] id=%d conf=%.3f dt=%.1fms", route_id, conf, dt_ms)


def main() -> None:
    rospy.init_node("nn_scancontext")
    _ = NNScancontextNode()
    rospy.spin()


if __name__ == "__main__":
    main()
