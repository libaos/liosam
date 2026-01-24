#!/usr/bin/env python3
"""ROS node for orchard point cloud segmentation and row fitting."""
import struct
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
import rospkg
import torch
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from orchard_row_mapping.segmentation import load_model, preprocess_points, run_inference


class OrchardSegmentationNode:
    """Subscribes to raw point clouds, runs RandLA-Net, and fits orchard rows."""

    def __init__(self) -> None:
        self.cloud_topic = rospy.get_param("~pointcloud_topic", "/velodyne_points")
        self.num_points = rospy.get_param("~num_points", 16384)
        self.sampling_method = rospy.get_param("~sampling", "random")
        self.sampling_seed = rospy.get_param("~sampling_seed", -1)
        self.num_classes = rospy.get_param("~num_classes", 2)
        self.tree_class = rospy.get_param("~tree_class", 0)
        self.tree_prob_threshold = rospy.get_param("~tree_prob_threshold", 0.0)
        self.min_points_per_row = rospy.get_param("~min_points_per_row", 200)
        self.line_alpha = rospy.get_param("~row_alpha", 1.0)
        self.line_width = rospy.get_param("~row_width", 0.2)
        self.split_axis = rospy.get_param("~split_axis", "y")  # y or x
        self.split_margin = rospy.get_param("~split_margin", 0.0)
        self.frame_override = rospy.get_param("~output_frame", "")
        self.use_gpu = rospy.get_param("~use_gpu", True)
        self.fit_x_min = rospy.get_param("~fit_x_min", -1.0e9)
        self.fit_x_max = rospy.get_param("~fit_x_max", 1.0e9)
        self.fit_y_abs_max = rospy.get_param("~fit_y_abs_max", 1.0e9)
        self.fit_z_min = rospy.get_param("~fit_z_min", -1.0e9)
        self.fit_z_max = rospy.get_param("~fit_z_max", 1.0e9)
        self.line_inlier_distance = rospy.get_param("~line_inlier_distance", 0.0)
        self.line_inlier_iterations = int(rospy.get_param("~line_inlier_iterations", 1))
        self.endpoint_percentile = rospy.get_param("~line_endpoint_percentile", 0.0)
        self.fixed_line_length = rospy.get_param("~fixed_line_length", 0.0)
        self.smoothing_alpha = rospy.get_param("~line_smoothing_alpha", 0.0)
        self.hold_last_seconds = rospy.get_param("~hold_last_seconds", 0.0)
        self.marker_lifetime = rospy.get_param("~marker_lifetime", 0.2)
        self.publish_tree_cloud = bool(rospy.get_param("~publish_tree_cloud", False))
        self.publish_segmented_cloud = bool(rospy.get_param("~publish_segmented_cloud", True))
        self.publish_row_markers = bool(rospy.get_param("~publish_row_markers", True))
        checkpoint_param = rospy.get_param("~model_path", "")
        self.color_map = self._load_color_map()

        self.device = self._select_device()
        checkpoint_path = self._resolve_checkpoint(checkpoint_param)
        if checkpoint_path is None:
            raise RuntimeError("No valid checkpoint file found for segmentation model")
        rospy.loginfo("[orchard_row_mapping] Loading model from %s on %s", checkpoint_path, self.device)
        self.model = load_model(self.num_classes, self.device, checkpoint_path)

        self.cloud_sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_callback, queue_size=1)
        self.seg_cloud_pub: Optional[rospy.Publisher] = None
        if self.publish_segmented_cloud:
            self.seg_cloud_pub = rospy.Publisher("~segmented_cloud", PointCloud2, queue_size=1)
        self.tree_cloud_pub: Optional[rospy.Publisher] = None
        if self.publish_tree_cloud:
            self.tree_cloud_pub = rospy.Publisher("~tree_cloud", PointCloud2, queue_size=1)
        self.row_marker_pub: Optional[rospy.Publisher] = None
        if self.publish_row_markers:
            self.row_marker_pub = rospy.Publisher("~row_markers", MarkerArray, queue_size=1)

        self._prev_lines: Dict[str, np.ndarray] = {}
        self._cached_markers: Optional[MarkerArray] = None
        self._cached_marker_time: Optional[rospy.Time] = None

        rospy.loginfo("[orchard_row_mapping] Ready. Listening on %s", self.cloud_topic)

    def _select_device(self) -> torch.device:
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _load_color_map(self) -> Dict[int, Tuple[int, int, int]]:
        colors = rospy.get_param("~label_colors", {"0": [0, 255, 0], "1": [180, 180, 180]})
        parsed: Dict[int, Tuple[int, int, int]] = {}
        for key, value in colors.items():
            try:
                idx = int(key)
                parsed[idx] = tuple(int(c) for c in value)
            except Exception:
                rospy.logwarn("[orchard_row_mapping] Invalid color entry for label %s", key)
        return parsed

    def _resolve_checkpoint(self, override: str) -> Optional[Path]:
        candidates: List[Path] = []
        if override:
            candidates.append(Path(override))
        try:
            pkg_path = Path(rospkg.RosPack().get_path("orchard_row_mapping"))
            candidates.append(pkg_path / "checkpoints" / "best_model.pth")
        except rospkg.ResourceNotFound:
            pass
        candidates.append(Path("/mysda/w/w/RandLA-Net-pytorch/noslam/checkpoints/best_model.pth"))
        candidates.append(Path("/mysda/w/w/RandLA-Net-pytorch/best_model.pth"))

        for path in candidates:
            if path and path.exists():
                return path
        return None

    def _cloud_callback(self, msg: PointCloud2) -> None:
        start_time = time.time()
        points_xyz = self._cloud_to_numpy(msg)
        if points_xyz.size == 0:
            rospy.logwarn_throttle(5.0, "[orchard_row_mapping] Received empty point cloud")
            return

        seed = None if int(self.sampling_seed) < 0 else int(self.sampling_seed)
        processed_points = preprocess_points(points_xyz, self.num_points, sampling=self.sampling_method, seed=seed)
        labels, probs = run_inference(self.model, processed_points, self.device, self.num_classes)

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.frame_override or msg.header.frame_id

        sampled_xyz = processed_points[:, :3]
        if self.publish_segmented_cloud:
            self._publish_segmented_cloud(header, sampled_xyz, labels)

        if self.publish_tree_cloud or self.publish_row_markers:
            if self.tree_class < 0 or self.tree_class >= probs.shape[1]:
                rospy.logwarn_throttle(
                    5.0,
                    "[orchard_row_mapping] tree_class %d out of range (C=%d)",
                    self.tree_class,
                    probs.shape[1],
                )
                self._maybe_hold_markers(header)
                return
            tree_points = self._compute_tree_points(sampled_xyz, labels, probs)
            if self.publish_tree_cloud:
                self._publish_tree_cloud(header, tree_points)
            if self.publish_row_markers:
                self._publish_row_markers(header, tree_points)

        duration = (time.time() - start_time) * 1000.0
        rospy.logdebug("[orchard_row_mapping] Inference completed in %.2f ms", duration)

    def _cloud_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        points = []
        for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append((x, y, z))
        if not points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    def _publish_segmented_cloud(self, header: Header, points: np.ndarray, labels: np.ndarray) -> None:
        if self.seg_cloud_pub is None:
            return
        fields = self._segmented_cloud_fields()
        cloud_points = []
        for point, label in zip(points, labels):
            color = self.color_map.get(int(label), (255, 255, 255))
            rgb_int = struct.unpack("I", struct.pack("BBBB", color[2], color[1], color[0], 255))[0]
            rgb_float = struct.unpack("f", struct.pack("I", rgb_int))[0]
            cloud_points.append((point[0], point[1], point[2], rgb_float, float(label)))
        cloud_msg = pc2.create_cloud(header, fields, cloud_points)
        self.seg_cloud_pub.publish(cloud_msg)

    def _segmented_cloud_fields(self) -> List[PointField]:
        return [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="label", offset=16, datatype=PointField.FLOAT32, count=1),
        ]

    def _publish_tree_cloud(self, header: Header, tree_points: np.ndarray) -> None:
        if self.tree_cloud_pub is None:
            return

        fields = self._segmented_cloud_fields()
        label_value = float(self.tree_class)
        tree_color = self.color_map.get(int(self.tree_class), (56, 188, 75))
        rgb_int = struct.unpack("I", struct.pack("BBBB", tree_color[2], tree_color[1], tree_color[0], 255))[0]
        rgb_float = struct.unpack("f", struct.pack("I", rgb_int))[0]
        cloud_points = [(float(p[0]), float(p[1]), float(p[2]), rgb_float, label_value) for p in tree_points]
        cloud_msg = pc2.create_cloud(header, fields, cloud_points)
        self.tree_cloud_pub.publish(cloud_msg)

    def _compute_tree_points(self, points: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
        tree_mask = labels == self.tree_class
        if self.tree_prob_threshold > 0.0:
            tree_mask &= probs[:, self.tree_class] >= float(self.tree_prob_threshold)
        tree_points = points[tree_mask]
        return self._filter_fit_bounds(tree_points)

    def _publish_row_markers(self, header: Header, tree_points: np.ndarray) -> None:
        if self.row_marker_pub is None:
            return
        if tree_points.shape[0] < self.min_points_per_row:
            rospy.logwarn_throttle(2.0, "[orchard_row_mapping] Not enough tree points for row fitting (%d)", tree_points.shape[0])
            self._maybe_hold_markers(header)
            return

        left_points, right_points = self._split_rows(tree_points)
        markers = MarkerArray()
        marker_id = 0
        if left_points is not None:
            left_points = self._smooth_line("left", left_points)
            marker = self._make_line_marker(header, marker_id, left_points, (0.1, 0.8, 0.1))
            markers.markers.append(marker)
            marker_id += 1
        if right_points is not None:
            right_points = self._smooth_line("right", right_points)
            marker = self._make_line_marker(header, marker_id, right_points, (0.9, 0.6, 0.1))
            markers.markers.append(marker)
        if not markers.markers:
            self._maybe_hold_markers(header)
            return

        self._cached_markers = markers
        self._cached_marker_time = rospy.Time.now()
        self.row_marker_pub.publish(markers)

    def _filter_fit_bounds(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        mask = (
            (points[:, 0] >= float(self.fit_x_min))
            & (points[:, 0] <= float(self.fit_x_max))
            & (np.abs(points[:, 1]) <= float(self.fit_y_abs_max))
            & (points[:, 2] >= float(self.fit_z_min))
            & (points[:, 2] <= float(self.fit_z_max))
        )
        return points[mask]

    def _maybe_hold_markers(self, header: Header) -> None:
        if self.row_marker_pub is None:
            return
        if self.hold_last_seconds <= 0.0:
            self._clear_markers(header)
            return
        if self._cached_markers is None or self._cached_marker_time is None:
            self._clear_markers(header)
            return
        if (rospy.Time.now() - self._cached_marker_time).to_sec() > float(self.hold_last_seconds):
            self._clear_markers(header)
            return
        for marker in self._cached_markers.markers:
            marker.header = header
        self.row_marker_pub.publish(self._cached_markers)

    def _split_rows(self, tree_points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        axis_idx = 1 if self.split_axis.lower() == "y" else 0
        axis_values = tree_points[:, axis_idx]
        median_val = np.median(axis_values)
        margin = float(self.split_margin)
        if margin > 0.0:
            left_mask = axis_values >= median_val + margin
            right_mask = axis_values <= median_val - margin
        else:
            left_mask = axis_values >= median_val
            right_mask = ~left_mask
        left_points = tree_points[left_mask]
        right_points = tree_points[right_mask]
        left_line = self._fit_line(left_points) if left_points.shape[0] >= self.min_points_per_row else None
        right_line = self._fit_line(right_points) if right_points.shape[0] >= self.min_points_per_row else None
        return left_line, right_line

    def _fit_line(self, points: np.ndarray) -> Optional[np.ndarray]:
        if points.shape[0] < 2:
            return None

        inlier_mask = np.ones(points.shape[0], dtype=bool)
        iterations = max(int(self.line_inlier_iterations), 0)
        distance_thresh = float(self.line_inlier_distance)

        for _ in range(iterations):
            candidate = points[inlier_mask]
            if candidate.shape[0] < 2 or distance_thresh <= 0.0:
                break
            centroid, direction = self._pca_direction(candidate[:, :2])
            distances = self._point_line_distance(points[:, :2], centroid, direction)
            new_mask = distances <= distance_thresh
            if new_mask.sum() < 2:
                break
            inlier_mask = new_mask

        inliers = points[inlier_mask]
        if inliers.shape[0] < 2:
            inliers = points
        centroid, direction = self._pca_direction(inliers[:, :2])

        if float(self.fixed_line_length) > 0.0:
            half_len = float(self.fixed_line_length) * 0.5
            low, high = -half_len, half_len
        else:
            projections = np.dot(inliers[:, :2] - centroid, direction)
            percentile = float(self.endpoint_percentile)
            percentile = max(0.0, min(percentile, 49.0))
            if percentile > 0.0:
                low = float(np.percentile(projections, percentile))
                high = float(np.percentile(projections, 100.0 - percentile))
            else:
                low = float(projections.min())
                high = float(projections.max())

        start_xy = centroid + direction * low
        end_xy = centroid + direction * high
        z_mean = float(np.mean(inliers[:, 2]))
        return np.array([[start_xy[0], start_xy[1], z_mean], [end_xy[0], end_xy[1], z_mean]], dtype=np.float32)

    def _pca_direction(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centroid = np.mean(xy, axis=0)
        centered = xy - centroid
        cov = np.dot(centered.T, centered) / max(centered.shape[0], 1)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        direction = eig_vecs[:, np.argmax(eig_vals)]
        norm = np.linalg.norm(direction)
        if norm < 1.0e-9:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        return centroid.astype(np.float32), direction.astype(np.float32)

    def _point_line_distance(self, xy: np.ndarray, centroid: np.ndarray, direction: np.ndarray) -> np.ndarray:
        diff = xy - centroid
        return np.abs(diff[:, 0] * direction[1] - diff[:, 1] * direction[0])

    def _smooth_line(self, key: str, line_points: np.ndarray) -> np.ndarray:
        alpha = float(self.smoothing_alpha)
        if alpha <= 0.0:
            self._prev_lines[key] = line_points
            return line_points
        alpha = max(0.0, min(alpha, 1.0))
        prev = self._prev_lines.get(key)
        if prev is None:
            self._prev_lines[key] = line_points
            return line_points
        prev_dir = prev[1, :2] - prev[0, :2]
        cur = line_points
        cur_dir = cur[1, :2] - cur[0, :2]
        if float(np.dot(prev_dir, cur_dir)) < 0.0:
            cur = cur[::-1].copy()
        smoothed = alpha * prev + (1.0 - alpha) * cur
        self._prev_lines[key] = smoothed
        return smoothed

    def _make_line_marker(self, header: Header, marker_id: int, line_points: np.ndarray, color: Tuple[float, float, float]) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "orchard_rows"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.line_width
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = self.line_alpha
        marker.points = [Point(x=line_points[0, 0], y=line_points[0, 1], z=line_points[0, 2]),
                         Point(x=line_points[1, 0], y=line_points[1, 1], z=line_points[1, 2])]
        marker.lifetime = rospy.Duration(float(self.marker_lifetime))
        return marker

    def _clear_markers(self, header: Header) -> None:
        if self.row_marker_pub is None:
            return
        marker = Marker()
        marker.header = header
        marker.action = Marker.DELETEALL
        marker.pose.orientation.w = 1.0
        self.row_marker_pub.publish(MarkerArray(markers=[marker]))


def main() -> None:
    rospy.init_node("orchard_segmentation_node")
    try:
        OrchardSegmentationNode()
    except rospy.ROSInterruptException:
        return
    rospy.spin()


if __name__ == "__main__":
    main()
