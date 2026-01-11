#!/usr/bin/env python3
"""Cluster a tree-only point cloud into per-tree circles (streaming).

This is the streaming counterpart of `orchard_tree_circles_node.py`:
- subscribes to a PointCloud2 topic (typically `/orchard_segmentation/tree_cloud`)
- clusters points per-frame using a simple 2D grid connected-components
- publishes circle markers for quick visualization in RViz.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class TreeCircle:
    x: float
    y: float
    z: float
    radius: float
    n_points: int


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


def _fit_direction_from_xy(points_xy: np.ndarray) -> Optional[np.ndarray]:
    if points_xy.shape[0] < 2:
        return None
    mean = np.mean(points_xy, axis=0)
    centered = points_xy - mean
    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return None
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = int(np.argmax(eig_vals))
    direction = eig_vecs[:, idx].astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-6:
        return None
    return (direction / norm).astype(np.float32)


def _normalize_direction(vec: np.ndarray) -> Optional[np.ndarray]:
    if vec.size != 2:
        return None
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-6:
        return None
    return (vec / norm).astype(np.float32)


def _load_prior_rows(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict[str, float]]]:
    p = Path(path).expanduser()
    if not p.is_file():
        return None, None, []
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = list(data.get("rows_uv") or data.get("rows") or [])
    dir_xy = np.array(data.get("direction_xy", []), dtype=np.float32).reshape(-1)
    perp_xy = np.array(data.get("perp_xy", []), dtype=np.float32).reshape(-1)
    dir_xy = _normalize_direction(dir_xy[:2]) if dir_xy.size >= 2 else None
    perp_xy = _normalize_direction(perp_xy[:2]) if perp_xy.size >= 2 else None
    return dir_xy, perp_xy, rows


def _kmeans_1d(values: np.ndarray, iters: int = 10) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if values.size < 2:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), float("nan"), float("nan")
    v0 = float(np.quantile(values, 0.25))
    v1 = float(np.quantile(values, 0.75))
    if abs(v1 - v0) < 1.0e-6:
        v0 = float(values.min())
        v1 = float(values.max())
    centers = np.array([v0, v1], dtype=np.float32)
    for _ in range(int(iters)):
        d0 = np.abs(values - centers[0])
        d1 = np.abs(values - centers[1])
        idx0 = np.where(d0 <= d1)[0]
        idx1 = np.where(d0 > d1)[0]
        if idx0.size:
            centers[0] = float(np.mean(values[idx0]))
        if idx1.size:
            centers[1] = float(np.mean(values[idx1]))
    d0 = np.abs(values - centers[0])
    d1 = np.abs(values - centers[1])
    idx0 = np.where(d0 <= d1)[0]
    idx1 = np.where(d0 > d1)[0]
    return idx0, idx1, float(centers[0]), float(centers[1])


def _cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _filter_points(
    points: np.ndarray,
    z_min: float,
    z_max: float,
    x_min: float,
    x_max: float,
    y_abs_max: float,
) -> np.ndarray:
    if points.size == 0:
        return points
    mask = (
        (points[:, 2] >= float(z_min))
        & (points[:, 2] <= float(z_max))
        & (points[:, 0] >= float(x_min))
        & (points[:, 0] <= float(x_max))
        & (np.abs(points[:, 1]) <= float(y_abs_max))
    )
    return points[mask]


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(points.shape[0], int(max_points), replace=False)
    return points[idx]


def _cluster_cells(
    xy: np.ndarray,
    cell_size: float,
    neighbor_range: int,
    min_points: int,
    max_clusters: int,
) -> List[np.ndarray]:
    cell_size = float(cell_size)
    if cell_size <= 0.0:
        raise ValueError("cell_size must be > 0")
    neighbor_range = max(0, int(neighbor_range))
    min_points = max(1, int(min_points))
    max_clusters = int(max_clusters)

    grid = np.floor(xy / cell_size).astype(np.int32)
    cells: Dict[Tuple[int, int], List[int]] = {}
    for idx, (cx, cy) in enumerate(grid.tolist()):
        cells.setdefault((int(cx), int(cy)), []).append(int(idx))

    visited: set[Tuple[int, int]] = set()
    clusters: List[np.ndarray] = []
    for cell in cells.keys():
        if cell in visited:
            continue
        queue: deque[Tuple[int, int]] = deque([cell])
        visited.add(cell)
        member_indices: List[int] = []
        while queue:
            cx, cy = queue.popleft()
            member_indices.extend(cells.get((cx, cy), []))
            for dx in range(-neighbor_range, neighbor_range + 1):
                for dy in range(-neighbor_range, neighbor_range + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in visited or nb not in cells:
                        continue
                    visited.add(nb)
                    queue.append(nb)

        if len(member_indices) < min_points:
            continue
        clusters.append(np.asarray(member_indices, dtype=np.int32))
        if max_clusters > 0 and len(clusters) >= max_clusters:
            break

    return clusters


def _parse_color(value, default_rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            r, g, b = float(value[0]), float(value[1]), float(value[2])
            if r > 1.5 or g > 1.5 or b > 1.5:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            return (
                float(max(0.0, min(r, 1.0))),
                float(max(0.0, min(g, 1.0))),
                float(max(0.0, min(b, 1.0))),
            )
    except Exception:
        pass
    return default_rgb


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    h = float(h) % 1.0
    s = float(max(0.0, min(s, 1.0)))
    v = float(max(0.0, min(v, 1.0)))
    i = int(h * 6.0)
    f = h * 6.0 - float(i)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return float(r), float(g), float(b)


def _cluster_id_to_rgb(cluster_id: int) -> Tuple[float, float, float]:
    phi = 0.618033988749895  # golden ratio
    h = (float(cluster_id) * phi) % 1.0
    return _hsv_to_rgb(h, 0.95, 1.0)


def _cloud_xyz(points_xyz: np.ndarray, header: Header) -> PointCloud2:
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud_points = [(float(p[0]), float(p[1]), float(p[2])) for p in points_xyz.astype(np.float32)]
    return pc2.create_cloud(header, fields, cloud_points)


class OrchardTreeClustersNode:
    def __init__(self) -> None:
        self.input_topic = str(rospy.get_param("~input_topic", "/orchard_segmentation/tree_cloud"))
        self.frame_override = str(rospy.get_param("~output_frame", "")).strip()
        self.target_frame = str(rospy.get_param("~target_frame", "")).strip()
        self.process_hz = float(rospy.get_param("~process_hz", 2.0))

        self.filter_z_min = float(rospy.get_param("~z_min", 0.7))
        self.filter_z_max = float(rospy.get_param("~z_max", 1.3))
        self.filter_x_min = float(rospy.get_param("~x_min", 0.0))
        self.filter_x_max = float(rospy.get_param("~x_max", 20.0))
        self.filter_y_abs_max = float(rospy.get_param("~y_abs_max", 6.0))
        self.max_points = int(rospy.get_param("~max_points", 20000))
        self.sample_seed = int(rospy.get_param("~sample_seed", 0))

        self.cluster_cell_size = float(rospy.get_param("~cluster_cell_size", 0.10))
        self.cluster_neighbor_range = int(rospy.get_param("~cluster_neighbor_range", 1))
        self.min_points_per_cluster = int(rospy.get_param("~min_points_per_tree", 40))
        self.max_clusters = int(rospy.get_param("~max_trees", 0))

        self.center_mode = str(rospy.get_param("~center_mode", "median")).strip().lower()

        self.radius_mode = str(rospy.get_param("~radius_mode", "quantile")).strip().lower()
        self.radius_constant = float(rospy.get_param("~radius_constant", 0.35))
        self.radius_quantile = float(rospy.get_param("~radius_quantile", 0.8))
        self.radius_min = float(rospy.get_param("~radius_min", 0.15))
        self.radius_max = float(rospy.get_param("~radius_max", 1.5))

        self.marker_type = str(rospy.get_param("~marker_type", "line_strip")).strip().lower()
        self.circle_segments = int(rospy.get_param("~circle_segments", 36))
        self.line_width = float(rospy.get_param("~line_width", 0.05))
        self.alpha = float(rospy.get_param("~alpha", 0.9))
        self.marker_z_offset = float(rospy.get_param("~marker_z_offset", 0.0))
        self.marker_lifetime = float(rospy.get_param("~marker_lifetime", 0.2))

        self.color_mode = str(rospy.get_param("~color_mode", "cluster")).strip().lower()
        self.color = _parse_color(rospy.get_param("~color", [0.0, 0.8, 0.0]), (0.0, 0.8, 0.0))

        self.publish_labels = bool(rospy.get_param("~publish_labels", False))
        self.label_height = float(rospy.get_param("~label_height", 0.6))
        self.label_z_offset = float(rospy.get_param("~label_z_offset", 0.8))
        self.label_start_index = int(rospy.get_param("~label_start_index", 1))

        self.publish_centers = bool(rospy.get_param("~publish_centers", False))

        self.prior_json = str(rospy.get_param("~prior_json", "")).strip()
        self.prior_bandwidth = float(rospy.get_param("~prior_bandwidth", 0.8))
        self.prior_u_padding = float(rospy.get_param("~prior_u_padding", 0.0))
        self._prior_dir: Optional[np.ndarray] = None
        self._prior_perp: Optional[np.ndarray] = None
        self._prior_rows: List[Dict[str, float]] = []
        if self.prior_json:
            self._prior_dir, self._prior_perp, self._prior_rows = _load_prior_rows(self.prior_json)
            if not self._prior_rows or self._prior_dir is None or self._prior_perp is None:
                rospy.logwarn("[orchard_row_mapping] Prior gating disabled (invalid prior_json: %s)", self.prior_json)
                self._prior_rows = []
                self._prior_dir = None
                self._prior_perp = None
            else:
                rospy.loginfo(
                    "[orchard_row_mapping] Prior gating enabled: %d rows, bandwidth=%.2f",
                    len(self._prior_rows),
                    float(self.prior_bandwidth),
                )

        self.publish_row_markers = bool(rospy.get_param("~publish_row_markers", False))
        self.row_split_method = str(rospy.get_param("~row_split_method", "auto")).strip().lower()
        self.row_split_margin = float(rospy.get_param("~row_split_margin", 0.0))
        self.row_min_separation = float(rospy.get_param("~row_min_separation", 1.0))
        self.row_min_centers = int(rospy.get_param("~row_min_centers_per_row", 4))
        self.row_line_width = float(rospy.get_param("~row_line_width", 0.08))
        self.row_line_alpha = float(rospy.get_param("~row_line_alpha", 1.0))
        self.row_line_fixed_length = float(rospy.get_param("~row_line_fixed_length", 0.0))
        self.row_line_endpoint_percentile = float(rospy.get_param("~row_line_endpoint_percentile", 0.0))
        self.row_line_z_offset = float(rospy.get_param("~row_line_z_offset", 0.0))
        self.row_left_color = _parse_color(rospy.get_param("~row_left_color", [0.1, 0.8, 0.1]), (0.1, 0.8, 0.1))
        self.row_right_color = _parse_color(rospy.get_param("~row_right_color", [0.9, 0.6, 0.1]), (0.9, 0.6, 0.1))

        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.2))
        self.tf_cache_time = float(rospy.get_param("~tf_cache_time", 30.0))
        if self.target_frame and self.tf_cache_time <= 0.0:
            raise ValueError("~tf_cache_time must be > 0")

        self._circles_pub = rospy.Publisher("~tree_circles", MarkerArray, queue_size=1)
        self._centers_pub = rospy.Publisher("~tree_centers", PointCloud2, queue_size=1) if self.publish_centers else None
        self._rows_pub = rospy.Publisher("~row_markers", MarkerArray, queue_size=1) if self.publish_row_markers else None

        self._last_pub_time: Optional[rospy.Time] = None
        self._last_circle_count = 0
        self._last_row_count = 0
        self._tf_buffer: Optional[tf2_ros.Buffer] = None
        self._tf_listener: Optional[tf2_ros.TransformListener] = None

        if self.target_frame:
            self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.tf_cache_time))
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
            if self.frame_override and self.frame_override != self.target_frame:
                rospy.logwarn(
                    "[orchard_row_mapping] output_frame=%s ignored (target_frame=%s)",
                    self.frame_override,
                    self.target_frame,
                )

        self._sub = rospy.Subscriber(self.input_topic, PointCloud2, self._on_cloud, queue_size=1)
        rospy.loginfo("[orchard_row_mapping] Tree clustering (streaming) listening on %s", self.input_topic)

    def _should_process(self, now: rospy.Time) -> bool:
        if self.process_hz <= 0.0:
            return True
        if self._last_pub_time is None:
            return True
        period = rospy.Duration.from_sec(1.0 / float(self.process_hz))
        return (now - self._last_pub_time) >= period

    def _compute_center(self, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return np.zeros((3,), dtype=np.float32)
        if self.center_mode in ("mean", "avg", "average"):
            return np.mean(pts, axis=0).astype(np.float32)
        return np.median(pts, axis=0).astype(np.float32)

    def _compute_radius(self, pts_xy: np.ndarray, center_xy: np.ndarray) -> float:
        radius_min = float(self.radius_min)
        radius_max = float(self.radius_max)
        if radius_max > 0.0 and radius_max < radius_min:
            radius_min, radius_max = radius_max, radius_min

        mode = (self.radius_mode or "constant").strip().lower()
        if mode in ("constant", "fixed"):
            radius = float(self.radius_constant)
        else:
            d = np.linalg.norm(pts_xy - center_xy.reshape(1, 2), axis=1)
            if d.size == 0:
                radius = float(self.radius_constant)
            elif mode in ("quantile", "percentile"):
                q = float(max(0.0, min(self.radius_quantile, 1.0)))
                radius = float(np.quantile(d, q))
            else:
                radius = float(np.median(d))

        if radius_min > 0.0:
            radius = max(radius, radius_min)
        if radius_max > 0.0:
            radius = min(radius, radius_max)
        return float(radius)

    def _circle_color(self, cluster_id: int) -> Tuple[float, float, float]:
        if self.color_mode in ("cluster", "id", "rainbow"):
            return _cluster_id_to_rgb(cluster_id)
        return self.color

    def _build_circle_marker(self, header: Header, circle: TreeCircle, marker_id: int) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "tree_circles"
        marker.id = int(marker_id)
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration.from_sec(max(0.0, float(self.marker_lifetime)))

        r, g, b = self._circle_color(marker_id)
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(max(0.0, min(self.alpha, 1.0)))

        cx, cy, cz, rad = float(circle.x), float(circle.y), float(circle.z), float(circle.radius)
        z = cz + float(self.marker_z_offset)

        marker_type = (self.marker_type or "line_strip").strip().lower()
        if marker_type in ("cylinder", "cyl"):
            marker.type = Marker.CYLINDER
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2.0 * rad
            marker.scale.y = 2.0 * rad
            marker.scale.z = 0.1
            return marker

        marker.type = Marker.LINE_STRIP
        marker.scale.x = float(self.line_width)
        segments = max(8, int(self.circle_segments))
        points: List[Point] = []
        for k in range(segments + 1):
            theta = (2.0 * math.pi) * float(k) / float(segments)
            points.append(Point(x=cx + rad * math.cos(theta), y=cy + rad * math.sin(theta), z=z))
        marker.points = points
        return marker

    def _build_label_marker(self, header: Header, circle: TreeCircle, marker_id: int, label_text: str) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "tree_labels"
        marker.id = int(marker_id)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration.from_sec(max(0.0, float(self.marker_lifetime)))

        marker.pose.position.x = float(circle.x)
        marker.pose.position.y = float(circle.y)
        marker.pose.position.z = float(circle.z) + float(self.label_z_offset)
        marker.pose.orientation.w = 1.0

        marker.scale.z = float(self.label_height)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.9
        marker.text = str(label_text)
        return marker

    def _delete_marker(self, header: Header, ns: str, marker_id: int) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = str(ns)
        marker.id = int(marker_id)
        marker.action = Marker.DELETE
        return marker

    def _filter_points_by_prior(self, points_xyz: np.ndarray) -> np.ndarray:
        if not self._prior_rows or self._prior_dir is None or self._prior_perp is None:
            return points_xyz
        xy = points_xyz[:, :2].astype(np.float32)
        u_vals = xy.dot(self._prior_dir.reshape(2))
        v_vals = xy.dot(self._prior_perp.reshape(2))
        keep = np.zeros((xy.shape[0],), dtype=bool)
        pad = float(self.prior_u_padding)
        band = float(self.prior_bandwidth)
        for row in self._prior_rows:
            u_min = float(row.get("u_min", -1.0e9)) - pad
            u_max = float(row.get("u_max", 1.0e9)) + pad
            v_center = float(row.get("v_center", 0.0))
            in_row = (u_vals >= u_min) & (u_vals <= u_max)
            in_band = np.abs(v_vals - v_center) <= band
            keep |= (in_row & in_band)
        return points_xyz[keep]

    def _transform_points(self, points_xyz: np.ndarray, header: Header) -> Tuple[np.ndarray, Header]:
        if not self.target_frame:
            return points_xyz, header
        if self._tf_buffer is None:
            return points_xyz, header

        src_frame = header.frame_id or self.target_frame
        if src_frame == self.target_frame:
            header.frame_id = self.target_frame
            return points_xyz, header

        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self.target_frame, src_frame, header.stamp, rospy.Duration(self.tf_timeout)
            )
        except Exception:
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self.target_frame, src_frame, rospy.Time(0), rospy.Duration(self.tf_timeout)
                )
            except Exception:
                rospy.logwarn_throttle(
                    2.0, "[orchard_row_mapping] TF lookup failed: %s -> %s", src_frame, self.target_frame
                )
                return np.empty((0, 3), dtype=np.float32), header

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        rot = _quat_to_rot(float(q.x), float(q.y), float(q.z), float(q.w))
        trans = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float32)
        points_xyz = (rot @ points_xyz.T).T + trans
        header.frame_id = self.target_frame
        return points_xyz, header

    def _split_row_indices(self, v_vals: np.ndarray) -> List[np.ndarray]:
        method = (self.row_split_method or "auto").strip().lower()
        min_per = max(1, int(self.row_min_centers))
        margin = float(self.row_split_margin)
        if method in ("auto", "v_sign"):
            left = np.where(v_vals < -margin)[0]
            right = np.where(v_vals > margin)[0]
            if left.size >= min_per and right.size >= min_per:
                return [left, right]
            if method == "v_sign":
                return []
        idx0, idx1, c0, c1 = _kmeans_1d(v_vals)
        if idx0.size < min_per or idx1.size < min_per:
            return []
        if abs(float(c0) - float(c1)) < float(self.row_min_separation):
            return []
        if float(np.mean(v_vals[idx0])) <= float(np.mean(v_vals[idx1])):
            return [idx0, idx1]
        return [idx1, idx0]

    def _fit_rows_from_centers(self, centers_xy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        if centers_xy.shape[0] < max(2, 2 * int(self.row_min_centers)):
            return []
        dir_xy = _fit_direction_from_xy(centers_xy)
        if dir_xy is None:
            return []
        perp_xy = np.array([-float(dir_xy[1]), float(dir_xy[0])], dtype=np.float32)
        u_vals = centers_xy.dot(dir_xy.reshape(2))
        v_vals = centers_xy.dot(perp_xy.reshape(2))
        groups = self._split_row_indices(v_vals)
        if not groups:
            return []
        rows: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for idxs in groups:
            if idxs.size < int(self.row_min_centers):
                continue
            u_row = u_vals[idxs]
            v_row = v_vals[idxs]
            v_center = float(np.median(v_row))
            if float(self.row_line_fixed_length) > 0.0:
                u_center = float(np.median(u_row))
                half = 0.5 * float(self.row_line_fixed_length)
                u_min = u_center - half
                u_max = u_center + half
            else:
                q = float(self.row_line_endpoint_percentile)
                if q > 0.0 and q < 0.49:
                    u_min = float(np.quantile(u_row, q))
                    u_max = float(np.quantile(u_row, 1.0 - q))
                else:
                    u_min = float(np.min(u_row))
                    u_max = float(np.max(u_row))
            if u_max <= u_min + 1.0e-4:
                continue
            p1 = dir_xy * float(u_min) + perp_xy * float(v_center)
            p2 = dir_xy * float(u_max) + perp_xy * float(v_center)
            rows.append((p1.astype(np.float32), p2.astype(np.float32), v_center))
        rows.sort(key=lambda r: float(r[2]))
        return rows[:2]

    def _build_row_marker(
        self,
        header: Header,
        marker_id: int,
        p1: np.ndarray,
        p2: np.ndarray,
        color: Tuple[float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "tree_rows"
        marker.id = int(marker_id)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration.from_sec(max(0.0, float(self.marker_lifetime)))
        marker.scale.x = float(self.row_line_width)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(max(0.0, min(self.row_line_alpha, 1.0)))
        z = float(self.row_line_z_offset)
        marker.points = [
            Point(x=float(p1[0]), y=float(p1[1]), z=z),
            Point(x=float(p2[0]), y=float(p2[1]), z=z),
        ]
        return marker

    def _on_cloud(self, msg: PointCloud2) -> None:
        now = rospy.Time.now()
        if not self._should_process(now):
            return

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        points_xyz = _cloud_to_xyz(msg)
        points_xyz = _filter_points(
            points_xyz,
            z_min=self.filter_z_min,
            z_max=self.filter_z_max,
            x_min=self.filter_x_min,
            x_max=self.filter_x_max,
            y_abs_max=self.filter_y_abs_max,
        )
        points_xyz, header = self._transform_points(points_xyz, header)
        if points_xyz.size == 0:
            return

        if not self.target_frame and self.frame_override:
            header.frame_id = self.frame_override

        points_xyz = self._filter_points_by_prior(points_xyz)
        points_xyz = _sample_points(points_xyz, self.max_points, seed=self.sample_seed)

        circles: List[TreeCircle] = []
        if points_xyz.size > 0:
            clusters = _cluster_cells(
                points_xyz[:, :2],
                cell_size=self.cluster_cell_size,
                neighbor_range=self.cluster_neighbor_range,
                min_points=self.min_points_per_cluster,
                max_clusters=self.max_clusters,
            )
            for cluster in clusters:
                pts = points_xyz[cluster]
                center = self._compute_center(pts)
                radius = self._compute_radius(pts[:, :2], center[:2])
                circles.append(
                    TreeCircle(
                        x=float(center[0]),
                        y=float(center[1]),
                        z=float(center[2]),
                        radius=float(radius),
                        n_points=int(pts.shape[0]),
                    )
                )

        markers = MarkerArray()
        for i, circle in enumerate(circles):
            markers.markers.append(self._build_circle_marker(header, circle, marker_id=i))
            if self.publish_labels:
                label_idx = int(self.label_start_index) + int(i)
                markers.markers.append(self._build_label_marker(header, circle, marker_id=i, label_text=str(label_idx)))

        # Delete stale markers when the count shrinks.
        current_count = len(circles)
        if self._last_circle_count > current_count:
            for stale_id in range(current_count, self._last_circle_count):
                markers.markers.append(self._delete_marker(header, "tree_circles", stale_id))
                if self.publish_labels:
                    markers.markers.append(self._delete_marker(header, "tree_labels", stale_id))
        self._last_circle_count = current_count

        self._circles_pub.publish(markers)
        if self._centers_pub is not None:
            centers = np.asarray([(c.x, c.y, c.z) for c in circles], dtype=np.float32)
            self._centers_pub.publish(_cloud_xyz(centers, header))

        if self._rows_pub is not None:
            centers_xy = np.asarray([(c.x, c.y) for c in circles], dtype=np.float32)
            rows = self._fit_rows_from_centers(centers_xy)
            row_markers = MarkerArray()
            colors = [self.row_left_color, self.row_right_color]
            for i, (p1, p2, _v) in enumerate(rows):
                row_markers.markers.append(self._build_row_marker(header, i, p1, p2, colors[i % 2]))
            if self._last_row_count > len(rows):
                for stale_id in range(len(rows), self._last_row_count):
                    row_markers.markers.append(self._delete_marker(header, "tree_rows", stale_id))
            self._last_row_count = len(rows)
            self._rows_pub.publish(row_markers)

        self._last_pub_time = now


def main() -> None:
    rospy.init_node("orchard_tree_clusters", anonymous=False)
    OrchardTreeClustersNode()
    rospy.spin()


if __name__ == "__main__":
    main()
