#!/usr/bin/env python3
"""ROS1 (rospy) orchard row boundary fitting from segmented fruit-tree PointCloud2.

Goal:
  Publish stable left/right "road side" row boundaries (line segments) in the vehicle frame.

Input:
  - sensor_msgs/PointCloud2 with fields at least: x, y, z, <label_field>
  - label==0 is fruit-tree; other labels ignored

Core pipeline (per frame):
  1) ROI crop + voxel downsample (tree points only)
  2) 2D grid connected-components instancing (XY projection) -> clusters
  3) Each cluster -> detection point (cx,cy) = median
  4) Row fitting uses last N frames of detection points (history)
  5) Optional TF motion compensation to a fixed frame (map/odom_est)
  6) Left/right grouping via deterministic k=2 clustering on lateral axis (PCA normal projection)
  7) Per-side robust PCA/TLS line fit + deterministic inlier pruning
  8) Clip infinite lines to ROI -> finite segments
  9) Temporal smoothing (EMA) + hold-last when insufficient points

Output:
  - std_msgs/String (~row_fit_json, default /tree_row_fit_json)
    JSON:
      {"frame_index":int,"stamp":float,"left":{...},"right":{...}}
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

# Reuse core utilities from the main tracker node to avoid duplicating point cloud parsing math.
import fruit_tree_tracker_node as _core


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _wrap_pi(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return float(a)


def _yaw_from_quat_z(qx: float, qy: float, qz: float, qw: float) -> float:
    # Standard yaw extraction (Z axis) from quaternion.
    siny_cosp = 2.0 * (float(qw) * float(qz) + float(qx) * float(qy))
    cosy_cosp = 1.0 - 2.0 * (float(qy) * float(qy) + float(qz) * float(qz))
    return float(math.atan2(siny_cosp, cosy_cosp))


def _pca_direction(points_xy: np.ndarray, *, last_v: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float32).reshape((-1, 2))
    if int(pts.shape[0]) < 2:
        return None
    mu = np.mean(pts, axis=0)
    x = pts - mu[None, :]
    cov = (x.T @ x).astype(np.float32, copy=False)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))].astype(np.float32, copy=False)
    nrm = float(np.linalg.norm(v))
    if not (nrm > 1e-12):
        return None
    v = (v / nrm).astype(np.float32, copy=False)

    # Stabilize sign ambiguity using last_v when available.
    if last_v is not None:
        lv = np.asarray(last_v, dtype=np.float32).reshape((2,))
        if float(lv[0] * v[0] + lv[1] * v[1]) < 0.0:
            v = (-v).astype(np.float32, copy=False)

    # Prefer pointing "forward" in the current coordinate frame when possible.
    if float(v[0]) < 0.0 or (abs(float(v[0])) < 1e-12 and float(v[1]) < 0.0):
        v = (-v).astype(np.float32, copy=False)

    return v


def _kmeans_1d_two_clusters(values: np.ndarray, *, iters: int = 10) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(values, dtype=np.float32).reshape((-1,))
    if int(x.shape[0]) == 0:
        return np.zeros((0,), dtype=bool), 0.0, 0.0
    c0 = float(np.min(x))
    c1 = float(np.max(x))
    if not (math.isfinite(c0) and math.isfinite(c1)) or abs(c1 - c0) < 1e-12:
        labels = np.zeros((int(x.shape[0]),), dtype=bool)
        return labels, float(c0), float(c1)

    labels = np.zeros((int(x.shape[0]),), dtype=bool)
    for _ in range(int(max(1, iters))):
        d0 = np.abs(x - float(c0))
        d1 = np.abs(x - float(c1))
        new_labels = d0 <= d1
        if bool(np.array_equal(new_labels, labels)):
            break
        labels = new_labels
        if not bool(np.any(labels)) or bool(np.all(labels)):
            break
        c0_new = float(np.mean(x[labels]))
        c1_new = float(np.mean(x[~labels]))
        if not (math.isfinite(c0_new) and math.isfinite(c1_new)):
            break
        if abs(c0_new - c0) < 1e-6 and abs(c1_new - c1) < 1e-6:
            c0, c1 = c0_new, c1_new
            break
        c0, c1 = c0_new, c1_new

    return labels, float(c0), float(c1)


def _line_dir_to_yaw(v: np.ndarray) -> float:
    vv = np.asarray(v, dtype=np.float32).reshape((2,))
    return float(math.atan2(float(vv[1]), float(vv[0])))


def _yaw_to_dir(yaw: float) -> np.ndarray:
    return np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)


@dataclass
class SmoothedLine:
    valid: bool = False
    p0: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=np.float32))
    v: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    inliers: int = 0
    total: int = 0
    rms: float = 0.0
    hold_frames: int = 0


class PointCloud2TreeParser:
    def __init__(self, *, label_field: str):
        self.label_field = str(label_field)
        self._dtype_cache_key = None
        self._dtype_cache = None

    def parse(self, msg) -> np.ndarray:
        from sensor_msgs.msg import PointField

        required = ("x", "y", "z", str(self.label_field))
        fields = {f.name: f for f in msg.fields}
        for name in required:
            if name not in fields:
                raise KeyError(f"missing field: {name}")

        cache_key = (
            tuple((f.name, int(f.offset), int(f.datatype), int(f.count)) for f in msg.fields),
            int(msg.point_step),
            int(msg.row_step),
            int(msg.width),
            int(msg.height),
            bool(msg.is_bigendian),
            required,
        )
        if cache_key != self._dtype_cache_key:
            endian = ">" if bool(msg.is_bigendian) else "<"
            dt_map = {
                PointField.INT8: "i1",
                PointField.UINT8: "u1",
                PointField.INT16: "i2",
                PointField.UINT16: "u2",
                PointField.INT32: "i4",
                PointField.UINT32: "u4",
                PointField.FLOAT32: "f4",
                PointField.FLOAT64: "f8",
            }

            names = []
            formats = []
            offsets = []
            for name in required:
                f = fields[name]
                if int(f.count) != 1:
                    raise ValueError(f"field {name} has count={int(f.count)}; expected 1")
                base = dt_map.get(int(f.datatype))
                if base is None:
                    raise ValueError(f"unsupported datatype for field {name}: {int(f.datatype)}")
                names.append(str(name))
                formats.append(np.dtype(endian + base))
                offsets.append(int(f.offset))

            self._dtype_cache = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(msg.point_step)})
            self._dtype_cache_key = cache_key

        dtype = self._dtype_cache
        if dtype is None:
            return np.empty((0, 3), dtype=np.float32)

        points_count = int(msg.width) * int(msg.height)
        if points_count <= 0:
            return np.empty((0, 3), dtype=np.float32)

        raw = memoryview(msg.data)
        if int(msg.height) == 1 or int(msg.row_step) == int(msg.point_step) * int(msg.width):
            arr = np.frombuffer(raw, dtype=dtype, count=points_count)
        else:
            rows = []
            width_bytes = int(msg.width) * int(msg.point_step)
            for r in range(int(msg.height)):
                start = int(r) * int(msg.row_step)
                row_view = raw[start : start + width_bytes]
                rows.append(np.frombuffer(row_view, dtype=dtype, count=int(msg.width)))
            arr = np.concatenate(rows, axis=0) if rows else np.empty((0,), dtype=dtype)

        x = arr["x"].astype(np.float32, copy=False)
        y = arr["y"].astype(np.float32, copy=False)
        z = arr["z"].astype(np.float32, copy=False)
        label = arr[str(self.label_field)]
        if label.dtype.kind == "f":
            label_i = np.rint(label).astype(np.int32)
        else:
            label_i = label.astype(np.int32, copy=False)

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        tree_mask = finite & (label_i == 0)
        if not bool(np.any(tree_mask)):
            return np.empty((0, 3), dtype=np.float32)

        return np.stack([x[tree_mask], y[tree_mask], z[tree_mask]], axis=1).astype(np.float32, copy=False)


class TreeRowFitNode:
    def __init__(self) -> None:
        import rospy
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String

        self.rospy = rospy
        self.PointCloud2 = PointCloud2
        self.String = String

        rospy.init_node("tree_row_fit", anonymous=False)

        self.input_topic = rospy.get_param("~input_topic", "/segmented_points")
        self.label_field = rospy.get_param("~label_field", "label")
        self.out_topic = rospy.get_param("~row_fit_json", "/tree_row_fit_json")

        roi = _core.RoiParams(
            x_min=float(rospy.get_param("~roi_x_min", 0.0)),
            x_max=float(rospy.get_param("~roi_x_max", 10.0)),
            y_min=float(rospy.get_param("~roi_y_min", -4.0)),
            y_max=float(rospy.get_param("~roi_y_max", 4.0)),
            z_min=float(rospy.get_param("~roi_z_min", -0.5)),
            z_max=float(rospy.get_param("~roi_z_max", 2.5)),
        )
        self.roi = roi
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.03))
        self.grid = _core.GridParams(
            cell_size=float(rospy.get_param("~cell_size", 0.10)),
            count_threshold=int(rospy.get_param("~grid_T", 5)),
        )
        self.instance_min_points = int(rospy.get_param("~instance_min_points", 6))
        self.instance_merge_distance = float(rospy.get_param("~instance_merge_distance", 0.35))

        self.row_fit_history_frames = int(rospy.get_param("~row_fit_history_frames", 20))
        self.row_fit_inlier_dist = float(rospy.get_param("~row_fit_inlier_dist", 0.20))
        self.row_fit_min_points = int(rospy.get_param("~row_fit_min_points", 3))
        self.row_fit_iters = int(rospy.get_param("~row_fit_iters", 2))
        self.row_fit_ema_alpha = float(rospy.get_param("~row_fit_ema_alpha", 0.30))
        self.row_fit_hold_max_frames = int(rospy.get_param("~row_fit_hold_max_frames", 15))
        self.row_fit_fixed_frame = rospy.get_param("~row_fit_fixed_frame", "")
        self.row_fit_fixed_frame_timeout = float(rospy.get_param("~row_fit_fixed_frame_timeout", 0.05))
        self.row_fit_reset_yaw_deg = float(rospy.get_param("~row_fit_reset_yaw_deg", 30.0))

        self._frame_index = 0
        self._pub = rospy.Publisher(str(self.out_topic), String, queue_size=1)
        self._parser = PointCloud2TreeParser(label_field=str(self.label_field))

        self._hist: Optional[Deque[np.ndarray]] = None
        if int(self.row_fit_history_frames) > 1:
            from collections import deque

            self._hist = deque(maxlen=int(self.row_fit_history_frames))

        self._tf_buffer = None
        self._tf_listener = None
        self._fixed_frame = ""
        self._fixed_timeout = float(max(0.0, float(self.row_fit_fixed_frame_timeout)))
        self._last_fixed_yaw: Optional[float] = None
        if isinstance(self.row_fit_fixed_frame, str) and self.row_fit_fixed_frame.strip():
            self._fixed_frame = str(self.row_fit_fixed_frame).strip()
            try:
                import tf2_ros

                self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
                self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
            except Exception as exc:
                rospy.logwarn("[tree_row_fit] row_fit_fixed_frame disabled (tf2_ros unavailable): %s", str(exc))
                self._fixed_frame = ""

        self._last_row_v_hist: Optional[np.ndarray] = None
        self._left = SmoothedLine()
        self._right = SmoothedLine()

        self._sub = rospy.Subscriber(self.input_topic, PointCloud2, self._on_cloud, queue_size=1, buff_size=2**24)
        rospy.loginfo("[tree_row_fit] Listening on %s (label_field=%s)", str(self.input_topic), str(self.label_field))
        rospy.loginfo("[tree_row_fit] Publishing %s", str(self.out_topic))
        if bool(self._fixed_frame):
            rospy.loginfo("[tree_row_fit] row_fit_fixed_frame=%s (motion-compensated history)", str(self._fixed_frame))

    def _merge_close_centers(self, centers_xy: np.ndarray, *, merge_distance: float) -> np.ndarray:
        merge_distance = float(max(0.0, merge_distance))
        pts = np.asarray(centers_xy, dtype=np.float32).reshape((-1, 2))
        if merge_distance <= 0.0 or int(pts.shape[0]) < 2:
            return pts
        merge2 = merge_distance * merge_distance

        n = int(pts.shape[0])
        parent = list(range(n))

        def _find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(a: int, b: int) -> None:
            ra = _find(a)
            rb = _find(b)
            if ra == rb:
                return
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                dx = float(pts[i, 0] - pts[j, 0])
                dy = float(pts[i, 1] - pts[j, 1])
                if dx * dx + dy * dy <= float(merge2):
                    _union(i, j)

        groups: Dict[int, list] = {}
        for i in range(n):
            r = _find(i)
            groups.setdefault(int(r), []).append(int(i))

        merged = []
        for r in sorted(groups.keys()):
            idxs = groups[r]
            merged.append(np.median(pts[idxs], axis=0))
        return np.asarray(merged, dtype=np.float32).reshape((-1, 2))

    def _update_line_state(self, state: SmoothedLine, *, p0: np.ndarray, v: np.ndarray, stats: Dict[str, float]) -> None:
        alpha = _clamp(float(self.row_fit_ema_alpha), 0.0, 1.0)
        p0 = np.asarray(p0, dtype=np.float32).reshape((2,))
        v = np.asarray(v, dtype=np.float32).reshape((2,))
        nrm = float(np.linalg.norm(v))
        if not (nrm > 1e-12):
            return
        v = (v / nrm).astype(np.float32, copy=False)
        if float(v[0]) < 0.0 or (abs(float(v[0])) < 1e-12 and float(v[1]) < 0.0):
            v = (-v).astype(np.float32, copy=False)

        if not bool(state.valid):
            state.valid = True
            state.p0 = p0
            state.v = v
        else:
            state.p0 = (alpha * p0 + (1.0 - alpha) * state.p0).astype(np.float32, copy=False)
            yaw_prev = _line_dir_to_yaw(state.v)
            yaw_new = _line_dir_to_yaw(v)
            yaw_sm = float(yaw_prev + alpha * _wrap_pi(float(yaw_new) - float(yaw_prev)))
            state.v = _yaw_to_dir(yaw_sm)

        state.inliers = int(stats.get("inliers", 0.0))
        state.total = int(stats.get("total", 0.0))
        state.rms = float(stats.get("rms", 0.0))
        state.hold_frames = 0

    def _hold_line_state(self, state: SmoothedLine) -> None:
        if not bool(state.valid):
            return
        state.hold_frames += 1
        if int(self.row_fit_hold_max_frames) > 0 and int(state.hold_frames) > int(self.row_fit_hold_max_frames):
            state.valid = False

    def _clip_segment(self, state: SmoothedLine) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not bool(state.valid):
            return None
        ok, pA, pB = _core.clip_line_to_roi(state.p0, state.v, self.roi)
        if not bool(ok):
            return None
        return pA, pB

    def _on_cloud(self, msg) -> None:
        rospy = self.rospy
        frame_index = int(self._frame_index)
        stamp_sec = float(msg.header.stamp.to_sec())
        try:
            try:
                pts = self._parser.parse(msg)
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "[tree_row_fit] Failed to parse PointCloud2: %s", str(exc))
                pts = np.empty((0, 3), dtype=np.float32)

            pts = pts.astype(np.float32, copy=False)
            pts = pts[_core._finite_xyz_mask(pts)]
            pts = _core._roi_crop(pts, self.roi)
            pts = _core._voxel_downsample(pts, float(self.voxel_size))

            observations = _core._grid_instances(pts, self.roi, self.grid)
            observations = [o for o in observations if int(o.point_count) >= int(max(1, self.instance_min_points))]
            centers_cur = np.array([[float(o.cx), float(o.cy)] for o in observations], dtype=np.float32).reshape((-1, 2))
            centers_cur = self._merge_close_centers(centers_cur, merge_distance=float(self.instance_merge_distance))

            use_fixed = bool(self._fixed_frame) and (self._tf_buffer is not None)
            tf_msg_to_fixed = None
            tf_fixed_to_msg = None
            centers_hist = None
            tf_ok = False

            if bool(use_fixed):
                try:
                    tf_msg_to_fixed = self._tf_buffer.lookup_transform(
                        str(self._fixed_frame),
                        str(msg.header.frame_id),
                        msg.header.stamp,
                        rospy.Duration(float(self._fixed_timeout)),
                    )

                    rot = tf_msg_to_fixed.transform.rotation
                    yaw = _yaw_from_quat_z(float(rot.x), float(rot.y), float(rot.z), float(rot.w))
                    if self._last_fixed_yaw is not None and float(self.row_fit_reset_yaw_deg) > 0.0:
                        diff = abs(_wrap_pi(float(yaw) - float(self._last_fixed_yaw)))
                        if float(diff) * 180.0 / math.pi > float(self.row_fit_reset_yaw_deg):
                            if self._hist is not None:
                                self._hist.clear()
                            self._last_row_v_hist = None
                            self._left = SmoothedLine()
                            self._right = SmoothedLine()
                            rospy.loginfo_throttle(
                                2.0,
                                "[tree_row_fit] history reset: yaw jump %.1f deg > %.1f deg",
                                float(diff) * 180.0 / math.pi,
                                float(self.row_fit_reset_yaw_deg),
                            )
                    self._last_fixed_yaw = float(yaw)

                    centers_fixed_cur = _core._apply_tf_xy(centers_cur, tf_msg_to_fixed)
                    if self._hist is not None:
                        self._hist.append(centers_fixed_cur)
                        centers_hist = (
                            np.concatenate(list(self._hist), axis=0) if len(self._hist) > 0 else np.empty((0, 2), dtype=np.float32)
                        )
                    else:
                        centers_hist = centers_fixed_cur

                    tf_fixed_to_msg = self._tf_buffer.lookup_transform(
                        str(msg.header.frame_id),
                        str(self._fixed_frame),
                        msg.header.stamp,
                        rospy.Duration(float(self._fixed_timeout)),
                    )
                    tf_ok = True
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "[tree_row_fit] TF lookup failed: %s", str(exc))
                    tf_msg_to_fixed = None
                    tf_fixed_to_msg = None

            if centers_hist is None:
                if bool(use_fixed):
                    # Do not mix coordinate frames in history if TF is unavailable; just reuse existing history.
                    if self._hist is not None and len(self._hist) > 0:
                        centers_hist = np.concatenate(list(self._hist), axis=0)
                    else:
                        centers_hist = np.empty((0, 2), dtype=np.float32)
                else:
                    if self._hist is not None:
                        self._hist.append(centers_cur)
                        centers_hist = (
                            np.concatenate(list(self._hist), axis=0) if len(self._hist) > 0 else np.empty((0, 2), dtype=np.float32)
                        )
                    else:
                        centers_hist = centers_cur

            centers_hist = np.asarray(centers_hist, dtype=np.float32).reshape((-1, 2))

            left_updated = False
            right_updated = False

            if bool(use_fixed) and not bool(tf_ok):
                # Fixed-frame mode requested but TF unavailable for this frame: hold last result.
                left_updated = False
                right_updated = False
            elif int(centers_hist.shape[0]) >= int(max(2, self.row_fit_min_points)):
                v_row = _pca_direction(centers_hist, last_v=self._last_row_v_hist)
                if v_row is not None:
                    self._last_row_v_hist = v_row
                    n = np.array([-float(v_row[1]), float(v_row[0])], dtype=np.float32)
                    nrm = float(np.linalg.norm(n))
                    if nrm > 1e-12:
                        n = (n / nrm).astype(np.float32, copy=False)
                        mu = np.mean(centers_hist, axis=0)
                        s = ((centers_hist - mu[None, :]) @ n.reshape((2, 1))).reshape((-1,))
                        labels, c0, c1 = _kmeans_1d_two_clusters(s, iters=10)
                        if int(labels.shape[0]) == int(centers_hist.shape[0]):
                            # labels==True -> cluster A
                            pts_a = centers_hist[labels]
                            pts_b = centers_hist[~labels]

                            # Decide which cluster is "left" in the current vehicle frame by transforming
                            # the per-cluster p0 into msg frame and comparing y.
                            cand = []
                            for pts_side in (pts_a, pts_b):
                                ok, p0, v, stats = _core.fit_line_pca(
                                    pts_side,
                                    inlier_dist=float(self.row_fit_inlier_dist),
                                    min_points=int(self.row_fit_min_points),
                                    iters=int(self.row_fit_iters),
                                )
                                if not bool(ok):
                                    cand.append((False, None, None, stats))
                                    continue
                                p0_use = p0
                                v_use = v
                                if bool(use_fixed) and tf_fixed_to_msg is not None and tf_msg_to_fixed is not None:
                                    p0m = _core._apply_tf_xy(p0_use.reshape((1, 2)), tf_fixed_to_msg)[0]
                                    p1m = _core._apply_tf_xy((p0_use + v_use).reshape((1, 2)), tf_fixed_to_msg)[0]
                                    vv = (p1m - p0m).astype(np.float32, copy=False)
                                    nv = float(np.linalg.norm(vv))
                                    if nv > 1e-12:
                                        vv = (vv / nv).astype(np.float32, copy=False)
                                    p0_use = p0m
                                    v_use = vv
                                cand.append((True, p0_use, v_use, stats))

                            # Assign by y (bigger y => left).
                            (ok0, p00, v0, st0), (ok1, p01, v1, st1) = cand[0], cand[1]
                            if bool(ok0) and bool(ok1):
                                if float(p00[1]) >= float(p01[1]):
                                    left_ok, left_p0, left_v, left_st = ok0, p00, v0, st0
                                    right_ok, right_p0, right_v, right_st = ok1, p01, v1, st1
                                else:
                                    left_ok, left_p0, left_v, left_st = ok1, p01, v1, st1
                                    right_ok, right_p0, right_v, right_st = ok0, p00, v0, st0
                            elif bool(ok0):
                                if float(p00[1]) >= 0.0:
                                    left_ok, left_p0, left_v, left_st = ok0, p00, v0, st0
                                    right_ok, right_p0, right_v, right_st = False, None, None, st1
                                else:
                                    left_ok, left_p0, left_v, left_st = False, None, None, st1
                                    right_ok, right_p0, right_v, right_st = ok0, p00, v0, st0
                            elif bool(ok1):
                                if float(p01[1]) >= 0.0:
                                    left_ok, left_p0, left_v, left_st = ok1, p01, v1, st1
                                    right_ok, right_p0, right_v, right_st = False, None, None, st0
                                else:
                                    left_ok, left_p0, left_v, left_st = False, None, None, st0
                                    right_ok, right_p0, right_v, right_st = ok1, p01, v1, st1
                            else:
                                left_ok = right_ok = False
                                left_p0 = left_v = right_p0 = right_v = None
                                left_st = st0
                                right_st = st1

                            if bool(left_ok) and left_p0 is not None and left_v is not None:
                                self._update_line_state(self._left, p0=left_p0, v=left_v, stats=left_st)
                                left_updated = True
                            if bool(right_ok) and right_p0 is not None and right_v is not None:
                                self._update_line_state(self._right, p0=right_p0, v=right_v, stats=right_st)
                                right_updated = True

            if not bool(left_updated):
                self._hold_line_state(self._left)
            if not bool(right_updated):
                self._hold_line_state(self._right)

            left_seg = self._clip_segment(self._left)
            right_seg = self._clip_segment(self._right)

            payload = {
                "frame_index": int(frame_index),
                "stamp": float(stamp_sec),
                "left": {
                    "valid": bool(left_seg is not None),
                    "held": bool(self._left.valid and not bool(left_updated)),
                    "segment": (
                        {"pA": [float(left_seg[0][0]), float(left_seg[0][1])], "pB": [float(left_seg[1][0]), float(left_seg[1][1])]}
                        if left_seg is not None
                        else None
                    ),
                    "inliers": int(self._left.inliers),
                    "total": int(self._left.total),
                    "rms": float(self._left.rms),
                },
                "right": {
                    "valid": bool(right_seg is not None),
                    "held": bool(self._right.valid and not bool(right_updated)),
                    "segment": (
                        {"pA": [float(right_seg[0][0]), float(right_seg[0][1])], "pB": [float(right_seg[1][0]), float(right_seg[1][1])]}
                        if right_seg is not None
                        else None
                    ),
                    "inliers": int(self._right.inliers),
                    "total": int(self._right.total),
                    "rms": float(self._right.rms),
                },
            }

            self._pub.publish(self.String(data=json.dumps(payload, ensure_ascii=False)))
        finally:
            self._frame_index += 1


def main() -> int:
    import rospy

    _ = TreeRowFitNode()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
