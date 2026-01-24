#!/usr/bin/env python3
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger, TriggerResponse

try:
    import cv2
except Exception:
    cv2 = None


class DebugExportNode:
    def __init__(self) -> None:
        self.tree_points_topic = rospy.get_param("~tree_points_topic", "/tree_points")
        self.bev_occ_topic = rospy.get_param("~bev_occ_topic", "/bev_occ")
        self.output_dir = rospy.get_param("~output_dir", "")
        self.save_mode = rospy.get_param("~save_mode", "trigger")
        self.save_rate_hz = float(rospy.get_param("~save_rate_hz", 1.0))
        self.max_files = int(rospy.get_param("~max_files", 0))

        self.grid_res = float(rospy.get_param("~grid_res", 0.05))
        self.grid_x_min = float(rospy.get_param("~grid_x_min", 0.0))
        self.grid_x_max = float(rospy.get_param("~grid_x_max", 12.0))
        self.grid_y_min = float(rospy.get_param("~grid_y_min", -4.0))
        self.grid_y_max = float(rospy.get_param("~grid_y_max", 4.0))
        self.point_size = int(rospy.get_param("~point_size", 1))
        self.save_occ_bev = bool(rospy.get_param("~save_occ_bev", True))
        self.use_occ_grid_dims = bool(rospy.get_param("~use_occ_grid_dims", True))

        self._tree_header = None
        self._tree_points = None
        self._bev_occ: Optional[OccupancyGrid] = None
        self._save_counter = 0
        self._saved_files = 0
        self._lock = threading.Lock()

        self._output_dir = self._resolve_output_dir(self.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._tree_sub = rospy.Subscriber(self.tree_points_topic, PointCloud2, self._tree_callback, queue_size=1)
        self._bev_sub = rospy.Subscriber(self.bev_occ_topic, OccupancyGrid, self._bev_callback, queue_size=1)

        self._srv_pcd = rospy.Service("save_tree_pcd_once", Trigger, self._handle_save_pcd)
        self._srv_bev = rospy.Service("save_tree_bev_once", Trigger, self._handle_save_bev)

        self._last_auto_save = rospy.Time(0)
        self._last_tree_stamp = None

        if self.save_mode == "auto" and self.save_rate_hz > 0:
            period = 1.0 / self.save_rate_hz
            self._timer = rospy.Timer(rospy.Duration(period), self._auto_save)

        rospy.loginfo("[debug_export] tree=%s occ=%s out=%s", self.tree_points_topic, self.bev_occ_topic, self._output_dir)

    def _resolve_output_dir(self, output_dir: str) -> Path:
        if output_dir and str(output_dir).strip():
            return Path(output_dir).expanduser().resolve()
        return Path(rospy.get_param("~ros_home", "~/.ros")).expanduser().resolve() / "corridor_debug"

    def _tree_callback(self, msg: PointCloud2) -> None:
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points_xyz = np.asarray(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)
        with self._lock:
            self._tree_header = msg.header
            self._tree_points = points_xyz
            self._last_tree_stamp = msg.header.stamp

    def _bev_callback(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._bev_occ = msg

    def _auto_save(self, _event) -> None:
        with self._lock:
            if self._last_tree_stamp is None:
                return
            if self._saved_files >= self.max_files and self.max_files > 0:
                return
            if self._tree_points is None:
                return
            if self._last_tree_stamp == self._last_auto_save:
                return
            self._last_auto_save = self._last_tree_stamp
            self._save_all_locked()

    def _handle_save_pcd(self, _req) -> TriggerResponse:
        with self._lock:
            if self._tree_points is None or self._tree_header is None:
                return TriggerResponse(success=False, message="no tree_points received yet")
            if self._saved_files >= self.max_files and self.max_files > 0:
                return TriggerResponse(success=False, message="max_files reached")
            path = self._save_pcd_locked()
            return TriggerResponse(success=True, message=str(path))

    def _handle_save_bev(self, _req) -> TriggerResponse:
        with self._lock:
            if self._tree_points is None or self._tree_header is None:
                return TriggerResponse(success=False, message="no tree_points received yet")
            if self._saved_files >= self.max_files and self.max_files > 0:
                return TriggerResponse(success=False, message="max_files reached")
            if cv2 is None:
                return TriggerResponse(success=False, message="cv2 not available for PNG export")
            paths = self._save_bev_locked()
            return TriggerResponse(success=True, message=",".join(paths))

    def _save_all_locked(self) -> None:
        if self._tree_points is None or self._tree_header is None:
            return
        if cv2 is None:
            rospy.logwarn_throttle(2.0, "[debug_export] cv2 not available; PNG export disabled")
        self._save_pcd_locked()
        if cv2 is not None:
            self._save_bev_locked()

    def _make_basename(self, prefix: str, header) -> str:
        stamp_ns = int(header.stamp.to_nsec()) if header is not None else int(rospy.Time.now().to_nsec())
        seq = int(getattr(header, "seq", 0)) if header is not None else 0
        name = f"{prefix}_{stamp_ns}_{seq}_{self._save_counter}"
        self._save_counter += 1
        return name

    def _save_pcd_locked(self) -> Path:
        base = self._make_basename("tree_points", self._tree_header)
        path = self._output_dir / f"{base}.pcd"
        self._write_pcd_xyz(path, self._tree_points)
        self._saved_files += 1
        rospy.loginfo("[debug_export] saved %s", path)
        return path

    def _save_bev_locked(self) -> Tuple[str, ...]:
        base = self._make_basename("tree_bev", self._tree_header)
        tree_path = self._output_dir / f"{base}.png"
        self._write_tree_bev_png(tree_path)

        saved = [str(tree_path)]
        if self.save_occ_bev and self._bev_occ is not None:
            occ_path = self._output_dir / f"occ_bev_{base.split('tree_bev_')[-1]}.png"
            self._write_occ_bev_png(occ_path, self._bev_occ)
            saved.append(str(occ_path))

        self._saved_files += 1
        rospy.loginfo("[debug_export] saved %s", ",".join(saved))
        return tuple(saved)

    def _write_pcd_xyz(self, path: Path, points: np.ndarray) -> None:
        points = points.astype(np.float32, copy=False)
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {points.shape[0]}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {points.shape[0]}\n"
            "DATA binary\n"
        ).encode("ascii")
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(points.tobytes())

    def _grid_params(self) -> Tuple[float, float, float, float, float, int, int]:
        if self.use_occ_grid_dims and self._bev_occ is not None:
            res = self._bev_occ.info.resolution
            x_min = self._bev_occ.info.origin.position.x
            y_min = self._bev_occ.info.origin.position.y
            width = int(self._bev_occ.info.width)
            height = int(self._bev_occ.info.height)
            x_max = x_min + width * res
            y_max = y_min + height * res
            return res, x_min, x_max, y_min, y_max, width, height
        res = self.grid_res
        x_min = self.grid_x_min
        x_max = self.grid_x_max
        y_min = self.grid_y_min
        y_max = self.grid_y_max
        width = int(np.ceil((x_max - x_min) / res))
        height = int(np.ceil((y_max - y_min) / res))
        return res, x_min, x_max, y_min, y_max, width, height

    def _write_tree_bev_png(self, path: Path) -> None:
        if cv2 is None:
            return
        res, x_min, x_max, y_min, y_max, width, height = self._grid_params()
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        if self._tree_points is None or self._tree_points.size == 0:
            cv2.imwrite(str(path), img)
            return
        x = self._tree_points[:, 0]
        y = self._tree_points[:, 1]
        ix = ((x - x_min) / res).astype(np.int32)
        iy = ((y_max - y) / res).astype(np.int32)
        mask = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        ix = ix[mask]
        iy = iy[mask]
        if ix.size == 0:
            cv2.imwrite(str(path), img)
            return
        if self.point_size <= 1:
            img[iy, ix] = (0, 0, 0)
        else:
            radius = max(1, int(self.point_size))
            for px, py in zip(ix.tolist(), iy.tolist()):
                cv2.circle(img, (int(px), int(py)), radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.imwrite(str(path), img)

    def _write_occ_bev_png(self, path: Path, occ: OccupancyGrid) -> None:
        if cv2 is None:
            return
        width = int(occ.info.width)
        height = int(occ.info.height)
        data = np.array(occ.data, dtype=np.int16).reshape((height, width))
        data = np.flipud(data)
        img = np.full((height, width), 255, dtype=np.uint8)
        img[data == -1] = 128
        img[data >= 50] = 0
        cv2.imwrite(str(path), img)


if __name__ == "__main__":
    rospy.init_node("debug_export")
    DebugExportNode()
    rospy.spin()
