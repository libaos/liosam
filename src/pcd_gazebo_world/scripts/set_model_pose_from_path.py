#!/usr/bin/env python3
"""Set a Gazebo model pose using the first point of a saved path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
from pathlib import Path
from typing import List, Optional, Tuple

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler


def _load_json(path_file: Path) -> List[Tuple[float, float, float, Optional[float]]]:
    data = json.loads(path_file.read_text(encoding="utf-8"))
    points = []
    for pt in data.get("points", []):
        x = float(pt.get("x", 0.0))
        y = float(pt.get("y", 0.0))
        z = float(pt.get("z", 0.0))
        yaw = pt.get("yaw", None)
        yaw = float(yaw) if yaw is not None else None
        points.append((x, y, z, yaw))
    if not points:
        raise RuntimeError(f"No points in JSON: {path_file}")
    return points


def _load_csv(path_file: Path) -> List[Tuple[float, float, float, Optional[float]]]:
    points = []
    with path_file.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].strip().startswith("#") or row[0].strip().lower() == "index":
                continue
            try:
                _idx, x, y, z, yaw = row[:5]
                points.append((float(x), float(y), float(z), float(yaw)))
            except Exception:
                continue
    if not points:
        raise RuntimeError(f"No points in CSV: {path_file}")
    return points


def _compute_yaw(points: List[Tuple[float, float, float, Optional[float]]], idx: int) -> float:
    if idx + 1 < len(points):
        x0, y0, _z0, _yaw0 = points[idx]
        x1, y1, _z1, _yaw1 = points[idx + 1]
        return math.atan2(y1 - y0, x1 - x0)
    if idx > 0:
        x0, y0, _z0, _yaw0 = points[idx - 1]
        x1, y1, _z1, _yaw1 = points[idx]
        return math.atan2(y1 - y0, x1 - x0)
    return 0.0


def _wait_for_model(model_name: str, timeout: float) -> bool:
    start = rospy.Time.now()
    while not rospy.is_shutdown():
        try:
            msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=1.0)
        except rospy.ROSException:
            msg = None
        if msg is not None and model_name in msg.name:
            return True
        if (rospy.Time.now() - start).to_sec() > timeout:
            return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Set Gazebo model pose from a path file")
    parser.add_argument("--path", type=str, default="maps/runs/rosbag_path.json", help="输入 JSON/CSV 路径文件")
    parser.add_argument("--model", type=str, default="scout", help="Gazebo model name")
    parser.add_argument("--index", type=int, default=0, help="使用的路径索引（-1=最后）")
    parser.add_argument("--z", type=float, default=0.2, help="覆盖 z 高度 (m)")
    parser.add_argument("--frame", type=str, default="world", help="参考坐标系")
    parser.add_argument("--wait", type=float, default=8.0, help="等待服务超时 (s)")
    args = parser.parse_args(rospy.myargv()[1:])

    path_file = Path(args.path).expanduser().resolve()
    if not path_file.is_file():
        raise SystemExit(f"path file not found: {path_file}")

    if path_file.suffix.lower() == ".json":
        points = _load_json(path_file)
    else:
        points = _load_csv(path_file)

    idx = int(args.index)
    if idx < 0:
        idx = len(points) - 1
    idx = max(0, min(idx, len(points) - 1))

    x, y, z, yaw = points[idx]
    if yaw is None:
        yaw = _compute_yaw(points, idx)

    z = float(args.z)

    rospy.init_node("set_model_pose_from_path", anonymous=True)
    wait_s = float(args.wait)
    rospy.wait_for_service("/gazebo/set_model_state", timeout=wait_s)
    if not _wait_for_model(str(args.model), wait_s):
        raise SystemExit(f"Model not found: {args.model}")
    service = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    state = ModelState()
    state.model_name = str(args.model)
    state.reference_frame = str(args.frame)
    state.pose.position.x = float(x)
    state.pose.position.y = float(y)
    state.pose.position.z = float(z)

    q = quaternion_from_euler(0.0, 0.0, float(yaw))
    state.pose.orientation.x = q[0]
    state.pose.orientation.y = q[1]
    state.pose.orientation.z = q[2]
    state.pose.orientation.w = q[3]

    resp = service(state)
    if not resp.success:
        raise SystemExit(f"SetModelState failed: {resp.status_message}")

    print(f"[OK] model={state.model_name} x={x:.3f} y={y:.3f} yaw={yaw:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
