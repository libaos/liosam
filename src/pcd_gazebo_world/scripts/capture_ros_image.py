#!/usr/bin/env python3
"""Capture a single ROS Image message and save it to disk.

Usage (with Gazebo running):
  python3 src/pcd_gazebo_world/scripts/capture_ros_image.py \
    --topic /orchard_cam/camera/image_raw \
    --out maps/runs/gazebo_orchard_cam.png
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one ROS image and write PNG/JPG")
    parser.add_argument("--topic", type=str, default="/orchard_cam/camera/image_raw", help="ROS image topic")
    parser.add_argument("--out", type=str, default="maps/runs/gazebo_orchard_cam.png", help="输出图片路径")
    parser.add_argument("--timeout", type=float, default=10.0, help="等待图像超时（秒）")
    parser.add_argument("--encoding", type=str, default="bgr8", help="cv_bridge 期望编码（常用 bgr8/rgb8/passthrough）")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rospy.init_node("capture_ros_image", anonymous=True, disable_signals=True)
    bridge = CvBridge()
    try:
        msg = rospy.wait_for_message(str(args.topic), Image, timeout=float(args.timeout))
    except rospy.ROSException as exc:
        raise SystemExit(f"Timeout waiting for {args.topic}: {exc}") from exc

    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=str(args.encoding))
    if cv_img is None:
        raise SystemExit(f"Failed to convert image from topic: {args.topic}")

    ok = cv2.imwrite(str(out_path), cv_img)
    if not ok:
        raise SystemExit(f"Failed to write image: {out_path}")

    print(f"[OK] wrote: {out_path}")
    print(f"[OK] topic: {args.topic}")
    print(f"[OK] size:  {int(cv_img.shape[1])}x{int(cv_img.shape[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
