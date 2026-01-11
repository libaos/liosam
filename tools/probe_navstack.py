#!/usr/bin/env python3
"""
Runtime probe for navigation prerequisites:
- TF must contain goal_frame -> robot_frame
- move_base (MoveBaseAction) action server must be available

This is meant to be run on a live ROS system (Gazebo or real robot).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import actionlib
import rospy
import tf2_ros
from move_base_msgs.msg import MoveBaseAction


def _check_action_server(action_name: str, timeout_s: float) -> bool:
    client = actionlib.SimpleActionClient(action_name, MoveBaseAction)
    ok = client.wait_for_server(rospy.Duration(timeout_s))
    return bool(ok)


def _check_tf(goal_frame: str, robot_frame: str, timeout_s: float) -> bool:
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
    _listener = tf2_ros.TransformListener(buf)
    return bool(buf.can_transform(goal_frame, robot_frame, rospy.Time(0), rospy.Duration(timeout_s)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Probe nav stack (TF + move_base action server).")
    p.add_argument("--action-name", default="move_base", help="move_base action name (default: move_base)")
    p.add_argument("--goal-frame", default="map", help="Goal frame to use (default: map)")
    p.add_argument("--robot-frame", default="base_link_est", help="Robot base frame (default: base_link_est)")
    p.add_argument("--timeout", type=float, default=3.0, help="Timeout seconds for each check (default: 3.0)")
    args = p.parse_args(argv)

    rospy.init_node("probe_navstack", anonymous=True, disable_signals=True)

    ok_tf = _check_tf(args.goal_frame, args.robot_frame, float(args.timeout))
    ok_action = _check_action_server(args.action_name, float(args.timeout))

    print("probe_navstack")
    print(f"  tf         : {args.goal_frame} -> {args.robot_frame}  {'OK' if ok_tf else 'MISSING'}")
    print(f"  action     : {args.action_name} (MoveBaseAction)  {'OK' if ok_action else 'MISSING'}")

    if not ok_tf:
        print("  hint(tf)   : check /tf tree; typical is map -> odom(_est) -> base_link(_est)")
    if not ok_action:
        print("  hint(action): check move_base is running (topics like /move_base/status should exist)")

    return 0 if (ok_tf and ok_action) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
