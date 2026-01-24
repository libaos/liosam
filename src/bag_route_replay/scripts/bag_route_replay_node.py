#!/usr/bin/env python3

import math
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import actionlib
import rosbag
import rospy
import tf
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Path


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    yaw: float


def distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def transform_xy_points(
    source_xy: Sequence[Tuple[float, float]],
    source_frame: str,
    target_frame: str,
    timeout_s: float = 5.0,
) -> List[Tuple[float, float]]:
    if source_frame == target_frame:
        return list(source_xy)
    if not source_xy:
        return []

    listener = tf.TransformListener()

    end_time = rospy.Time.now() + rospy.Duration(timeout_s)
    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        try:
            listener.waitForTransform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(0.2)
            )
            break
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.sleep(0.05)
    else:
        raise RuntimeError(f"TF not available: {target_frame} -> {source_frame}")

    out: List[Tuple[float, float]] = []
    for (x, y) in source_xy:
        ps = PoseStamped()
        ps.header.stamp = rospy.Time(0)
        ps.header.frame_id = source_frame
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0

        try:
            ps_t = listener.transformPose(target_frame, ps)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            raise RuntimeError(f"TF transformPose failed: {source_frame} -> {target_frame}: {exc}") from exc

        out.append((ps_t.pose.position.x, ps_t.pose.position.y))
    return out


def pick_best_path_topic(bag_path: str, requested_topic: Optional[str]) -> str:
    if requested_topic:
        return requested_topic

    with rosbag.Bag(bag_path) as bag:
        topics: Dict[str, object] = bag.get_type_and_topic_info().topics

    path_topics = [
        topic
        for topic, info in topics.items()
        if getattr(info, "msg_type", None) == "nav_msgs/Path"
    ]
    if not path_topics:
        raise RuntimeError("No nav_msgs/Path topics found in bag; pass ~path_topic.")

    priority = [
        "/lio_sam/mapping/path",
        "/liorl/mapping/path",
        "/liorf/mapping/path",
    ]
    for candidate in priority:
        if candidate in path_topics:
            return candidate

    mapping_paths = [t for t in path_topics if t.endswith("/mapping/path")]
    if mapping_paths:
        return sorted(mapping_paths)[0]

    return sorted(path_topics)[0]


def load_last_path(bag_path: str, topic: str) -> Path:
    last_msg = None
    with rosbag.Bag(bag_path) as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            last_msg = msg

    if last_msg is None:
        raise RuntimeError(f"Topic not found in bag: {topic}")
    if not hasattr(last_msg, "poses") or not hasattr(last_msg, "header"):
        raise RuntimeError(f"Topic is not nav_msgs/Path: {topic}")
    if not last_msg.poses:
        raise RuntimeError(f"Path is empty on topic: {topic}")
    return last_msg


def downsample_poses(poses: Sequence[PoseStamped], min_dist_m: float) -> List[PoseStamped]:
    if min_dist_m <= 0:
        return list(poses)
    if not poses:
        return []

    selected = [poses[0]]
    last_xy = (poses[0].pose.position.x, poses[0].pose.position.y)
    for ps in poses[1:-1]:
        xy = (ps.pose.position.x, ps.pose.position.y)
        if distance_xy(xy, last_xy) >= min_dist_m:
            selected.append(ps)
            last_xy = xy
    if poses[-1] is not selected[-1]:
        selected.append(poses[-1])
    return selected


def cap_waypoints_uniform(poses: Sequence[PoseStamped], max_waypoints: int) -> List[PoseStamped]:
    if max_waypoints <= 0:
        return list(poses)
    if max_waypoints < 2:
        raise ValueError("max_waypoints must be >= 2")
    if len(poses) <= max_waypoints:
        return list(poses)

    out = [poses[0]]
    inner = poses[1:-1]
    need_inner = max_waypoints - 2
    step = max(1, len(inner) // need_inner)
    out.extend(inner[::step][:need_inner])
    out.append(poses[-1])
    return out


def xy_to_waypoints(xy_points: Sequence[Tuple[float, float]]) -> List[Waypoint]:
    if not xy_points:
        return []

    waypoints: List[Waypoint] = []
    last_yaw = 0.0

    for i, xy in enumerate(xy_points):
        if len(xy_points) >= 2:
            if i < len(xy_points) - 1:
                dx = xy_points[i + 1][0] - xy_points[i][0]
                dy = xy_points[i + 1][1] - xy_points[i][1]
                if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                    yaw = last_yaw
                else:
                    yaw = math.atan2(dy, dx)
            else:
                yaw = last_yaw
        else:
            yaw = 0.0

        last_yaw = yaw
        waypoints.append(Waypoint(x=float(xy[0]), y=float(xy[1]), yaw=float(yaw)))

    return waypoints


def yaw_to_quaternion(yaw: float):
    qx, qy, qz, qw = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
    return qx, qy, qz, qw


def publish_waypoints_path(pub: rospy.Publisher, waypoints: Sequence[Waypoint], frame_id: str) -> None:
    msg = Path()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    for wp in waypoints:
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = wp.x
        ps.pose.position.y = wp.y
        ps.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(wp.yaw)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        msg.poses.append(ps)
    pub.publish(msg)


def pick_start_index_nearest(
    waypoints: Sequence[Waypoint],
    goal_frame: str,
    robot_frame: str,
    tf_timeout_s: float = 5.0,
) -> int:
    if not waypoints:
        return 0

    listener = tf.TransformListener()
    end_time = rospy.Time.now() + rospy.Duration(tf_timeout_s)
    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        try:
            listener.waitForTransform(
                goal_frame, robot_frame, rospy.Time(0), rospy.Duration(0.2)
            )
            (trans, _rot) = listener.lookupTransform(goal_frame, robot_frame, rospy.Time(0))
            robot_xy = (float(trans[0]), float(trans[1]))
            break
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.sleep(0.05)
    else:
        raise RuntimeError(f"TF not available: {goal_frame} -> {robot_frame}")

    best_i = 0
    best_d = float("inf")
    for i, wp in enumerate(waypoints):
        d = distance_xy((wp.x, wp.y), robot_xy)
        if d < best_d:
            best_d = d
            best_i = i
    rospy.loginfo("Start from nearest waypoint: %d (dist=%.3fm)", best_i, best_d)
    return best_i


def main() -> None:
    rospy.init_node("bag_route_replay")

    bag_path = str(rospy.get_param("~bag_path", "")).strip()
    if not bag_path:
        raise RuntimeError("~bag_path is required")
    bag_path = os.path.abspath(os.path.expanduser(bag_path))
    if not os.path.isfile(bag_path):
        raise RuntimeError(f"Bag file not found: {bag_path}")

    path_topic_param = str(rospy.get_param("~path_topic", "")).strip()
    path_topic = pick_best_path_topic(bag_path, path_topic_param or None)

    min_dist = float(rospy.get_param("~min_dist", 1.0))
    max_waypoints = int(rospy.get_param("~max_waypoints", 0))
    start_nearest = bool(rospy.get_param("~start_nearest", True))
    start_index = int(rospy.get_param("~start_index", 0))
    goal_timeout = float(rospy.get_param("~goal_timeout", 60.0))
    skip_failed = bool(rospy.get_param("~skip_failed", True))
    loop = bool(rospy.get_param("~loop", False))
    publish_path = bool(rospy.get_param("~publish_path", True))

    action_name = str(rospy.get_param("~action_name", "move_base")).strip() or "move_base"
    robot_frame = str(rospy.get_param("~robot_frame", "base_link_est")).strip() or "base_link_est"
    goal_frame_param = str(rospy.get_param("~goal_frame", "")).strip()
    transform_timeout = float(rospy.get_param("~transform_timeout", 5.0))

    rospy.loginfo("Bag        : %s", bag_path)
    rospy.loginfo("Path topic  : %s", path_topic)

    path_msg = load_last_path(bag_path, path_topic)
    source_frame = str(path_msg.header.frame_id).strip()
    if not source_frame and path_msg.poses:
        source_frame = str(path_msg.poses[0].header.frame_id).strip()
    if not source_frame:
        source_frame = "map"

    goal_frame = goal_frame_param or source_frame
    rospy.loginfo("Source frame: %s", source_frame)
    rospy.loginfo("Goal frame  : %s", goal_frame)

    poses = downsample_poses(path_msg.poses, min_dist_m=min_dist)
    poses = cap_waypoints_uniform(poses, max_waypoints=max_waypoints)

    source_xy = [(ps.pose.position.x, ps.pose.position.y) for ps in poses]
    if goal_frame != source_frame:
        rospy.loginfo(
            "Transforming %d points: %s -> %s (timeout=%.1fs)",
            len(source_xy),
            source_frame,
            goal_frame,
            transform_timeout,
        )
        goal_xy = transform_xy_points(
            source_xy,
            source_frame=source_frame,
            target_frame=goal_frame,
            timeout_s=transform_timeout,
        )
    else:
        goal_xy = source_xy

    waypoints = xy_to_waypoints(goal_xy)
    if not waypoints:
        raise RuntimeError("No waypoints extracted from path")
    rospy.loginfo("Waypoints   : %d (min_dist=%.3f, max_waypoints=%d)", len(waypoints), min_dist, max_waypoints)

    path_pub = rospy.Publisher("~waypoints_path", Path, queue_size=1, latch=True)
    if publish_path:
        publish_waypoints_path(path_pub, waypoints, frame_id=goal_frame)

    if start_nearest:
        start_index = pick_start_index_nearest(waypoints, goal_frame=goal_frame, robot_frame=robot_frame)
    else:
        start_index = max(0, min(start_index, len(waypoints) - 1))
        rospy.loginfo("Start index : %d", start_index)

    client = actionlib.SimpleActionClient(action_name, MoveBaseAction)
    rospy.loginfo("Waiting for action server: %s", action_name)
    if not client.wait_for_server(rospy.Duration(30.0)):
        raise RuntimeError(f"move_base action server not available: {action_name}")
    rospy.loginfo("Connected: %s", action_name)

    i = start_index
    while not rospy.is_shutdown():
        if i >= len(waypoints):
            if loop:
                rospy.loginfo("Reached end; looping to start")
                i = 0
            else:
                rospy.loginfo("Reached end; done")
                break

        wp = waypoints[i]
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = goal_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = wp.x
        goal.target_pose.pose.position.y = wp.y
        goal.target_pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(wp.yaw)
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw

        rospy.loginfo("Sending waypoint %d/%d: x=%.3f y=%.3f yaw=%.3f", i + 1, len(waypoints), wp.x, wp.y, wp.yaw)
        client.send_goal(goal)

        if goal_timeout > 0:
            finished = client.wait_for_result(rospy.Duration(goal_timeout))
        else:
            finished = client.wait_for_result()

        if not finished:
            client.cancel_goal()
            rospy.logwarn("Waypoint %d timeout (%.1fs)", i + 1, goal_timeout)
            if not skip_failed:
                break
            i += 1
            continue

        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("Waypoint %d reached", i + 1)
            i += 1
            continue

        rospy.logwarn("Waypoint %d failed: state=%d", i + 1, state)
        if not skip_failed:
            break
        i += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        rospy.logfatal("%s", exc)
        raise
