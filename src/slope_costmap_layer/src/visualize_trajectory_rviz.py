#!/usr/bin/env python3

import rospy
import numpy as np
import sys
import os
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

def load_trajectory_data(csv_file):
    """Load trajectory data from CSV file"""
    if not os.path.exists(csv_file):
        rospy.logerr(f"File not found: {csv_file}")
        return None
    
    try:
        # Load data from CSV (format: timestamp,x,y,z)
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        rospy.loginfo(f"Loaded {len(data)} trajectory points from {csv_file}")
        return data
    except Exception as e:
        rospy.logerr(f"Error loading trajectory data: {e}")
        return None

def create_path_message(data, frame_id):
    """Create a Path message from trajectory data"""
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = rospy.Time.now()
    
    for i in range(len(data)):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = data[i, 1]  # x
        pose.pose.position.y = data[i, 2]  # y
        pose.pose.position.z = data[i, 3]  # z
        
        # Use identity quaternion
        pose.pose.orientation.w = 1.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        
        path_msg.poses.append(pose)
    
    return path_msg

def create_marker_array(data, frame_id, ns="trajectory", id_start=0, rgba=(1.0, 0.0, 0.0, 1.0)):
    """Create a MarkerArray for trajectory visualization"""
    markers = MarkerArray()
    
    # Line strip marker for the trajectory
    line_marker = Marker()
    line_marker.header.frame_id = frame_id
    line_marker.header.stamp = rospy.Time.now()
    line_marker.ns = ns
    line_marker.id = id_start
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    
    # Set the scale
    line_marker.scale.x = 0.1  # Line width
    
    # Set the color
    line_marker.color.r = rgba[0]
    line_marker.color.g = rgba[1]
    line_marker.color.b = rgba[2]
    line_marker.color.a = rgba[3]
    
    # Add points to the line strip
    for i in range(len(data)):
        p = Point()
        p.x = data[i, 1]  # x
        p.y = data[i, 2]  # y
        p.z = data[i, 3]  # z
        line_marker.points.append(p)
    
    markers.markers.append(line_marker)
    
    # Add start and end markers
    start_marker = Marker()
    start_marker.header.frame_id = frame_id
    start_marker.header.stamp = rospy.Time.now()
    start_marker.ns = ns
    start_marker.id = id_start + 1
    start_marker.type = Marker.SPHERE
    start_marker.action = Marker.ADD
    start_marker.pose.position.x = data[0, 1]
    start_marker.pose.position.y = data[0, 2]
    start_marker.pose.position.z = data[0, 3]
    start_marker.scale.x = 0.5
    start_marker.scale.y = 0.5
    start_marker.scale.z = 0.5
    start_marker.color.r = 0.0
    start_marker.color.g = 1.0
    start_marker.color.b = 0.0
    start_marker.color.a = 1.0
    
    end_marker = Marker()
    end_marker.header.frame_id = frame_id
    end_marker.header.stamp = rospy.Time.now()
    end_marker.ns = ns
    end_marker.id = id_start + 2
    end_marker.type = Marker.SPHERE
    end_marker.action = Marker.ADD
    end_marker.pose.position.x = data[-1, 1]
    end_marker.pose.position.y = data[-1, 2]
    end_marker.pose.position.z = data[-1, 3]
    end_marker.scale.x = 0.5
    end_marker.scale.y = 0.5
    end_marker.scale.z = 0.5
    end_marker.color.r = 1.0
    end_marker.color.g = 0.0
    end_marker.color.b = 0.0
    end_marker.color.a = 1.0
    
    markers.markers.append(start_marker)
    markers.markers.append(end_marker)
    
    return markers

def main():
    rospy.init_node("trajectory_visualizer")
    
    # Get parameters
    csv_file = rospy.get_param("~csv_file", "")
    frame_id = rospy.get_param("~frame_id", "map")
    publish_rate = rospy.get_param("~rate", 1.0)  # Hz
    
    if not csv_file:
        rospy.logerr("No CSV file specified. Use _csv_file:=path_to_csv")
        return
    
    # Load trajectory data
    data = load_trajectory_data(csv_file)
    if data is None:
        return
    
    # Create publishers
    path_pub = rospy.Publisher("trajectory_path", Path, queue_size=1, latch=True)
    markers_pub = rospy.Publisher("trajectory_markers", MarkerArray, queue_size=1, latch=True)
    
    rate = rospy.Rate(publish_rate)
    
    # Create messages
    path_msg = create_path_message(data, frame_id)
    markers_msg = create_marker_array(data, frame_id)
    
    rospy.loginfo(f"Publishing trajectory visualization at {publish_rate} Hz")
    
    while not rospy.is_shutdown():
        # Update timestamps
        path_msg.header.stamp = rospy.Time.now()
        for marker in markers_msg.markers:
            marker.header.stamp = rospy.Time.now()
        
        # Publish messages
        path_pub.publish(path_msg)
        markers_pub.publish(markers_msg)
        
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 