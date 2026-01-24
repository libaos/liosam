cmake_minimum_required(VERSION 3.0.2)
project(orchard_corridor)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  nav_msgs
  std_msgs
  std_srvs
  tf2_ros
  tf2_sensor_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/pointcloud_preprocess_node.py
  scripts/bev_occupancy_node.py
  scripts/debug_export_node.py
  scripts/centerline_node.py
  scripts/boundary_from_centerline_node.py
  scripts/route_id_gate_node.py
  scripts/controller_node.py
  scripts/cmd_vel_mux_node.py
  scripts/analyze_turns_on_bag.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
