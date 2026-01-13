cmake_minimum_required(VERSION 3.0.2)
project(pcd_gazebo_world)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  geometry_msgs
  gazebo_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/capture_ros_image.py
    scripts/circles_to_occupancy_map.py
    scripts/follow_path_goals.py
    scripts/generate_orchard_world.py
    scripts/pcd_to_mesh.py
    scripts/pcd_to_orchard_world.py
    scripts/pid_path_follower.py
    scripts/plot_pcd_tree_centers_bev.py
    scripts/plot_reference_vs_replay.py
    scripts/plot_tree_centers_vs_rosbag.py
    scripts/publish_path.py
    scripts/publish_via_points_window.py
    scripts/record_trajectory.py
    scripts/rosbag_path_to_json.py
    scripts/rosbag_registered_cloud_to_orchard_world.py
    scripts/rosbag_tree_cloud_to_orchard_world.py
    scripts/set_map_to_odom_from_path.py
    scripts/set_model_pose_from_path.py
    scripts/validate_tree_centers.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch config maps models worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  DIRECTORY tools
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
  PATTERN "__pycache__" EXCLUDE
)
