cmake_minimum_required(VERSION 3.0.2)
project(orchard_row_mapping)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  pcl_ros
)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/orchard_segmentation_node.py
    scripts/orchard_row_prior_node.py
    scripts/orchard_tree_circles_node.py
    scripts/orchard_tree_map_builder_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
