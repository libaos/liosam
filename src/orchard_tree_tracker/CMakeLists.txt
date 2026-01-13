cmake_minimum_required(VERSION 3.0.2)
project(orchard_tree_tracker)

find_package(
  catkin REQUIRED
  COMPONENTS
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
    geometry_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(
  PROGRAMS
    scripts/fruit_tree_tracker_node.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
