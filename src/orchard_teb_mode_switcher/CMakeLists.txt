cmake_minimum_required(VERSION 3.0.2)
project(orchard_teb_mode_switcher)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  dynamic_reconfigure
)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/teb_mode_switcher_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
