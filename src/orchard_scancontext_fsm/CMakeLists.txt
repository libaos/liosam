cmake_minimum_required(VERSION 3.0.2)
project(orchard_scancontext_fsm)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  nav_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/scancontext_fsm_node.py
    scripts/build_scancontext_db.py
    scripts/annotate_bag_with_fsm_mode.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
