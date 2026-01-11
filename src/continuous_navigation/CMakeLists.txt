cmake_minimum_required(VERSION 3.0.2)
project(continuous_navigation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  move_base_msgs
  actionlib
  tf
)

find_package(yaml-cpp REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs move_base_msgs actionlib tf
  DEPENDS yaml-cpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

# Create the waypoint_navigator node
add_executable(waypoint_navigator 
  src/waypoint_navigator_node.cpp
  src/waypoint_navigator.cpp
)
target_link_libraries(waypoint_navigator
  ${catkin_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

# Create the simple two points navigator node
add_executable(simple_two_points_navigator 
  src/simple_two_points_navigator.cpp
)
target_link_libraries(simple_two_points_navigator
  ${catkin_LIBRARIES}
)

# Install targets
install(TARGETS waypoint_navigator
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
) 