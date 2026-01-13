cmake_minimum_required(VERSION 3.0.2)
project(liosam_bringup)

find_package(catkin REQUIRED COMPONENTS
  roscpp
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(smoke_node src/smoke_node.cpp)
target_link_libraries(smoke_node ${catkin_LIBRARIES})

install(
  TARGETS smoke_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
