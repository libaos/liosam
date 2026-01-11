cmake_minimum_required(VERSION 3.0.2)
project(slope_costmap_layer)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  costmap_2d
  pluginlib
  pcl_ros
  grid_map_ros
  grid_map_pcl
)

find_package(PCL REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES slope_costmap_layer
  CATKIN_DEPENDS roscpp costmap_2d pluginlib pcl_ros grid_map_ros grid_map_pcl
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(slope_costmap_layer
  src/slope_costmap_layer.cpp
)
target_link_libraries(slope_costmap_layer
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(process_global_map src/process_global_map.cpp)
target_link_libraries(process_global_map
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(move_base_style_costmap src/move_base_style_costmap.cpp)
target_link_libraries(move_base_style_costmap
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(slope_visualization_costmap src/slope_visualization_costmap.cpp)
target_link_libraries(slope_visualization_costmap
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(slope_normal_visualization src/slope_normal_visualization.cpp)
target_link_libraries(slope_normal_visualization
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(ground_slope_visualization src/ground_slope_visualization.cpp)
target_link_libraries(ground_slope_visualization
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(trajectory_costmap_visualizer src/trajectory_with_costmap.cpp)
target_link_libraries(trajectory_costmap_visualizer
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS slope_costmap_layer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(FILES slope_costmap_layer_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)
