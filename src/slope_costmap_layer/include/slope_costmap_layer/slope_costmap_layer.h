#ifndef SLOPE_COSTMAP_LAYER_H_
#define SLOPE_COSTMAP_LAYER_H_

#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/costmap_layer.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

namespace slope_costmap_layer
{

class SlopeCostmapLayer : public costmap_2d::CostmapLayer
{
public:
  SlopeCostmapLayer();
  virtual ~SlopeCostmapLayer();

  virtual void onInitialize();
  virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, 
                           double* min_x, double* min_y, double* max_x, double* max_y);
  virtual void updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j);
  virtual void reset();

private:
  void processPointCloud();
  void calculateSlope(grid_map::GridMap& map);
  void gridMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);
  
  // ROS parameters
  bool use_pcd_map_;
  std::string pcd_map_path_;
  double max_slope_angle_;
  double slope_threshold_;
  double slope_weight_;
  double max_roughness_;
  double roughness_threshold_;
  double roughness_weight_;
  double resolution_;
  bool visualize_;
  
  // ROS subscribers and publishers
  ros::Subscriber grid_map_sub_;
  ros::Publisher costmap_pub_;
  
  // Data storage
  grid_map::GridMap slope_map_;
  bool map_received_;
  bool initialized_;
};

} // namespace slope_costmap_layer

#endif // SLOPE_COSTMAP_LAYER_H_ 