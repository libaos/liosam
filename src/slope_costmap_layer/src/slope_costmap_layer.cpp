#include <pluginlib/class_list_macros.h>
#include "slope_costmap_layer/slope_costmap_layer.h"
#include <tf2/LinearMath/Quaternion.h>

PLUGINLIB_EXPORT_CLASS(slope_costmap_layer::SlopeCostmapLayer, costmap_2d::Layer)

namespace slope_costmap_layer
{

SlopeCostmapLayer::SlopeCostmapLayer() 
  : map_received_(false), initialized_(false)
{
}

SlopeCostmapLayer::~SlopeCostmapLayer()
{
}

void SlopeCostmapLayer::onInitialize()
{
  ros::NodeHandle nh("~/" + name_);
  current_ = true;
  default_value_ = costmap_2d::NO_INFORMATION;
  
  // Get parameters
  nh.param("use_pcd_map", use_pcd_map_, true);
  nh.param("pcd_map_path", pcd_map_path_, std::string("/root/shared_dir/LIO-SAM/shangpo2map0.01/GlobalMap.pcd"));
  nh.param("max_slope_angle", max_slope_angle_, 30.0);
  nh.param("slope_threshold", slope_threshold_, 25.0);
  nh.param("slope_weight", slope_weight_, 0.7);
  nh.param("max_roughness", max_roughness_, 0.1);
  nh.param("roughness_threshold", roughness_threshold_, 0.05);
  nh.param("roughness_weight", roughness_weight_, 0.3);
  nh.param("resolution", resolution_, 0.05);
  nh.param("visualize", visualize_, true);
  
  // Subscribe to grid map topic
  grid_map_sub_ = nh.subscribe("/process_global_map/slope_grid_map", 1, &SlopeCostmapLayer::gridMapCallback, this);
  
  // Publisher for visualization
  if (visualize_) {
    costmap_pub_ = nh.advertise<grid_map_msgs::GridMap>("slope_costmap", 1);
  }
  
  // Initialize the costmap
  matchSize();
  
  // If using PCD map directly, process it
  if (use_pcd_map_) {
    processPointCloud();
  }
  
  initialized_ = true;
}

void SlopeCostmapLayer::processPointCloud()
{
  ROS_INFO("Processing point cloud from file: %s", pcd_map_path_.c_str());
  
  // Load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_map_path_, *cloud) == -1) {
    ROS_ERROR("Failed to load PCD file: %s", pcd_map_path_.c_str());
    return;
  }
  
  // Create grid map
  slope_map_.setFrameId("map");
  slope_map_.setGeometry(grid_map::Length(200.0, 200.0), resolution_);
  
  // Add layers
  slope_map_.add("elevation");
  
  // Fill grid map with point cloud data
  // We'll manually fill the grid map since GridMapPclLoader has compatibility issues
  for (const auto& point : cloud->points) {
    grid_map::Position position(point.x, point.y);
    grid_map::Index index;
    if (slope_map_.getIndex(position, index)) {
      // Update elevation with maximum height at this position
      if (!slope_map_.isValid(index) || point.z > slope_map_.at("elevation", index)) {
        slope_map_.at("elevation", index) = point.z;
      }
    }
  }
  
  // Calculate slope
  calculateSlope(slope_map_);
  
  map_received_ = true;
  
  ROS_INFO("Point cloud processed and slope map created");
}

void SlopeCostmapLayer::calculateSlope(grid_map::GridMap& map)
{
  // Create a slope layer
  map.add("slope");
  
  // Get the height layer
  const grid_map::Matrix& elevation = map["elevation"];
  grid_map::Matrix& slope = map["slope"];
  
  // Calculate the slope using central differences
  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
    const grid_map::Index index(*it);
    
    // Skip if elevation is NaN
    if (!std::isfinite(elevation(index(0), index(1)))) {
      slope(index(0), index(1)) = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
    
    // Get neighboring indices
    grid_map::Index indexNorth = index + grid_map::Index(-1, 0);
    grid_map::Index indexSouth = index + grid_map::Index(1, 0);
    grid_map::Index indexWest = index + grid_map::Index(0, -1);
    grid_map::Index indexEast = index + grid_map::Index(0, 1);
    
    // Check if indices are within map bounds
    bool isNorthValid = map.isValid(indexNorth);
    bool isSouthValid = map.isValid(indexSouth);
    bool isWestValid = map.isValid(indexWest);
    bool isEastValid = map.isValid(indexEast);
    
    // Calculate slope using available neighbors
    double dzdx = 0.0;
    double dzdy = 0.0;
    int validX = 0;
    int validY = 0;
    
    if (isNorthValid && std::isfinite(elevation(indexNorth(0), indexNorth(1)))) {
      dzdy += elevation(index(0), index(1)) - elevation(indexNorth(0), indexNorth(1));
      validY++;
    }
    
    if (isSouthValid && std::isfinite(elevation(indexSouth(0), indexSouth(1)))) {
      dzdy += elevation(indexSouth(0), indexSouth(1)) - elevation(index(0), index(1));
      validY++;
    }
    
    if (isWestValid && std::isfinite(elevation(indexWest(0), indexWest(1)))) {
      dzdx += elevation(index(0), index(1)) - elevation(indexWest(0), indexWest(1));
      validX++;
    }
    
    if (isEastValid && std::isfinite(elevation(indexEast(0), indexEast(1)))) {
      dzdx += elevation(indexEast(0), indexEast(1)) - elevation(index(0), index(1));
      validX++;
    }
    
    // Normalize by the number of valid neighbors and cell size
    if (validX > 0) dzdx /= (validX * map.getResolution());
    if (validY > 0) dzdy /= (validY * map.getResolution());
    
    // Calculate slope angle in degrees
    double slopeAngle = std::atan(std::sqrt(dzdx*dzdx + dzdy*dzdy)) * 180.0 / M_PI;
    slope(index(0), index(1)) = slopeAngle;
  }
}

void SlopeCostmapLayer::gridMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg)
{
  ROS_INFO("Received grid map message");
  
  // Convert message to grid map
  grid_map::GridMapRosConverter::fromMessage(*msg, slope_map_);
  
  // Check if slope layer exists, if not calculate it
  if (!slope_map_.exists("slope")) {
    calculateSlope(slope_map_);
  }
  
  map_received_ = true;
}

void SlopeCostmapLayer::updateBounds(double robot_x, double robot_y, double robot_yaw, 
                                    double* min_x, double* min_y, double* max_x, double* max_y)
{
  if (!enabled_ || !map_received_) {
    return;
  }
  
  // Set bounds to the entire map
  *min_x = -std::numeric_limits<double>::max();
  *min_y = -std::numeric_limits<double>::max();
  *max_x = std::numeric_limits<double>::max();
  *max_y = std::numeric_limits<double>::max();
}

void SlopeCostmapLayer::updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_ || !map_received_) {
    return;
  }
  
  // Get costmap dimensions and resolution
  double costmap_resolution = master_grid.getResolution();
  double costmap_origin_x = master_grid.getOriginX();
  double costmap_origin_y = master_grid.getOriginY();
  unsigned int costmap_size_x = master_grid.getSizeInCellsX();
  unsigned int costmap_size_y = master_grid.getSizeInCellsY();
  
  // Iterate through each cell in the costmap
  for (unsigned int i = 0; i < costmap_size_x; ++i) {
    for (unsigned int j = 0; j < costmap_size_y; ++j) {
      // Get world coordinates of cell
      double world_x = costmap_origin_x + (i + 0.5) * costmap_resolution;
      double world_y = costmap_origin_y + (j + 0.5) * costmap_resolution;
      
      // Check if point is in slope map
      grid_map::Position position(world_x, world_y);
      if (slope_map_.isInside(position)) {
        try {
          // Get slope value at position
          float slope_value = slope_map_.atPosition("slope", position);
          
          // Skip if slope value is NaN
          if (!std::isfinite(slope_value)) {
            continue;
          }
          
          // Calculate cost based on slope
          unsigned char cost = 0;
          if (slope_value > slope_threshold_) {
            // Scale cost between lethal (254) and inscribed (128) based on slope
            double slope_ratio = std::min(1.0, (slope_value - slope_threshold_) / (max_slope_angle_ - slope_threshold_));
            cost = static_cast<unsigned char>(128 + 126 * slope_ratio);
          } else if (slope_value > 20.0) {
            // Very steep slopes (20-25 degrees) - high cost
            double slope_ratio = std::min(1.0, (slope_value - 20.0) / (slope_threshold_ - 20.0));
            cost = static_cast<unsigned char>(100 + 28 * slope_ratio);
          } else if (slope_value > 15.0) {
            // Steep slopes (15-20 degrees) - medium-high cost
            double slope_ratio = std::min(1.0, (slope_value - 15.0) / 5.0);
            cost = static_cast<unsigned char>(70 + 30 * slope_ratio);
          } else if (slope_value > 10.0) {
            // Moderate slopes (10-15 degrees) - medium cost
            double slope_ratio = std::min(1.0, (slope_value - 10.0) / 5.0);
            cost = static_cast<unsigned char>(40 + 30 * slope_ratio);
          } else {
            // Gentle slopes (0-10 degrees) - low cost
            double slope_ratio = std::min(1.0, slope_value / 10.0);
            cost = static_cast<unsigned char>(40 * slope_ratio);
          }
          
          // Update master grid with cost
          unsigned char old_cost = master_grid.getCost(i, j);
          if (old_cost < cost) {
            master_grid.setCost(i, j, cost);
          }
        } catch (const std::out_of_range& e) {
          // Position is outside of grid map, skip
          continue;
        }
      }
    }
  }
  
  // Publish costmap for visualization
  if (visualize_) {
    grid_map_msgs::GridMap msg;
    grid_map::GridMapRosConverter::toMessage(slope_map_, msg);
    costmap_pub_.publish(msg);
  }
}

void SlopeCostmapLayer::reset()
{
  // Reset the map
  map_received_ = false;
  
  // Call the parent reset function
  costmap_2d::CostmapLayer::reset();
  
  // Reinitialize if needed
  if (initialized_) {
    onInitialize();
  }
}

} // namespace slope_costmap_layer 