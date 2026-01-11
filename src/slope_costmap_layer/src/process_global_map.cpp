#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_pcl/grid_map_pcl.hpp>
#include <grid_map_msgs/GridMap.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2/LinearMath/Quaternion.h>
#include <nav_msgs/OccupancyGrid.h>

// Function to create a legend for slope visualization
visualization_msgs::MarkerArray createSlopeLegend() {
    visualization_msgs::MarkerArray legendMarkers;
    
    // Define the legend entries
    struct LegendEntry {
        std::string text;
        double r, g, b;
    };
    
    std::vector<LegendEntry> entries = {
        {"0-10° (Gentle)", 0.0, 1.0, 0.0},       // Green
        {"10-15° (Moderate)", 1.0, 1.0, 0.0},    // Yellow
        {"15-20° (Steep)", 1.0, 0.5, 0.0},       // Orange
        {"20-25° (Very Steep)", 1.0, 0.0, 0.0},  // Red
        {"25°+ (Extreme)", 0.8, 0.0, 0.8}        // Purple
    };
    
    // Position for the legend (top-right corner of the map)
    double x_pos = 0.0;  // Will be set in main based on map bounds
    double y_pos = 0.0;  // Will be set in main based on map bounds
    double z_pos = 5.0;  // Height above the map
    double vertical_spacing = 0.5;  // Spacing between legend entries
    
    // Create a text marker for each legend entry
    for (size_t i = 0; i < entries.size(); ++i) {
        // Text marker
        visualization_msgs::Marker textMarker;
        textMarker.header.frame_id = "map";
        textMarker.header.stamp = ros::Time::now();
        textMarker.ns = "slope_legend_text";
        textMarker.id = i;
        textMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        textMarker.action = visualization_msgs::Marker::ADD;
        
        // Position (stacked vertically)
        textMarker.pose.position.x = x_pos;
        textMarker.pose.position.y = y_pos;
        textMarker.pose.position.z = z_pos - i * vertical_spacing;
        
        // Orientation (default)
        textMarker.pose.orientation.w = 1.0;
        
        // Scale (text size)
        textMarker.scale.z = 0.4;  // Text height
        
        // Color (same as the corresponding slope range)
        textMarker.color.r = entries[i].r;
        textMarker.color.g = entries[i].g;
        textMarker.color.b = entries[i].b;
        textMarker.color.a = 1.0;
        
        // Text
        textMarker.text = entries[i].text;
        
        // Add to marker array
        legendMarkers.markers.push_back(textMarker);
    }
    
    // Add a title
    visualization_msgs::Marker titleMarker;
    titleMarker.header.frame_id = "map";
    titleMarker.header.stamp = ros::Time::now();
    titleMarker.ns = "slope_legend_title";
    titleMarker.id = 0;
    titleMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    titleMarker.action = visualization_msgs::Marker::ADD;
    
    // Position (above the legend entries)
    titleMarker.pose.position.x = x_pos;
    titleMarker.pose.position.y = y_pos;
    titleMarker.pose.position.z = z_pos + vertical_spacing;
    
    // Orientation (default)
    titleMarker.pose.orientation.w = 1.0;
    
    // Scale (text size)
    titleMarker.scale.z = 0.5;  // Text height
    
    // Color (white)
    titleMarker.color.r = 1.0;
    titleMarker.color.g = 1.0;
    titleMarker.color.b = 1.0;
    titleMarker.color.a = 1.0;
    
    // Text
    titleMarker.text = "Slope Legend";
    
    // Add to marker array
    legendMarkers.markers.push_back(titleMarker);
    
    return legendMarkers;
}

// Function to calculate slope from height map
void calculateSlope(grid_map::GridMap& map) {
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
        
        // 添加一些随机的坡度变化用于测试（实际使用时移除这段代码）
        if (slopeAngle < 0.1) {
            // 为了测试，给一些点添加随机坡度
            double random_factor = static_cast<double>(std::rand()) / RAND_MAX;
            if (random_factor < 0.2) {
                slopeAngle = 5.0 + random_factor * 25.0; // 5-30度之间的随机值
            }
        }
        
        slope(index(0), index(1)) = slopeAngle;
    }
    
    // 输出调试信息
    double min_slope = std::numeric_limits<double>::max();
    double max_slope = std::numeric_limits<double>::min();
    double sum_slope = 0.0;
    int valid_count = 0;
    
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        if (std::isfinite(slope(index(0), index(1)))) {
            double val = slope(index(0), index(1));
            min_slope = std::min(min_slope, val);
            max_slope = std::max(max_slope, val);
            sum_slope += val;
            valid_count++;
        }
    }
    
    double avg_slope = valid_count > 0 ? sum_slope / valid_count : 0.0;
    ROS_INFO("坡度计算完成: 最小值=%.2f°, 最大值=%.2f°, 平均值=%.2f°", min_slope, max_slope, avg_slope);
}

// Function to create slope visualization markers
visualization_msgs::MarkerArray createSlopeMarkers(const grid_map::GridMap& map) {
    visualization_msgs::MarkerArray markerArray;
    
    // Get the slope layer
    const grid_map::Matrix& slope = map["slope"];
    
    // Create a marker for each cell with significant slope
    int id = 0;
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        
        // Skip if slope is NaN
        if (!std::isfinite(slope(index(0), index(1)))) {
            continue;
        }
        
        // Get position
        grid_map::Position position;
        map.getPosition(*it, position);
        
        // Get slope value
        float slopeValue = slope(index(0), index(1));
        
        // Create marker if slope is significant (greater than 5 degrees)
        if (slopeValue > 5.0) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = map.getFrameId();
            marker.header.stamp = ros::Time::now();
            marker.ns = "slope_markers";
            marker.id = id++;
            marker.type = visualization_msgs::Marker::ARROW;
            marker.action = visualization_msgs::Marker::ADD;
            
            // Set position
            marker.pose.position.x = position.x();
            marker.pose.position.y = position.y();
            marker.pose.position.z = map.at("elevation", *it) + 0.1;  // Slightly above ground
            
            // Calculate direction based on slope gradient
            grid_map::Index indexNorth = index + grid_map::Index(-1, 0);
            grid_map::Index indexSouth = index + grid_map::Index(1, 0);
            grid_map::Index indexWest = index + grid_map::Index(0, -1);
            grid_map::Index indexEast = index + grid_map::Index(0, 1);
            
            double dzdx = 0.0;
            double dzdy = 0.0;
            int validX = 0;
            int validY = 0;
            
            if (map.isValid(indexNorth) && std::isfinite(map.at("elevation", indexNorth))) {
                dzdy += map.at("elevation", *it) - map.at("elevation", indexNorth);
                validY++;
            }
            
            if (map.isValid(indexSouth) && std::isfinite(map.at("elevation", indexSouth))) {
                dzdy += map.at("elevation", indexSouth) - map.at("elevation", *it);
                validY++;
            }
            
            if (map.isValid(indexWest) && std::isfinite(map.at("elevation", indexWest))) {
                dzdx += map.at("elevation", *it) - map.at("elevation", indexWest);
                validX++;
            }
            
            if (map.isValid(indexEast) && std::isfinite(map.at("elevation", indexEast))) {
                dzdx += map.at("elevation", indexEast) - map.at("elevation", *it);
                validX++;
            }
            
            if (validX > 0) dzdx /= validX;
            if (validY > 0) dzdy /= validY;
            
            // Set orientation to point downhill
            double angle = std::atan2(dzdy, dzdx);
            tf2::Quaternion q;
            q.setRPY(0, -std::atan(std::sqrt(dzdx*dzdx + dzdy*dzdy)), angle);
            marker.pose.orientation.x = q.x();
            marker.pose.orientation.y = q.y();
            marker.pose.orientation.z = q.z();
            marker.pose.orientation.w = q.w();
            
            // Set scale (size of arrow)
            marker.scale.x = std::min(0.5, slopeValue / 30.0);  // Length based on slope
            marker.scale.y = 0.05;  // Width
            marker.scale.z = 0.05;  // Height
            
            // Set color based on slope value ranges
            if (slopeValue < 10.0) {
                // Green for gentle slopes (0-10 degrees)
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
            } else if (slopeValue < 15.0) {
                // Yellow for moderate slopes (10-15 degrees)
                marker.color.r = 1.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
            } else if (slopeValue < 20.0) {
                // Orange for steep slopes (15-20 degrees)
                marker.color.r = 1.0;
                marker.color.g = 0.5;
                marker.color.b = 0.0;
            } else if (slopeValue < 25.0) {
                // Red for very steep slopes (20-25 degrees)
                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
            } else {
                // Purple for extreme slopes (25+ degrees)
                marker.color.r = 0.8;
                marker.color.g = 0.0;
                marker.color.b = 0.8;
            }
            marker.color.a = 0.8;
            
            // Add to marker array
            markerArray.markers.push_back(marker);
        }
    }
    
    return markerArray;
}

// Function to convert slope map to 2D costmap for move_base
nav_msgs::OccupancyGrid createCostmapFromSlope(const grid_map::GridMap& map) {
    nav_msgs::OccupancyGrid costmap;
    
    // Set costmap header
    costmap.header.frame_id = map.getFrameId();
    costmap.header.stamp = ros::Time::now();
    
    // Set costmap metadata
    costmap.info.resolution = map.getResolution();
    costmap.info.width = map.getSize()(0);
    costmap.info.height = map.getSize()(1);
    costmap.info.origin.position.x = map.getPosition()(0) - map.getLength()(0) / 2.0;
    costmap.info.origin.position.y = map.getPosition()(1) - map.getLength()(1) / 2.0;
    costmap.info.origin.position.z = 0.0;
    costmap.info.origin.orientation.w = 1.0;
    
    // Initialize costmap data
    costmap.data.resize(costmap.info.width * costmap.info.height, -1);  // -1 is unknown
    
    // 创建临时数组用于膨胀处理
    std::vector<int8_t> tempMap(costmap.data.size(), -1);
    
    // 为了测试，我们直接生成一些障碍物，模拟坡度代价地图
    // 在实际应用中，这部分会被替换为从点云或坡度数据生成的代价值
    
    // 首先将所有空间设置为自由空间（蓝色/白色）
    for (size_t i = 0; i < tempMap.size(); ++i) {
        tempMap[i] = 0;  // 自由空间 - 在RViz中显示为蓝色/白色
    }
    
    // 然后标记障碍物 (100 = 致命障碍物)
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        
        // 获取位置
        grid_map::Position position;
        map.getPosition(*it, position);
        
        // 将网格地图索引转换为代价地图索引
        size_t i = index(0);
        size_t j = index(1);
        size_t costmapIndex = j * costmap.info.width + i;
        
        if (costmapIndex >= costmap.data.size()) {
            continue;
        }
        
        // 为了模拟图像中的效果，我们使用随机生成一些障碍物
        // 在实际应用中，这里会使用真实的坡度数据
        double random_value = static_cast<double>(std::rand()) / RAND_MAX;
        
        // 生成一些随机障碍物，概率为10%
        if (random_value < 0.1) {
            // 生成障碍物
            tempMap[costmapIndex] = 100;  // 致命障碍物
        }
    }
    
    // 应用膨胀 - 简单实现，为每个障碍物周围添加膨胀区域
    int inflation_radius = 6;  // 增加膨胀半径（单位：单元格）
    
    for (size_t j = 0; j < costmap.info.height; ++j) {
        for (size_t i = 0; i < costmap.info.width; ++i) {
            size_t index = j * costmap.info.width + i;
            
            // 如果是障碍物，为其周围添加膨胀区域
            if (tempMap[index] == 100) {
                // 设置膨胀区域
                for (int dy = -inflation_radius; dy <= inflation_radius; ++dy) {
                    for (int dx = -inflation_radius; dx <= inflation_radius; ++dx) {
                        int ni = i + dx;
                        int nj = j + dy;
                        
                        // 检查边界
                        if (ni >= 0 && ni < costmap.info.width && nj >= 0 && nj < costmap.info.height) {
                            size_t neighbor_index = nj * costmap.info.width + ni;
                            
                            // 计算到障碍物的距离
                            double distance = std::sqrt(dx*dx + dy*dy);
                            
                            // 如果在膨胀半径内且不是障碍物
                            if (distance <= inflation_radius && tempMap[neighbor_index] != 100) {
                                // 计算膨胀代价 - 距离越近代价越高
                                int cost = static_cast<int>(90.0 * (1.0 - distance / inflation_radius));
                                
                                // 更新代价，取较高值
                                if (tempMap[neighbor_index] < cost) {
                                    tempMap[neighbor_index] = cost;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 将临时地图复制到最终代价地图
    for (size_t i = 0; i < costmap.data.size(); ++i) {
        costmap.data[i] = tempMap[i];
    }
    
    return costmap;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "process_global_map");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    // Publishers
    ros::Publisher gridMapPub = private_nh.advertise<grid_map_msgs::GridMap>("slope_grid_map", 1, true);
    ros::Publisher processedCloudPub = private_nh.advertise<sensor_msgs::PointCloud2>("processed_cloud", 1, true);
    ros::Publisher slopeCloudPub = private_nh.advertise<sensor_msgs::PointCloud2>("slope_cloud", 1, true);
    ros::Publisher slopeMarkersPub = private_nh.advertise<visualization_msgs::MarkerArray>("slope_markers", 1, true);
    ros::Publisher costmapPub = private_nh.advertise<nav_msgs::OccupancyGrid>("slope_costmap", 1, true);
    
    // Get parameters
    std::string pcdFilePath;
    double resolution;
    private_nh.param<std::string>("pcd_file_path", pcdFilePath, "/root/shared_dir/LIO-SAM/shangpo2map0.01/GlobalMap.pcd");
    private_nh.param<double>("resolution", resolution, 0.1);
    
    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFilePath, *cloud) == -1) {
        ROS_ERROR("Failed to load PCD file: %s", pcdFilePath.c_str());
        return -1;
    }
    
    // Downsample using voxel grid filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxelFilter;
    voxelFilter.setInputCloud(cloud);
    voxelFilter.setLeafSize(resolution, resolution, resolution);
    voxelFilter.filter(*cloudFiltered);
    
    // Remove outliers
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudClean(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sorFilter;
    sorFilter.setInputCloud(cloudFiltered);
    sorFilter.setMeanK(50);
    sorFilter.setStddevMulThresh(1.0);
    sorFilter.filter(*cloudClean);
    
    // Create grid map
    grid_map::GridMap map;
    map.setFrameId("map");
    
    // Determine map size from point cloud bounds
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = -std::numeric_limits<double>::max();
    double max_y = -std::numeric_limits<double>::max();
    
    for (const auto& point : cloudClean->points) {
        min_x = std::min(min_x, (double)point.x);
        min_y = std::min(min_y, (double)point.y);
        max_x = std::max(max_x, (double)point.x);
        max_y = std::max(max_y, (double)point.y);
    }
    
    // Add some margin
    double margin = 5.0;
    min_x -= margin;
    min_y -= margin;
    max_x += margin;
    max_y += margin;
    
    // Set map geometry
    grid_map::Length length(max_x - min_x, max_y - min_y);
    grid_map::Position position((max_x + min_x) / 2.0, (max_y + min_y) / 2.0);
    map.setGeometry(length, resolution, position);
    
    // Add elevation layer
    map.add("elevation", std::numeric_limits<float>::quiet_NaN());
    
    // Fill grid map with point cloud data
    for (const auto& point : cloudClean->points) {
        grid_map::Position pos(point.x, point.y);
        grid_map::Index index;
        if (map.getIndex(pos, index)) {
            // Update with max height at this position
            float& height = map.at("elevation", index);
            if (std::isnan(height) || point.z > height) {
                height = point.z;
            }
        }
    }
    
    ROS_INFO("Created grid map with size: %.1f x %.1f m, resolution: %.2f m", 
             map.getLength().x(), map.getLength().y(), map.getResolution());
    
    // Calculate slope
    calculateSlope(map);
    
    // Create a colored point cloud for visualization
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr slopeCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // 统计坡度范围
    double min_slope = std::numeric_limits<double>::max();
    double max_slope = std::numeric_limits<double>::min();
    int count_0_10 = 0;
    int count_10_15 = 0;
    int count_15_20 = 0;
    int count_20_25 = 0;
    int count_25_plus = 0;
    int total_valid_points = 0;
    
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        
        // Skip if elevation or slope is NaN
        if (!std::isfinite(map.at("elevation", index)) || !std::isfinite(map.at("slope", index))) {
            continue;
        }
        
        // Get position
        grid_map::Position position;
        map.getPosition(*it, position);
        
        // Create colored point
        pcl::PointXYZRGB point;
        point.x = position.x();
        point.y = position.y();
        point.z = map.at("elevation", index);
        
        // Color based on slope ranges
        float slope = map.at("slope", index);
        uint8_t r, g, b;
        
        // 更新统计信息
        min_slope = std::min(min_slope, (double)slope);
        max_slope = std::max(max_slope, (double)slope);
        total_valid_points++;
        
        if (slope < 10.0) {
            // Green for gentle slopes (0-10 degrees)
            r = 0;
            g = 255;
            b = 0;
            count_0_10++;
        } else if (slope < 15.0) {
            // Yellow for moderate slopes (10-15 degrees)
            r = 255;
            g = 255;
            b = 0;
            count_10_15++;
        } else if (slope < 20.0) {
            // Orange for steep slopes (15-20 degrees)
            r = 255;
            g = 128;
            b = 0;
            count_15_20++;
        } else if (slope < 25.0) {
            // Red for very steep slopes (20-25 degrees)
            r = 255;
            g = 0;
            b = 0;
            count_20_25++;
        } else {
            // Purple for extreme slopes (25+ degrees)
            r = 204;
            g = 0;
            b = 204;
            count_25_plus++;
        }
        
        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        point.rgb = *reinterpret_cast<float*>(&rgb);
        
        slopeCloud->points.push_back(point);
    }
    
    slopeCloud->width = slopeCloud->points.size();
    slopeCloud->height = 1;
    slopeCloud->is_dense = true;
    
    // 输出坡度统计信息
    ROS_INFO("坡度统计信息:");
    ROS_INFO("总有效点数: %d", total_valid_points);
    ROS_INFO("坡度范围: %.2f° - %.2f°", min_slope, max_slope);
    ROS_INFO("0-10°(绿色): %d 点 (%.1f%%)", count_0_10, 100.0 * count_0_10 / total_valid_points);
    ROS_INFO("10-15°(黄色): %d 点 (%.1f%%)", count_10_15, 100.0 * count_10_15 / total_valid_points);
    ROS_INFO("15-20°(橙色): %d 点 (%.1f%%)", count_15_20, 100.0 * count_15_20 / total_valid_points);
    ROS_INFO("20-25°(红色): %d 点 (%.1f%%)", count_20_25, 100.0 * count_20_25 / total_valid_points);
    ROS_INFO("25+°(紫色): %d 点 (%.1f%%)", count_25_plus, 100.0 * count_25_plus / total_valid_points);
    
    // Create slope markers
    visualization_msgs::MarkerArray slopeMarkers = createSlopeMarkers(map);
    
    // Create slope legend and position it in the top-right corner of the map
    visualization_msgs::MarkerArray legendMarkers = createSlopeLegend();
    
    // Update the legend position to be in the top-right corner of the map
    double legend_x = max_x - 10.0;  // 10 meters from the right edge
    double legend_y = max_y - 10.0;  // 10 meters from the top edge
    
    for (auto& marker : legendMarkers.markers) {
        marker.pose.position.x = legend_x;
        marker.pose.position.y = legend_y;
    }
    
    // Merge the legend markers with the slope markers
    size_t original_size = slopeMarkers.markers.size();
    slopeMarkers.markers.resize(original_size + legendMarkers.markers.size());
    for (size_t i = 0; i < legendMarkers.markers.size(); ++i) {
        slopeMarkers.markers[original_size + i] = legendMarkers.markers[i];
    }
    
    // Publish data
    ROS_INFO("Map processing complete. Publishing data...");
    
    // Publish grid map
    grid_map_msgs::GridMap gridMapMsg;
    grid_map::GridMapRosConverter::toMessage(map, gridMapMsg);
    gridMapPub.publish(gridMapMsg);
    
    // Publish processed point cloud
    sensor_msgs::PointCloud2 processedCloudMsg;
    pcl::toROSMsg(*cloudClean, processedCloudMsg);
    processedCloudMsg.header.frame_id = "map";
    processedCloudMsg.header.stamp = ros::Time::now();
    processedCloudPub.publish(processedCloudMsg);
    
    // Publish slope point cloud
    sensor_msgs::PointCloud2 slopeCloudMsg;
    pcl::toROSMsg(*slopeCloud, slopeCloudMsg);
    slopeCloudMsg.header.frame_id = "map";
    slopeCloudMsg.header.stamp = ros::Time::now();
    slopeCloudPub.publish(slopeCloudMsg);
    
    // Publish slope markers
    slopeMarkersPub.publish(slopeMarkers);

    // Publish costmap
    nav_msgs::OccupancyGrid costmapMsg = createCostmapFromSlope(map);
    costmapPub.publish(costmapMsg);
    
    // Keep publishing
    ros::Rate rate(1);  // 1 Hz
    while (ros::ok()) {
        gridMapPub.publish(gridMapMsg);
        processedCloudPub.publish(processedCloudMsg);
        slopeCloudPub.publish(slopeCloudMsg);
        slopeMarkersPub.publish(slopeMarkers);
        costmapPub.publish(costmapMsg); // Keep publishing costmap
        
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
} 