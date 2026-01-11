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
#include <map_msgs/OccupancyGridUpdate.h>
#include <algorithm>
#include <cmath>

// 坡度范围颜色定义
struct SlopeColorRange {
    double min_angle;
    double max_angle;
    std::string name;
    double r, g, b;
};

// 定义不同坡度范围的颜色
const std::vector<SlopeColorRange> SLOPE_COLORS = {
    {0.0,  10.0, "平坦区域",   0.0, 1.0, 0.0},  // 绿色 - 平坦区域
    {10.0, 15.0, "缓坡区域",   1.0, 1.0, 0.0},  // 黄色 - 缓坡区域
    {15.0, 20.0, "中等坡度",   1.0, 0.5, 0.0},  // 橙色 - 中等坡度
    {20.0, 25.0, "陡峭区域",   1.0, 0.0, 0.0},  // 红色 - 陡峭区域
    {25.0, 90.0, "极陡区域",   0.8, 0.0, 0.8}   // 紫色 - 极陡区域
};

// 创建坡度图例
visualization_msgs::MarkerArray createSlopeLegend(double x_pos, double y_pos) {
    visualization_msgs::MarkerArray legendMarkers;
    
    double z_pos = 5.0;  // 地图上方的高度
    double vertical_spacing = 0.6;  // 图例条目之间的间距
    
    // 为每个坡度范围创建图例
    for (size_t i = 0; i < SLOPE_COLORS.size(); ++i) {
        // 文本标记
        visualization_msgs::Marker textMarker;
        textMarker.header.frame_id = "map";
        textMarker.header.stamp = ros::Time::now();
        textMarker.ns = "slope_legend_text";
        textMarker.id = i;
        textMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        textMarker.action = visualization_msgs::Marker::ADD;
        
        // 位置（垂直堆叠）
        textMarker.pose.position.x = x_pos;
        textMarker.pose.position.y = y_pos;
        textMarker.pose.position.z = z_pos - i * vertical_spacing;
        
        // 方向（默认）
        textMarker.pose.orientation.w = 1.0;
        
        // 比例（文本大小）
        textMarker.scale.z = 0.4;  // 文本高度
        
        // 颜色（与相应的坡度范围相同）
        textMarker.color.r = SLOPE_COLORS[i].r;
        textMarker.color.g = SLOPE_COLORS[i].g;
        textMarker.color.b = SLOPE_COLORS[i].b;
        textMarker.color.a = 1.0;
        
        // 文本（角度范围与描述）
        std::stringstream ss;
        ss << SLOPE_COLORS[i].min_angle << "°-" << SLOPE_COLORS[i].max_angle << "° (" << SLOPE_COLORS[i].name << ")";
        textMarker.text = ss.str();
        
        // 添加到标记数组
        legendMarkers.markers.push_back(textMarker);
    }
    
    // 添加标题
    visualization_msgs::Marker titleMarker;
    titleMarker.header.frame_id = "map";
    titleMarker.header.stamp = ros::Time::now();
    titleMarker.ns = "slope_legend_title";
    titleMarker.id = 0;
    titleMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    titleMarker.action = visualization_msgs::Marker::ADD;
    
    // 位置（图例条目上方）
    titleMarker.pose.position.x = x_pos;
    titleMarker.pose.position.y = y_pos;
    titleMarker.pose.position.z = z_pos + vertical_spacing;
    
    // 方向（默认）
    titleMarker.pose.orientation.w = 1.0;
    
    // 比例（文本大小）
    titleMarker.scale.z = 0.5;  // 文本高度
    
    // 颜色（白色）
    titleMarker.color.r = 1.0;
    titleMarker.color.g = 1.0;
    titleMarker.color.b = 1.0;
    titleMarker.color.a = 1.0;
    
    // 文本
    titleMarker.text = "坡度图例";
    
    // 添加到标记数组
    legendMarkers.markers.push_back(titleMarker);
    
    return legendMarkers;
}

// 计算点云的坡度
void calculateSlope(grid_map::GridMap& map) {
    // 创建坡度层
    map.add("slope");
    map.add("slope_color");
    
    // 获取高程层
    const grid_map::Matrix& elevation = map["elevation"];
    grid_map::Matrix& slope = map["slope"];
    grid_map::Matrix& slope_color = map["slope_color"];
    
    // 使用中心差分计算坡度
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        
        // 如果高程是NaN则跳过
        if (!std::isfinite(elevation(index(0), index(1)))) {
            slope(index(0), index(1)) = std::numeric_limits<float>::quiet_NaN();
            slope_color(index(0), index(1)) = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        
        // 获取相邻索引
        grid_map::Index indexNorth = index + grid_map::Index(-1, 0);
        grid_map::Index indexSouth = index + grid_map::Index(1, 0);
        grid_map::Index indexWest = index + grid_map::Index(0, -1);
        grid_map::Index indexEast = index + grid_map::Index(0, 1);
        
        // 检查索引是否在地图边界内
        bool isNorthValid = map.isValid(indexNorth);
        bool isSouthValid = map.isValid(indexSouth);
        bool isWestValid = map.isValid(indexWest);
        bool isEastValid = map.isValid(indexEast);
        
        // 使用可用邻居计算坡度
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
        
        // 根据有效邻居数和单元格大小进行归一化
        if (validX > 0) dzdx /= (validX * map.getResolution());
        if (validY > 0) dzdy /= (validY * map.getResolution());
        
        // 计算坡度角度（度）
        double slopeAngle = std::atan(std::sqrt(dzdx*dzdx + dzdy*dzdy)) * 180.0 / M_PI;
        
        slope(index(0), index(1)) = slopeAngle;
        
        // 根据坡度范围设置颜色值（0-1之间的值，用于可视化）
        for (const auto& range : SLOPE_COLORS) {
            if (slopeAngle >= range.min_angle && slopeAngle < range.max_angle) {
                // 将颜色编码为单个浮点数，以便在网格地图中存储
                // 范围从0到SLOPE_COLORS.size()，代表不同的颜色
                slope_color(index(0), index(1)) = 
                    static_cast<float>(std::distance(SLOPE_COLORS.data(), &range)) / 
                    static_cast<float>(SLOPE_COLORS.size() - 1);
                break;
            }
        }
    }
    
    // 计算坡度统计信息
    double min_slope = std::numeric_limits<double>::max();
    double max_slope = std::numeric_limits<double>::lowest();
    double sum_slope = 0.0;
    int valid_count = 0;
    
    // 统计不同坡度范围的点数
    std::vector<int> slope_counts(SLOPE_COLORS.size(), 0);
    
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        if (std::isfinite(slope(index(0), index(1)))) {
            double val = slope(index(0), index(1));
            min_slope = std::min(min_slope, val);
            max_slope = std::max(max_slope, val);
            sum_slope += val;
            valid_count++;
            
            // 统计各坡度范围的点数
            for (size_t i = 0; i < SLOPE_COLORS.size(); ++i) {
                if (val >= SLOPE_COLORS[i].min_angle && val < SLOPE_COLORS[i].max_angle) {
                    slope_counts[i]++;
                    break;
                }
            }
        }
    }
    
    // 输出坡度统计信息
    if (valid_count > 0) {
        double avg_slope = sum_slope / valid_count;
        ROS_INFO("坡度统计信息: 最小值=%.2f°, 最大值=%.2f°, 平均值=%.2f°", 
                 min_slope, max_slope, avg_slope);
        
        ROS_INFO("坡度范围统计:");
        ROS_INFO("总有效点数: %d", valid_count);
        ROS_INFO("坡度分布范围: %.2f° - %.2f°", min_slope, max_slope);
        
        for (size_t i = 0; i < SLOPE_COLORS.size(); ++i) {
            double percentage = 100.0 * static_cast<double>(slope_counts[i]) / valid_count;
            ROS_INFO("%g-%g°(%s): %d 点 (%.1f%%)", 
                     SLOPE_COLORS[i].min_angle, SLOPE_COLORS[i].max_angle, 
                     SLOPE_COLORS[i].name.c_str(), slope_counts[i], percentage);
        }
    }
}

// 从坡度地图创建代价地图（白底蓝色膨胀层）
nav_msgs::OccupancyGrid createMoveBaseStyleCostmap(const grid_map::GridMap& map) {
    nav_msgs::OccupancyGrid costmap;
    
    // 设置代价地图头部
    costmap.header.frame_id = map.getFrameId();
    costmap.header.stamp = ros::Time::now();
    
    // 设置代价地图元数据
    costmap.info.resolution = map.getResolution();
    costmap.info.width = map.getSize()(0);
    costmap.info.height = map.getSize()(1);
    costmap.info.origin.position.x = map.getPosition()(0) - map.getLength()(0) / 2.0;
    costmap.info.origin.position.y = map.getPosition()(1) - map.getLength()(1) / 2.0;
    costmap.info.origin.position.z = 0.0;
    costmap.info.origin.orientation.w = 1.0;
    
    // 在RViz的map配色方案中:
    // -1 (255): 未知区域 - 灰色
    //  0 (0): 自由空间 - 白色 
    // 100 (100): 障碍物 - 黑色
    // 1-99: 膨胀区域 - 从浅蓝色到深蓝色的渐变
    
    // 初始化代价地图为自由空间(0) - 在map配色方案中显示为白色
    costmap.data.resize(costmap.info.width * costmap.info.height, 0);
    
    // 获取坡度层和高程层
    const grid_map::Matrix& slope = map["slope"];
    const grid_map::Matrix& elevation = map["elevation"];
    
    // 计算高程统计信息，用于地面滤波
    double min_elevation = std::numeric_limits<double>::max();
    double max_elevation = std::numeric_limits<double>::lowest();
    std::vector<double> elevation_values;
    
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        if (std::isfinite(elevation(index(0), index(1)))) {
            double elev = elevation(index(0), index(1));
            min_elevation = std::min(min_elevation, elev);
            max_elevation = std::max(max_elevation, elev);
            elevation_values.push_back(elev);
        }
    }
    
    // 计算高程阈值 - 使用简单统计方法确定地面高度
    std::sort(elevation_values.begin(), elevation_values.end());
    double ground_height = 0.0;
    if (!elevation_values.empty()) {
        // 使用中位数作为地面高度参考
        ground_height = elevation_values[elevation_values.size() / 2];
    }
    
    // 定义地面过滤参数
    double ground_threshold = 0.3; // 超过地面高度0.3米的点被视为障碍物
    
    // 坡度阈值定义
    const double lethal_threshold = 25.0;  // 25°+ -> 致命障碍物 (100)
    
    // 创建临时数组，用于存储障碍物信息
    std::vector<int8_t> obstacle_map(costmap.data.size(), 0);
    
    // 统计信息
    int total_cells = 0;
    int obstacle_cells = 0;
    int ground_cells = 0;
    int unknown_cells = 0;
    
    // 第一步：标记障碍物位置（100 = 黑色）
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        
        // 将网格地图索引转换为代价地图索引
        size_t i = index(0);
        size_t j = index(1);
        size_t costmapIndex = j * costmap.info.width + i;
        
        if (costmapIndex >= costmap.data.size()) {
            continue;
        }
        
        total_cells++;
        
        // 获取高程值 - 用于地面过滤
        bool is_ground = true;
        if (std::isfinite(elevation(index(0), index(1)))) {
            double elev = elevation(index(0), index(1));
            // 高于地面阈值的点被视为障碍物
            if (elev > ground_height + ground_threshold) {
                obstacle_map[costmapIndex] = 100;  // 障碍物 = 黑色
                obstacle_cells++;
                is_ground = false;
            }
        } else {
            unknown_cells++;
            is_ground = false;
        }
        
        if (is_ground) {
            ground_cells++;
        }
        
        // 获取坡度值
        if (std::isfinite(slope(index(0), index(1)))) {
            double slopeValue = slope(index(0), index(1));
            
            // 根据坡度值设置障碍物 (100 = 黑色)
            if (slopeValue >= lethal_threshold) {
                obstacle_map[costmapIndex] = 100;
                if (is_ground) {
                    ground_cells--;
                    obstacle_cells++;
                }
            }
        }
    }
    
    ROS_INFO("Ground filtering statistics:");
    ROS_INFO("Total cells: %d", total_cells);
    ROS_INFO("Ground cells: %d (%.1f%%)", ground_cells, 100.0 * ground_cells / total_cells);
    ROS_INFO("Obstacle cells: %d (%.1f%%)", obstacle_cells, 100.0 * obstacle_cells / total_cells);
    ROS_INFO("Unknown cells: %d (%.1f%%)", unknown_cells, 100.0 * unknown_cells / total_cells);
    ROS_INFO("Ground height reference: %.2f m, threshold: %.2f m", ground_height, ground_threshold);
    
    // 第二步：创建膨胀层
    // 定义膨胀半径（单元格数）- 增大以使效果更明显
    int inflation_radius = 30;
    
    // 调整膨胀代价衰减因子 - 降低以使膨胀效果更明显
    double cost_scaling_factor = 0.7;
    
    // 复制一份代价地图，用于膨胀处理
    std::vector<int8_t> inflated_map = obstacle_map;
    
    // 应用膨胀算法
    for (size_t j = 0; j < costmap.info.height; ++j) {
        for (size_t i = 0; i < costmap.info.width; ++i) {
            size_t index = j * costmap.info.width + i;
            
            // 如果是障碍物，为其周围添加膨胀区域
            if (obstacle_map[index] == 100) {
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
                            if (distance <= inflation_radius && obstacle_map[neighbor_index] != 100) {
                                // 计算膨胀代价 - 距离越近代价越高
                                // 使用幂函数衰减，确保在map配色方案下有良好的蓝色渐变效果
                                double factor = 1.0 - (distance / inflation_radius);
                                
                                // 代价范围: 10-99
                                // 99(深蓝色)最接近障碍物，10(浅蓝色)最远离障碍物
                                // 最小值设为10，使蓝色更加明显
                                int cost = static_cast<int>(99.0 * std::pow(factor, cost_scaling_factor));
                                
                                // 确保最小为10，有更明显的蓝色
                                cost = std::max(10, cost);
                                
                                // 取最大值，避免覆盖更高的代价
                                if (inflated_map[neighbor_index] < cost) {
                                    inflated_map[neighbor_index] = cost;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 将膨胀后的地图复制到最终代价地图
    costmap.data = inflated_map;
    
    return costmap;
}

// 主函数
int main(int argc, char** argv) {
    ros::init(argc, argv, "slope_visualization_costmap");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    // 发布器
    ros::Publisher gridMapPub = private_nh.advertise<grid_map_msgs::GridMap>("slope_grid_map", 1, true);
    ros::Publisher costmapPub = private_nh.advertise<nav_msgs::OccupancyGrid>("slope_costmap", 1, true);
    ros::Publisher processedCloudPub = private_nh.advertise<sensor_msgs::PointCloud2>("processed_cloud", 1, true);
    ros::Publisher legendPub = private_nh.advertise<visualization_msgs::MarkerArray>("slope_legend", 1, true);
    
    // 获取参数
    std::string pcdFilePath;
    double resolution;
    private_nh.param<std::string>("pcd_file_path", pcdFilePath, "/root/shared_dir/LIO-SAM/shangpo2map0.01/GlobalMap.pcd");
    private_nh.param<double>("resolution", resolution, 0.1);
    
    // 加载点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFilePath, *cloud) == -1) {
        ROS_ERROR("Failed to load PCD file: %s", pcdFilePath.c_str());
        return -1;
    }
    
    // 下采样使用体素网格滤波器
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxelFilter;
    voxelFilter.setInputCloud(cloud);
    voxelFilter.setLeafSize(resolution, resolution, resolution);
    voxelFilter.filter(*cloudFiltered);
    
    // 去除离群点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudClean(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sorFilter;
    sorFilter.setInputCloud(cloudFiltered);
    sorFilter.setMeanK(50);
    sorFilter.setStddevMulThresh(1.0);
    sorFilter.filter(*cloudClean);
    
    // 创建网格地图
    grid_map::GridMap map;
    map.setFrameId("map");
    
    // 根据点云边界确定地图大小
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
    
    // 添加一些边距
    double margin = 5.0;
    min_x -= margin;
    min_y -= margin;
    max_x += margin;
    max_y += margin;
    
    // 设置地图几何形状
    grid_map::Length length(max_x - min_x, max_y - min_y);
    grid_map::Position position((max_x + min_x) / 2.0, (max_y + min_y) / 2.0);
    map.setGeometry(length, resolution, position);
    
    // 添加高程层
    map.add("elevation", std::numeric_limits<float>::quiet_NaN());
    
    // 用点云数据填充网格地图
    for (const auto& point : cloudClean->points) {
        grid_map::Position pos(point.x, point.y);
        grid_map::Index index;
        if (map.getIndex(pos, index)) {
            // 更新此位置的最大高度
            float& height = map.at("elevation", index);
            if (std::isnan(height) || point.z > height) {
                height = point.z;
            }
        }
    }
    
    ROS_INFO("Created grid map with size: %.1f x %.1f m, resolution: %.2f m", 
             map.getLength().x(), map.getLength().y(), map.getResolution());
    
    // 计算坡度
    calculateSlope(map);
    
    // 创建带有膨胀层的坡度代价地图
    nav_msgs::OccupancyGrid costmap = createMoveBaseStyleCostmap(map);
    
    // 将处理后的点云转换为ROS消息
    sensor_msgs::PointCloud2 processedCloudMsg;
    pcl::toROSMsg(*cloudClean, processedCloudMsg);
    processedCloudMsg.header.frame_id = "map";
    processedCloudMsg.header.stamp = ros::Time::now();
    
    // 将网格地图转换为ROS消息
    grid_map_msgs::GridMap gridMapMsg;
    grid_map::GridMapRosConverter::toMessage(map, gridMapMsg);
    
    // 创建坡度图例 - 位置在地图右上角
    visualization_msgs::MarkerArray legend = createSlopeLegend(max_x - 10.0, max_y - 10.0);
    
    // 发布频率 (Hz)
    double publish_rate = 2.0;
    ros::Rate rate(publish_rate);
    
    ROS_INFO("Map processing complete. Publishing data...");
    
    // 不断发布数据
    while (ros::ok()) {
        // 更新时间戳
        gridMapMsg.info.header.stamp = ros::Time::now();
        costmap.header.stamp = ros::Time::now();
        processedCloudMsg.header.stamp = ros::Time::now();
        for (auto& marker : legend.markers) {
            marker.header.stamp = ros::Time::now();
        }
        
        // 发布数据
        gridMapPub.publish(gridMapMsg);
        costmapPub.publish(costmap);
        processedCloudPub.publish(processedCloudMsg);
        legendPub.publish(legend);
        
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
} 