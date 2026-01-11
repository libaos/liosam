#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_pcl/grid_map_pcl.hpp>
#include <grid_map_msgs/GridMap.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2/LinearMath/Quaternion.h>
#include <fstream>
#include <string>
#include <vector>

class TrajectoryCostmapVisualizer
{
public:
    TrajectoryCostmapVisualizer() : nh_("~")
    {
        // 参数获取
        std::string pcd_file_path;
        double resolution;
        nh_.param("pcd_file_path", pcd_file_path, std::string("/root/shared_dir/LIO-SAM/shangpo2map0.01/GlobalMap.pcd"));
        nh_.param("resolution", resolution, 0.1);
        nh_.param("trajectory_csv", trajectory_csv_, std::string("/root/lio_ws/trajectory_data/2025-07-09-15-57-19__liorl_mapping_path.csv"));
        nh_.param("obstacle_threshold", obstacle_threshold_, 0.3);
        nh_.param("inflation_radius", inflation_radius_, 0.5);

        // 发布者
        costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/trajectory_costmap", 1, true);
        map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("/trajectory_grid_map", 1, true);
        path_pub_ = nh_.advertise<nav_msgs::Path>("/visualization_trajectory", 1, true);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/trajectory_markers", 1, true);

        // 处理点云和创建代价地图
        processPointCloudAndCreateCostmap(pcd_file_path, resolution);
        
        // 加载轨迹数据
        loadTrajectoryFromCSV(trajectory_csv_);
        
        // 发布轨迹标记
        publishTrajectoryMarkers();
        
        // 发布轨迹路径
        publishTrajectoryPath();

        // 定时发布数据
        timer_ = nh_.createTimer(ros::Duration(1.0), &TrajectoryCostmapVisualizer::timerCallback, this);
    }

private:
    // 处理点云数据并创建代价地图
    void processPointCloudAndCreateCostmap(const std::string &pcd_file_path, double resolution)
    {
        // 加载点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path, *cloud) == -1)
        {
            ROS_ERROR("无法加载点云文件: %s", pcd_file_path.c_str());
            return;
        }
        
        ROS_INFO("加载了 %ld 个点云点", cloud->points.size());
        
        // 下采样点云
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(resolution, resolution, resolution);
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_grid.filter(*filtered_cloud);
        
        ROS_INFO("下采样后 %ld 个点云点", filtered_cloud->points.size());
        
        // 创建网格地图
        grid_map_.setFrameId("map");
        grid_map_.setGeometry(grid_map::Length(200.0, 200.0), resolution);
        grid_map_.add("elevation");
        grid_map_.add("obstacle");
        
        // 点云转为高程图和障碍物图层
        for (const auto &point : filtered_cloud->points)
        {
            grid_map::Position position(point.x, point.y);
            if (!grid_map_.isInside(position))
                continue;
                
            grid_map::Index index;
            grid_map_.getIndex(position, index);
            // 必须使用cast()方法进行类型转换
            
            // 更新高程图层
            float &elevation = grid_map_.at("elevation", index);
            if (std::isnan(elevation) || point.z > elevation)
                elevation = point.z;
                
            // 对于高于地面一定高度的点标记为障碍物
            if (point.z > obstacle_threshold_)
            {
                grid_map_.at("obstacle", index) = 1.0;  // 1.0表示有障碍物
            }
        }
        
        // 进行障碍物膨胀处理，创建代价地图
        inflateObstacles();
        
        // 将网格地图转换为占用栅格地图
        grid_map_msgs::GridMap grid_map_msg;
        grid_map::GridMapRosConverter::toMessage(grid_map_, grid_map_msg);
        map_pub_.publish(grid_map_msg);
        
        // 转换为占用栅格地图
        nav_msgs::OccupancyGrid occupancy_grid;
        grid_map::GridMapRosConverter::toOccupancyGrid(grid_map_, "obstacle", 0.0, 1.0, occupancy_grid);
        occupancy_grid.header.stamp = ros::Time::now();
        occupancy_grid.header.frame_id = "map";
        
        // 保存
        costmap_ = occupancy_grid;
        
        ROS_INFO("创建了网格地图，大小: %.1f x %.1f m，分辨率: %.2f m", 
                 grid_map_.getLength().x(), grid_map_.getLength().y(), grid_map_.getResolution());
    }
    
    // 膨胀障碍物
    void inflateObstacles()
    {
        grid_map::GridMap inflated_map = grid_map_;
        inflated_map.add("inflated_obstacle", 0.0);
        
        // 获取膨胀半径的栅格数量
        int inflation_radius_cells = std::ceil(inflation_radius_ / grid_map_.getResolution());
        
        // 遍历所有栅格
        for (grid_map::GridMapIterator it(grid_map_); !it.isPastEnd(); ++it)
        {
            const grid_map::Index center_index(*it);
            
            // 如果是障碍物
            if (grid_map_.at("obstacle", center_index) > 0.5)
            {
                // 将周围膨胀区域内的栅格都标记为障碍物
                // 获取中心点位置
                grid_map::Position center_position;
                grid_map_.getPosition(center_index, center_position);
                
                // 计算膨胀半径对应的单元格数量
                int radius_cells = std::ceil(inflation_radius_ / grid_map_.getResolution());
                
                // 遍历中心点周围的正方形区域
                for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
                    for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
                        // 检查位置是否在圆形范围内
                        grid_map::Index neighbor_index(center_index(0) + dx, center_index(1) + dy);
                        
                        // 检查索引是否有效
                        if (!grid_map_.isValid(neighbor_index)) {
                            continue;
                        }
                        
                        // 计算实际位置
                        grid_map::Position current_position;
                        grid_map_.getPosition(neighbor_index, current_position);
                        
                        // 计算与中心点的距离
                        double distance = (center_position - current_position).norm();
                    
                        // 如果在圆形范围内
                        if (distance <= inflation_radius_) {
                            // 根据距离计算代价值 (越近代价越高)
                            double cost = 1.0 - distance / inflation_radius_;
                            if (cost < 0.0) cost = 0.0;
                            
                            // 更新代价值，取最大值
                            float &current_cost = inflated_map.at("inflated_obstacle", neighbor_index);
                            current_cost = std::max(current_cost, static_cast<float>(cost));
                        }
                    }
                }
            }
        }
        
        // 复制膨胀后的障碍物层到原始地图中
        grid_map_["obstacle"] = inflated_map["inflated_obstacle"];
    }

    // 从CSV文件加载轨迹数据
    void loadTrajectoryFromCSV(const std::string& csv_file)
    {
        std::ifstream file(csv_file);
        if (!file.is_open())
        {
            ROS_ERROR("无法打开轨迹文件: %s", csv_file.c_str());
            return;
        }
        
        // 跳过CSV头行
        std::string line;
        std::getline(file, line);
        
        // 读取每一行数据
        while (std::getline(file, line))
        {
            std::istringstream ss(line);
            std::string token;
            std::vector<double> values;
            
            // 解析CSV行，格式: timestamp,x,y,z
            while (std::getline(ss, token, ','))
            {
                values.push_back(std::stod(token));
            }
            
            if (values.size() >= 4)
            {
                // 格式: timestamp, x, y, z
                double timestamp = values[0];
                double x = values[1];
                double y = values[2];
                double z = values[3];
                
                trajectory_points_.push_back(std::make_tuple(timestamp, x, y, z));
            }
        }
        
        ROS_INFO("从CSV文件加载了 %ld 个轨迹点", trajectory_points_.size());
    }

    // 发布轨迹标记
    void publishTrajectoryMarkers()
    {
        visualization_msgs::MarkerArray marker_array;
        
        // 轨迹线条标记
        visualization_msgs::Marker line_marker;
        line_marker.header.frame_id = "map";
        line_marker.header.stamp = ros::Time::now();
        line_marker.ns = "trajectory";
        line_marker.id = 0;
        line_marker.type = visualization_msgs::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::Marker::ADD;
        line_marker.scale.x = 0.1;  // 线宽
        line_marker.color.r = 0.0;
        line_marker.color.g = 0.0;
        line_marker.color.b = 1.0;  // 蓝色
        line_marker.color.a = 1.0;
        line_marker.pose.orientation.w = 1.0;
        line_marker.lifetime = ros::Duration(0);  // 永久显示
        
        // 添加轨迹点
        for (const auto& point : trajectory_points_)
        {
            geometry_msgs::Point p;
            p.x = std::get<1>(point);  // x
            p.y = std::get<2>(point);  // y
            p.z = std::get<3>(point);  // z
            line_marker.points.push_back(p);
        }
        
        marker_array.markers.push_back(line_marker);
        
        // 起点标记
        if (!trajectory_points_.empty())
        {
            visualization_msgs::Marker start_marker;
            start_marker.header.frame_id = "map";
            start_marker.header.stamp = ros::Time::now();
            start_marker.ns = "trajectory";
            start_marker.id = 1;
            start_marker.type = visualization_msgs::Marker::SPHERE;
            start_marker.action = visualization_msgs::Marker::ADD;
            start_marker.pose.position.x = std::get<1>(trajectory_points_.front());
            start_marker.pose.position.y = std::get<2>(trajectory_points_.front());
            start_marker.pose.position.z = std::get<3>(trajectory_points_.front());
            start_marker.pose.orientation.w = 1.0;
            start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.5;
            start_marker.color.r = 0.0;
            start_marker.color.g = 1.0;  // 绿色
            start_marker.color.b = 0.0;
            start_marker.color.a = 1.0;
            start_marker.lifetime = ros::Duration(0);
            
            // 终点标记
            visualization_msgs::Marker end_marker = start_marker;
            end_marker.id = 2;
            end_marker.pose.position.x = std::get<1>(trajectory_points_.back());
            end_marker.pose.position.y = std::get<2>(trajectory_points_.back());
            end_marker.pose.position.z = std::get<3>(trajectory_points_.back());
            end_marker.color.r = 1.0;  // 红色
            end_marker.color.g = 0.0;
            end_marker.color.b = 0.0;
            
            marker_array.markers.push_back(start_marker);
            marker_array.markers.push_back(end_marker);
        }
        
        marker_pub_.publish(marker_array);
    }

    // 发布轨迹路径
    void publishTrajectoryPath()
    {
        nav_msgs::Path path;
        path.header.frame_id = "map";
        path.header.stamp = ros::Time::now();
        
        for (const auto& point : trajectory_points_)
        {
            geometry_msgs::PoseStamped pose;
            pose.header = path.header;
            pose.pose.position.x = std::get<1>(point);
            pose.pose.position.y = std::get<2>(point);
            pose.pose.position.z = std::get<3>(point);
            
            // 创建一个简单的朝向，沿着轨迹方向
            tf2::Quaternion q;
            q.setRPY(0, 0, 0);  // 简单的默认朝向
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();
            
            path.poses.push_back(pose);
        }
        
        path_pub_.publish(path);
    }

    // 定时回调函数
    void timerCallback(const ros::TimerEvent& event)
    {
        // 定期更新时间戳并重新发布数据
        costmap_.header.stamp = ros::Time::now();
        costmap_pub_.publish(costmap_);
        
        // 重新发布轨迹
        publishTrajectoryPath();
        publishTrajectoryMarkers();
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher costmap_pub_;
    ros::Publisher map_pub_;
    ros::Publisher path_pub_;
    ros::Publisher marker_pub_;
    ros::Timer timer_;
    
    grid_map::GridMap grid_map_;
    nav_msgs::OccupancyGrid costmap_;
    
    std::vector<std::tuple<double, double, double, double>> trajectory_points_;  // timestamp, x, y, z
    
    std::string trajectory_csv_;
    double obstacle_threshold_;
    double inflation_radius_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "trajectory_costmap_visualizer");
    TrajectoryCostmapVisualizer visualizer;
    ros::spin();
    return 0;
} 