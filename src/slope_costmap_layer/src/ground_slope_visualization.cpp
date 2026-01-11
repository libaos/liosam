#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>
#include <sensor_msgs/PointCloud2.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_pcl/grid_map_pcl.hpp>
#include <grid_map_msgs/GridMap.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2/LinearMath/Quaternion.h>
#include <nav_msgs/OccupancyGrid.h>
#include <thread>
#include <mutex>
#include <cmath>

// 坡度范围颜色定义
struct SlopeColorRange {
    double min_angle;
    double max_angle;
    std::string name;
    uint8_t r, g, b;
};

// 定义不同坡度范围的颜色
const std::vector<SlopeColorRange> SLOPE_COLORS = {
    {0.0,  3.0, "平坦区域",   40, 180, 50},    // 浅绿色 - 非常平坦
    {3.0,  6.0, "轻微坡度",   120, 220, 50},   // 绿色 - 轻微坡度
    {6.0,  10.0, "缓坡区域",   255, 255, 0},   // 黄色 - 缓坡区域
    {10.0, 15.0, "中等坡度",   255, 180, 0},   // 橙黄色 - 中等坡度
    {15.0, 20.0, "明显坡度",   255, 128, 0},   // 橙色 - 明显坡度
    {20.0, 25.0, "陡峭区域",   255, 0, 0},     // 红色 - 陡峭区域
    {25.0, 30.0, "很陡区域",   200, 0, 100},   // 紫红色 - 很陡区域
    {30.0, 90.0, "极陡区域",   128, 0, 128}    // 紫色 - 极陡区域
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
        textMarker.scale.z = 0.5;  // 文本高度
        
        // 颜色（与相应的坡度范围相同）
        textMarker.color.r = SLOPE_COLORS[i].r / 255.0;
        textMarker.color.g = SLOPE_COLORS[i].g / 255.0;
        textMarker.color.b = SLOPE_COLORS[i].b / 255.0;
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
    titleMarker.scale.z = 0.6;  // 文本高度
    
    // 颜色（白色）
    titleMarker.color.r = 1.0;
    titleMarker.color.g = 1.0;
    titleMarker.color.b = 1.0;
    titleMarker.color.a = 1.0;
    
    // 文本
    titleMarker.text = "地面坡度图例";
    
    // 添加到标记数组
    legendMarkers.markers.push_back(titleMarker);
    
    return legendMarkers;
}

// 提取地面点云
pcl::PointCloud<pcl::PointXYZ>::Ptr extractGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                                                double distance_threshold = 0.2,
                                                bool* success = nullptr) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // 如果输入点云为空，直接返回
    if (cloud->empty()) {
        if (success) *success = false;
        ROS_WARN("输入点云为空");
        return ground_cloud;
    }
    
    // 使用RANSAC找到主要平面（地面）
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    // 创建分割对象
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(200);
    seg.setDistanceThreshold(distance_threshold);
    
    // 设置主轴垂直方向约束（z轴向上）
    Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
    seg.setAxis(axis);
    seg.setEpsAngle(30.0 * (M_PI / 180.0)); // 30度容忍度
    
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    
    if (inliers->indices.empty()) {
        if (success) *success = false;
        ROS_WARN("找不到地面平面");
        return ground_cloud;
    }
    
    // 提取地面点
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*ground_cloud);
    
    ROS_INFO("地面平面方程: %fx + %fy + %fz + %f = 0", 
             coefficients->values[0], coefficients->values[1], 
             coefficients->values[2], coefficients->values[3]);
    ROS_INFO("提取的地面点数: %zu (占比: %.2f%%)", 
             ground_cloud->size(), 100.0 * ground_cloud->size() / cloud->size());
    
    if (success) *success = true;
    return ground_cloud;
}

// 细分地形：对地面点云进行聚类并计算法向量和坡度
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr analyzeGroundSlopes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& ground_cloud,
                                                               double normal_radius = 0.5,
                                                               double cluster_tolerance = 0.5,
                                                               int min_cluster_size = 50,
                                                               int max_cluster_size = 100000) {
    // 创建一个新的带法向量和颜色的点云
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    
    // 如果地面点云为空，直接返回
    if (ground_cloud->empty()) {
        ROS_WARN("地面点云为空");
        return colored_cloud;
    }
    
    // 创建KdTree用于法向量计算和聚类
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    
    // 计算法向量
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    
    ne.setInputCloud(ground_cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(normal_radius);
    ne.compute(*normals);
    
    ROS_INFO("法向量计算完成，点云大小: %zu, 法向量大小: %zu", ground_cloud->size(), normals->size());
    
    // 对地面点云进行聚类，将相似的区域分组
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_cluster_size);
    ec.setMaxClusterSize(max_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(ground_cloud);
    ec.extract(cluster_indices);
    
    ROS_INFO("找到 %zu 个地形区域", cluster_indices.size());
    
    // 统计不同坡度范围的点数
    std::vector<int> slope_counts(SLOPE_COLORS.size(), 0);
    double min_slope = std::numeric_limits<double>::max();
    double max_slope = std::numeric_limits<double>::lowest();
    double sum_slope = 0.0;
    
    // 调整点云大小以匹配原始点云
    colored_cloud->resize(ground_cloud->size());
    
    // 将原始点云和法向量合并，并计算坡度
    for (size_t i = 0; i < ground_cloud->size(); ++i) {
        pcl::PointXYZRGBNormal& p = colored_cloud->points[i];
        
        // 复制位置
        p.x = ground_cloud->points[i].x;
        p.y = ground_cloud->points[i].y;
        p.z = ground_cloud->points[i].z;
        
        // 复制法向量
        p.normal_x = normals->points[i].normal_x;
        p.normal_y = normals->points[i].normal_y;
        p.normal_z = normals->points[i].normal_z;
        p.curvature = normals->points[i].curvature;
        
        // 计算法向量与垂直向上向量的夹角（以度为单位）
        double dot_product = p.normal_z;  // 假设法向量已经归一化，与(0,0,1)的点积就是z分量
        if (dot_product > 1.0) dot_product = 1.0;
        if (dot_product < -1.0) dot_product = -1.0;
        
        double slope_angle = std::acos(std::fabs(dot_product)) * 180.0 / M_PI;  // 使用绝对值，因为我们只关心倾斜度
        
        // 统计坡度信息
        min_slope = std::min(min_slope, slope_angle);
        max_slope = std::max(max_slope, slope_angle);
        sum_slope += slope_angle;
        
        // 根据坡度角度设置颜色
        uint8_t r = 255, g = 255, b = 255;  // 默认白色
        
        for (size_t j = 0; j < SLOPE_COLORS.size(); ++j) {
            if (slope_angle >= SLOPE_COLORS[j].min_angle && slope_angle < SLOPE_COLORS[j].max_angle) {
                r = SLOPE_COLORS[j].r;
                g = SLOPE_COLORS[j].g;
                b = SLOPE_COLORS[j].b;
                slope_counts[j]++;
                break;
            }
        }
        
        // 设置RGB颜色
        p.r = r;
        p.g = g;
        p.b = b;
    }
    
    // 输出坡度统计信息
    double avg_slope = sum_slope / ground_cloud->size();
    ROS_INFO("坡度统计信息: 最小值=%.2f°, 最大值=%.2f°, 平均值=%.2f°", 
             min_slope, max_slope, avg_slope);
    
    ROS_INFO("坡度范围统计:");
    ROS_INFO("总点数: %zu", ground_cloud->size());
    ROS_INFO("坡度分布范围: %.2f° - %.2f°", min_slope, max_slope);
    
    for (size_t i = 0; i < SLOPE_COLORS.size(); ++i) {
        double percentage = 100.0 * static_cast<double>(slope_counts[i]) / ground_cloud->size();
        ROS_INFO("%g-%g°(%s): %d 点 (%.1f%%)", 
                 SLOPE_COLORS[i].min_angle, SLOPE_COLORS[i].max_angle, 
                 SLOPE_COLORS[i].name.c_str(), slope_counts[i], percentage);
    }
    
    return colored_cloud;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "ground_slope_visualization");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    // 发布器
    ros::Publisher ground_cloud_pub = private_nh.advertise<sensor_msgs::PointCloud2>("ground_colored_cloud", 1, true);
    ros::Publisher full_cloud_pub = private_nh.advertise<sensor_msgs::PointCloud2>("full_cloud", 1, true);
    ros::Publisher normals_pub = private_nh.advertise<sensor_msgs::PointCloud2>("normals", 1, true);
    ros::Publisher legend_pub = private_nh.advertise<visualization_msgs::MarkerArray>("slope_legend", 1, true);
    
    // 获取参数
    std::string pcd_file_path;
    double resolution, normal_radius, ground_distance_threshold;
    bool visualize_normals;
    private_nh.param<std::string>("pcd_file_path", pcd_file_path, "/root/shared_dir/LIO-SAM/shangpo2map0.01/GlobalMap.pcd");
    private_nh.param<double>("resolution", resolution, 0.1);
    private_nh.param<double>("normal_radius", normal_radius, 0.5);
    private_nh.param<double>("ground_distance_threshold", ground_distance_threshold, 0.2);
    private_nh.param<bool>("visualize_normals", visualize_normals, true);
    
    // 加载点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path, *cloud) == -1) {
        ROS_ERROR("Failed to load PCD file: %s", pcd_file_path.c_str());
        return -1;
    }
    
    ROS_INFO("已加载点云，原始点数: %zu", cloud->size());
    
    // 下采样使用体素网格滤波器
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxelFilter;
    voxelFilter.setInputCloud(cloud);
    voxelFilter.setLeafSize(resolution, resolution, resolution);
    voxelFilter.filter(*cloudFiltered);
    
    ROS_INFO("下采样后点数: %zu", cloudFiltered->size());
    
    // 去除离群点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudClean(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sorFilter;
    sorFilter.setInputCloud(cloudFiltered);
    sorFilter.setMeanK(50);
    sorFilter.setStddevMulThresh(1.0);
    sorFilter.filter(*cloudClean);
    
    ROS_INFO("去除离群点后点数: %zu", cloudClean->size());
    
    // 提取地面点云
    bool ground_found = false;
    pcl::PointCloud<pcl::PointXYZ>::Ptr groundCloud = extractGround(cloudClean, ground_distance_threshold, &ground_found);
    
    if (!ground_found || groundCloud->empty()) {
        ROS_ERROR("无法提取地面点云，退出");
        return -1;
    }
    
    // 计算地面坡度和法向量，并添加颜色
    ROS_INFO("正在计算地面坡度和法向量...");
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_ground_cloud = analyzeGroundSlopes(groundCloud, normal_radius);
    
    // 根据点云边界确定地图大小（用于图例位置）
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = -std::numeric_limits<double>::max();
    double max_y = -std::numeric_limits<double>::max();
    
    for (const auto& point : groundCloud->points) {
        min_x = std::min(min_x, (double)point.x);
        min_y = std::min(min_y, (double)point.y);
        max_x = std::max(max_x, (double)point.x);
        max_y = std::max(max_y, (double)point.y);
    }
    
    // 创建原始点云消息
    sensor_msgs::PointCloud2 full_cloud_msg;
    pcl::toROSMsg(*cloudClean, full_cloud_msg);
    full_cloud_msg.header.frame_id = "map";
    full_cloud_msg.header.stamp = ros::Time::now();
    
    // 创建带有颜色的地面点云消息
    sensor_msgs::PointCloud2 colored_ground_cloud_msg;
    pcl::toROSMsg(*colored_ground_cloud, colored_ground_cloud_msg);
    colored_ground_cloud_msg.header.frame_id = "map";
    colored_ground_cloud_msg.header.stamp = ros::Time::now();
    
    // 如果需要可视化法向量
    sensor_msgs::PointCloud2 normals_msg;
    if (visualize_normals) {
        // 创建一个专门用于可视化法向量的点云
        pcl::PointCloud<pcl::PointNormal>::Ptr normal_cloud(new pcl::PointCloud<pcl::PointNormal>());
        normal_cloud->resize(colored_ground_cloud->size());
        
        for (size_t i = 0; i < colored_ground_cloud->size(); ++i) {
            const pcl::PointXYZRGBNormal& p = colored_ground_cloud->points[i];
            pcl::PointNormal& pn = normal_cloud->points[i];
            
            pn.x = p.x;
            pn.y = p.y;
            pn.z = p.z;
            pn.normal_x = p.normal_x;
            pn.normal_y = p.normal_y;
            pn.normal_z = p.normal_z;
            pn.curvature = p.curvature;
        }
        
        pcl::toROSMsg(*normal_cloud, normals_msg);
        normals_msg.header.frame_id = "map";
        normals_msg.header.stamp = ros::Time::now();
    }
    
    // 创建坡度图例（放在地图的右上角）
    visualization_msgs::MarkerArray legend = createSlopeLegend(max_x - 10.0, max_y - 10.0);
    
    // 发布频率 (Hz)
    double publish_rate = 2.0;
    ros::Rate rate(publish_rate);
    
    ROS_INFO("处理完成。开始发布数据...");
    
    // 不断发布数据
    while (ros::ok()) {
        // 更新时间戳
        colored_ground_cloud_msg.header.stamp = ros::Time::now();
        full_cloud_msg.header.stamp = ros::Time::now();
        
        if (visualize_normals) {
            normals_msg.header.stamp = ros::Time::now();
        }
        
        for (auto& marker : legend.markers) {
            marker.header.stamp = ros::Time::now();
        }
        
        // 发布数据
        ground_cloud_pub.publish(colored_ground_cloud_msg);
        full_cloud_pub.publish(full_cloud_msg);
        
        if (visualize_normals) {
            normals_pub.publish(normals_msg);
        }
        legend_pub.publish(legend);
        
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
} 