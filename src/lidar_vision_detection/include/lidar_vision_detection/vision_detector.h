#ifndef VISION_DETECTOR_H
#define VISION_DETECTOR_H

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// Include generated message types and utilities
#include "lidar_vision_detection/DetectedObject.h"
#include "lidar_vision_detection/DetectedObjectArray.h"
#include "lidar_vision_detection/visualization.h"

namespace lidar_vision_detection {

// Class to store detection results
class Detection {
public:
    cv::Rect bbox;      // Bounding box in image
    std::string label;  // Class label
    float confidence;   // Detection confidence
    
    Detection() : confidence(0.0f) {}
    Detection(const cv::Rect& r, const std::string& l, float conf)
        : bbox(r), label(l), confidence(conf) {}
};

class VisionDetector {
public:
    VisionDetector();
    ~VisionDetector();
    
    // Initialize the detector
    bool initialize(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    
    // Process incoming image
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    
    // Process incoming point cloud (optional)
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    
private:
    // ROS communication
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher detection_pub_;
    
    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // Detector parameters
    float confidence_threshold_;
    float nms_threshold_;
    int input_width_;
    int input_height_;
    std::string model_path_;
    std::string config_path_;
    std::string camera_frame_id_;
    std::vector<std::string> class_names_;
    
    // OpenCV DNN
    cv::dnn::Net net_;
    std::vector<std::string> output_names_;
    
    // Current data
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud_;
    
    // Visualization
    std::shared_ptr<DetectionVisualizer> visualizer_;
    
    // Detector methods
    bool loadNetwork();
    std::vector<Detection> runInference(const cv::Mat& image);
    void postProcessDetections(const std::vector<cv::Mat>& outputs, const cv::Mat& image, std::vector<Detection>& detections);
    lidar_vision_detection::DetectedObjectArray createDetectionMessage(const std::vector<Detection>& detections, const std_msgs::Header& header);
};

} // namespace lidar_vision_detection

#endif // VISION_DETECTOR_H 