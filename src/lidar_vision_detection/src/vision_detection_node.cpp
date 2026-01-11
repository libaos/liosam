#include "lidar_vision_detection/vision_detector.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "vision_detection_node");
    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    lidar_vision_detection::VisionDetector detector;
    
    if (!detector.initialize(nh, private_nh)) {
        ROS_ERROR("Failed to initialize vision detector");
        return 1;
    }
    
    ros::spin();
    
    return 0;
} 