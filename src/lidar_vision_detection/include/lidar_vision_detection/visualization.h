#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Point.h>

#include "lidar_vision_detection/DetectedObject.h"
#include "lidar_vision_detection/DetectedObjectArray.h"

namespace lidar_vision_detection {

class DetectionVisualizer {
public:
    DetectionVisualizer(ros::NodeHandle& nh);
    ~DetectionVisualizer();
    
    // Create marker from detected objects
    visualization_msgs::MarkerArray createMarkerArray(
        const lidar_vision_detection::DetectedObjectArray& detections);
    
private:
    ros::Publisher marker_pub_;
    
    // Generate a color based on the class label
    std_msgs::ColorRGBA getColor(const std::string& label);
};

} // namespace lidar_vision_detection

#endif // VISUALIZATION_H 