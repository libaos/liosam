#include "lidar_vision_detection/visualization.h"

namespace lidar_vision_detection {

DetectionVisualizer::DetectionVisualizer(ros::NodeHandle& nh) {
    // Create the marker publisher
    marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("detection_markers", 1);
}

DetectionVisualizer::~DetectionVisualizer() {
}

visualization_msgs::MarkerArray DetectionVisualizer::createMarkerArray(
    const lidar_vision_detection::DetectedObjectArray& detections) {
    
    visualization_msgs::MarkerArray marker_array;
    
    // Create markers for each detection
    for (size_t i = 0; i < detections.objects.size(); ++i) {
        const auto& obj = detections.objects[i];
        
        // Bounding box marker
        visualization_msgs::Marker box_marker;
        box_marker.header = obj.header;
        box_marker.ns = "bounding_boxes";
        box_marker.id = static_cast<int>(i);
        box_marker.type = visualization_msgs::Marker::LINE_STRIP;
        box_marker.action = visualization_msgs::Marker::ADD;
        
        // If 3D pose is available
        if (obj.dimensions.x > 0 && obj.dimensions.y > 0 && obj.dimensions.z > 0) {
            // Create a 3D bounding box
            geometry_msgs::Point p;
            
            // Bottom rectangle
            p.x = obj.pose.position.x - obj.dimensions.x/2;
            p.y = obj.pose.position.y - obj.dimensions.y/2;
            p.z = obj.pose.position.z - obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x + obj.dimensions.x/2;
            p.y = obj.pose.position.y - obj.dimensions.y/2;
            p.z = obj.pose.position.z - obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x + obj.dimensions.x/2;
            p.y = obj.pose.position.y + obj.dimensions.y/2;
            p.z = obj.pose.position.z - obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x - obj.dimensions.x/2;
            p.y = obj.pose.position.y + obj.dimensions.y/2;
            p.z = obj.pose.position.z - obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x - obj.dimensions.x/2;
            p.y = obj.pose.position.y - obj.dimensions.y/2;
            p.z = obj.pose.position.z - obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            // Top rectangle
            p.x = obj.pose.position.x - obj.dimensions.x/2;
            p.y = obj.pose.position.y - obj.dimensions.y/2;
            p.z = obj.pose.position.z + obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x + obj.dimensions.x/2;
            p.y = obj.pose.position.y - obj.dimensions.y/2;
            p.z = obj.pose.position.z + obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x + obj.dimensions.x/2;
            p.y = obj.pose.position.y + obj.dimensions.y/2;
            p.z = obj.pose.position.z + obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x - obj.dimensions.x/2;
            p.y = obj.pose.position.y + obj.dimensions.y/2;
            p.z = obj.pose.position.z + obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x - obj.dimensions.x/2;
            p.y = obj.pose.position.y - obj.dimensions.y/2;
            p.z = obj.pose.position.z + obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            // Connect bottom and top rectangles
            p.x = obj.pose.position.x + obj.dimensions.x/2;
            p.y = obj.pose.position.y - obj.dimensions.y/2;
            p.z = obj.pose.position.z + obj.dimensions.z/2;
            box_marker.points.push_back(p);
            p.x = obj.pose.position.x + obj.dimensions.x/2;
            p.y = obj.pose.position.y - obj.dimensions.y/2;
            p.z = obj.pose.position.z - obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x + obj.dimensions.x/2;
            p.y = obj.pose.position.y + obj.dimensions.y/2;
            p.z = obj.pose.position.z + obj.dimensions.z/2;
            box_marker.points.push_back(p);
            p.x = obj.pose.position.x + obj.dimensions.x/2;
            p.y = obj.pose.position.y + obj.dimensions.y/2;
            p.z = obj.pose.position.z - obj.dimensions.z/2;
            box_marker.points.push_back(p);
            
            p.x = obj.pose.position.x - obj.dimensions.x/2;
            p.y = obj.pose.position.y + obj.dimensions.y/2;
            p.z = obj.pose.position.z + obj.dimensions.z/2;
            box_marker.points.push_back(p);
            p.x = obj.pose.position.x - obj.dimensions.x/2;
            p.y = obj.pose.position.y + obj.dimensions.y/2;
            p.z = obj.pose.position.z - obj.dimensions.z/2;
            box_marker.points.push_back(p);
        }
        else if (!obj.polygon.empty()) {
            // Use 2D polygon in image coordinates
            for (size_t j = 0; j < obj.polygon.size(); ++j) {
                box_marker.points.push_back(obj.polygon[j]);
            }
            // Close the loop
            box_marker.points.push_back(obj.polygon[0]);
        }
        
        // Set marker properties
        box_marker.scale.x = 0.05; // Line width
        box_marker.color = getColor(obj.label);
        box_marker.lifetime = ros::Duration(0.1);
        
        // Text marker for label and confidence
        visualization_msgs::Marker text_marker;
        text_marker.header = obj.header;
        text_marker.ns = "labels";
        text_marker.id = static_cast<int>(i);
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        if (obj.dimensions.x > 0) {
            text_marker.pose.position.x = obj.pose.position.x;
            text_marker.pose.position.y = obj.pose.position.y;
            text_marker.pose.position.z = obj.pose.position.z + obj.dimensions.z/2 + 0.2;
        }
        else if (!obj.polygon.empty()) {
            // Use the top-left corner of the polygon
            text_marker.pose.position = obj.polygon[0];
            text_marker.pose.position.z += 0.2;
        }
        
        text_marker.text = obj.label + " (" + std::to_string(static_cast<int>(obj.score * 100)) + "%)";
        text_marker.scale.z = 0.4;  // Font size
        text_marker.color = getColor(obj.label);
        text_marker.lifetime = ros::Duration(0.1);
        
        // Add markers to array
        marker_array.markers.push_back(box_marker);
        marker_array.markers.push_back(text_marker);
    }
    
    // Publish markers
    marker_pub_.publish(marker_array);
    
    return marker_array;
}

std_msgs::ColorRGBA DetectionVisualizer::getColor(const std::string& label) {
    std_msgs::ColorRGBA color;
    color.a = 1.0;
    
    // Simple hash function to create consistent colors for labels
    size_t hash = 0;
    for (size_t i = 0; i < label.length(); ++i) {
        hash = label[i] + (hash << 6) + (hash << 16) - hash;
    }
    
    // Use hash to generate color
    color.r = static_cast<float>((hash & 0xFF)) / 255.0f;
    color.g = static_cast<float>(((hash >> 8) & 0xFF)) / 255.0f;
    color.b = static_cast<float>(((hash >> 16) & 0xFF)) / 255.0f;
    
    // Ensure color is not too dark
    float min_val = 0.3f;
    color.r = std::max(color.r, min_val);
    color.g = std::max(color.g, min_val);
    color.b = std::max(color.b, min_val);
    
    return color;
}

} // namespace lidar_vision_detection 