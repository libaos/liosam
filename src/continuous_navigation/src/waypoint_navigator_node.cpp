#include <ros/ros.h>
#include "continuous_navigation/waypoint_navigator.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "waypoint_navigator");
    ros::NodeHandle nh("~");
    
    ROS_INFO("Starting Waypoint Navigator node");
    
    continuous_navigation::WaypointNavigator navigator(nh);
    
    if (!navigator.initialize()) {
        ROS_ERROR("Failed to initialize Waypoint Navigator");
        return 1;
    }
    
    // Start the navigation
    navigator.start();
    
    // Let ROS handle all callbacks
    ros::spin();
    
    return 0;
} 