#ifndef WAYPOINT_NAVIGATOR_H
#define WAYPOINT_NAVIGATOR_H

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <tf/transform_datatypes.h>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <fstream>

namespace continuous_navigation {

/**
 * @brief Structure to represent a waypoint with position and orientation
 */
struct Waypoint {
    double x;
    double y;
    double z;
    double yaw;  // Orientation in radians
    std::string name;  // Optional name/label for the waypoint
};

/**
 * @brief Class to manage continuous navigation through a sequence of waypoints
 */
class WaypointNavigator {
public:
    /**
     * @brief Constructor
     * @param nh ROS node handle
     */
    WaypointNavigator(ros::NodeHandle& nh);
    
    /**
     * @brief Destructor
     */
    ~WaypointNavigator();
    
    /**
     * @brief Initialize the navigator
     * @return True if initialization was successful
     */
    bool initialize();
    
    /**
     * @brief Start the navigation process
     */
    void start();

private:
    /**
     * @brief Load waypoints from a YAML file
     * @param filename Path to the YAML file
     * @return True if loading was successful
     */
    bool loadWaypoints(const std::string& filename);
    
    /**
     * @brief Send the next waypoint to move_base
     */
    void sendNextWaypoint();
    
    /**
     * @brief Callback for when a goal is completed
     * @param state The state of the action
     * @param result The result of the action
     */
    void goalCompletedCallback(const actionlib::SimpleClientGoalState& state,
                              const move_base_msgs::MoveBaseResultConstPtr& result);
    
    /**
     * @brief Callback for when a goal is active
     */
    void goalActiveCallback();
    
    /**
     * @brief Callback for goal feedback
     * @param feedback The feedback from the action
     */
    void goalFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback);
    
    /**
     * @brief Convert a waypoint to a MoveBaseGoal
     * @param waypoint The waypoint to convert
     * @return The corresponding MoveBaseGoal
     */
    move_base_msgs::MoveBaseGoal waypointToMoveBaseGoal(const Waypoint& waypoint);

    /**
     * @brief Callback for receiving manual goals from user
     * @param goal The goal from user
     */
    void manualGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal);
    
    /**
     * @brief Save manual waypoints to a YAML file
     * @param filename Path to the YAML file
     * @return True if saving was successful
     */
    bool saveManualWaypoints(const std::string& filename);
    
    /**
     * @brief Convert quaternion to yaw angle
     * @param q Quaternion to convert
     * @return Yaw angle in radians
     */
    double quaternionToYaw(const geometry_msgs::Quaternion& q);

    // ROS node handle
    ros::NodeHandle& nh_;
    
    // Action client for move_base
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> move_base_client_;
    
    // Subscriber for manual goals
    ros::Subscriber manual_goal_sub_;
    
    // List of waypoints
    std::vector<Waypoint> waypoints_;
    
    // List of manual waypoints
    std::vector<geometry_msgs::PoseStamped> manual_waypoints_;
    
    // Current waypoint index
    size_t current_waypoint_index_;
    
    // Parameters
    std::string waypoints_file_;
    std::string global_frame_;
    double goal_timeout_;
    bool cycle_waypoints_;  // Whether to loop through waypoints continuously
    bool pause_between_waypoints_;  // Whether to pause between waypoints
    double pause_duration_;  // Duration to pause between waypoints in seconds
    int num_manual_goals_;   // Number of manual goals to wait for
    int manual_goals_received_;  // Number of manual goals received
    bool auto_start_;  // Whether to start navigation automatically after receiving manual goals
    std::string manual_waypoints_file_;  // File to save manual waypoints
    bool save_manual_waypoints_;  // Whether to save manual waypoints to file
    
    // Timer for pausing between waypoints
    ros::Timer pause_timer_;
    
    // Callback for pause timer
    void pauseTimerCallback(const ros::TimerEvent&);
};

} // namespace continuous_navigation

#endif // WAYPOINT_NAVIGATOR_H 