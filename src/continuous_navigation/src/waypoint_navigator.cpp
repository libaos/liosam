#include "continuous_navigation/waypoint_navigator.h"

namespace continuous_navigation {

WaypointNavigator::WaypointNavigator(ros::NodeHandle& nh)
    : nh_(nh),
      move_base_client_("move_base", true),
      current_waypoint_index_(0),
      manual_goals_received_(0) {
}

WaypointNavigator::~WaypointNavigator() {
    // Cancel any active goals before shutting down
    if (move_base_client_.isServerConnected()) {
        move_base_client_.cancelAllGoals();
    }
}

bool WaypointNavigator::initialize() {
    // Load parameters
    nh_.param<std::string>("waypoints_file", waypoints_file_, "waypoints.yaml");
    nh_.param<std::string>("global_frame", global_frame_, "map");
    nh_.param<double>("goal_timeout", goal_timeout_, 60.0);
    nh_.param<bool>("cycle_waypoints", cycle_waypoints_, true);
    nh_.param<bool>("pause_between_waypoints", pause_between_waypoints_, false);
    nh_.param<double>("pause_duration", pause_duration_, 2.0);
    nh_.param<int>("num_manual_goals", num_manual_goals_, 2);
    nh_.param<bool>("auto_start", auto_start_, true);
    nh_.param<std::string>("manual_waypoints_file", manual_waypoints_file_, "manual_waypoints.yaml");
    nh_.param<bool>("save_manual_waypoints", save_manual_waypoints_, true);

    ROS_INFO("Waiting for move_base action server...");
    if (!move_base_client_.waitForServer(ros::Duration(10.0))) {
        ROS_ERROR("Could not connect to move_base action server");
        return false;
    }
    ROS_INFO("Connected to move_base action server");

    // Load waypoints
    if (!loadWaypoints(waypoints_file_)) {
        ROS_ERROR("Failed to load waypoints from file: %s", waypoints_file_.c_str());
        return false;
    }
    
    if (waypoints_.empty()) {
        ROS_ERROR("No waypoints loaded");
        return false;
    }
    
    ROS_INFO("Loaded %zu waypoints", waypoints_.size());
    
    // Subscribe to manual goals
    if (num_manual_goals_ > 0) {
        manual_goal_sub_ = nh_.subscribe("/move_base_simple/goal", 10, 
                                        &WaypointNavigator::manualGoalCallback, this);
        ROS_INFO("Waiting for user to provide %d manual goals via RViz '2D Nav Goal' button", 
                num_manual_goals_);
    }
    
    return true;
}

void WaypointNavigator::manualGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal) {
    if (manual_goals_received_ >= num_manual_goals_) {
        ROS_INFO("Already received all manual goals. Ignoring this one.");
        return; // Ignore additional goals
    }
    
    manual_goals_received_++;
    manual_waypoints_.push_back(*goal);
    
    ROS_INFO("Received manual goal %d/%d: (%.2f, %.2f)", 
             manual_goals_received_, num_manual_goals_,
             goal->pose.position.x, goal->pose.position.y);
    
    // Save the manual waypoints to file if requested
    if (save_manual_waypoints_) {
        if (saveManualWaypoints(manual_waypoints_file_)) {
            ROS_INFO("Saved manual waypoints to %s", manual_waypoints_file_.c_str());
        } else {
            ROS_WARN("Failed to save manual waypoints to %s", manual_waypoints_file_.c_str());
        }
    }
    
    // If this is the first manual goal and auto_start is true, start navigation
    if (manual_goals_received_ == 1 && auto_start_) {
        ROS_INFO("Starting navigation to first manual goal");
        
        // Send the goal with callbacks
        move_base_msgs::MoveBaseGoal move_base_goal;
        move_base_goal.target_pose = manual_waypoints_[0];
        
        move_base_client_.sendGoal(
            move_base_goal,
            boost::bind(&WaypointNavigator::goalCompletedCallback, this, _1, _2),
            boost::bind(&WaypointNavigator::goalActiveCallback, this),
            boost::bind(&WaypointNavigator::goalFeedbackCallback, this, _1)
        );
    } else if (manual_goals_received_ == num_manual_goals_) {
        ROS_INFO("Received all manual goals. %s", 
                auto_start_ ? "Navigation will continue automatically." : 
                "Call start() to begin navigation.");
    }
}

bool WaypointNavigator::saveManualWaypoints(const std::string& filename) {
    try {
        YAML::Node root;
        YAML::Node waypoints_node = root["waypoints"];
        
        for (size_t i = 0; i < manual_waypoints_.size(); i++) {
            const geometry_msgs::PoseStamped& pose = manual_waypoints_[i];
            YAML::Node wp;
            wp["name"] = "manual_waypoint_" + std::to_string(i);
            wp["x"] = pose.pose.position.x;
            wp["y"] = pose.pose.position.y;
            wp["z"] = pose.pose.position.z;
            wp["yaw"] = quaternionToYaw(pose.pose.orientation);
            waypoints_node.push_back(wp);
        }
        
        root["waypoints"] = waypoints_node;
        
        // Create output file stream
        std::ofstream fout(filename.c_str());
        if (!fout.is_open()) {
            ROS_ERROR("Failed to open file for writing: %s", filename.c_str());
            return false;
        }
        
        fout << root;
        fout.close();
        
        return true;
    } catch (const std::exception& e) {
        ROS_ERROR("Error saving manual waypoints: %s", e.what());
        return false;
    }
}

double WaypointNavigator::quaternionToYaw(const geometry_msgs::Quaternion& q) {
    tf::Quaternion tfq(q.x, q.y, q.z, q.w);
    tf::Matrix3x3 m(tfq);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return yaw;
}

bool WaypointNavigator::loadWaypoints(const std::string& filename) {
    try {
        YAML::Node config = YAML::LoadFile(filename);
        
        if (!config["waypoints"]) {
            ROS_ERROR("No 'waypoints' section found in the YAML file");
            return false;
        }
        
        const YAML::Node& waypoints = config["waypoints"];
        for (size_t i = 0; i < waypoints.size(); i++) {
            const YAML::Node& wp = waypoints[i];
            
            Waypoint waypoint;
            waypoint.x = wp["x"].as<double>();
            waypoint.y = wp["y"].as<double>();
            waypoint.z = wp["z"].as<double>(0.0);  // Default to 0.0 if not specified
            waypoint.yaw = wp["yaw"].as<double>(0.0);  // Default to 0.0 if not specified
            
            // Optional name
            if (wp["name"]) {
                waypoint.name = wp["name"].as<std::string>();
            } else {
                waypoint.name = "waypoint_" + std::to_string(i);
            }
            
            waypoints_.push_back(waypoint);
            ROS_INFO("Loaded waypoint %s: (%.2f, %.2f, %.2f), yaw: %.2f",
                     waypoint.name.c_str(), waypoint.x, waypoint.y, waypoint.z, waypoint.yaw);
        }
        
        return true;
    } catch (const YAML::Exception& e) {
        ROS_ERROR("Error parsing waypoints file: %s", e.what());
        return false;
    } catch (const std::exception& e) {
        ROS_ERROR("Error loading waypoints: %s", e.what());
        return false;
    }
}

void WaypointNavigator::start() {
    if (waypoints_.empty() && manual_waypoints_.empty()) {
        ROS_ERROR("No waypoints to navigate to");
        return;
    }
    
    if (num_manual_goals_ > 0 && manual_goals_received_ < num_manual_goals_) {
        ROS_INFO("Still waiting for manual goals. Received %d/%d", 
                manual_goals_received_, num_manual_goals_);
        return;
    }
    
    // If we have manual waypoints and haven't started navigation yet, start with the first manual waypoint
    if (!manual_waypoints_.empty() && !auto_start_) {
        ROS_INFO("Starting navigation with first manual waypoint");
        
        move_base_msgs::MoveBaseGoal move_base_goal;
        move_base_goal.target_pose = manual_waypoints_[0];
        
        move_base_client_.sendGoal(
            move_base_goal,
            boost::bind(&WaypointNavigator::goalCompletedCallback, this, _1, _2),
            boost::bind(&WaypointNavigator::goalActiveCallback, this),
            boost::bind(&WaypointNavigator::goalFeedbackCallback, this, _1)
        );
    } else if (manual_waypoints_.empty()) {
        // If no manual waypoints, start with the first waypoint from file
        current_waypoint_index_ = 0;
        sendNextWaypoint();
    }
}

void WaypointNavigator::sendNextWaypoint() {
    // Check if we still have manual waypoints to navigate to
    if (current_waypoint_index_ < manual_waypoints_.size()) {
        ROS_INFO("Navigating to manual waypoint %zu/%zu", 
                current_waypoint_index_ + 1, manual_waypoints_.size());
        
        move_base_msgs::MoveBaseGoal move_base_goal;
        move_base_goal.target_pose = manual_waypoints_[current_waypoint_index_];
        
        move_base_client_.sendGoal(
            move_base_goal,
            boost::bind(&WaypointNavigator::goalCompletedCallback, this, _1, _2),
            boost::bind(&WaypointNavigator::goalActiveCallback, this),
            boost::bind(&WaypointNavigator::goalFeedbackCallback, this, _1)
        );
        return;
    }
    
    // If we've completed all manual waypoints, move on to the waypoints from file
    size_t file_waypoint_index = current_waypoint_index_ - manual_waypoints_.size();
    
    if (file_waypoint_index >= waypoints_.size()) {
        if (cycle_waypoints_) {
            ROS_INFO("Reached end of waypoints, cycling back to the beginning");
            current_waypoint_index_ = manual_waypoints_.size(); // Reset to start of file waypoints
            file_waypoint_index = 0;
        } else {
            ROS_INFO("Reached end of waypoints, navigation complete");
            return;
        }
    }
    
    const Waypoint& waypoint = waypoints_[file_waypoint_index];
    ROS_INFO("Navigating to waypoint %s (%zu/%zu): (%.2f, %.2f, %.2f), yaw: %.2f",
             waypoint.name.c_str(), file_waypoint_index + 1, waypoints_.size(),
             waypoint.x, waypoint.y, waypoint.z, waypoint.yaw);
    
    move_base_msgs::MoveBaseGoal goal = waypointToMoveBaseGoal(waypoint);
    
    // Send the goal with callbacks
    move_base_client_.sendGoal(
        goal,
        boost::bind(&WaypointNavigator::goalCompletedCallback, this, _1, _2),
        boost::bind(&WaypointNavigator::goalActiveCallback, this),
        boost::bind(&WaypointNavigator::goalFeedbackCallback, this, _1)
    );
}

move_base_msgs::MoveBaseGoal WaypointNavigator::waypointToMoveBaseGoal(const Waypoint& waypoint) {
    move_base_msgs::MoveBaseGoal goal;
    
    // Set the frame ID and timestamp
    goal.target_pose.header.frame_id = global_frame_;
    goal.target_pose.header.stamp = ros::Time::now();
    
    // Set the position
    goal.target_pose.pose.position.x = waypoint.x;
    goal.target_pose.pose.position.y = waypoint.y;
    goal.target_pose.pose.position.z = waypoint.z;
    
    // Set the orientation
    goal.target_pose.pose.orientation = tf::createQuaternionMsgFromYaw(waypoint.yaw);
    
    return goal;
}

void WaypointNavigator::goalCompletedCallback(const actionlib::SimpleClientGoalState& state,
                                            const move_base_msgs::MoveBaseResultConstPtr& result) {
    if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
        if (current_waypoint_index_ < manual_waypoints_.size()) {
            ROS_INFO("Manual waypoint %zu reached successfully", current_waypoint_index_ + 1);
        } else {
            size_t file_waypoint_index = current_waypoint_index_ - manual_waypoints_.size();
            ROS_INFO("Waypoint %zu reached successfully", file_waypoint_index + 1);
        }
        
        current_waypoint_index_++;
        
        if (pause_between_waypoints_) {
            ROS_INFO("Pausing for %.1f seconds before next waypoint", pause_duration_);
            pause_timer_ = nh_.createTimer(
                ros::Duration(pause_duration_),
                &WaypointNavigator::pauseTimerCallback,
                this,
                true  // oneshot
            );
        } else {
            sendNextWaypoint();
        }
    } else {
        ROS_WARN("Failed to reach waypoint %zu: %s", 
                current_waypoint_index_ + 1, state.toString().c_str());
        
        // Optionally retry or skip to next waypoint
        current_waypoint_index_++;
        sendNextWaypoint();
    }
}

void WaypointNavigator::goalActiveCallback() {
    ROS_INFO("Goal is now active");
}

void WaypointNavigator::goalFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
    // This could be used to monitor progress, but we'll keep it simple for now
}

void WaypointNavigator::pauseTimerCallback(const ros::TimerEvent&) {
    ROS_INFO("Pause complete, continuing to next waypoint");
    sendNextWaypoint();
}

} // namespace continuous_navigation 