#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>
#include <std_srvs/Trigger.h>

class TwoPointsNavigator {
public:
    TwoPointsNavigator() : 
        nh_("~"),
        move_base_client_("move_base", true),
        first_goal_received_(false),
        second_goal_received_(false),
        first_goal_sent_(false),
        first_goal_reached_(false),
        navigation_in_progress_(false) {
        
        // 订阅来自RViz的导航点
        goal_sub_ = nh_.subscribe("/move_base_simple/goal", 1, &TwoPointsNavigator::goalCallback, this);
        
        // 创建服务以触发导航
        start_navigation_service_ = nh_.advertiseService("start_navigation", &TwoPointsNavigator::startNavigationCallback, this);
        reset_goals_service_ = nh_.advertiseService("reset_goals", &TwoPointsNavigator::resetGoalsCallback, this);
        
        // 等待move_base服务器
        ROS_INFO("Waiting for move_base action server...");
        if (!move_base_client_.waitForServer(ros::Duration(10.0))) {
            ROS_ERROR("Could not connect to move_base action server");
            return;
        }
        ROS_INFO("Connected to move_base action server");
        
        ROS_INFO("Please set the first navigation point using RViz 2D Nav Goal");
    }
    
    // 处理来自RViz的导航点
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal) {
        if (!first_goal_received_) {
            first_goal_ = *goal;
            first_goal_received_ = true;
            ROS_INFO("Received first goal: (%.2f, %.2f)", 
                    first_goal_.pose.position.x, first_goal_.pose.position.y);
            
            ROS_INFO("Please set the second navigation point using RViz 2D Nav Goal");
        } else if (!second_goal_received_) {
            second_goal_ = *goal;
            second_goal_received_ = true;
            ROS_INFO("Received second goal: (%.2f, %.2f)", 
                    second_goal_.pose.position.x, second_goal_.pose.position.y);
            
            ROS_INFO("Both goals received. Use service 'start_navigation' to begin navigation.");
        } else {
            // 更新已有的点
            if (navigation_in_progress_) {
                ROS_WARN("Navigation in progress. Cannot update goals now.");
                return;
            }
            
            // 更新第一个点
            first_goal_ = *goal;
            ROS_INFO("Updated first goal: (%.2f, %.2f)", 
                    first_goal_.pose.position.x, first_goal_.pose.position.y);
            ROS_INFO("Please set the second navigation point using RViz 2D Nav Goal");
            second_goal_received_ = false;  // 需要重新设置第二个点
        }
    }
    
    // 服务回调：开始导航
    bool startNavigationCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        if (!first_goal_received_ || !second_goal_received_) {
            res.success = false;
            res.message = "Cannot start navigation: not all goals are set";
            ROS_WARN("%s", res.message.c_str());
            return true;
        }
        
        if (navigation_in_progress_) {
            res.success = false;
            res.message = "Navigation already in progress";
            ROS_WARN("%s", res.message.c_str());
            return true;
        }
        
        // 重置导航状态
        first_goal_sent_ = false;
        first_goal_reached_ = false;
        navigation_in_progress_ = true;
        
        // 开始导航到第一个点
        sendFirstGoal();
        
        res.success = true;
        res.message = "Navigation started";
        return true;
    }
    
    // 服务回调：重置导航点
    bool resetGoalsCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        if (navigation_in_progress_) {
            res.success = false;
            res.message = "Cannot reset goals while navigation is in progress";
            ROS_WARN("%s", res.message.c_str());
            return true;
        }
        
        first_goal_received_ = false;
        second_goal_received_ = false;
        
        ROS_INFO("Goals have been reset. Please set new goals using RViz 2D Nav Goal");
        
        res.success = true;
        res.message = "Goals reset successfully";
        return true;
    }
    
    // 发送第一个导航点
    void sendFirstGoal() {
        if (!first_goal_received_) {
            ROS_WARN("Cannot send first goal: not received yet");
            navigation_in_progress_ = false;
            return;
        }
        
        if (first_goal_sent_) {
            ROS_WARN("First goal already sent");
            return;
        }
        
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose = first_goal_;
        
        ROS_INFO("Sending first goal to move_base");
        
        move_base_client_.sendGoal(
            goal,
            boost::bind(&TwoPointsNavigator::firstGoalDoneCallback, this, _1, _2),
            boost::bind(&TwoPointsNavigator::activeCallback, this),
            boost::bind(&TwoPointsNavigator::feedbackCallback, this, _1)
        );
        
        first_goal_sent_ = true;
    }
    
    // 当第一个导航点完成时的回调
    void firstGoalDoneCallback(const actionlib::SimpleClientGoalState& state,
                              const move_base_msgs::MoveBaseResultConstPtr& result) {
        if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
            ROS_INFO("First goal reached successfully");
            first_goal_reached_ = true;
            
            // 自动发送第二个点
            if (second_goal_received_) {
                sendSecondGoal();
            } else {
                ROS_WARN("Second goal not set. Navigation sequence incomplete.");
                navigation_in_progress_ = false;
            }
        } else {
            ROS_WARN("Failed to reach first goal: %s", state.toString().c_str());
            navigation_in_progress_ = false;
        }
    }
    
    // 发送第二个导航点
    void sendSecondGoal() {
        if (!second_goal_received_) {
            ROS_WARN("Cannot send second goal: not received yet");
            navigation_in_progress_ = false;
            return;
        }
        
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose = second_goal_;
        
        ROS_INFO("Sending second goal to move_base");
        
        move_base_client_.sendGoal(
            goal,
            boost::bind(&TwoPointsNavigator::secondGoalDoneCallback, this, _1, _2),
            boost::bind(&TwoPointsNavigator::activeCallback, this),
            boost::bind(&TwoPointsNavigator::feedbackCallback, this, _1)
        );
    }
    
    // 当第二个导航点完成时的回调
    void secondGoalDoneCallback(const actionlib::SimpleClientGoalState& state,
                               const move_base_msgs::MoveBaseResultConstPtr& result) {
        if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
            ROS_INFO("Second goal reached successfully");
            ROS_INFO("Navigation completed!");
        } else {
            ROS_WARN("Failed to reach second goal: %s", state.toString().c_str());
        }
        navigation_in_progress_ = false;
    }
    
    // 当目标变为活动状态时的回调
    void activeCallback() {
        ROS_INFO("Goal is now active");
    }
    
    // 接收反馈的回调
    void feedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
        // 可以在这里处理导航反馈，例如打印当前位置
    }
    
private:
    ros::NodeHandle nh_;
    ros::Subscriber goal_sub_;
    ros::ServiceServer start_navigation_service_;
    ros::ServiceServer reset_goals_service_;
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> move_base_client_;
    
    geometry_msgs::PoseStamped first_goal_;
    geometry_msgs::PoseStamped second_goal_;
    
    bool first_goal_received_;
    bool second_goal_received_;
    bool first_goal_sent_;
    bool first_goal_reached_;
    bool navigation_in_progress_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "two_points_navigator");
    
    TwoPointsNavigator navigator;
    
    ros::spin();
    
    return 0;
} 