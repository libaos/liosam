#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "liosam_smoke");
  ros::NodeHandle nh;
  ROS_INFO("liosam_smoke: node started (workspace launch sanity check).");
  ros::spin();
  return 0;
}
