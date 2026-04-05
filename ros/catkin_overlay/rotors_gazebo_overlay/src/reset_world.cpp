#include <ros/ros.h>
#include <std_srvs/Empty.h>

int main(int argc, char** argv) {
  
  ros::init(argc, argv, "reset_world");
  ros::NodeHandle nh;

  std_srvs::Empty srv;
  
  //bool paused = ros::service::call("/gazebo/pause_physics", srv);

  ros::Duration(1).sleep();

  bool reset_world = ros::service::call("/gazebo/reset_world", srv);

  //bool unpaused = ros::service::call("/gazebo/unpause_physics", srv);

  ros::spinOnce();
  ros::shutdown();
  
  return 0;
}