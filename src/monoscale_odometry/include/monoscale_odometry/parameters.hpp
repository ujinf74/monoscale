// Every parameter the estimator takes, declared in one place.
//
// The node and the offline replay both come through here, so a configuration
// cannot mean one thing live and another when it is scored. The defaults and
// the reasoning behind them are carried over from the Python unchanged: what
// each number costs when it is wrong is the most expensive thing this project
// knows, and it lives in these comments.

#ifndef MONOSCALE_ODOMETRY_CPP__PARAMETERS_HPP_
#define MONOSCALE_ODOMETRY_CPP__PARAMETERS_HPP_

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "monoscale_core/estimator.hpp"

namespace monoscale_ros
{

// Where the data comes from and goes, which the core knows nothing about.
struct Topics
{
  std::string imu = "/sensing/imu/imu_data";
  std::string odometry = "/localization/kinematic_state";
  std::string pose = "/current_pose";
  std::string ground_points = "/perception/ground_points";
  std::string map_frame = "map";
  // REP-105 puts dead reckoning here and localisation above it: odom->base_link
  // is continuous and drifts, map->odom is the correction and may jump. This
  // estimator is the first of those, so its pose belongs in odom.
  std::string odom_frame = "odom";
  std::string base_frame = "base_link";
  std::string track_prefix;
  std::vector<std::string> image_topics;
  std::vector<std::string> camera_info_topics;

  bool use_camera_info = true;
  bool publish_tf = true;
  bool publish_map_to_odom = true;
  bool extrinsics_from_tf = true;
  double tf_lookup_timeout_sec = 5.0;

  std::string input_reliability = "best_effort";
  int input_queue_depth = 5;
  std::string input_history = "keep_last";
  std::string input_durability = "volatile";
  std::string imu_reliability = "best_effort";
  int imu_queue_depth = 200;
  int odometry_queue_depth = 10;

  double solve_timer_hz = 200.0;
  int executor_threads = 1;
};

struct Configuration
{
  monoscale::EstimatorSettings estimator;
  Topics topics;
};

// Declares every parameter on `node` and reads the result back.
Configuration declare_and_read(rclcpp::Node & node);

}  // namespace monoscale_ros

#endif  // MONOSCALE_ODOMETRY_CPP__PARAMETERS_HPP_
