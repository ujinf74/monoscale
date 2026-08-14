// Score the estimator against a recorded drive, without a ROS graph.
//
// Replaying a bag through live nodes leaves the answer depending on the order
// callbacks happened to fire in, which is enough to move drift by half a point
// between two runs of the same configuration. That is larger than most of the
// differences worth measuring.
//
// Here the bag is read directly and the estimator is driven in recorded order,
// so the same bag and the same parameters give the same number every time --
// and, since nothing in the library reads a clock or a random seed, the same
// number on the desktop and on the Orin. That is what makes this the tool the
// port is judged with: the Python equivalent
// (monoscale_evaluation/offline_replay.py) reads the same bag the same way.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "monoscale_core/estimator.hpp"
#include "monoscale_odometry/parameters.hpp"

namespace
{

using monoscale::Pose2;

double wrap_pi(double angle)
{
  return std::remainder(angle, 2.0 * M_PI);
}

struct Sample
{
  double stamp;
  double x;
  double y;
  double yaw;
};

double yaw_of(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

// `pose` expressed in the frame of `origin`.
Sample relative_pose(const Sample & origin, const Sample & pose)
{
  const double c = std::cos(origin.yaw);
  const double s = std::sin(origin.yaw);
  const double dx = pose.x - origin.x;
  const double dy = pose.y - origin.y;
  return Sample{
    pose.stamp, c * dx + s * dy, -s * dx + c * dy, wrap_pi(pose.yaw - origin.yaw)};
}

Sample compose_pose(const Sample & base, const Sample & delta)
{
  const double c = std::cos(base.yaw);
  const double s = std::sin(base.yaw);
  return Sample{
    delta.stamp,
    base.x + c * delta.x - s * delta.y,
    base.y + s * delta.x + c * delta.y,
    wrap_pi(base.yaw + delta.yaw)};
}

// Accumulates absolute drift and segment-wise relative error, the way
// monoscale_evaluation does. The ground truth is rigidly anchored to the
// estimate at the first matched sample and no further alignment is applied, so
// what comes out is accumulated drift rather than a fitting residual.
class TrajectoryMetrics
{
public:
  explicit TrajectoryMetrics(double segment)
  : segment_(std::max(segment, 1e-3)) {}

  void add(const Sample & estimate, const Sample & truth)
  {
    const double error = std::hypot(estimate.x - truth.x, estimate.y - truth.y);
    const double yaw_error = std::abs(wrap_pi(estimate.yaw - truth.yaw));
    ++samples_;
    squared_position_ += error * error;
    max_position_ = std::max(max_position_, error);
    final_position_ = error;
    squared_yaw_ += yaw_error * yaw_error;

    if (have_previous_) {
      truth_distance_ += std::hypot(truth.x - previous_truth_.x, truth.y - previous_truth_.y);
      estimate_distance_ +=
        std::hypot(estimate.x - previous_estimate_.x, estimate.y - previous_estimate_.y);
    }
    previous_truth_ = truth;
    previous_estimate_ = estimate;
    have_previous_ = true;

    if (!have_anchor_) {
      anchor_truth_ = truth;
      anchor_estimate_ = estimate;
      anchor_distance_ = truth_distance_;
      have_anchor_ = true;
      return;
    }
    if (truth_distance_ - anchor_distance_ < segment_) {
      return;
    }
    const Sample truth_step = relative_pose(anchor_truth_, truth);
    const Sample estimate_step = relative_pose(anchor_estimate_, estimate);
    const double travelled = std::hypot(truth_step.x, truth_step.y);
    // Straight line displacement per segment, which unlike a summed path
    // length is not inflated by per sample noise.
    segment_truth_ += travelled;
    segment_estimate_ += std::hypot(estimate_step.x, estimate_step.y);
    if (travelled > 1e-6) {
      relative_.push_back(
        std::hypot(estimate_step.x - truth_step.x, estimate_step.y - truth_step.y) /
        travelled);
    }
    anchor_truth_ = truth;
    anchor_estimate_ = estimate;
    anchor_distance_ = truth_distance_;
  }

  int samples() const {return samples_;}
  double truth_distance() const {return truth_distance_;}

  void report(const std::string & label) const
  {
    if (samples_ == 0) {
      std::printf("--- %s\nno matched samples\n", label.c_str());
      return;
    }
    double relative_mean = 0.0;
    for (double value : relative_) {
      relative_mean += value;
    }
    relative_mean = relative_.empty() ? 0.0 : relative_mean / relative_.size();
    std::printf("--- %s\n", label.c_str());
    std::printf(
      "n=%d gt=%.1fm scale=%.3f ate_rmse=%.4fm final=%.4fm drift=%.2f%% "
      "yaw_rmse=%.2fdeg rpe=%.2f%%\n",
      samples_, truth_distance_,
      truth_distance_ > 1e-6 ? estimate_distance_ / truth_distance_ : 0.0,
      std::sqrt(squared_position_ / samples_), final_position_,
      truth_distance_ > 1e-6 ? 100.0 * final_position_ / truth_distance_ : 0.0,
      std::sqrt(squared_yaw_ / samples_) * 180.0 / M_PI, 100.0 * relative_mean);
  }

private:
  double segment_;
  int samples_ = 0;
  double squared_position_ = 0.0;
  double max_position_ = 0.0;
  double final_position_ = 0.0;
  double squared_yaw_ = 0.0;
  double truth_distance_ = 0.0;
  double estimate_distance_ = 0.0;
  double segment_truth_ = 0.0;
  double segment_estimate_ = 0.0;
  std::vector<double> relative_;
  Sample previous_truth_{};
  Sample previous_estimate_{};
  Sample anchor_truth_{};
  Sample anchor_estimate_{};
  double anchor_distance_ = 0.0;
  bool have_previous_ = false;
  bool have_anchor_ = false;
};

template<typename Message>
Message deserialize(const rclcpp::SerializedMessage & serialized)
{
  Message message;
  rclcpp::Serialization<Message> serialization;
  serialization.deserialize_message(&serialized, &message);
  return message;
}

double stamp_of(const std_msgs::msg::Header & header)
{
  return header.stamp.sec + header.stamp.nanosec * 1e-9;
}

// The tracker's flat layout:
//   [stamp, count, width, height, id, prev_x, prev_y, cur_x, cur_y, ...]
bool parse_tracks(const std_msgs::msg::Float32MultiArray & message, monoscale::TrackFrame & out)
{
  const auto & data = message.data;
  if (data.size() < 4) {
    return false;
  }
  out.stamp = data[0];
  const int count = static_cast<int>(data[1]);
  out.width = static_cast<int>(data[2]);
  out.height = static_cast<int>(data[3]);
  if (out.stamp <= 0.0 || out.width <= 0 || out.height <= 0 ||
    data.size() < static_cast<size_t>(4 + 5 * count))
  {
    return false;
  }
  out.ids.resize(count);
  out.previous_pixels.resize(count, 2);
  out.pixels.resize(count, 2);
  for (int i = 0; i < count; ++i) {
    const size_t at = static_cast<size_t>(4 + 5 * i);
    out.ids(i) = static_cast<int64_t>(data[at]);
    out.previous_pixels(i, 0) = data[at + 1];
    out.previous_pixels(i, 1) = data[at + 2];
    out.pixels(i, 0) = data[at + 3];
    out.pixels(i, 1) = data[at + 4];
  }
  return true;
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string bag;
  std::string truth_topic = "/carla/ground_truth/pose";
  std::string label;
  std::string tum_directory;
  double distance = 34.0;
  double segment = 1.0;
  double tolerance = 0.05;
  double wheel_base = 2.8192;
  std::vector<std::string> ros_arguments{"--ros-args"};

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto next = [&]() {return i + 1 < argc ? std::string(argv[++i]) : std::string();};
    if (argument == "--params") {
      ros_arguments.push_back("--params-file");
      ros_arguments.push_back(next());
    } else if (argument == "--set") {
      ros_arguments.push_back("-p");
      ros_arguments.push_back(next());
    } else if (argument == "--truth-topic") {
      truth_topic = next();
    } else if (argument == "--label") {
      label = next();
    } else if (argument == "--tum") {
      tum_directory = next();
    } else if (argument == "--distance") {
      distance = std::stod(next());
    } else if (argument == "--segment") {
      segment = std::stod(next());
    } else if (argument == "--tolerance") {
      tolerance = std::stod(next());
    } else if (argument == "--wheel-base") {
      wheel_base = std::stod(next());
    } else if (bag.empty()) {
      bag = argument;
    }
  }
  if (bag.empty()) {
    std::cerr << "사용법: monoscale_replay <bag> --params <yaml> [--set name:=value] ...\n";
    return 2;
  }

  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.arguments(ros_arguments);
  // The node exists only to own the parameters; it is never spun.
  auto node = std::make_shared<rclcpp::Node>("monoscale_replay", options);
  const auto configuration = monoscale_ros::declare_and_read(*node);

  monoscale::Estimator estimator(configuration.estimator);

  std::map<std::string, size_t> track_topics;
  std::map<std::string, size_t> info_topics;
  for (size_t i = 0; i < configuration.estimator.cameras.size(); ++i) {
    const std::string & name = configuration.estimator.cameras[i].name;
    if (!configuration.topics.track_prefix.empty()) {
      track_topics[configuration.topics.track_prefix + "/" + name] = i;
    }
    if (configuration.topics.use_camera_info && i < configuration.topics.camera_info_topics.size()) {
      info_topics[configuration.topics.camera_info_topics[i]] = i;
    }
  }
  if (track_topics.empty()) {
    std::cerr << "track_topic_prefix 가 비어 있다. 이 경로는 트랙 입력만 받는다.\n";
    return 2;
  }

  rosbag2_cpp::Reader reader;
  reader.open(bag);

  std::vector<Sample> estimates;
  std::vector<Sample> truth;
  while (reader.has_next()) {
    const auto message = reader.read_next();
    rclcpp::SerializedMessage serialized(*message->serialized_data);

    const auto track = track_topics.find(message->topic_name);
    if (track != track_topics.end()) {
      monoscale::TrackFrame frame;
      if (parse_tracks(deserialize<std_msgs::msg::Float32MultiArray>(serialized), frame)) {
        estimator.ingest_tracks(track->second, frame);
      }
    } else if (message->topic_name == configuration.topics.imu) {
      const auto imu = deserialize<sensor_msgs::msg::Imu>(serialized);
      monoscale::ImuSample sample;
      sample.stamp = stamp_of(imu.header);
      sample.orientation = Eigen::Vector4d(
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w);
      sample.angular_velocity = Eigen::Vector3d(
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
      sample.linear_acceleration = Eigen::Vector3d(
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z);
      estimator.ingest_imu(sample);
    } else if (info_topics.count(message->topic_name) > 0) {
      const auto info = deserialize<sensor_msgs::msg::CameraInfo>(serialized);
      if (info.width > 0 && info.height > 0 && info.k[0] > 0.0) {
        Eigen::Matrix3d k;
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) {
            k(r, c) = info.k[static_cast<size_t>(3 * r + c)];
          }
        }
        Eigen::VectorXd d(static_cast<Eigen::Index>(info.d.size()));
        for (size_t n = 0; n < info.d.size(); ++n) {
          d(static_cast<Eigen::Index>(n)) = info.d[n];
        }
        estimator.set_calibration(
          info_topics[message->topic_name], k, d,
          monoscale::lens_from_name(info.distortion_model),
          static_cast<int>(info.width), static_cast<int>(info.height));
      }
      continue;
    } else if (message->topic_name == truth_topic) {
      const auto pose =
        deserialize<geometry_msgs::msg::PoseWithCovarianceStamped>(serialized);
      const double yaw = yaw_of(pose.pose.pose.orientation);
      // The truth reports the vehicle centre; the stack estimates the rear
      // axle, so it is shifted back along the heading.
      truth.push_back(
        Sample{
          stamp_of(pose.header),
          pose.pose.pose.position.x - 0.5 * wheel_base * std::cos(yaw),
          pose.pose.pose.position.y - 0.5 * wheel_base * std::sin(yaw), yaw});
      continue;
    } else {
      continue;
    }

    for (const auto & update : estimator.take_updates()) {
      if (!update.pose_valid) {
        continue;
      }
      estimates.push_back(
        Sample{update.stamp, update.pose.x, update.pose.y, update.pose.yaw});
    }
  }
  rclcpp::shutdown();

  TrajectoryMetrics metrics(segment);
  std::vector<std::pair<Sample, Sample>> rows;
  if (!estimates.empty() && !truth.empty()) {
    size_t index = 0;
    bool have_origin = false;
    Sample origin_truth{};
    Sample origin_estimate{};
    for (const auto & estimate : estimates) {
      while (index + 1 < truth.size() && truth[index + 1].stamp <= estimate.stamp) {
        ++index;
      }
      const size_t next = std::min(index + 1, truth.size() - 1);
      const size_t nearest =
        std::abs(truth[index].stamp - estimate.stamp) <=
        std::abs(truth[next].stamp - estimate.stamp) ? index : next;
      if (std::abs(truth[nearest].stamp - estimate.stamp) > tolerance) {
        continue;
      }
      if (!have_origin) {
        origin_truth = truth[nearest];
        origin_estimate = estimate;
        have_origin = true;
      }
      const Sample aligned =
        compose_pose(origin_estimate, relative_pose(origin_truth, truth[nearest]));
      metrics.add(estimate, aligned);
      rows.emplace_back(estimate, aligned);
      if (distance > 0.0 && metrics.truth_distance() >= distance) {
        break;
      }
    }
  }

  metrics.report(label.empty() ? bag : label);
  const auto & diagnostics = estimator.diagnostics();
  std::printf(
    "pairs=%ld solves=%ld estimates=%zu failures=%ld (nosolve=%ld trans=%ld yaw=%ld) "
    "coasted=%ld anchors=%d map_frames=%ld evicted=%ld\n",
    diagnostics.pairs_seen, diagnostics.frames_processed, estimates.size(),
    diagnostics.motion_failures, diagnostics.fail_no_solve, diagnostics.fail_translation,
    diagnostics.fail_yaw, diagnostics.coasted, diagnostics.anchors,
    diagnostics.map_aligned_frames, diagnostics.frames_evicted);
  std::string stages;
  for (const auto & [name, seconds] : diagnostics.stage_seconds) {
    if (seconds <= 0.0) {
      continue;
    }
    char buffer[64];
    std::snprintf(
      buffer, sizeof(buffer), "%s=%.2f ", name.c_str(),
      1000.0 * seconds / std::max<int64_t>(diagnostics.frames_processed, 1));
    stages += buffer;
  }
  std::printf("단계별 ms/solve: %s\n", stages.c_str());
  if (diagnostics.last_nis != 0.0 || diagnostics.gyro_bias != 0.0) {
    std::printf(
      "필터: 자이로바이어스=%+.5f rad/s  헤딩끌림=%+.5f rad/hop  마지막 NIS=%.3f  "
      "게이트기각=%ld  버림=%ld\n",
      diagnostics.gyro_bias, diagnostics.heading_drift, diagnostics.last_nis,
      diagnostics.filter_rejections, diagnostics.filter_dropped);
  }

  if (!tum_directory.empty() && !rows.empty()) {
    std::ofstream estimate_file(tum_directory + "/estimate.tum");
    std::ofstream truth_file(tum_directory + "/truth.tum");
    for (const auto & [estimate, aligned] : rows) {
      const auto line = [](std::ofstream & file, const Sample & pose) {
          const double yaw = wrap_pi(pose.yaw);
          file << std::fixed;
          file.precision(6);
          file << pose.stamp << " " << pose.x << " " << pose.y
               << " 0.000000 0.000000 0.000000 " << std::sin(0.5 * yaw) << " "
               << std::cos(0.5 * yaw) << "\n";
        };
      line(estimate_file, estimate);
      line(truth_file, aligned);
    }
  }
  return 0;
}
