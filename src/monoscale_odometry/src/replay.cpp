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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

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
bool parse_tracks(const std_msgs::msg::Float64MultiArray & message, monoscale::TrackFrame & out)
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
    // The parallax block: four values a cell, then the cell count, then the
    // marker. Read before the clarity block because it sits behind it.
    size_t tail = data.size();  // NOLINT
    if (tail > 2 && data[tail - 1] == -8.125e7) {
      const int cells = static_cast<int>(data[tail - 2]);
      const size_t need = static_cast<size_t>(cells) * 4;
      if (cells > 0 && tail >= need + 2 && tail - need - 2 >= static_cast<size_t>(4 + 5 * count)) {
        const size_t first = tail - need - 2;
        out.parallax.resize(cells, 4);
        for (int i = 0; i < cells; ++i) {
          for (int j = 0; j < 4; ++j) {
            out.parallax(i, j) = data[first + static_cast<size_t>(i) * 4 + j];
          }
        }
        tail = first;
      }
    }
  // The clarity block, self-describing, appended after everything else. See
  // the same parse in odometry_node.
  if (count > 0 && tail >= static_cast<size_t>(4 + 5 * count + count + 1) &&
    static_cast<int>(data[tail - 1]) == count)
  {
    const size_t first = tail - 1 - static_cast<size_t>(count);
    if (first >= static_cast<size_t>(4 + 5 * count)) {
      out.clarity.resize(count);
      for (int i = 0; i < count; ++i) {
        out.clarity(i) = data[first + static_cast<size_t>(i)];
      }
    }
  }
  const size_t after = static_cast<size_t>(4 + 5 * count);
  if (data.size() > after) {
    out.photometric_step = data[after];
    if (data.size() > after + 1) {
      out.photometric_score = data[after + 1];
    }
    if (data.size() > after + 2) {
      out.photometric_spread = data[after + 2];
    }
    if (data.size() > after + 12) {
      out.band_near = data[after + 3];
      out.band_far = data[after + 4];
      out.band_left = data[after + 7];
      out.band_right = data[after + 8];
      out.band_near_range = data[after + 9];
      out.band_far_range = data[after + 10];
      out.band_left_lateral = data[after + 11];
      out.band_right_lateral = data[after + 12];
    }
    if (data.size() > after + 14) {
      out.band_near_forward = data[after + 13];
      out.band_far_forward = data[after + 14];
    }
    if (data.size() > after + 17) {
      out.esm_yaw = data[after + 15];
      out.esm_pitch = data[after + 16];
      out.esm_roll = data[after + 17];
    }
  }
  return true;
}

// The bag, deserialized once.
//
// Reading and parsing the bag is the whole cost of a replay: scoring one metre
// of a 109 m drive takes the same 1.6 s as scoring all of it, because the bag
// is walked either way, while the estimator itself is 0.36 ms a pair. So the
// events are decoded once into these vectors and the estimator is run over
// them -- once normally, or once per line of `--configs`, which is what makes
// a parameter sweep affordable.
//
// Nothing here depends on the estimator's settings. The transforms that depend
// on the command line rather than on settings (`--gyro-yaw`, `--until`,
// `--no-imu`) are applied while filling this, because they describe the input
// stream and not the estimator.
struct Buffered
{
  enum class Kind { Track, Imu, Calibration, Tilt };
  struct Event
  {
    Kind kind;
    size_t index;
  };
  struct Calibration
  {
    size_t camera;
    Eigen::Matrix3d k;
    Eigen::VectorXd d;
    monoscale::Lens lens;
    int width;
    int height;
  };
  std::vector<Event> order;
  std::vector<std::pair<size_t, monoscale::TrackFrame>> tracks;
  std::vector<monoscale::ImuSample> imu;
  std::vector<Calibration> calibrations;
  std::vector<std::pair<double, double>> tilts;
  std::vector<Sample> truth;
};

// What one pass over the buffer produces.
struct Outcome
{
  std::vector<Sample> estimates;
  std::vector<monoscale::Update> hops;
  std::vector<monoscale::LabelledPoint> points;
  double claimed_position = 0.0;
  double claimed_yaw = 0.0;
};

void replay_into(
  monoscale::Estimator & estimator, const Buffered & buffered, Outcome & outcome,
  bool keep_points)
{
  for (const auto & event : buffered.order) {
    if (event.kind == Buffered::Kind::Calibration) {
      const auto & c = buffered.calibrations[event.index];
      estimator.set_calibration(c.camera, c.k, c.d, c.lens, c.width, c.height);
      continue;
    }
    if (event.kind == Buffered::Kind::Tilt) {
      estimator.override_tilt(
        buffered.tilts[event.index].first, buffered.tilts[event.index].second);
      continue;
    }
    if (event.kind == Buffered::Kind::Track) {
      const auto & entry = buffered.tracks[event.index];
      estimator.ingest_tracks(entry.first, entry.second);
    } else {
      estimator.ingest_imu(buffered.imu[event.index]);
    }
    for (const auto & update : estimator.take_updates()) {
      if (keep_points && !update.points.empty()) {
        outcome.points.insert(
          outcome.points.end(), update.points.begin(), update.points.end());
      }
      if (update.hops_valid && update.previous_stamp > 0.0) {
        outcome.hops.push_back(update);
      }
      if (!update.pose_valid) {
        continue;
      }
      outcome.estimates.push_back(
        Sample{update.stamp, update.pose.x, update.pose.y, update.pose.yaw});
      if (update.covariance_valid) {
        outcome.claimed_position = std::sqrt(
          std::max(update.pose_covariance(0, 0) + update.pose_covariance(1, 1), 0.0));
        outcome.claimed_yaw = std::sqrt(std::max(update.pose_covariance(2, 2), 0.0));
      }
    }
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string bag;
  std::string truth_topic = "/carla/ground_truth/pose";
  std::string label;
  std::string tum_directory;
  // One parameter set per line, each a whitespace separated list of
  // `name:=value`. The bag is decoded once and every line is run over that one
  // buffer, which is the whole point: the decode is 1.4 s and the estimator is
  // 0.05 s, so a sweep that re-reads the bag per configuration spends 96% of
  // itself on the same work. Writes `<tum>/<line>/estimate.tum` and
  // `truth.tum`, one directory per line, and prints the line's index and its
  // ate_rmse. `--distance` is not applied here; a sweep scores whole drives.
  std::string configs_path;
  // Score the whole drive. This was 34.0, which fitted the approach bags
  // (24-32 m) and silently truncated anything longer: a 110 m straight scored
  // its first 34 m and reported that as the drive. The cut is still available
  // as --distance for comparing against a shorter recording.
  double distance = 0.0;
  double segment = 1.0;
  double tolerance = 0.05;
  // Where the truth pose is reported, ahead of the point the stack estimates.
  //
  // Half the wheelbase is the obvious guess and it is not the right quantity:
  // CARLA reports the actor origin, which is the centre of the bounding box,
  // and the box centre of the vehicle this rig drives sits 1.399 m ahead of
  // the rear axle against a 2.8192 m wheelbase. Confirmed independently from
  // the truth alone -- regressing the reported point's lateral velocity on the
  // yaw rate locates the point that does not slide sideways, and on the
  // best-conditioned drives that lands at 1.29 to 1.35 m, biased forward by
  // whatever side-slip the manoeuvre has.
  //
  // Worth 0.3% of mean ATE here, which is inside the noise; it is applied
  // because it is the right number, not because it measured better.
  double truth_offset_m = 1.399;
  bool truth_tilt = false;
  // Do not feed the estimator any inertial sample at all. The point is not to
  // switch off one consumer but to ask what the cameras can do alone, and on
  // this simulator the instrument's orientation is the recorded truth, so
  // every score taken with it is a score no real vehicle would repeat.
  bool no_imu = false;
  // Replace the instrument's orientation with its own gyro, integrated from the
  // first sample. CARLA reports the recorded truth as the orientation -- the
  // yaw error against truth measures 0.000000 degrees -- so every score taken
  // with it is a score no vehicle with a real part would repeat. Integrated,
  // the same drives drift 0.006 to 0.29 degrees over a minute, which is what a
  // heading actually costs.
  bool gyro_yaw = false;
  std::string hops_path;
  std::string points_path;
  std::string revisits_path;
  std::string anchors_path;
  double until_seconds = 0.0;
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
    } else if (argument == "--configs") {
      configs_path = next();
    } else if (argument == "--tum") {
      tum_directory = next();
    } else if (argument == "--distance") {
      distance = std::stod(next());
    } else if (argument == "--segment") {
      segment = std::stod(next());
    } else if (argument == "--tolerance") {
      tolerance = std::stod(next());
    } else if (argument == "--truth-tilt") {
      truth_tilt = true;
    } else if (argument == "--no-imu") {
      no_imu = true;
    } else if (argument == "--gyro-yaw") {
      gyro_yaw = true;
    } else if (argument == "--points") {
      points_path = next();
    } else if (argument == "--hops") {
      hops_path = next();
    } else if (argument == "--anchors") {
      anchors_path = next();
    } else if (argument == "--revisits") {
      revisits_path = next();
    } else if (argument == "--until") {
      // Six of the eleven recorded drives end in a contact the simulator
      // reports as tens of g. Three of them destroy the measurement -- s8 goes
      // from 0.68% hop error to 39.20% -- so the benchmark stops before it
      // rather than scoring the wreck.
      until_seconds = std::stod(next());
    } else if (argument == "--truth-offset") {
      truth_offset_m = std::stod(next());
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

  Buffered buffered;
  Outcome outcome;
  // What the estimator claims to know about its own pose by the end, so the
  // claim can be laid beside the error it actually made. A covariance that
  // does not match is worse than none: it is a lie a consumer will act on.
  // Header time of the first IMU sample. `--until` is measured from there, so
  // it is on the same clock as everything the diagnostics report.
  double first_header_stamp = 0.0;
  bool stop_reading = false;
  while (reader.has_next() && !stop_reading) {
    const auto message = reader.read_next();
    rclcpp::SerializedMessage serialized(*message->serialized_data);

    const auto track = track_topics.find(message->topic_name);
    if (track != track_topics.end()) {
      monoscale::TrackFrame frame;
      if (parse_tracks(deserialize<std_msgs::msg::Float64MultiArray>(serialized), frame)) {
        buffered.order.push_back({Buffered::Kind::Track, buffered.tracks.size()});
        buffered.tracks.emplace_back(track->second, std::move(frame));
      }
    } else if (message->topic_name == configuration.topics.imu) {
      const auto imu = deserialize<sensor_msgs::msg::Imu>(serialized);
      const double header = stamp_of(imu.header);
      if (first_header_stamp == 0.0) {
        first_header_stamp = header;
      }
      if (until_seconds > 0.0 && header - first_header_stamp > until_seconds) {
        stop_reading = true;
        continue;
      }
      monoscale::ImuSample sample;
      sample.stamp = stamp_of(imu.header);
      sample.orientation = Eigen::Vector4d(
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w);
      sample.angular_velocity = Eigen::Vector3d(
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
      sample.linear_acceleration = Eigen::Vector3d(
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z);
      if (gyro_yaw) {
        // Seeded on the first sample and carried by the rate from there. Roll
        // and pitch are left as reported: this asks about the heading, and
        // changing two things at once would not answer it.
        static bool seeded = false;
        static double heading = 0.0;
        static double last = 0.0;
        const double w = 2.0 * std::atan2(sample.orientation.z(), sample.orientation.w());
        if (!seeded) {
          heading = w;
          last = sample.stamp;
          seeded = true;
        } else if (sample.stamp > last) {
          heading += sample.angular_velocity.z() * (sample.stamp - last);
          last = sample.stamp;
        }
        sample.orientation = Eigen::Vector4d(
          0.0, 0.0, std::sin(0.5 * heading), std::cos(0.5 * heading));
      }
      if (!no_imu) {
        buffered.order.push_back({Buffered::Kind::Imu, buffered.imu.size()});
        buffered.imu.push_back(sample);
      }
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
        buffered.order.push_back(
          {Buffered::Kind::Calibration, buffered.calibrations.size()});
        buffered.calibrations.push_back(
          Buffered::Calibration{
            info_topics[message->topic_name], k, d,
            monoscale::lens_from_name(info.distortion_model),
            static_cast<int>(info.width), static_cast<int>(info.height)});
      }
      continue;
    } else if (message->topic_name == truth_topic) {
      const auto pose =
        deserialize<geometry_msgs::msg::PoseWithCovarianceStamped>(serialized);
      const double yaw = yaw_of(pose.pose.pose.orientation);
      if (truth_tilt) {
        // A ceiling, not a mode: hand the estimator the attitude it is trying
        // to estimate and see what the rest of it is worth.
        const auto & q = pose.pose.pose.orientation;
        const double sr = 2.0 * (q.w * q.x + q.y * q.z);
        const double cr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        const double sp = std::clamp(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0);
        buffered.order.push_back({Buffered::Kind::Tilt, buffered.tilts.size()});
        buffered.tilts.emplace_back(std::atan2(sr, cr), std::asin(sp));
      }
      // The truth reports the box centre; the stack estimates the rear axle,
      // so it is shifted back along the heading.
      buffered.truth.push_back(
        Sample{
          stamp_of(pose.header),
          pose.pose.pose.position.x - truth_offset_m * std::cos(yaw),
          pose.pose.pose.position.y - truth_offset_m * std::sin(yaw), yaw});
      continue;
    } else {
      continue;
    }
  }

  if (!configs_path.empty()) {
    if (tum_directory.empty()) {
      std::cerr << "--configs 는 --tum <디렉터리> 를 함께 요구한다.\n";
      return 2;
    }
    std::ifstream list(configs_path);
    std::string line;
    int index = -1;
    while (std::getline(list, line)) {
      ++index;
      std::vector<std::string> arguments = ros_arguments;
      std::istringstream tokens(line);
      std::string token;
      while (tokens >> token) {
        arguments.push_back("-p");
        arguments.push_back(token);
      }
      rclcpp::NodeOptions per_config;
      per_config.arguments(arguments);
      auto config_node =
        std::make_shared<rclcpp::Node>("monoscale_replay_sweep", per_config);
      const auto settings = monoscale_ros::declare_and_read(*config_node);
      monoscale::Estimator one(settings.estimator);
      Outcome result;
      replay_into(one, buffered, result, false);

      // The same origin alignment the scoring below uses, without the metrics
      // it also accumulates: a sweep only needs the matched pairs.
      const std::string directory = tum_directory + "/" + std::to_string(index);
      std::filesystem::create_directories(directory);
      std::ofstream estimate_file(directory + "/estimate.tum");
      std::ofstream truth_file(directory + "/truth.tum");
      double square = 0.0;
      size_t matched = 0;
      size_t at = 0;
      bool have_origin = false;
      Sample origin_truth{};
      Sample origin_estimate{};
      for (const auto & estimate : result.estimates) {
        if (buffered.truth.empty()) {
          break;
        }
        while (at + 1 < buffered.truth.size() &&
          buffered.truth[at + 1].stamp <= estimate.stamp)
        {
          ++at;
        }
        const size_t next_index = std::min(at + 1, buffered.truth.size() - 1);
        const size_t nearest =
          std::abs(buffered.truth[at].stamp - estimate.stamp) <=
          std::abs(buffered.truth[next_index].stamp - estimate.stamp) ? at : next_index;
        if (std::abs(buffered.truth[nearest].stamp - estimate.stamp) > tolerance) {
          continue;
        }
        if (!have_origin) {
          origin_truth = buffered.truth[nearest];
          origin_estimate = estimate;
          have_origin = true;
        }
        const Sample aligned = compose_pose(
          origin_estimate, relative_pose(origin_truth, buffered.truth[nearest]));
        const double dx = estimate.x - aligned.x;
        const double dy = estimate.y - aligned.y;
        square += dx * dx + dy * dy;
        ++matched;
        const auto write = [](std::ofstream & file, const Sample & pose) {
            const double yaw = wrap_pi(pose.yaw);
            file << std::fixed;
            file.precision(6);
            file << pose.stamp << " " << pose.x << " " << pose.y
                 << " 0.000000 0.000000 0.000000 " << std::sin(0.5 * yaw) << " "
                 << std::cos(0.5 * yaw) << "\n";
          };
        write(estimate_file, estimate);
        write(truth_file, aligned);
      }
      std::printf(
        "%d ate_rmse=%.6f n=%zu\n", index,
        matched > 0 ? std::sqrt(square / static_cast<double>(matched)) : 0.0, matched);
      std::fflush(stdout);
    }
    rclcpp::shutdown();
    return 0;
  }

  replay_into(estimator, buffered, outcome, !points_path.empty());
  const std::vector<Sample> & estimates = outcome.estimates;
  const std::vector<monoscale::Update> & hops = outcome.hops;
  const std::vector<monoscale::LabelledPoint> & collected_points = outcome.points;
  const std::vector<Sample> & truth = buffered.truth;
  const double claimed_position = outcome.claimed_position;
  const double claimed_yaw = outcome.claimed_yaw;
  rclcpp::shutdown();

  // Each camera's hop against the hop the vehicle actually made. Nothing else
  // in the stack measures a camera against anything but the other camera.
  if (!hops.empty() && truth.size() > 2) {
    const auto at = [&truth](double stamp) {
        size_t i = 0;
        while (i + 2 < truth.size() && truth[i + 1].stamp < stamp) {
          ++i;
        }
        const double span = truth[i + 1].stamp - truth[i].stamp;
        const double t = span > 1e-9
          ? std::clamp((stamp - truth[i].stamp) / span, 0.0, 1.0) : 0.0;
        Sample out{};
        out.stamp = stamp;
        out.x = truth[i].x + t * (truth[i + 1].x - truth[i].x);
        out.y = truth[i].y + t * (truth[i + 1].y - truth[i].y);
        out.yaw = truth[i].yaw + t * wrap_pi(truth[i + 1].yaw - truth[i].yaw);
        return out;
      };
    const size_t cameras = hops.front().camera_hops.size();
    std::vector<double> sum(cameras + 1, 0.0);
    std::vector<double> square(cameras + 1, 0.0);
    std::vector<int64_t> seen(cameras + 1, 0);
    std::vector<double> cross(cameras, 0.0);
    std::vector<double> pair_a(cameras, 0.0);
    std::vector<double> pair_b(cameras, 0.0);
    int64_t paired = 0;
    double travelled = 0.0;
    double cross_sum = 0.0;
    double cross_square = 0.0;
    double world_along = 0.0;
    double world_along_y = 0.0;
    double world_across_x = 0.0;
    double world_across_y = 0.0;
    double world_error_x = 0.0;
    double world_error_y = 0.0;
    double applied_along = 0.0;
    int64_t applied_n = 0;
    int64_t coasted = 0;
    double lever_xy = 0.0;
    double lever_xx = 0.0;
    double rect_xy = 0.0;
    double rect_xx = 0.0;
    for (const auto & hop : hops) {
      if (hop.previous_stamp < truth.front().stamp || hop.stamp > truth.back().stamp) {
        continue;
      }
      const Sample before = at(hop.previous_stamp);
      const Sample after = at(hop.stamp);
      const Sample step = relative_pose(before, after);
      const double length = std::hypot(step.x, step.y);
      // Two centimetres, not epsilon. A hop of half a millimetre divides a
      // millimetre of error into a four-figure percentage, and the parking
      // drives are full of them.
      if (length < 0.02) {
        continue;
      }
      travelled += length;
      const double ux = step.x / length;
      const double uy = step.y / length;
      // Error along the direction the vehicle actually went, as a fraction of
      // how far it went. Cross track is left out on purpose: the scale is what
      // this estimator gets wrong.
      // The applied hop, not the one the cameras voted for.
      const Eigen::Vector2d moved = hop.applied_valid ? hop.applied_hop : hop.fused_hop;
      const auto along = [&](const Eigen::Vector2d & v) {
          return (v.x() * ux + v.y() * uy - length) / length;
        };
      // Sideways of where the vehicle actually went. On a straight this
      // averages out; on a weave it does not, because the path's direction
      // turns underneath it and the error is carried into the world frame
      // pointing somewhere new each time.
      const auto across = [&](const Eigen::Vector2d & v) {
          return (-v.x() * uy + v.y() * ux) / length;
        };
      std::vector<double> errors(cameras, std::numeric_limits<double>::quiet_NaN());
      for (size_t i = 0; i < cameras; ++i) {
        if (!hop.camera_hops[i].allFinite()) {
          continue;
        }
        errors[i] = along(hop.camera_hops[i]);
        sum[i] += errors[i];
        square[i] += errors[i] * errors[i];
        ++seen[i];
      }
      const double fused = along(moved);
      sum[cameras] += fused;
      square[cameras] += fused * fused;
      ++seen[cameras];
      const double sideways = across(moved);
      cross_sum += sideways;
      cross_square += sideways * sideways;
      // Cross error in metres against the signed turn. The slope is a length:
      // it is the lever arm of whatever point the estimator is really tracking
      // against the one the truth is quoted at, because a point offset by e
      // from the rear axle carries a sideslip of e*omega/v.
      lever_xy += (sideways * length) * step.yaw;
      lever_xx += step.yaw * step.yaw;
      // And against the unsigned turn, which a sideslip cannot produce.
      const double turn = std::abs(step.yaw);
      rect_xy += (sideways * length) * turn;
      rect_xx += turn * turn;
      // What the error actually costs the pose. The hop error is a vector in
      // the earlier body frame; carrying it into the world needs that frame's
      // heading, and only then do the along and across parts stop being about
      // this hop and start being about the map.
      if (hop.coasted) {
        ++coasted;
      }
      applied_along += (moved.x() * ux + moved.y() * uy - length) / length;
      ++applied_n;
      const double ex = moved.x() - step.x;
      const double ey = moved.y() - step.y;
      const double heading = before.yaw;
      const double cw = std::cos(heading);
      const double sw = std::sin(heading);
      world_error_x += cw * ex - sw * ey;
      world_error_y += sw * ex + cw * ey;
      // Split the same vector by what it was in the body frame, so the two
      // contributions can be compared as world displacements.
      const double along_x = fused * length * ux;
      const double along_y = fused * length * uy;
      world_along += cw * along_x - sw * along_y;
      world_along_y += sw * along_x + cw * along_y;
      const double across_x = -sideways * length * uy;
      const double across_y = sideways * length * ux;
      world_across_x += cw * across_x - sw * across_y;
      world_across_y += sw * across_x + cw * across_y;
      if (cameras >= 2 && std::isfinite(errors[0]) && std::isfinite(errors[1])) {
        cross[0] += errors[0] * errors[1];
        pair_a[0] += errors[0] * errors[0];
        pair_b[0] += errors[1] * errors[1];
        ++paired;
      }
    }
    // The same hops split by whether the vehicle was turning. The drives that
    // turn read long and the drives that do not read short, so the two want
    // separating before anything is concluded about scale.
    {
      double straight_sum = 0.0;
      double turning_sum = 0.0;
      int64_t straight_n = 0;
      int64_t turning_n = 0;
      double turning_rate = 0.0;
      for (const auto & hop : hops) {
        if (hop.previous_stamp < truth.front().stamp || hop.stamp > truth.back().stamp) {
          continue;
        }
        const Sample step = relative_pose(at(hop.previous_stamp), at(hop.stamp));
        const double length = std::hypot(step.x, step.y);
        if (length < 0.02) {
          continue;
        }
        const double error =
          (hop.fused_hop.x() * step.x / length + hop.fused_hop.y() * step.y / length -
          length) / length;
        // Radians per metre, so the split is about the path's curvature rather
        // than about how fast the drive happened to be going.
        const double curvature = std::abs(step.yaw) / length;
        if (curvature < 0.01) {
          straight_sum += error;
          ++straight_n;
        } else {
          turning_sum += error;
          turning_rate += curvature;
          ++turning_n;
        }
      }
      if (straight_n > 0 || turning_n > 0) {
        std::printf(
          "  곡률별 융합 편향: 직진 %+.2f%% (n=%ld)   선회 %+.2f%% (n=%ld, 평균 %.3f rad/m)\n",
          straight_n > 0 ? 100.0 * straight_sum / straight_n : 0.0, straight_n,
          turning_n > 0 ? 100.0 * turning_sum / turning_n : 0.0, turning_n,
          turning_n > 0 ? turning_rate / turning_n : 0.0);
      }
      // The shape of it, and whether the turn's direction matters. A lever arm
      // would flip sign with the turn; a rectification would not.
      constexpr int kBands = 6;
      const double edges[kBands] = {0.0, 0.02, 0.05, 0.10, 0.20, 0.40};
      double band_sum[kBands] = {};
      double band_yaw[kBands] = {};
      int64_t band_n[kBands] = {};
      double left_sum = 0.0;
      double right_sum = 0.0;
      double left_yaw = 0.0;
      double right_yaw = 0.0;
      int64_t left_n = 0;
      int64_t right_n = 0;
      std::vector<double> per_camera_sum(4, 0.0);
      std::vector<int64_t> per_camera_n(4, 0);
      for (const auto & hop : hops) {
        if (hop.previous_stamp < truth.front().stamp || hop.stamp > truth.back().stamp) {
          continue;
        }
        const Sample step = relative_pose(at(hop.previous_stamp), at(hop.stamp));
        const double length = std::hypot(step.x, step.y);
        if (length < 0.02) {
          continue;
        }
        const double ux = step.x / length;
        const double uy = step.y / length;
        const double error =
          (hop.fused_hop.x() * ux + hop.fused_hop.y() * uy - length) / length;
        const double curvature = std::abs(step.yaw) / length;
        int band = 0;
        for (int b = kBands - 1; b >= 0; --b) {
          if (curvature >= edges[b]) {
            band = b;
            break;
          }
        }
        band_sum[band] += error;
        band_yaw[band] += curvature;
        ++band_n[band];
        if (curvature >= 0.05) {
          if (step.yaw > 0.0) {
            left_sum += error;
            left_yaw += curvature;
            ++left_n;
          } else {
            right_sum += error;
            right_yaw += curvature;
            ++right_n;
          }
          for (size_t c = 0; c < hop.camera_hops.size() && c < 2; ++c) {
            if (hop.camera_hops[c].allFinite()) {
              per_camera_sum[c] +=
                (hop.camera_hops[c].x() * ux + hop.camera_hops[c].y() * uy - length) / length;
              ++per_camera_n[c];
            }
          }
        }
      }
      if (!anchors_path.empty()) {
    std::vector<std::array<double, 7>> polar;
    estimator.anchor_polar(polar);
    std::ofstream out(anchors_path);
    out << "range,bearing,weight,unseen\n";
    for (const auto & a : polar) {
      out << a[0] << "," << a[1] << "," << a[2] << "," << a[3] << "\n";
    }
  }
  if (!revisits_path.empty()) {
    std::ofstream out(revisits_path);
    out << "t0,t1,edx,edy,odx,ody,w\n";
    for (const auto & r : estimator.revisit_audit()) {
      out << r.time_from << "," << r.time_to << "," << r.edge_dx << "," << r.edge_dy
          << "," << r.odometry_dx << "," << r.odometry_dy << "," << r.weight << "\n";
    }
  }
  if (!points_path.empty() && !collected_points.empty()) {
    std::FILE * f = std::fopen(points_path.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "x,y,z,label,origin_x,origin_y\n");
      for (const auto & point : collected_points) {
        std::fprintf(
          f, "%.4f,%.4f,%.4f,%.0f,%.4f,%.4f\n",
          point.x, point.y, point.z, point.label, point.origin_x, point.origin_y);
      }
      std::fclose(f);
      std::printf("점 %zu 개 -> %s\n", collected_points.size(), points_path.c_str());
    }
  }
  if (!hops_path.empty()) {
        // One row per solve, so the bias can be regressed against candidate
        // drivers instead of binned by the one the correction already assumes.
        std::FILE * f = std::fopen(hops_path.c_str(), "w");
        if (f != nullptr) {
          std::fprintf(f, "t0,t1,dt,length,dyaw,curvature,bias,lat,front,rear,fmap,rmap,fwd,yawsign,fcond,fweak,rcond,rweak,fhx,fhy,fpx,fpy,rhx,rhy,rpx,rpy,frng,fn,rrng,rn,pdist,flen,fh,fp,rh,rp,roll,pitch,byaw,broll,bpitch,btx,bty\n");
          for (const auto & hop : hops) {
            if (hop.previous_stamp < truth.front().stamp || hop.stamp > truth.back().stamp) {
              continue;
            }
            const Sample step = relative_pose(at(hop.previous_stamp), at(hop.stamp));
            const double length = std::hypot(step.x, step.y);
            if (length < 0.02) {
              continue;
            }
            const double ux = step.x / length;
            const double uy = step.y / length;
            const auto along = [&](const Eigen::Vector2d & v) {
                return (v.x() * ux + v.y() * uy - length) / length;
              };
            // Across the direction of travel, same normalisation. Roll tilts
            // the plane sideways, so if it reaches the hop at all this is where
            // it lands -- straight-ahead points barely move under roll and the
            // left and right ones move opposite ways.
            const auto across = [&](const Eigen::Vector2d & v) {
                return (v.x() * -uy + v.y() * ux) / length;
              };
            const auto camera = [&](size_t c) {
                return c < hop.camera_hops.size() && hop.camera_hops[c].allFinite()
                  ? along(hop.camera_hops[c]) : std::numeric_limits<double>::quiet_NaN();
              };
            const auto from_map = [&](size_t c) {
                return c < hop.camera_from_map.size() ? int(hop.camera_from_map[c]) : -1;
              };
            const auto gain = [](const std::vector<Eigen::Vector2d> & v, size_t c, int a) {
                return c < v.size() ? v[c](a) : std::numeric_limits<double>::quiet_NaN();
              };
            const auto pick = [](const std::vector<double> & v, size_t c) {
                return c < v.size() ? v[c] : std::numeric_limits<double>::quiet_NaN();
              };
            std::fprintf(
              f, "%.6f,%.6f,%.4f,%.5f,%.6f,%.5f,%.6f,%.6f,%.6f,%.6f,%d,%d,%.5f,%+d,"
              "%.4f,%.5f,%.4f,%.5f,"
              "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
              "%.4f,%.0f,%.4f,%.0f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
              hop.previous_stamp, hop.stamp, hop.stamp - hop.previous_stamp, length, std::abs(step.yaw),
              std::abs(step.yaw) / length, along(hop.fused_hop), across(hop.fused_hop),
              camera(0), camera(1),
              from_map(0), from_map(1), step.x, step.yaw >= 0.0 ? 1 : -1,
              pick(hop.camera_condition, 0), pick(hop.camera_weak_bearing, 0),
              pick(hop.camera_condition, 1), pick(hop.camera_weak_bearing, 1),
              gain(hop.camera_height_gain, 0, 0), gain(hop.camera_height_gain, 0, 1),
              gain(hop.camera_pitch_gain, 0, 0), gain(hop.camera_pitch_gain, 0, 1),
              gain(hop.camera_height_gain, 1, 0), gain(hop.camera_height_gain, 1, 1),
              gain(hop.camera_pitch_gain, 1, 0), gain(hop.camera_pitch_gain, 1, 1),
              pick(hop.camera_mean_range, 0), pick(hop.camera_point_count, 0),
              pick(hop.camera_mean_range, 1), pick(hop.camera_point_count, 1),
              hop.photometric_distance, hop.fused_length,
              pick(hop.radial_height, 0), pick(hop.radial_pitch, 0),
              pick(hop.radial_height, 1), pick(hop.radial_pitch, 1),
              hop.roll, hop.pitch, hop.bearing_yaw,
              hop.bearing_roll_raw, hop.bearing_pitch_raw,
              hop.bearing_tx, hop.bearing_ty);
          }
          std::fclose(f);
        }
      }
      std::printf("  곡률 구간별 편향:");
      for (int b = 0; b < kBands; ++b) {
        if (band_n[b] < 10) {
          continue;
        }
        std::printf(
          "  %.3f:%+.2f%%(%ld)", band_yaw[b] / band_n[b],
          100.0 * band_sum[b] / band_n[b], band_n[b]);
      }
      std::printf("\n");
      if (left_n >= 10 || right_n >= 10) {
        std::printf(
          "  선회 방향별: 좌 %+.2f%% (n=%ld, %.3f)  우 %+.2f%% (n=%ld, %.3f)"
          "  | 선회구간 카메라 front %+.2f%% rear %+.2f%%\n",
          left_n > 0 ? 100.0 * left_sum / left_n : 0.0, left_n,
          left_n > 0 ? left_yaw / left_n : 0.0,
          right_n > 0 ? 100.0 * right_sum / right_n : 0.0, right_n,
          right_n > 0 ? right_yaw / right_n : 0.0,
          per_camera_n[0] > 0 ? 100.0 * per_camera_sum[0] / per_camera_n[0] : 0.0,
          per_camera_n[1] > 0 ? 100.0 * per_camera_sum[1] / per_camera_n[1] : 0.0);
      }
    }
    std::printf("hop 대 진값 (진행 방향, 홉 길이 대비):\n");
    for (size_t i = 0; i <= cameras; ++i) {
      if (seen[i] == 0) {
        continue;
      }
      const double n = static_cast<double>(seen[i]);
      const double mean = sum[i] / n;
      const double rms = std::sqrt(square[i] / n);
      const double noise = std::sqrt(std::max(rms * rms - mean * mean, 0.0));
      std::printf(
        "  %-7s n=%5ld  bias=%+7.2f%%  noise=%6.2f%%  rms=%6.2f%%\n",
        i == cameras ? "fused" : (i == 0 ? "front" : "rear"), seen[i],
        100.0 * mean, 100.0 * noise, 100.0 * rms);
    }
    // The same numbers per ten metres of true travel. The whole-drive figure
    // says the two cameras end up mirroring each other; this says where in the
    // drive they start to, which is the only place a cause can be.
    if (cameras >= 2) {
      struct Bin
      {
        double travel = 0.0;
        double a = 0.0;
        double b = 0.0;
        double aa = 0.0;
        double bb = 0.0;
        double ab = 0.0;
        int64_t map0 = 0;
        int64_t map1 = 0;
        int64_t n = 0;
      };
      std::vector<Bin> bins;
      struct SplitRow
      {
        double split;
        double speed;
        double turn_rate;
        double signed_rate;
      };
      std::vector<SplitRow> split_rows;
      struct CommonRow
      {
        double symmetric;    // half the sum of the two cameras' errors
        double fused;        // what the fusion actually produced
        double speed;
        double turn_rate;
        double hop;
      };
      std::vector<CommonRow> common_rows;
      double run = 0.0;
      for (const auto & hop : hops) {
        if (hop.previous_stamp < truth.front().stamp || hop.stamp > truth.back().stamp) {
          continue;
        }
        const Sample step = relative_pose(at(hop.previous_stamp), at(hop.stamp));
        const double length = std::hypot(step.x, step.y);
        if (length < 0.02 || !hop.camera_hops[0].allFinite() ||
          !hop.camera_hops[1].allFinite())
        {
          continue;
        }
        const double ux = step.x / length;
        const double uy = step.y / length;
        const double e0 =
          (hop.camera_hops[0].x() * ux + hop.camera_hops[0].y() * uy - length) / length;
        const double e1 =
          (hop.camera_hops[1].x() * ux + hop.camera_hops[1].y() * uy - length) / length;
        if (bins.empty() || run >= 10.0) {
          bins.emplace_back();
          run = 0.0;
        }
        run += length;
        Bin & bin = bins.back();
        bin.travel += length;
        bin.a += e0;
        bin.b += e1;
        bin.aa += e0 * e0;
        bin.bb += e1 * e1;
        bin.ab += e0 * e1;
        if (hop.camera_from_map.size() >= 2) {
          bin.map0 += hop.camera_from_map[0];
          bin.map1 += hop.camera_from_map[1];
        }
        ++bin.n;
        // What the split is, and what might explain it.
        //
        // A pitch error accounts for 67% of it across the eleven drives, so a
        // third is something else -- and the two drives it fails on are the
        // fastest and the sharpest turn, which suggests two mechanisms rather
        // than one residue. A camera sync offset costs speed times the offset,
        // so it scales with speed; anything about the lever arms or the
        // rotation scales with yaw rate.
        const double dt = hop.stamp - hop.previous_stamp;
        if (dt > 1e-4) {
          split_rows.push_back(
            {0.5 * (e0 - e1), length / dt, std::abs(step.yaw) / dt, step.yaw / dt});
          // And the part that actually survives the fusion.
          //
          // The split is the antisymmetric half and averaging the two cameras
          // removes it by construction, which is why correcting it never moved
          // the fused hop. What propagates is the symmetric half -- both
          // cameras wrong the same way -- and until now nothing measured it.
          const Eigen::Vector2d fused =
            hop.applied_valid ? hop.applied_hop : hop.fused_hop;
          const double fused_error =
            (fused.x() * ux + fused.y() * uy - length) / length;
          common_rows.push_back(
            {0.5 * (e0 + e1), fused_error, length / dt, std::abs(step.yaw) / dt, length});
        }
      }
      if (!split_rows.empty()) {
        // Least squares of the split on each candidate alone, so the two can be
        // told apart by which one carries the variance on which drive.
        const auto fit = [&split_rows](int which) {
            double sxx = 0.0;
            double sxy = 0.0;
            double sx = 0.0;
            double sy = 0.0;
            const double n = static_cast<double>(split_rows.size());
            for (const auto & r : split_rows) {
              const double x = which == 0 ? r.speed : (which == 1 ? r.turn_rate : r.signed_rate);
              sx += x;
              sy += r.split;
              sxx += x * x;
              sxy += x * r.split;
            }
            const double denominator = n * sxx - sx * sx;
            if (std::abs(denominator) < 1e-12) {
              return std::pair<double, double>{0.0, 0.0};
            }
            const double slope = (n * sxy - sx * sy) / denominator;
            const double intercept = (sy - slope * sx) / n;
            double residual = 0.0;
            double total = 0.0;
            const double mean = sy / n;
            for (const auto & r : split_rows) {
              const double x = which == 0 ? r.speed : (which == 1 ? r.turn_rate : r.signed_rate);
              const double predicted = slope * x + intercept;
              residual += (r.split - predicted) * (r.split - predicted);
              total += (r.split - mean) * (r.split - mean);
            }
            return std::pair<double, double>{
              slope, total > 1e-18 ? 1.0 - residual / total : 0.0};
          };
        double mean_split = 0.0;
        for (const auto & r : split_rows) {
          mean_split += r.split;
        }
        mean_split /= static_cast<double>(split_rows.size());
        const auto by_speed = fit(0);
        const auto by_rate = fit(1);
        const auto by_signed = fit(2);
        std::printf(
          "  split 평균 %+.2f%%  n=%zu | 속도 기울기 %+.3f%%/(m/s) R2=%.3f"
          " | |요율| %+.3f%%/(rad/s) R2=%.3f | 부호요율 %+.3f R2=%.3f\n",
          100.0 * mean_split, split_rows.size(),
          100.0 * by_speed.first, by_speed.second,
          100.0 * by_rate.first, by_rate.second,
          100.0 * by_signed.first, by_signed.second);
      }
      if (!common_rows.empty()) {
        const auto stat = [](const std::vector<double> & v) {
            double mean = 0.0;
            for (double x : v) {
              mean += x;
            }
            mean /= static_cast<double>(v.size());
            double var = 0.0;
            for (double x : v) {
              var += (x - mean) * (x - mean);
            }
            return std::pair<double, double>{
              mean, std::sqrt(var / static_cast<double>(v.size()))};
          };
        std::vector<double> sym;
        std::vector<double> fus;
        for (const auto & r : common_rows) {
          sym.push_back(r.symmetric);
          fus.push_back(r.fused);
        }
        const auto s = stat(sym);
        const auto f = stat(fus);
        std::printf(
          "  대칭성분(누적되는 것): 편향 %+.3f%% 잡음 %.3f%% | 융합결과: 편향 %+.3f%%"
          " 잡음 %.3f%%  n=%zu\n",
          100.0 * s.first, 100.0 * s.second, 100.0 * f.first, 100.0 * f.second,
          common_rows.size());
      }
      if (bins.size() >= 3) {
        std::printf("  10 m 구간별 front편향/rear편향 | 맵사용률 front/rear:\n   ");
        for (const auto & bin : bins) {
          if (bin.n < 5) {
            continue;
          }
          const double n = static_cast<double>(bin.n);
          const double ma = bin.a / n;
          const double mb = bin.b / n;
          const double va = bin.aa / n - ma * ma;
          const double vb = bin.bb / n - mb * mb;
          const double cov = bin.ab / n - ma * mb;
          (void)va;
          (void)vb;
          (void)cov;
          std::printf(
            " %+.0f/%+.0f|%.0f/%.0f", 100.0 * ma, 100.0 * mb,
            100.0 * bin.map0 / n, 100.0 * bin.map1 / n);
        }
        std::printf("\n");
      }
    }
    if (seen[cameras] > 0) {
      const double n = static_cast<double>(seen[cameras]);
      const double mean = cross_sum / n;
      const double rms = std::sqrt(cross_square / n);
      std::printf(
        "  fused 횡: bias=%+.2f%% noise=%.2f%%  | 누적 %.3f m (진행분 %.3f, 횡분 %.3f)"
        "  coasted %ld/%ld\n",
        100.0 * mean, 100.0 * std::sqrt(std::max(rms * rms - mean * mean, 0.0)),
        std::hypot(world_error_x, world_error_y),
        std::hypot(world_along, world_along_y),
        std::hypot(world_across_x, world_across_y), coasted, applied_n);
      std::printf(
        "  횡오차 회귀: 부호있는 요 -> %+.4f m (레버암),  |요| -> %+.4f m\n",
        lever_xx > 1e-12 ? lever_xy / lever_xx : 0.0,
        rect_xx > 1e-12 ? rect_xy / rect_xx : 0.0);
    }
    if (paired > 0 && pair_a[0] > 0.0 && pair_b[0] > 0.0) {
      std::printf(
        "  두 카메라 오차 상관 r=%+.3f (n=%ld), 평균 홉 %.3f m\n",
        cross[0] / std::sqrt(pair_a[0] * pair_b[0]), paired,
        travelled / std::max<int64_t>(seen[cameras], 1));
    }
  }

  TrajectoryMetrics metrics(segment);
  double along_error = 0.0;
  double along_square = 0.0;
  double cross_error = 0.0;
  double cross_square = 0.0;
  int64_t error_count = 0;
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
      // The position error split along the direction the vehicle is pointing
      // and across it. ate_rmse is the two in quadrature, so this says which of
      // them it is made of.
      {
        const double ex = estimate.x - aligned.x;
        const double ey = estimate.y - aligned.y;
        const double ch = std::cos(aligned.yaw);
        const double sh = std::sin(aligned.yaw);
        const double a = ex * ch + ey * sh;
        const double c = -ex * sh + ey * ch;
        along_error += a;
        along_square += a * a;
        cross_error += c;
        cross_square += c * c;
        ++error_count;
      }
      rows.emplace_back(estimate, aligned);
      if (distance > 0.0 && metrics.truth_distance() >= distance) {
        break;
      }
    }
  }

  metrics.report(label.empty() ? bag : label);
  if (error_count > 0) {
    const double n = static_cast<double>(error_count);
    const double am = along_error / n;
    const double cm = cross_error / n;
    const double ar = std::sqrt(along_square / n);
    const double cr = std::sqrt(cross_square / n);
    std::printf(
      "위치오차 분해: 종 rms=%.4f sd=%.4f mean=%+.4f | 횡 rms=%.4f sd=%.4f mean=%+.4f"
      "  (합성 %.4f, n=%ld)\n",
      ar, std::sqrt(std::max(ar * ar - am * am, 0.0)), am,
      cr, std::sqrt(std::max(cr * cr - cm * cm, 0.0)), cm,
      std::hypot(ar, cr), error_count);
  }
  {
    const auto & d = estimator.diagnostics();
    if (d.nis_samples > 0) {
      std::printf(
        "필터 NIS 평균 %.3f (정직하면 2.0)   갱신 %ld회\n",
        d.nis_total / static_cast<double>(d.nis_samples), d.nis_samples);
    }
  }
  if (claimed_position > 0.0) {
    std::printf(
      "주장 불확실성: 위치 1시그마 %.4f m   헤딩 1시그마 %.3f deg\n",
      claimed_position, claimed_yaw * 180.0 / M_PI);
  }
  const auto & diagnostics = estimator.diagnostics();
  std::printf(
    "pairs=%ld solves=%ld estimates=%zu failures=%ld (nosolve=%ld trans=%ld yaw=%ld) "
    "sector[0-30 30-60 60-90 90-120 120-150 150-180] "
    "n=%ld/%ld/%ld/%ld/%ld/%ld w=%.0f/%.0f/%.0f/%.0f/%.0f/%.0f "
    "r=%.1f/%.1f/%.1f/%.1f/%.1f/%.1f\n"
    "seen[1 2 3-4 5-8 9-16 17+]=%ld/%ld/%ld/%ld/%ld/%ld road_anchors=%ld usable=%ld parallax_pts=%ld scatter[<1e-6 1e-5 1e-4 1e-3 1e-2 +]=%ld/%ld/%ld/%ld/%ld/%ld\n"
    "coasted=%ld yaw_misses=%ld anchors=%d(도달 %d, 종평균 %+.1fm 종퍼짐 %.1fm) "
    "map_frames=%ld evicted=%ld "
    "link[n=%ld gap=%.4fm range=%.2fm gap/range=%.5f]\n",
    diagnostics.pairs_seen, diagnostics.frames_processed, estimates.size(),
    diagnostics.motion_failures, diagnostics.fail_no_solve, diagnostics.fail_translation,
    diagnostics.fail_yaw,
    diagnostics.anchor_sector_count[0], diagnostics.anchor_sector_count[1],
    diagnostics.anchor_sector_count[2], diagnostics.anchor_sector_count[3],
    diagnostics.anchor_sector_count[4], diagnostics.anchor_sector_count[5],
    diagnostics.anchor_sector_weight[0], diagnostics.anchor_sector_weight[1],
    diagnostics.anchor_sector_weight[2], diagnostics.anchor_sector_weight[3],
    diagnostics.anchor_sector_weight[4], diagnostics.anchor_sector_weight[5],
    diagnostics.anchor_sector_count[0] > 0
      ? diagnostics.anchor_sector_range[0] / diagnostics.anchor_sector_count[0] : 0.0,
    diagnostics.anchor_sector_count[1] > 0
      ? diagnostics.anchor_sector_range[1] / diagnostics.anchor_sector_count[1] : 0.0,
    diagnostics.anchor_sector_count[2] > 0
      ? diagnostics.anchor_sector_range[2] / diagnostics.anchor_sector_count[2] : 0.0,
    diagnostics.anchor_sector_count[3] > 0
      ? diagnostics.anchor_sector_range[3] / diagnostics.anchor_sector_count[3] : 0.0,
    diagnostics.anchor_sector_count[4] > 0
      ? diagnostics.anchor_sector_range[4] / diagnostics.anchor_sector_count[4] : 0.0,
    diagnostics.anchor_sector_count[5] > 0
      ? diagnostics.anchor_sector_range[5] / diagnostics.anchor_sector_count[5] : 0.0,
    diagnostics.anchor_sightings[0], diagnostics.anchor_sightings[1],
    diagnostics.anchor_sightings[2], diagnostics.anchor_sightings[3],
    diagnostics.anchor_sightings[4], diagnostics.anchor_sightings[5],
    diagnostics.anchor_road, diagnostics.anchor_usable, diagnostics.parallax_points,
    diagnostics.anchor_scatter[0], diagnostics.anchor_scatter[1],
    diagnostics.anchor_scatter[2], diagnostics.anchor_scatter[3],
    diagnostics.anchor_scatter[4], diagnostics.anchor_scatter[5],
    diagnostics.coasted, diagnostics.imu_yaw_misses, diagnostics.anchors,
    diagnostics.anchors_within, diagnostics.anchors_along, diagnostics.anchors_across,
    diagnostics.map_aligned_frames, diagnostics.frames_evicted,
    diagnostics.anchors_adopted, diagnostics.link_gap_m, diagnostics.link_range_m,
    diagnostics.link_gap_per_m);
  if (diagnostics.camera_travel.size() >= 2) {
    const double a = diagnostics.camera_travel[0];
    const double b = diagnostics.camera_travel[1];
    std::printf(
      "camera travel: %.2f m vs %.2f m  ratio %.5f\n", a, b, b > 1e-9 ? a / b : 0.0);
  }
  for (size_t i = 0; i < diagnostics.camera_solves.size(); ++i) {
    const double n = static_cast<double>(std::max<int64_t>(diagnostics.camera_solves[i], 1));
    std::printf(
      "  camera[%zu]: solves=%ld anchored=%ld travel=%.2fm inliers=%.0f spread=%.4f"
      "  usable=%.0f known=%.0f (%.0f%%) bearing=%.1fdeg scale=%.4f\n",
      i, diagnostics.camera_solves[i], diagnostics.camera_anchored[i],
      diagnostics.camera_travel[i], diagnostics.camera_inliers[i] / n,
      diagnostics.camera_spread[i] / n,
      i < diagnostics.camera_usable.size() && diagnostics.camera_looks[i] > 0
      ? diagnostics.camera_usable[i] / diagnostics.camera_looks[i] : 0.0,
      i < diagnostics.camera_known.size() && diagnostics.camera_looks[i] > 0
      ? diagnostics.camera_known[i] / diagnostics.camera_looks[i] : 0.0,
      i < diagnostics.camera_usable.size() && diagnostics.camera_usable[i] > 0
      ? 100.0 * diagnostics.camera_known[i] / diagnostics.camera_usable[i] : 0.0,
      i < diagnostics.camera_bearing.size() ? diagnostics.camera_bearing[i] : -1.0,
      i < diagnostics.camera_scale.size() ? diagnostics.camera_scale[i] : -1.0);
  }
  if (diagnostics.travel_bins.size() >= 3) {
    // A split that is a calibration reads the same in every stretch; one that
    // accumulates does not. Measured: str_v3 holds 0.98 the whole way while
    // str_v2 runs 1.09 to 1.95, so this one is a divergence, not a mount.
    std::printf("  10 m 구간별 카메라 비: ");
    const size_t group =
      (diagnostics.travel_bins.size() + 11) / 12;
    double front = 0.0;
    double rear = 0.0;
    for (size_t i = 0; i < diagnostics.travel_bins.size(); ++i) {
      front += diagnostics.travel_bins[i](1);
      rear += diagnostics.travel_bins[i](2);
      if ((i + 1) % group == 0 || i + 1 == diagnostics.travel_bins.size()) {
        std::printf("%.3f ", rear > 1e-9 ? front / rear : -1.0);
        front = 0.0;
        rear = 0.0;
      }
    }
    std::printf("\n");
  }
  if (diagnostics.crossings > 0) {
    std::printf(
      "카메라 교차: n=%ld  along=%+.4f m  baseline=%.2f m  기울기->scale %.4f  절편=%+.4f m\n",
      diagnostics.crossings, diagnostics.crossing_along_m,
      diagnostics.crossing_travel_m, diagnostics.crossing_scale,
      diagnostics.crossing_offset_m);
  }
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
  if (diagnostics.remembered_sightings > 0) {
    const double slots = static_cast<double>(diagnostics.anchors);
    const double bytes = slots * 16.0 * (4.0 + 8.0 + 4.0) + slots * 4.0;
    std::printf(
      "관측별 포즈: %ld개 기억, 앵커 %ld, 포즈 이력 %ld, 저장 %.1f MB, 재구성 %.2f ms\n",
      diagnostics.remembered_sightings, diagnostics.anchors,
      diagnostics.pose_history, bytes / 1048576.0, diagnostics.rebuild_ms);
    std::printf("  재구성이 앵커를 옮기는 거리 평균 %.4f m, 관측 포즈 폭 %.1f\n",
      diagnostics.rebuild_shift_m, diagnostics.sighting_span);
    std::printf("  광도 step: 적용 %ld / 기회 %ld, 맵없음 %ld\n",
      diagnostics.photometric_uses, diagnostics.photometric_chances,
      diagnostics.photometric_mapless);
    std::printf("  포즈그래프: 재방문 구속 %ld개, 포즈 평균 이동 %.4f m\n",
      diagnostics.pose_graph_loops, diagnostics.pose_graph_shift_m);
  }
  for (size_t i = 0; i < diagnostics.pair_radial_samples.size(); ++i) {
    if (diagnostics.pair_radial_samples[i] == 0) {
      continue;
    }
    std::printf(
      "2프레임 dh/h[%zu]: %+.5f  n=%ld\n",
      i, diagnostics.pair_radial[i], diagnostics.pair_radial_samples[i]);
  }
  for (size_t i = 0; i < diagnostics.radial_samples.size(); ++i) {
    if (diagnostics.radial_samples[i] == 0) {
      continue;
    }
    std::printf(
      "지면 기울기[%zu]: %+.5f  n=%ld\n",
      i, diagnostics.radial_linear[i], diagnostics.radial_samples[i]);
  }
  if (diagnostics.levelled > 0) {
    std::printf(
      "자세: roll=%+.3fdeg pitch=%+.3fdeg 높이=%+.3fm 수평보정=%ld 지면스케일=%.5f\n",
      diagnostics.roll * 180.0 / M_PI, diagnostics.pitch * 180.0 / M_PI,
      diagnostics.height, diagnostics.levelled, diagnostics.range_scale);
    std::printf("hop 잔차: 갱신직후=%.4f 최종=%.4f m\n", diagnostics.hop_taken, diagnostics.hop_residual);
  }
  if (diagnostics.last_nis != 0.0 || diagnostics.gyro_bias != 0.0 ||
    diagnostics.heading_updates != 0)
  {
    std::printf(
      "필터: 자이로바이어스=%+.5f rad/s  헤딩끌림=%+.5f rad/hop  마지막 NIS=%.3f  "
      "게이트기각=%ld  버림=%ld  헤딩갱신=%ld\n",
      diagnostics.gyro_bias, diagnostics.heading_drift, diagnostics.last_nis,
      diagnostics.filter_rejections, diagnostics.filter_dropped,
      diagnostics.heading_updates);
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
