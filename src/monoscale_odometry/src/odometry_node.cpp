// The ROS side of the estimator: subscriptions, QoS, transforms and
// publishers. Everything that decides where the vehicle is lives in
// monoscale_core; this file only carries messages across.
//
// Two things are deliberately different from the Python this replaces.
//
// There is no solve timer. The estimator drains its queues on every ingest, so
// the 200 Hz timer that used to close pairs is not needed -- and that timer,
// running in its own callback group beside the IMU's, was calling the same
// state machine concurrently. The two are visible together in a replay: the
// solve tears down the tracked set while the IMU-driven call is halfway
// through reading it.
//
// And every ingest is serialised on one lock. The estimator holds per-camera
// queues, an anchor map and a filter chain, none of which is safe to enter
// twice at once. A solve costs well under a millisecond here, so the lock is
// cheaper than the alternative was to debug.

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "monoscale_core/estimator.hpp"
#include "monoscale_odometry/parameters.hpp"

namespace monoscale_ros
{

namespace
{

rclcpp::QoS input_qos(const Topics & topics)
{
  rclcpp::QoS qos = topics.input_history == "keep_all"
    ? rclcpp::QoS(rclcpp::KeepAll())
    : rclcpp::QoS(static_cast<size_t>(std::max(topics.input_queue_depth, 1)));
  if (topics.input_reliability == "reliable") {
    qos.reliable();
  } else {
    qos.best_effort();
  }
  if (topics.input_durability == "transient_local") {
    qos.transient_local();
  }
  return qos;
}

Eigen::Matrix3d rotation_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (norm < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }
  const double s = 2.0 / norm;
  Eigen::Matrix3d rotation;
  rotation <<
    1.0 - s * (q.y * q.y + q.z * q.z), s * (q.x * q.y - q.z * q.w),
    s * (q.x * q.z + q.y * q.w),
    s * (q.x * q.y + q.z * q.w), 1.0 - s * (q.x * q.x + q.z * q.z),
    s * (q.y * q.z - q.x * q.w),
    s * (q.x * q.z - q.y * q.w), s * (q.y * q.z + q.x * q.w),
    1.0 - s * (q.x * q.x + q.y * q.y);
  return rotation;
}

}  // namespace

namespace
{

geometry_msgs::msg::Quaternion orientation_of(double yaw, double roll, double pitch)
{
  const Eigen::Quaterniond q =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}

// Roll and pitch are estimated by a complementary filter, which carries no
// covariance to report. Scored against the recorded truth attitude it runs
// 0.39 to 0.45 degrees RMS across the eleven drives, so one degree is carried
// here: constant, conservative, and honest about being a measurement of the
// filter rather than an output of it.
constexpr double kTiltSigma = 1.0 * M_PI / 180.0;
// Height is not estimated at all -- the solve is planar. Say so with a number
// large enough that nobody fuses it.
constexpr double kUnestimated = 1.0e6;

void fill_covariance(nav_msgs::msg::Odometry & odometry, const monoscale::Update & update)
{
  if (!update.covariance_valid) {
    return;
  }
  auto & pose = odometry.pose.covariance;
  // Row-major 6x6 over x, y, z, roll, pitch, yaw.
  pose[0] = update.pose_covariance(0, 0);
  pose[1] = update.pose_covariance(0, 1);
  pose[6] = update.pose_covariance(1, 0);
  pose[7] = update.pose_covariance(1, 1);
  pose[35] = update.pose_covariance(2, 2);
  pose[14] = kUnestimated;
  pose[21] = update.tilt_valid ? kTiltSigma * kTiltSigma : kUnestimated;
  pose[28] = update.tilt_valid ? kTiltSigma * kTiltSigma : kUnestimated;

  auto & twist = odometry.twist.covariance;
  twist[0] = update.twist_covariance(0, 0);
  twist[1] = update.twist_covariance(0, 1);
  twist[6] = update.twist_covariance(1, 0);
  twist[7] = update.twist_covariance(1, 1);
  twist[35] = update.twist_covariance(2, 2);
  twist[14] = kUnestimated;
  twist[21] = kUnestimated;
  twist[28] = kUnestimated;
}

}  // namespace

class OdometryNode : public rclcpp::Node
{
public:
  OdometryNode()
  : rclcpp::Node("monoscale_odometry")
  {
    configuration_ = declare_and_read(*this);
    estimator_ = std::make_unique<monoscale::Estimator>(configuration_.estimator);

    if (configuration_.topics.track_prefix.empty()) {
      RCLCPP_FATAL(
        get_logger(),
        "track_topic_prefix is empty. This node takes tracks from "
        "monoscale_tracker; the optical flow front end the Python offered "
        "instead is not part of this path.");
      throw std::runtime_error("track_topic_prefix is required");
    }

    if (configuration_.topics.extrinsics_from_tf) {
      // On its own thread, which here means its own callback group on its own
      // executor -- rclcpp's listener adds the group, not the node. The Python
      // equivalent hands the whole node over instead, which put the solve
      // timer and the IMU callback into two executors at once and let them
      // tear down the same tracked set; that is a hazard this side does not
      // have.
      buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
      listener_ = std::make_unique<tf2_ros::TransformListener>(*buffer_, this, true);
      // And a deadline. A rig that publishes no tree is a supported rig, but
      // one that is asked forever never says so: the mounts silently stay at
      // whatever the parameters hold, which is the same failure as a wrong
      // calibration and just as quiet.
      settle_ = create_wall_timer(
        std::chrono::duration<double>(
          std::max(configuration_.topics.tf_lookup_timeout_sec, 0.1)),
        [this]() {settle_extrinsics();});
    }

    const auto qos = input_qos(configuration_.topics);
    for (size_t i = 0; i < configuration_.estimator.cameras.size(); ++i) {
      const std::string & name = configuration_.estimator.cameras[i].name;
      groups_.push_back(
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive));
      rclcpp::SubscriptionOptions options;
      options.callback_group = groups_.back();
      tracks_.push_back(
        create_subscription<std_msgs::msg::Float64MultiArray>(
          configuration_.topics.track_prefix + "/" + name, qos,
          [this, i](std_msgs::msg::Float64MultiArray::ConstSharedPtr message) {
            on_tracks(i, *message);
          },
          options));
      if (configuration_.topics.use_camera_info &&
        i < configuration_.topics.camera_info_topics.size())
      {
        infos_.push_back(
          create_subscription<sensor_msgs::msg::CameraInfo>(
            configuration_.topics.camera_info_topics[i], qos,
            [this, i](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
              on_camera_info(i, *message);
            }));
      }
    }

    if (configuration_.estimator.use_imu_yaw) {
      // Tracking a frame pair blocks a node for tens of milliseconds, which is
      // several IMU samples. A shallow queue silently drops them and the yaw
      // over that interval is then unrecoverable.
      rclcpp::QoS imu_qos(
        static_cast<size_t>(std::max(configuration_.topics.imu_queue_depth, 1)));
      if (configuration_.topics.imu_reliability == "reliable") {
        imu_qos.reliable();
      } else {
        imu_qos.best_effort();
      }
      groups_.push_back(
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive));
      rclcpp::SubscriptionOptions options;
      options.callback_group = groups_.back();
      imu_ = create_subscription<sensor_msgs::msg::Imu>(
        configuration_.topics.imu, imu_qos,
        [this](sensor_msgs::msg::Imu::ConstSharedPtr message) {on_imu(*message);},
        options);
    }

    rclcpp::QoS reliable(
      static_cast<size_t>(std::max(configuration_.topics.odometry_queue_depth, 1)));
    reliable.reliable();
    odometry_ = create_publisher<nav_msgs::msg::Odometry>(
      configuration_.topics.odometry, reliable);
    pose_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      configuration_.topics.pose, reliable);
    rclcpp::QoS latched(1);
    latched.reliable().transient_local();
    points_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      configuration_.topics.ground_points, latched);

    if (configuration_.topics.publish_tf) {
      broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }
    report_ = create_wall_timer(
      std::chrono::seconds(2), [this]() {report();});
  }

  int executor_threads() const {return configuration_.topics.executor_threads;}

private:
  void on_tracks(size_t camera, const std_msgs::msg::Float64MultiArray & message)
  {
    // Counted before any validity check, so a message rejected here can be told
    // apart from one that never arrived.
    ++raw_[camera];
    const auto & data = message.data;
    if (data.size() < 4) {
      return;
    }
    monoscale::TrackFrame frame;
    frame.stamp = data[0];
    const int count = static_cast<int>(data[1]);
    frame.width = static_cast<int>(data[2]);
    frame.height = static_cast<int>(data[3]);
    if (frame.stamp <= 0.0 || frame.width <= 0 || frame.height <= 0 || count < 0 ||
      data.size() < static_cast<size_t>(4 + 5 * count))
    {
      return;
    }
    frame.ids.resize(count);
    frame.previous_pixels.resize(count, 2);
    frame.pixels.resize(count, 2);
    for (int i = 0; i < count; ++i) {
      const size_t at = static_cast<size_t>(4 + 5 * i);
      frame.ids(i) = static_cast<int64_t>(data[at]);
      frame.previous_pixels(i, 0) = data[at + 1];
      frame.previous_pixels(i, 1) = data[at + 2];
      frame.pixels(i, 0) = data[at + 3];
      frame.pixels(i, 1) = data[at + 4];
    }
    // The parallax block: four values a cell, then the cell count, then the
    // marker. Read before the clarity block because it sits behind it.
    size_t tail = data.size();
    if (tail > 2 && data[tail - 1] == -8.125e7) {
      const int cells = static_cast<int>(data[tail - 2]);
      const size_t need = static_cast<size_t>(cells) * 4;
      if (cells > 0 && tail >= need + 2 && tail - need - 2 >= static_cast<size_t>(4 + 5 * count)) {
        const size_t first = tail - need - 2;
        frame.parallax.resize(cells, 4);
        for (int i = 0; i < cells; ++i) {
          for (int j = 0; j < 4; ++j) {
            frame.parallax(i, j) = data[first + static_cast<size_t>(i) * 4 + j];
          }
        }
        tail = first;
      }
    }
    // The clarity block, if the tracker appended one: `count` values then the
    // count itself, so a message without it is read exactly as before and one
    // with it identifies itself rather than being inferred from a length.
    const size_t after_features = static_cast<size_t>(4 + 5 * count);
    if (count > 0 && tail >= after_features + static_cast<size_t>(count) + 1 &&
      static_cast<int>(data[tail - 1]) == count)
    {
      const size_t first = tail - 1 - static_cast<size_t>(count);
      if (first >= after_features) {
        frame.clarity.resize(count);
        for (int i = 0; i < count; ++i) {
          frame.clarity(i) = data[first + static_cast<size_t>(i)];
        }
      }
    }
    // The photometric block the tracker appends after the features. The live
    // path read none of it until now, so every measurement built on the road
    // region existed only under replay -- the gain that consumes it defaults
    // to zero, so parsing it here changes nothing on its own.
    const size_t after = static_cast<size_t>(4 + 5 * count);
    if (data.size() > after) {
      frame.photometric_step = data[after];
      if (data.size() > after + 1) {
        frame.photometric_score = data[after + 1];
      }
      if (data.size() > after + 2) {
        frame.photometric_spread = data[after + 2];
      }
      if (data.size() > after + 12) {
        frame.band_near = data[after + 3];
        frame.band_far = data[after + 4];
        frame.band_left = data[after + 7];
        frame.band_right = data[after + 8];
        frame.band_near_range = data[after + 9];
        frame.band_far_range = data[after + 10];
        frame.band_left_lateral = data[after + 11];
        frame.band_right_lateral = data[after + 12];
      }
      if (data.size() > after + 14) {
        frame.band_near_forward = data[after + 13];
        frame.band_far_forward = data[after + 14];
      }
      if (data.size() > after + 17) {
        frame.esm_yaw = data[after + 15];
        frame.esm_pitch = data[after + 16];
        frame.esm_roll = data[after + 17];
      }
    }
    ++received_[camera];

    // Published under the same lock the updates came out of. Each camera and
    // the instrument sit in their own callback group, so with more than one
    // executor thread two of them can be here at once -- and taking the
    // updates in order then publishing them outside the lock lets the later
    // batch reach the topic first. Odometry that goes backwards in time is
    // worse than odometry that arrives late.
    std::lock_guard<std::mutex> guard(lock_);
    estimator_->ingest_tracks(camera, frame);
    publish(estimator_->take_updates());
  }

  void on_imu(const sensor_msgs::msg::Imu & message)
  {
    monoscale::ImuSample sample;
    sample.stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9;
    sample.orientation = Eigen::Vector4d(
      message.orientation.x, message.orientation.y, message.orientation.z,
      message.orientation.w);
    sample.angular_velocity = Eigen::Vector3d(
      message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z);
    last_rate_ = sample.angular_velocity;
    sample.linear_acceleration = Eigen::Vector3d(
      message.linear_acceleration.x, message.linear_acceleration.y,
      message.linear_acceleration.z);

    // Under the lock, for the ordering reason in on_tracks.
    std::lock_guard<std::mutex> guard(lock_);
    estimator_->ingest_imu(sample);
    publish(estimator_->take_updates());
  }

  void on_camera_info(size_t camera, const sensor_msgs::msg::CameraInfo & message)
  {
    ++infos_seen_[camera];
    mount_from_tf(camera, message.header.frame_id);
    if (message.width == 0 || message.height == 0 || message.k[0] <= 0.0 ||
      message.k[4] <= 0.0)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Ignoring invalid %s CameraInfo; using the configured K",
        configuration_.estimator.cameras[camera].name.c_str());
      return;
    }
    Eigen::Matrix3d k;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        k(r, c) = message.k[static_cast<size_t>(3 * r + c)];
      }
    }
    Eigen::VectorXd d(static_cast<Eigen::Index>(message.d.size()));
    for (size_t i = 0; i < message.d.size(); ++i) {
      d(static_cast<Eigen::Index>(i)) = message.d[i];
    }
    std::lock_guard<std::mutex> guard(lock_);
    estimator_->set_calibration(
      camera, k, d, monoscale::lens_from_name(message.distortion_model),
      static_cast<int>(message.width), static_cast<int>(message.height));
  }

  // Take the mount off the transform tree. Asked once per camera and then
  // cached: where a camera is bolted does not change while the vehicle drives.
  // A rig with no tree keeps the parameters, and says which it used rather than
  // leaving it to be guessed from behaviour.
  void mount_from_tf(size_t camera, const std::string & frame_id)
  {
    if (!buffer_ || frame_id.empty() || mounted_.count(camera) > 0) {
      return;
    }
    geometry_msgs::msg::TransformStamped found;
    try {
      found = buffer_->lookupTransform(
        configuration_.topics.base_frame, frame_id, tf2::TimePointZero);
    } catch (const tf2::TransformException & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(),
        static_cast<int>(1000 * configuration_.topics.tf_lookup_timeout_sec),
        "No transform %s <- %s for %s; using the configured mount",
        configuration_.topics.base_frame.c_str(), frame_id.c_str(),
        configuration_.estimator.cameras[camera].name.c_str());
      return;
    }
    const Eigen::Vector3d translation(
      found.transform.translation.x, found.transform.translation.y,
      found.transform.translation.z);
    std::lock_guard<std::mutex> guard(lock_);
    estimator_->set_mount(
      camera, rotation_from_quaternion(found.transform.rotation), translation);
    mounted_.insert(camera);
    RCLCPP_INFO(
      get_logger(), "%s mounted from TF: %s <- %s",
      configuration_.estimator.cameras[camera].name.c_str(),
      configuration_.topics.base_frame.c_str(), frame_id.c_str());
    if (mounted_.size() == configuration_.estimator.cameras.size()) {
      // Every camera answered. Where a camera is bolted does not change while
      // the vehicle drives, so one answer each is the whole job.
      stop_listening();
    }
  }

  // Stop waiting for a transform tree that is not coming, and say which mounts
  // are being used instead of leaving it to be guessed from behaviour.
  void settle_extrinsics()
  {
    std::string waiting;
    for (size_t i = 0; i < configuration_.estimator.cameras.size(); ++i) {
      if (mounted_.count(i) == 0) {
        waiting += (waiting.empty() ? "" : ", ") + configuration_.estimator.cameras[i].name;
      }
    }
    if (!waiting.empty()) {
      RCLCPP_WARN(
        get_logger(), "No transform for %s after %.1fs; the configured mounts stand",
        waiting.c_str(), configuration_.topics.tf_lookup_timeout_sec);
    }
    stop_listening();
  }

  void stop_listening()
  {
    if (settle_) {
      // Cancelled, not destroyed: this runs from inside the timer's own
      // callback and tearing it down there stalls the executor.
      settle_->cancel();
      settle_.reset();
    }
    // The listener owns a thread; letting it go joins it. Holding one nobody
    // is waiting for is how a run ends without the process ending. It borrows
    // the buffer, so it goes first.
    listener_.reset();
    buffer_.reset();
  }

  void publish(const std::vector<monoscale::Update> & updates)
  {
    for (const auto & update : updates) {
      builtin_interfaces::msg::Time stamp;
      stamp.sec = static_cast<int32_t>(update.stamp);
      stamp.nanosec = static_cast<uint32_t>(
        std::llround((update.stamp - stamp.sec) * 1e9));
      last_stamp_ = stamp;

      if (update.pose_valid) {
        nav_msgs::msg::Odometry odometry;
        odometry.header.stamp = stamp;
        // REP-105: an odometry source owns odom->base_link, which is
        // continuous and allowed to drift. Publishing the pose in `map` said
        // this estimator had localised, which it has not -- the anchor map
        // binds the drift, it does not remove it.
        odometry.header.frame_id = configuration_.topics.odom_frame;
        odometry.child_frame_id = configuration_.topics.base_frame;
        odometry.pose.pose.position.x = update.pose.x;
        odometry.pose.pose.position.y = update.pose.y;
        // Roll and pitch come from the attitude filter and yaw from the planar
        // solve, which is the split the estimator is built on. Publishing yaw
        // alone told every consumer the vehicle is permanently level, while the
        // tilt it withheld is the quantity the ground projection turns on.
        odometry.pose.pose.orientation =
          orientation_of(update.pose.yaw, update.tilt_valid ? update.roll : 0.0,
            update.tilt_valid ? update.pitch : 0.0);
        odometry.twist.twist.linear.x = update.twist.x();
        odometry.twist.twist.linear.y = update.twist.y();
        // The instrument's, resolved into the body: the planar solve owns only
        // the third one, and the other two are measured and were being dropped.
        odometry.twist.twist.angular.x = last_rate_.x();
        odometry.twist.twist.angular.y = last_rate_.y();
        odometry.twist.twist.angular.z = update.twist.z();
        fill_covariance(odometry, update);
        odometry_->publish(odometry);

        geometry_msgs::msg::PoseStamped pose;
        pose.header = odometry.header;
        pose.pose = odometry.pose.pose;
        pose_->publish(pose);

        if (broadcaster_) {
          geometry_msgs::msg::TransformStamped transform;
          transform.header.stamp = stamp;
          transform.header.frame_id = configuration_.topics.odom_frame;
          transform.child_frame_id = configuration_.topics.base_frame;
          transform.transform.translation.x = update.pose.x;
          transform.transform.translation.y = update.pose.y;
          // The same orientation the odometry carries. base_link is the body and
          // the body leans; publishing a level transform beside a tilted pose
          // would make the two disagree about the same vehicle.
          transform.transform.rotation = odometry.pose.pose.orientation;
          broadcaster_->sendTransform(transform);

          if (configuration_.topics.publish_map_to_odom) {
            // Identity. There is nothing in this stack that localises against
            // anything absolute, so map and odom coincide -- this exists so the
            // tree is whole and so a localiser can be dropped in above it
            // without anything below changing. When one arrives it owns this
            // transform and `publish_map_to_odom` goes false.
            geometry_msgs::msg::TransformStamped correction;
            correction.header.stamp = stamp;
            correction.header.frame_id = configuration_.topics.map_frame;
            correction.child_frame_id = configuration_.topics.odom_frame;
            correction.transform.rotation.w = 1.0;
            broadcaster_->sendTransform(correction);
          }
        }
      }

      if (!update.points.empty()) {
        publish_points(stamp, update.points);
      }
    }
  }

  void publish_points(
    const builtin_interfaces::msg::Time & stamp,
    const std::vector<monoscale::LabelledPoint> & points)
  {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    // These are dead reckoned like the pose, so they carry the same frame.
    // Labelling them `map` claimed an absolute position they do not have.
    cloud.header.frame_id = configuration_.topics.odom_frame;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(points.size());
    const char * names[] = {"x", "y", "z", "label", "origin_x", "origin_y"};
    for (int i = 0; i < 6; ++i) {
      sensor_msgs::msg::PointField field;
      field.name = names[i];
      field.offset = static_cast<uint32_t>(4 * i);
      field.datatype = sensor_msgs::msg::PointField::FLOAT32;
      field.count = 1;
      cloud.fields.push_back(field);
    }
    cloud.is_bigendian = false;
    cloud.point_step = 24;
    cloud.row_step = 24 * static_cast<uint32_t>(points.size());
    cloud.is_dense = true;
    cloud.data.resize(cloud.row_step);
    std::memcpy(cloud.data.data(), points.data(), cloud.row_step);
    points_->publish(cloud);
  }

  void report()
  {
    monoscale::Diagnostics diagnostics;
    monoscale::Pose2 pose;
    {
      std::lock_guard<std::mutex> guard(lock_);
      diagnostics = estimator_->diagnostics();
      pose = estimator_->pose();
    }
    std::string counts;
    for (size_t i = 0; i < configuration_.estimator.cameras.size(); ++i) {
      counts += (i ? "/" : "") + std::to_string(received_[i]);
    }
    RCLCPP_INFO(
      get_logger(),
      "vision odometry: solves=%ld pairs=%ld evicted=%ld rx=%s anchors=%d "
      "map_frames=%ld failures=%ld (nosolve=%ld trans=%ld yaw=%ld) coasted=%ld "
      "filter_rej=%ld imu_misses=%ld pose=(%.2f, %.2f, %.2f) "
      "obst[usable=%ld ready=%ld noslip=%ld band=%ld out=%ld]",
      diagnostics.frames_processed, diagnostics.pairs_seen, diagnostics.frames_evicted,
      counts.c_str(), diagnostics.anchors, diagnostics.map_aligned_frames,
      diagnostics.motion_failures, diagnostics.fail_no_solve,
      diagnostics.fail_translation, diagnostics.fail_yaw, diagnostics.coasted,
      diagnostics.filter_rejections, diagnostics.imu_yaw_misses, pose.x, pose.y, pose.yaw,
      diagnostics.obstacle_usable, diagnostics.obstacle_ready, diagnostics.obstacle_no_slip,
      diagnostics.obstacle_out_of_band, diagnostics.obstacle_points);
  }

  Configuration configuration_;
  std::unique_ptr<monoscale::Estimator> estimator_;
  std::mutex lock_;

  std::vector<rclcpp::CallbackGroup::SharedPtr> groups_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr> tracks_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> infos_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_;
  std::unique_ptr<tf2_ros::Buffer> buffer_;
  std::unique_ptr<tf2_ros::TransformListener> listener_;
  // The last rates the instrument reported, for the two twist axes the
  // planar solve does not estimate.
  Eigen::Vector3d last_rate_ = Eigen::Vector3d::Zero();

  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
  rclcpp::TimerBase::SharedPtr report_;
  // Fires once if the transform tree has not answered by then.
  rclcpp::TimerBase::SharedPtr settle_;

  std::map<size_t, int64_t> raw_;
  std::map<size_t, int64_t> received_;
  std::map<size_t, int64_t> infos_seen_;
  std::set<size_t> mounted_;
  builtin_interfaces::msg::Time last_stamp_;
};

}  // namespace monoscale_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<monoscale_ros::OdometryNode>();
  const int threads = node->executor_threads();
  if (threads > 1) {
    rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), static_cast<size_t>(threads));
    executor.add_node(node);
    executor.spin();
  } else {
    rclcpp::spin(node);
  }
  rclcpp::shutdown();
  return 0;
}
