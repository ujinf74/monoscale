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
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "monoscale_core/estimator.hpp"
#include "monoscale_odometry_cpp/parameters.hpp"

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
        create_subscription<std_msgs::msg::Float32MultiArray>(
          configuration_.topics.track_prefix + "/" + name, qos,
          [this, i](std_msgs::msg::Float32MultiArray::ConstSharedPtr message) {
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
  void on_tracks(size_t camera, const std_msgs::msg::Float32MultiArray & message)
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
    ++received_[camera];

    std::vector<monoscale::Update> updates;
    {
      std::lock_guard<std::mutex> guard(lock_);
      estimator_->ingest_tracks(camera, frame);
      updates = estimator_->take_updates();
    }
    publish(updates);
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
    sample.linear_acceleration = Eigen::Vector3d(
      message.linear_acceleration.x, message.linear_acceleration.y,
      message.linear_acceleration.z);

    std::vector<monoscale::Update> updates;
    {
      std::lock_guard<std::mutex> guard(lock_);
      estimator_->ingest_imu(sample);
      updates = estimator_->take_updates();
    }
    publish(updates);
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
        odometry.header.frame_id = configuration_.topics.map_frame;
        odometry.child_frame_id = configuration_.topics.base_frame;
        odometry.pose.pose.position.x = update.pose.x;
        odometry.pose.pose.position.y = update.pose.y;
        odometry.pose.pose.orientation.z = std::sin(0.5 * update.pose.yaw);
        odometry.pose.pose.orientation.w = std::cos(0.5 * update.pose.yaw);
        odometry.twist.twist.linear.x = update.twist.x();
        odometry.twist.twist.linear.y = update.twist.y();
        odometry.twist.twist.angular.z = update.twist.z();
        odometry_->publish(odometry);

        geometry_msgs::msg::PoseStamped pose;
        pose.header = odometry.header;
        pose.pose = odometry.pose.pose;
        pose_->publish(pose);

        if (broadcaster_) {
          geometry_msgs::msg::TransformStamped transform;
          transform.header.stamp = stamp;
          transform.header.frame_id = configuration_.topics.map_frame;
          transform.child_frame_id = configuration_.topics.base_frame;
          transform.transform.translation.x = update.pose.x;
          transform.transform.translation.y = update.pose.y;
          transform.transform.rotation.z = odometry.pose.pose.orientation.z;
          transform.transform.rotation.w = odometry.pose.pose.orientation.w;
          broadcaster_->sendTransform(transform);
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
    cloud.header.frame_id = configuration_.topics.map_frame;
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
      "filter_rej=%ld imu_misses=%ld pose=(%.2f, %.2f, %.2f)",
      diagnostics.frames_processed, diagnostics.pairs_seen, diagnostics.frames_evicted,
      counts.c_str(), diagnostics.anchors, diagnostics.map_aligned_frames,
      diagnostics.motion_failures, diagnostics.fail_no_solve,
      diagnostics.fail_translation, diagnostics.fail_yaw, diagnostics.coasted,
      diagnostics.filter_rejections, diagnostics.imu_yaw_misses, pose.x, pose.y, pose.yaw);
  }

  Configuration configuration_;
  std::unique_ptr<monoscale::Estimator> estimator_;
  std::mutex lock_;

  std::vector<rclcpp::CallbackGroup::SharedPtr> groups_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr> tracks_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> infos_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_;
  std::unique_ptr<tf2_ros::Buffer> buffer_;
  std::unique_ptr<tf2_ros::TransformListener> listener_;
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
