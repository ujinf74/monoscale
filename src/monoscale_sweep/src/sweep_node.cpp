// The online occupancy node.
//
// Subscribes to the two fisheye images, their CameraInfo, and the estimator's
// Odometry; keyframes every keyframe_travel of driven path, sweeps the ring of
// buffered frames the way plane_sweep.py does offline, and publishes the
// accumulated grid on a timer. The pose comes from odometry alone -- this node
// does no estimation and holds no transform tree, which is the seam that let
// the whole thing be debugged offline (see docs/점유지도_노드_설계.md).

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "monoscale_sweep/sweep.hpp"

namespace
{

double yaw_of(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

// One camera's ring of recent frames, each tagged with the pose it was seen
// from. Bounded by travelled distance rather than count so a stop does not
// flush it.
struct Frame
{
  double stamp = 0.0;
  cv::Mat gray;      // CV_32F at processing width
  monoscale_sweep::Pose5 pose;
  double travelled = 0.0;
};

}  // namespace

class SweepNode : public rclcpp::Node
{
  using Lens = monoscale_sweep::Lens;

public:
  SweepNode()
  : rclcpp::Node("monoscale_sweep")
  {
    cameras_ = declare_parameter<std::vector<std::string>>(
      "cameras", std::vector<std::string>{"front", "rear"});
    image_topics_ = declare_parameter<std::vector<std::string>>(
      "image_topics", std::vector<std::string>{
        "/sensing/camera/front/fisheye/image_raw",
        "/sensing/camera/rear/fisheye/image_raw"});
    info_topics_ = declare_parameter<std::vector<std::string>>(
      "info_topics", std::vector<std::string>{
        "/sensing/camera/front/fisheye/camera_info",
        "/sensing/camera/rear/fisheye/camera_info"});
    odometry_topic_ = declare_parameter<std::string>(
      "odometry_topic", "/localization/odometry");
    grid_topic_ = declare_parameter<std::string>(
      "occupancy_topic", "/perception/occupancy_grid_map");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    processing_width_ = declare_parameter<int>("processing_width", 1280);
    const double rate = declare_parameter<double>("publish_rate_hz", 5.0);

    // The mounts, one block per camera, the same numbers the offline path
    // reads out of vision_fisheye.param.yaml.
    for (const auto & name : cameras_) {
      Lens lens;
      const auto k = declare_parameter<std::vector<double>>(
        name + ".k", std::vector<double>{});
      const auto rot = declare_parameter<std::vector<double>>(
        name + ".rotation_base_from_camera", std::vector<double>{});
      const auto trans = declare_parameter<std::vector<double>>(
        name + ".translation_base_from_camera", std::vector<double>{});
      calibration_width_[name] = declare_parameter<double>(
        name + ".calibration_width", 2560.0);
      if (k.size() == 9) {
        lens.focal = k[0];
        lens.cx = k[2];
        lens.cy = k[5];
      }
      if (rot.size() == 9) {
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) {
            lens.rotation_base_from_camera(r, c) = rot[r * 3 + c];
          }
        }
      }
      if (trans.size() == 3) {
        lens.translation_base_from_camera =
          Eigen::Vector3d(trans[0], trans[1], trans[2]);
      }
      base_lens_[name] = lens;
    }

    monoscale_sweep::SweepSettings settings;  // operating-point defaults
    settings_ = settings;
    for (const auto & name : cameras_) {
      grids_[name].reset(settings_);
    }

    rclcpp::QoS sensor(10);
    sensor.best_effort();
    for (size_t i = 0; i < cameras_.size(); ++i) {
      const std::string name = cameras_[i];
      image_subs_.push_back(create_subscription<sensor_msgs::msg::Image>(
        image_topics_[i], sensor,
        [this, name](sensor_msgs::msg::Image::ConstSharedPtr m) {on_image(name, *m);}));
      info_subs_.push_back(create_subscription<sensor_msgs::msg::CameraInfo>(
        info_topics_[i], sensor,
        [this, name](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {on_info(name, *m);}));
    }
    rclcpp::QoS reliable(50);
    reliable.reliable();
    odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odometry_topic_, reliable,
      [this](nav_msgs::msg::Odometry::ConstSharedPtr m) {on_odometry(*m);});

    rclcpp::QoS latched(1);
    latched.reliable().transient_local();
    grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(grid_topic_, latched);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(rate, 0.1)),
      [this]() {publish();});
    report_ = create_wall_timer(std::chrono::seconds(2), [this]() {report();});
  }

private:
  void on_info(const std::string & name, const sensor_msgs::msg::CameraInfo & message)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (base_lens_.count(name) && base_lens_[name].focal == 0.0 && message.k[0] != 0.0) {
      base_lens_[name].focal = message.k[0];
      base_lens_[name].cx = message.k[2];
      base_lens_[name].cy = message.k[5];
    }
  }

  void on_odometry(const nav_msgs::msg::Odometry & message)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    const double stamp = rclcpp::Time(message.header.stamp).seconds();
    // Anchor the map to the first pose, the way the offline path does: the
    // grid is fixed at origin (-30,-30) and the drive has to start near it, or
    // an absolute CARLA pose at (-48, 6) lands the whole track off the grid.
    const double raw_x = message.pose.pose.position.x;
    const double raw_y = message.pose.pose.position.y;
    const double raw_yaw = yaw_of(message.pose.pose.orientation);
    if (!have_origin_) {
      origin_x_ = raw_x;
      origin_y_ = raw_y;
      origin_yaw_ = raw_yaw;
      have_origin_ = true;
    }
    const double c0 = std::cos(-origin_yaw_);
    const double s0 = std::sin(-origin_yaw_);
    const double dx = raw_x - origin_x_;
    const double dy = raw_y - origin_y_;
    monoscale_sweep::Pose5 pose;
    pose.x = c0 * dx - s0 * dy;
    pose.y = s0 * dx + c0 * dy;
    pose.yaw = raw_yaw - origin_yaw_;
    // Roll and pitch from the same quaternion: the sweep places the road as a
    // world-horizontal plane, and a pitched vehicle over a flat road needs
    // them or the ground ten metres out moves by metres.
    const auto & q = message.pose.pose.orientation;
    pose.roll = std::atan2(2.0 * (q.w * q.x + q.y * q.z),
      1.0 - 2.0 * (q.x * q.x + q.y * q.y));
    const double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    pose.pitch = std::abs(sinp) >= 1.0 ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);
    odometry_.push_back({stamp, pose});
    while (odometry_.size() > 4000) {odometry_.pop_front();}
  }

  // The pose at an image stamp, interpolated from the odometry ring. Nullopt
  // until odometry brackets the stamp -- a frame with no pose is not keyframed.
  std::optional<monoscale_sweep::Pose5> pose_at(double stamp) const
  {
    if (odometry_.size() < 2) {return std::nullopt;}
    if (stamp < odometry_.front().first || stamp > odometry_.back().first) {
      return std::nullopt;
    }
    size_t hi = 1;
    while (hi < odometry_.size() && odometry_[hi].first < stamp) {++hi;}
    const auto & a = odometry_[hi - 1];
    const auto & b = odometry_[hi];
    const double span = b.first - a.first;
    const double w = span > 1e-9 ? (stamp - a.first) / span : 0.0;
    monoscale_sweep::Pose5 pose;
    pose.x = a.second.x + (b.second.x - a.second.x) * w;
    pose.y = a.second.y + (b.second.y - a.second.y) * w;
    const double dyaw = std::atan2(std::sin(b.second.yaw - a.second.yaw),
      std::cos(b.second.yaw - a.second.yaw));
    pose.yaw = a.second.yaw + dyaw * w;
    pose.roll = a.second.roll + (b.second.roll - a.second.roll) * w;
    pose.pitch = a.second.pitch + (b.second.pitch - a.second.pitch) * w;
    return pose;
  }

  void on_image(const std::string & name, const sensor_msgs::msg::Image & message)
  {
    const double stamp = rclcpp::Time(message.header.stamp).seconds();
    cv::Mat gray;
    try {
      gray = cv_bridge::toCvCopy(
        std::make_shared<sensor_msgs::msg::Image>(message), "mono8")->image;
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "cv_bridge: %s", error.what());
      return;
    }
    const double scale = processing_width_ / static_cast<double>(gray.cols);
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::Mat gray32;
    resized.convertTo(gray32, CV_32F);

    std::lock_guard<std::mutex> guard(mutex_);
    auto pose = pose_at(stamp);
    if (!pose) {return;}

    auto & ring = rings_[name];
    double travelled = 0.0;
    if (!ring.empty()) {
      const auto & last = ring.back();
      travelled = last.travelled +
        std::hypot(pose->x - last.pose.x, pose->y - last.pose.y);
    }
    Frame frame;
    frame.stamp = stamp;
    frame.gray = gray32;
    frame.pose = *pose;
    frame.travelled = travelled;
    ring.push_back(std::move(frame));
    // Keep enough baseline for the widest source offset, plus a margin.
    while (ring.size() > 2 &&
      travelled - ring.front().travelled > 6.0)
    {
      ring.pop_front();
    }
    ensure_sweep(name);
    maybe_keyframe(name);
  }

  void ensure_sweep(const std::string & name)
  {
    if (sweeps_.count(name)) {return;}
    Lens lens = base_lens_[name];
    // The calibration is quoted at the spawn width; scale to processing width.
    const double ratio = processing_width_ / calibration_width_[name];
    lens.focal *= ratio;
    lens.cx *= ratio;
    lens.cy *= ratio;
    sweeps_.emplace(name, std::make_unique<monoscale_sweep::Sweep>(settings_, lens));
  }

  void maybe_keyframe(const std::string & name)
  {
    auto & ring = rings_[name];
    const Frame & reference = ring.back();
    if (reference.travelled < next_at_[name]) {return;}

    // Sources at the configured offsets of camera travel, nearest frame in
    // the ring within tolerance. Offsets are along the driven path here (the
    // ring is short), matching the offline --source-offsets without
    // --baseline-select.
    std::vector<cv::Mat> source_grays;
    std::vector<monoscale_sweep::Pose5> source_poses;
    for (double offset : settings_.source_offsets) {
      const double want = reference.travelled + offset;
      const Frame * best = nullptr;
      double gap = settings_.source_tolerance;
      for (const auto & frame : ring) {
        const double error = std::abs(frame.travelled - want);
        if (error < gap && &frame != &reference) {gap = error; best = &frame;}
      }
      if (best) {
        source_grays.push_back(best->gray);
        source_poses.push_back(best->pose);
      }
    }
    if (source_grays.size() < 2) {return;}
    next_at_[name] = reference.travelled + settings_.keyframe_travel;
    sweeps_[name]->keyframe(
      reference.gray, reference.pose, source_grays, source_poses, grids_[name]);
    ++keyframes_;
    last_stamp_ = reference.stamp;
    have_stamp_ = true;
  }

  void publish()
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!have_stamp_) {return;}
    std::vector<monoscale_sweep::CameraGrid *> grids;
    for (const auto & name : cameras_) {grids.push_back(&grids_[name]);}
    const cv::Mat values = monoscale_sweep::publish(settings_, grids);

    nav_msgs::msg::OccupancyGrid message;
    message.header.stamp = rclcpp::Time(static_cast<int64_t>(last_stamp_ * 1e9));
    message.header.frame_id = map_frame_;
    message.info.resolution = static_cast<float>(settings_.resolution);
    message.info.width = static_cast<uint32_t>(settings_.grid_width);
    message.info.height = static_cast<uint32_t>(settings_.grid_height);
    message.info.origin.position.x = settings_.origin_x;
    message.info.origin.position.y = settings_.origin_y;
    message.info.origin.orientation.w = 1.0;
    message.data.resize(static_cast<size_t>(values.total()));
    std::memcpy(message.data.data(), values.data, values.total());
    grid_pub_->publish(message);
    published_ = static_cast<int>(values.total());
  }

  void report()
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (keyframes_ == 0) {return;}
    RCLCPP_INFO(get_logger(), "sweep: keyframes=%ld published=%d cells", keyframes_, published_);
  }

  std::vector<std::string> cameras_;
  std::vector<std::string> image_topics_;
  std::vector<std::string> info_topics_;
  std::string odometry_topic_;
  std::string grid_topic_;
  std::string map_frame_;
  int processing_width_ = 1280;

  monoscale_sweep::SweepSettings settings_;
  std::map<std::string, Lens> base_lens_;
  std::map<std::string, double> calibration_width_;
  std::map<std::string, std::unique_ptr<monoscale_sweep::Sweep>> sweeps_;
  std::map<std::string, monoscale_sweep::CameraGrid> grids_;
  std::map<std::string, std::deque<Frame>> rings_;
  std::map<std::string, double> next_at_;
  std::deque<std::pair<double, monoscale_sweep::Pose5>> odometry_;

  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> info_subs_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr report_;

  std::mutex mutex_;
  int64_t keyframes_ = 0;
  double last_stamp_ = 0.0;
  bool have_stamp_ = false;
  int published_ = 0;
  bool have_origin_ = false;
  double origin_x_ = 0.0;
  double origin_y_ = 0.0;
  double origin_yaw_ = 0.0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SweepNode>());
  rclcpp::shutdown();
  return 0;
}
