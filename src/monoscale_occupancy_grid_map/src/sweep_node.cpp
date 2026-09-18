// The online occupancy node.
//
// Subscribes to the two fisheye images, their CameraInfo, and the estimator's
// Odometry; keyframes every keyframe_travel of driven path, sweeps the ring of
// buffered frames the way plane_sweep.py does offline, and publishes the
// accumulated grid on a timer. The pose comes from odometry alone -- this node
// does no estimation and holds no transform tree, which is the seam that let
// the whole thing be debugged offline (the node design note).

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
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

#include "monoscale_occupancy_grid_map/sweep.hpp"

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
  monoscale_occupancy::Pose5 pose;
  double travelled = 0.0;
};

}  // namespace

// One camera's sweep of one keyframe, lifted out of the node's lock.
//
// The keyframe is 160 ms of work and it used to run with `mutex_` held, so the
// two cameras took turns at it even under a multi-threaded executor. Nothing in
// it is shared between cameras: the `Sweep` and the `CameraGrid` are per camera,
// and the images are refcounted `cv::Mat` handles the sweep only reads. So the
// selection stays under the lock -- it walks the ring, which the odometry
// callback mutates -- and the sweep itself is carried out here, after it.
//
// The pointers are resolved under the lock too. `sweeps_` is a node-based map
// that `ensure_sweep` inserts into, so a reference to one camera's entry
// survives another camera's insertion, but only if the lookup itself did not
// race with it.
struct Job
{
  monoscale_occupancy::Sweep * sweep = nullptr;
  monoscale_occupancy::CameraGrid * grid = nullptr;
  cv::Mat gray;
  monoscale_occupancy::Pose5 pose;
  std::vector<cv::Mat> source_grays;
  std::vector<monoscale_occupancy::Pose5> source_poses;

  void run() const
  {
    sweep->keyframe(gray, pose, source_grays, source_poses, *grid);
  }
};

class SweepNode : public rclcpp::Node
{
  using Lens = monoscale_occupancy::Lens;

public:
  SweepNode()
  : rclcpp::Node("monoscale_occupancy_grid_map")
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

    monoscale_occupancy::SweepSettings settings;  // operating-point defaults
    settings_ = settings;
    for (const auto & name : cameras_) {
      grids_[name].reset(settings_);
    }

    rclcpp::QoS sensor(10);
    sensor.best_effort();
    for (size_t i = 0; i < cameras_.size(); ++i) {
      const std::string name = cameras_[i];
      // A group per camera, so a sweep on one does not stop frames arriving
      // on the other, and so a frame keeps being ingested while a sweep runs.
      // The node's default group is mutually exclusive, so without this the
      // multi-threaded executor below would change nothing: a keyframe holds
      // its callback for about 110 ms, and at 22 Hz on a best-effort queue ten
      // deep the sweep would drop the frames it is about to need.
      auto group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      image_groups_.push_back(group);
      rclcpp::SubscriptionOptions image_options;
      image_options.callback_group = group;
      image_subs_.push_back(create_subscription<sensor_msgs::msg::Image>(
        image_topics_[i], sensor,
        [this, name](sensor_msgs::msg::Image::ConstSharedPtr m) {on_image(name, *m);},
        image_options));
      info_subs_.push_back(create_subscription<sensor_msgs::msg::CameraInfo>(
        info_topics_[i], sensor,
        [this, name](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {on_info(name, *m);}));
    }
    rclcpp::QoS reliable(50);
    reliable.reliable();
    // Its own group as well: this is the callback that actually sweeps, since
    // every frame arrives before the pose that places it.
    odometry_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions odometry_options;
    odometry_options.callback_group = odometry_group_;
    odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odometry_topic_, reliable,
      [this](nav_msgs::msg::Odometry::ConstSharedPtr m) {on_odometry(*m);},
      odometry_options);

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
    std::vector<Job> jobs;
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
    monoscale_occupancy::Pose5 pose;
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
    // The pose that just arrived is what the held frames were waiting for.
    //
    // And this is where the sweeps actually happen in deployment, not in
    // `on_image`: the odometry is solved from these very images, so every
    // frame arrives ahead of the newest pose -- 400 of 400, by a median of
    // 0.70 s -- and is held until a pose reaches it. So both cameras keyframe
    // from this one callback, which is why it has to be the concurrent one.
    for (const auto & name : cameras_) {drain(name, jobs);}
    }
    run_jobs(jobs);
  }

  // The pose at an image stamp, interpolated from the odometry ring. Nullopt
  // until odometry brackets the stamp -- a frame with no pose is not keyframed.
  std::optional<monoscale_occupancy::Pose5> pose_at(double stamp) const
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
    monoscale_occupancy::Pose5 pose;
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

    std::vector<Job> jobs;
    {
    std::lock_guard<std::mutex> guard(mutex_);
    // Hold the frame rather than resolving its pose now. The odometry is
    // solved from these very images, so its stamps can never lead them: at
    // arrival every frame is newer than the newest pose, `pose_at` returns
    // nothing, and dropping it here meant the live node made no keyframes at
    // all -- ever, at any speed. Measured 09-08 on the driving rig: 400 of
    // 400 frames arrived ahead of the newest pose, by a median of 0.70 s.
    // Offline never saw this because there the whole trajectory is loaded
    // before the first frame is placed.
    auto & queue = pending_[name];
    queue.push_back({stamp, gray32});
    while (queue.size() > kPendingFrames) {queue.pop_front();}
    drain(name, jobs);
    }
    run_jobs(jobs);
  }

  // The collected sweeps, one thread each, the first on this one.
  //
  // In deployment both cameras keyframe out of the same odometry callback, so
  // running the list in order is what made the two cameras take turns. They
  // write disjoint grids, and the device is not saturated by one of them:
  // offline, the identical change took a 674-keyframe run from 129.2 s to
  // 74.5 s -- 1.735x -- with the published map bit identical, 0 of 360000
  // cells moved.
  //
  // `publish_shared_` is held across them, which excludes the publisher and
  // not each other: two keyframes touch two different `CameraGrid`s, but
  // `publish` reads both and must not read one mid-write.
  //
  // A thread per keyframe, against 160 ms of work in it. That is about 20
  // creations a second for tens of microseconds each; a pool would be tidier
  // and would save nothing measurable.
  void run_jobs(std::vector<Job> & jobs)
  {
    if (jobs.empty()) {return;}
    std::shared_lock<std::shared_mutex> hold(publish_shared_);
    std::vector<std::thread> workers;
    workers.reserve(jobs.size() - 1);
    for (size_t i = 1; i < jobs.size(); ++i) {
      workers.emplace_back([&jobs, i]() {jobs[i].run();});
    }
    jobs[0].run();
    for (auto & worker : workers) {worker.join();}
  }

  // Admit every held frame the odometry can now place, oldest first so the
  // ring stays ordered and `travelled` keeps accumulating along the path.
  void drain(const std::string & name, std::vector<Job> & jobs)
  {
    auto & queue = pending_[name];
    while (!queue.empty()) {
      const double stamp = queue.front().first;
      const auto pose = pose_at(stamp);
      if (!pose) {
        if (!odometry_.empty() && stamp < odometry_.front().first) {
          // Older than any pose still held: it can never be placed now.
          queue.pop_front();
          continue;
        }
        return;  // Newer than the newest pose. Wait for the odometry.
      }
      admit(name, stamp, queue.front().second, *pose, jobs);
      queue.pop_front();
    }
  }

  void admit(
    const std::string & name, const double stamp, const cv::Mat & gray,
    const monoscale_occupancy::Pose5 & pose, std::vector<Job> & jobs)
  {
    auto & ring = rings_[name];
    double travelled = 0.0;
    if (!ring.empty()) {
      const auto & last = ring.back();
      travelled = last.travelled +
        std::hypot(pose.x - last.pose.x, pose.y - last.pose.y);
    }
    Frame frame;
    frame.stamp = stamp;
    frame.gray = gray;
    frame.pose = pose;
    frame.travelled = travelled;
    // A frame from where the last one already stands carries no baseline the
    // ring does not have, so take its place instead of joining it. Without
    // this a stationary vehicle grows the ring at the frame rate for ever --
    // the travel-based trim below cannot fire when travel does not advance,
    // and the node reached 8.3 GB in three minutes of standing still.
    const bool moved = ring.empty() ||
      travelled - ring.back().travelled > kRingMinStep;
    if (moved) {
      ring.push_back(std::move(frame));
    } else {
      ring.back() = std::move(frame);
    }
    // Keep enough baseline for the widest source offset, plus a margin.
    while (ring.size() > 2 &&
      travelled - ring.front().travelled > 6.0)
    {
      ring.pop_front();
    }
    // Backstop: the trim above is distance-based, so anything that keeps the
    // travel small keeps the ring long. Each frame is a float image.
    while (ring.size() > kRingFrames) {ring.pop_front();}
    ++placed_;
    ensure_sweep(name);
    maybe_keyframe(name, jobs);
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
    sweeps_.emplace(name, std::make_unique<monoscale_occupancy::Sweep>(settings_, lens));
  }

  void maybe_keyframe(const std::string & name, std::vector<Job> & jobs)
  {
    auto & ring = rings_[name];
    const Frame & reference = ring.back();
    if (reference.travelled < next_at_[name]) {return;}

    // Sources at the configured offsets of camera travel, nearest frame in
    // the ring within tolerance. Offsets are along the driven path here (the
    // ring is short), matching the offline --source-offsets without
    // --baseline-select.
    std::vector<cv::Mat> source_grays;
    std::vector<monoscale_occupancy::Pose5> source_poses;
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
    jobs.push_back(
      {sweeps_[name].get(), &grids_[name], reference.gray, reference.pose,
        std::move(source_grays), std::move(source_poses)});
    ++keyframes_;
    last_stamp_ = reference.stamp;
    have_stamp_ = true;
  }

  void publish()
  {
    std::unique_lock<std::shared_mutex> alone(publish_shared_);
    std::lock_guard<std::mutex> guard(mutex_);
    if (!have_stamp_) {return;}
    std::vector<monoscale_occupancy::CameraGrid *> grids;
    for (const auto & name : cameras_) {grids.push_back(&grids_[name]);}
    // No extent asked for: the map published is the part of the world that
    // has actually been mapped, so it grows with the drive instead of being
    // the 60 m box the run started in.
    const auto published = monoscale_occupancy::publish(settings_, grids);
    if (published.window.empty()) {return;}
    const cv::Mat & values = published.values;

    nav_msgs::msg::OccupancyGrid message;
    message.header.stamp = rclcpp::Time(static_cast<int64_t>(last_stamp_ * 1e9));
    message.header.frame_id = map_frame_;
    message.info.resolution = static_cast<float>(settings_.resolution);
    message.info.width = static_cast<uint32_t>(published.window.width);
    message.info.height = static_cast<uint32_t>(published.window.height);
    message.info.origin.position.x = published.window.origin_x;
    message.info.origin.position.y = published.window.origin_y;
    message.info.origin.orientation.w = 1.0;
    message.data.resize(static_cast<size_t>(values.total()));
    std::memcpy(message.data.data(), values.data, values.total());
    grid_pub_->publish(message);
    published_ = static_cast<int>(values.total());
  }

  void report()
  {
    std::lock_guard<std::mutex> guard(mutex_);
    // Said even at zero. A node silent until it has already succeeded cannot
    // be told from a dead one, and that is how the live keyframe fault stayed
    // invisible: no frame was ever placed, so nothing was ever logged.
    size_t held = 0;
    size_t ringed = 0;
    for (const auto & entry : pending_) {held += entry.second.size();}
    for (const auto & entry : rings_) {ringed += entry.second.size();}
    RCLCPP_INFO(
      get_logger(),
      "sweep: keyframes=%ld placed=%ld held=%zu ring=%zu poses=%zu published=%d cells",
      keyframes_, placed_, held, ringed, odometry_.size(), published_);
  }

  std::vector<std::string> cameras_;
  std::vector<std::string> image_topics_;
  std::vector<std::string> info_topics_;
  std::string odometry_topic_;
  std::string grid_topic_;
  std::string map_frame_;
  int processing_width_ = 1280;

  monoscale_occupancy::SweepSettings settings_;
  std::map<std::string, Lens> base_lens_;
  std::map<std::string, double> calibration_width_;
  std::map<std::string, std::unique_ptr<monoscale_occupancy::Sweep>> sweeps_;
  std::map<std::string, monoscale_occupancy::CameraGrid> grids_;
  std::map<std::string, std::deque<Frame>> rings_;
  // Frames waiting for odometry to reach their stamp. At 22 Hz this bounds
  // the wait at about twenty seconds, well past the 0.7 s actually seen.
  static constexpr size_t kPendingFrames = 512;
  // A frame closer than this to the ring's newest replaces it rather than
  // extending the ring: half the smallest source offset's tolerance.
  static constexpr double kRingMinStep = 0.02;
  // Hard cap on ring length. 6 m of baseline at the 5 cm keyframe spacing
  // needs 120; this is double that, and bounds the ring near 1 GB per camera.
  static constexpr size_t kRingFrames = 256;
  std::map<std::string, std::deque<std::pair<double, cv::Mat>>> pending_;
  std::map<std::string, double> next_at_;
  std::deque<std::pair<double, monoscale_occupancy::Pose5>> odometry_;

  std::vector<rclcpp::CallbackGroup::SharedPtr> image_groups_;
  rclcpp::CallbackGroup::SharedPtr odometry_group_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> info_subs_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr report_;

  std::mutex mutex_;
  // Excludes `publish` against the sweeps. A sweep takes it shared -- the
  // sweeps do not exclude one another -- and `publish` takes it alone. Never
  // taken while `mutex_` is held, which is why the two cannot deadlock.
  std::shared_mutex publish_shared_;
  int64_t keyframes_ = 0;
  int64_t placed_ = 0;
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
  // Multi-threaded, because a keyframe holds its callback for about 110 ms and
  // the frames arriving in that window are what the next one sweeps. The
  // groups above are what make it concurrent; the sweeps themselves are made
  // concurrent by `run_jobs`, which works under either executor.
  auto node = std::make_shared<SweepNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
