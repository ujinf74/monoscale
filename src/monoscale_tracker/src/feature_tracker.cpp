// Ground feature tracking, moved out of Python.
//
// The estimator's geometry was never the bottleneck: a single optical flow
// call over 600 points at 640x360 measures 2.45 ms, while the same hop costs
// about 12 ms once it has been through rclpy and numpy. That 9.5 ms of
// overhead is what stops the vehicle above roughly two metres a second, and no
// amount of tuning inside Python moved it -- making feature detection 54 times
// cheaper changed the throughput not at all.
//
// This node does only the part that has to keep up: take images, carry the
// tracked set forward, and publish where each feature went. Everything that
// gives the answer its meaning -- ground projection, the anchor map, the
// registration -- stays where it is.
//
// Published layout, one message per camera, flat float64:
//   [stamp_sec, count, width, height, id0, prev_x0, prev_y0, cur_x0, cur_y0, ...]
//
// float64, not float32. Two things in here are integers that do not survive a
// 24 bit mantissa. A Unix timestamp of 1.79e9 quantises to about 106 seconds,
// which collapses every frame of a live run onto the same instant and takes
// dt, the frame pairing and the IMU interpolation with it -- invisible offline
// only because the bags carry simulation stamps of a few tens of seconds.
// Feature identities are the other: at road speed this tracker mints half a
// million of them a minute, and past 2^24 consecutive integers land on the
// same float, which the identity matcher assumes cannot happen. A 53 bit
// mantissa holds both with room to spare, at twice the bytes on a link that
// carries 2.4 MB/s.
// Identities persist for as long as a feature is followed, which is what lets
// the estimator match a sighting against the anchor it built earlier.
//
// The frame size is in the message because the pixels are measured in the
// downscaled frame, and the estimator has to scale the intrinsics to match.
// Leaving it to both sides reading the same processing_width parameter makes
// a silent geometry error out of a configuration mismatch.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#ifdef MONOSCALE_TRACKER_HAS_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace
{

// An Image as a single channel Mat, without cv_bridge.
//
// cv_bridge links against the OpenCV the ROS distribution was built with, and
// this node may be pointed at another one that has the CUDA modules. Two
// OpenCVs in one process is an ABI gamble; the conversion it was doing for us
// is this.
bool to_gray(const sensor_msgs::msg::Image & message, cv::Mat & gray)
{
  const std::string & encoding = message.encoding;
  int channels = 0;
  int code = -1;
  if (encoding == "mono8" || encoding == "8UC1") {
    channels = 1;
  } else if (encoding == "bgr8" || encoding == "8UC3") {
    channels = 3;
    code = cv::COLOR_BGR2GRAY;
  } else if (encoding == "rgb8") {
    channels = 3;
    code = cv::COLOR_RGB2GRAY;
  } else if (encoding == "bgra8" || encoding == "8UC4") {
    channels = 4;
    code = cv::COLOR_BGRA2GRAY;
  } else if (encoding == "rgba8") {
    channels = 4;
    code = cv::COLOR_RGBA2GRAY;
  } else {
    return false;
  }

  const int rows = static_cast<int>(message.height);
  const int columns = static_cast<int>(message.width);
  if (rows <= 0 || columns <= 0 ||
    message.data.size() < static_cast<size_t>(rows) * message.step)
  {
    return false;
  }
  // A view over the caller's buffer, honouring the row stride.
  const cv::Mat view(
    rows, columns, CV_8UC(channels), const_cast<uint8_t *>(message.data.data()),
    message.step);
  if (channels == 1) {
    view.copyTo(gray);
  } else {
    cv::cvtColor(view, gray, code);
  }
  return true;
}

#ifdef MONOSCALE_TRACKER_HAS_CUDA
// A page-locked buffer of at least `count` entries, handed back as a header
// over exactly that many. Grown when it has to be and never shrunk, so a
// steady feature count pays the allocation once and the rest of the drive
// reuses it. Page-locked because a pageable transfer is staged by the driver
// anyway and cannot overlap with a kernel; these are what let the copies ride
// the stream instead of stalling it.
cv::Mat pinned(cv::cuda::HostMem & memory, int count, int type)
{
  if (memory.empty() || memory.cols < count || memory.type() != type) {
    memory.create(1, std::max(count, 1), type);
  }
  return memory.createMatHeader().colRange(0, count);
}
#endif

struct TrackState
{
  cv::Mat previous_gray;
  // ov_core's TrackKLT keeps the pyramid it built for the last image and
  // hands it straight to the next flow call. Passing raw images instead makes
  // OpenCV rebuild both sides on every call, and there are two calls a frame
  // -- four builds where one is needed.
  std::vector<cv::Mat> previous_pyramid;
  std::vector<cv::Point2f> points;
  // Where each point moved on the hop before this one, used to start the
  // search in the right place rather than at the feature's last position.
  std::vector<cv::Point2f> velocities;
  // The same thing said once for the whole frame instead of once per feature.
  // Image motion of a plane between two views is exactly a homography, and the
  // ground is a plane, so one 3x3 describes every ground feature's hop -- new
  // ones included, which a per-feature velocity cannot. It also carries the
  // perspective: apply it to a point that has moved closer and it returns a
  // larger step, which is the acceleration a constant pixel velocity misses.
  // Empty until two frames have been seen.
  cv::Mat plane_motion;
  std::vector<int64_t> identities;
  int64_t next_identity = 0;
  // When the last frame arrived and how long the hop before it took. A
  // dropped frame doubles the gap, and a prediction that does not know that
  // is wrong by a whole frame of motion.
  double stamp = 0.0;
  double interval = 0.0;
  // Virtual features carried by a photometric fit of the road region instead
  // of by per-corner flow. The road is a rapidly moving, low-textured surface
  // -- sparse corners on it die in a median of eight frames and their
  // aggregate carries a frame-wide error ten to twenty times what independent
  // point noise would give. One warp fitted over thousands of road pixels does
  // not have that: it is the same cue Song and Chandraker call plane-guided
  // dense stereo (CVPR 2014), reported there as the term that "vastly improves
  // camera self-localization".
  std::vector<cv::Point2f> road_points;
  std::vector<int64_t> road_identities;
  std::vector<int> road_ages;
  // Previous frame to current, in ROI coordinates. Kept as the warm start for
  // the next fit, which is most of why the iteration converges at all.
  cv::Mat road_warp;

  // How many features this camera is currently trying to hold. Owned by the
  // one thread that runs this camera's callback, so it needs no guard.
  int target = 0;

#ifdef MONOSCALE_TRACKER_HAS_CUDA
  // The same two things the CPU path keeps, on the device. The pyramid is held
  // rather than rebuilt for the same reason: two flow calls a frame would
  // otherwise build four pyramids where two are needed.
  std::vector<cv::cuda::GpuMat> previous_device_pyramid;
  // One stream per camera, so a frame's whole conversation with the device --
  // the image up, both flow calls, every result back -- is queued once and
  // waited on once. It belongs to the camera and not to the node because the
  // two callbacks run at the same time on their own threads.
  cv::cuda::Stream stream;
  // Page-locked staging for everything that crosses the bus.
  cv::cuda::HostMem host_image;
  cv::cuda::HostMem host_source;
  cv::cuda::HostMem host_guess;
  cv::cuda::HostMem host_forward;
  cv::cuda::HostMem host_backward;
  cv::cuda::HostMem host_status;
  cv::cuda::HostMem host_backward_status;
  cv::cuda::HostMem host_error;
  // Held across frames so the device allocator is not asked for the same
  // handful of buffers on every hop.
  cv::cuda::GpuMat device_source;
  cv::cuda::GpuMat device_forward;
  cv::cuda::GpuMat device_status;
  cv::cuda::GpuMat device_error;
  cv::cuda::GpuMat device_backward;
  cv::cuda::GpuMat device_backward_status;
#endif
};

// What the flow did on one hop. Only the report reads it, but the survivors'
// mean hop on its own cannot tell a camera that is not moving apart from one
// whose moving features are all being filtered out.
struct FollowStats
{
  double raw = 0.0;     // mean hop over every point the flow kept
  size_t lost = 0;      // the flow gave up, forward or backward
  size_t noisy = 0;     // patch error above the threshold
  size_t drifted = 0;   // failed the forward-backward check
};

// Where a frame's time went. Which stage grows with the image and which with
// the feature count is not obvious from the code: the flow is charged per
// point, but the pyramid it builds first is charged per pixel.
struct StageTimes
{
  double prep = 0.0;      // decode and resize
  double follow = 0.0;    // the two flow calls and the checks
  double trim = 0.0;
  double detect = 0.0;
  double publish = 0.0;
};

}  // namespace

class FeatureTracker : public rclcpp::Node
{
public:
  // Totals at teardown, because the periodic line is a snapshot: the
  // estimator reported receiving more messages than the last status said had
  // been published, which is only possible if the counter was read early.
  ~FeatureTracker()
  {
    std::string totals;
    for (const auto & entry : published_) {
      totals += " " + entry.first + "=" + std::to_string(entry.second);
    }
    RCLCPP_INFO(get_logger(), "published total:%s", totals.c_str());
  }

  FeatureTracker()
  : Node("feature_tracker")
  {
    // Negative leaves OpenCV's own default, which is what to use. Pinning the
    // flow to a single thread on the theory that two cameras plus the
    // estimator were oversubscribing the machine made the frame cost four
    // times worse -- 60-79 ms against 15-21 -- and the drive scored 20.9 m
    // against 4.0. The internal parallelism is earning its keep.
    cv::setNumThreads(declare_parameter<int>("opencv_threads", -1));
    processing_width_ = declare_parameter<int>("processing_width", 640);
    // 600 was set at parking speed and starves the ground registration at a
    // steady 8 m/s: a third of solves returned nothing, 3.478 m against 0.250.
    // 2000 fixed that, but once detection spread the features over the frame
    // instead of letting them pile up, 1200 turned out to be both cheaper and
    // more accurate -- 0.041 m against 0.076 at 8 m/s, 0.096 against 0.138 at
    // 2.5 m/s. Well placed features are worth more than many of them.
    //
    // 700 is where the frame fits the budget. At 1200 the front camera needs
    // 21 ms against the 16.7 a 60 Hz frame allows, so it runs at 50 Hz, and
    // everything downstream reads as a different bug: a sixth of the pairs
    // rejected on the translation gate, the two cameras disagreeing, the
    // trajectory finishing a fifth short, 2.8 m of position error. None of it
    // survives the frame fitting. At 700 the cameras hold 59 Hz, one pair in
    // the drive is rejected instead of thirty, and the real-time score is the
    // offline score -- 0.041 m either way.
    max_features_ = declare_parameter<int>("max_features_per_camera", 700);
    // What one frame is allowed to cost, in milliseconds, and how far the
    // feature count may be cut to stay under it. A fixed count cannot serve
    // both ends of the range: 2000 scores 0.095 m at 2.5 m/s and 2.8 m at
    // 8 m/s, 700 scores 0.041 m at 8 m/s and 0.131 at 2.5. The budget is the
    // thing that actually differs -- at parking speed the hops are small and
    // the frame is cheap, at road speed it is not -- so let the measurement
    // set the count instead of guessing it. 0 holds max_features_ fixed.
    //
    // Off by default, because a steady count turns out to be worth more than
    // a well chosen one: letting it move scored 0.059-0.064 m at 8 m/s where
    // a fixed 700 scored 0.038-0.042, even though the controller settled on
    // more features than that. Features that come and go have short lives,
    // and the anchor map is built out of long ones. Worth turning on where
    // the speed range is wider than these drives cover -- it evens the two
    // ends out, 0.060 and 0.109 against 0.040 and 0.131.
    frame_budget_ms_ = declare_parameter<double>("frame_budget_ms", 0.0);
    min_features_ = declare_parameter<int>("min_features_per_camera", 500);
    quality_level_ = declare_parameter<double>("quality_level", 0.01);
    min_distance_ = declare_parameter<double>("min_feature_distance_px", 8.0);
    // 15, not 21: the search costs the square of this and 21 bought nothing
    // measurable. Together with the feature count it halves the frame cost,
    // which is the difference between holding 60 Hz and not.
    window_ = declare_parameter<int>("lk_window_px", 15);
    levels_ = declare_parameter<int>("lk_pyramid_levels", 4);
    // The backward check earns its cost: dropping it, or even running it on
    // every other hop, let mistracked points survive to the solve and moved
    // position error from 0.10 m to 0.73 and 0.29 respectively.
    backward_threshold_ = declare_parameter<double>(
      "lk_forward_backward_threshold_px", 1.0);
    error_threshold_ = declare_parameter<double>("lk_error_threshold", 30.0);
    // Applied per grid cell, not to the frame as a whole -- see detect().
    refill_ratio_ = declare_parameter<double>("feature_refill_ratio", 0.7);
    // 4x3 is enough to keep the horizon from crowding out the near ground,
    // and no finer: 8x5 tracked just as well and cost 50 ms a frame against
    // 15, all of it per-cell detection overhead rather than pixels.
    grid_columns_ = declare_parameter<int>("detection_grid_columns", 4);
    grid_rows_ = declare_parameter<int>("detection_grid_rows", 3);
    warm_start_ = declare_parameter<bool>("lk_warm_start", true);
    predict_by_plane_ =
      declare_parameter<bool>("lk_predict_by_plane", false);
    seed_new_from_cell_ =
      declare_parameter<bool>("lk_seed_new_from_cell", false);
    seed_min_flow_ =
      static_cast<float>(declare_parameter<double>("lk_seed_min_flow_px", 8.0));
    // Photometric alignment of the road region, emitted as virtual features.
    road_alignment_ = declare_parameter<bool>("road_alignment", false);
    road_roi_ = {
      declare_parameter<double>("road_roi_x0", 0.20),
      declare_parameter<double>("road_roi_y0", 0.50),
      declare_parameter<double>("road_roi_x1", 0.80),
      declare_parameter<double>("road_roi_y1", 1.00)};
    road_columns_ = declare_parameter<int>("road_grid_columns", 3);
    road_rows_ = declare_parameter<int>("road_grid_rows", 3);
    road_iterations_ = declare_parameter<int>("road_ecc_iterations", 30);
    road_homography_ = declare_parameter<bool>("road_homography", false);
    // How many frames a grid point may be carried before it is dropped and
    // reseeded. Each frame multiplies one more warp onto its position, so a
    // long-lived point accumulates the fit's error; on a curve that error has
    // a consistent direction and does not average out. 0 keeps them until they
    // leave the region.
    road_max_age_ = declare_parameter<int>("road_max_age", 0);
    road_epsilon_ = declare_parameter<double>("road_ecc_epsilon", 1.0e-4);
    // Run the optical flow on the GPU.
    //
    // This is the one stage in the whole stack where a GPU has a case to make.
    // Measured on an Orin Nano at 25 W, a 640 px frame with 1600 features costs
    // 42-43 ms, of which the two flow calls are 35-38; the 30 Hz budget is
    // 33.3. Everything else -- detection, the trim, the publish -- is under
    // 5 ms together, and the estimator downstream is under 3.
    //
    // Off by default, and it stays off until it has been measured on the board
    // it is meant to help. The CPU path is what every recorded number came
    // from, and a GPU path that is merely plausible is worth less than a CPU
    // path that is known.
    use_cuda_ = declare_parameter<bool>("use_cuda", false);
    // How many iterations the GPU flow runs per pyramid level.
    //
    // The CPU stops at whichever comes first, thirty iterations or a step under
    // 0.01 px. OpenCV's CUDA kernel has no epsilon and always runs the count,
    // so on an ambiguous patch it keeps walking after the CPU would have
    // stopped. Before the gates, the GPU's mean hop reads 18.5 px against the
    // CPU's 8.6 on the same frames, which is what that looks like.
    //
    // Which makes lowering it look obvious, and it is a trap. Position RMSE
    // over the same two drives, tracks recorded from each arm and replayed
    // through the same estimator:
    //
    //                    CPU      GPU 10    GPU 30
    //   clean60_release  0.0989   0.0773    0.1106
    //   fish8            0.0618   0.1750    0.0593
    //
    // Ten iterations is a 22% win on one drive and a 2.8x loss on the other.
    // Thirty -- OpenCV's default, and the count the CPU would reach if its
    // epsilon never fired -- sits in the same band as the CPU on both. So the
    // default stays where it is, and the drop to ten stays here as a warning
    // rather than as a setting anyone should reach for.
    cuda_iterations_ = declare_parameter<int>("lk_cuda_iterations", 30);
    if (use_cuda_) {
#ifdef MONOSCALE_TRACKER_HAS_CUDA
      if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cuda_active_ = true;
        forward_flow_ = cv::cuda::SparsePyrLKOpticalFlow::create(
          cv::Size(window_, window_), levels_, cuda_iterations_, true);
        backward_flow_ = cv::cuda::SparsePyrLKOpticalFlow::create(
          cv::Size(window_, window_), levels_, cuda_iterations_, false);
        RCLCPP_INFO(
          get_logger(), "optical flow on the GPU: %s",
          cv::cuda::DeviceInfo(0).name());
      } else {
        RCLCPP_WARN(get_logger(), "use_cuda asked for, but no CUDA device; staying on the CPU");
      }
#else
      RCLCPP_WARN(
        get_logger(),
        "use_cuda asked for, but this OpenCV has no cudaoptflow; staying on the CPU");
#endif
    }
    // Fraction of the image, measured from the top, to leave out of feature
    // detection. The estimator keeps only what projects onto the ground
    // between ground_min_distance_m and ground_max_distance_m, so anything
    // above the horizon -- sky, buildings, the road far ahead -- is tracked
    // at full cost and then discarded.
    //
    // This was once justified by the front camera costing 29-34 ms a frame
    // against the rear's 12-14. That comparison was worthless: the rear was
    // holding a stationary band of horizon features and doing no work at all.
    // With both cameras tracking, the two cost the same. 0 keeps the whole
    // image, which is what has been measured for accuracy.
    skip_top_ = declare_parameter<double>("detection_skip_top_fraction", 0.0);
    dump_dir_ = declare_parameter<std::string>("debug_dump_directory", "");
    // Best effort with a shallow queue is right against a live camera, where
    // a late frame is worth less than the one behind it. It is wrong when the
    // input is a bag being replayed for measurement: the rear camera kept
    // under half its frames that way, and a configuration cannot be compared
    // against another on a different subset of the drive. Reliable with a
    // deep queue makes the replay lossless and the simulator wait instead.
    const auto reliability = declare_parameter<std::string>(
      "input_reliability", "best_effort");
    // Half a second at 60 Hz, where five frames was 83 ms. A frame costs
    // 9-10 ms here, so this is never a backlog being worked through -- it is
    // room for the scheduler to be late without the frame being lost. Five
    // was not enough room: the number of pairs the estimator managed to form
    // from the same recording swung by a tenth between runs, and with it the
    // score.
    const int depth = static_cast<int>(declare_parameter<int>("input_queue_depth", 30));
    rclcpp::QoS input_qos(static_cast<size_t>(std::max(depth, 1)));
    if (reliability == "reliable") {
      input_qos.reliable();
    } else {
      input_qos.best_effort();
    }

    // The input comes from a camera and best effort is right for it: a late
    // frame is worth less than the one behind it, and nothing can be done to
    // make a camera wait. The output goes to the estimator, which is ours,
    // and there is no reason to lose any of it -- every track dropped on that
    // link is a pair the estimator never forms, and it showed up as the same
    // recording scoring differently run to run.
    const auto output_reliability = declare_parameter<std::string>(
      "output_reliability", "reliable");
    const int output_depth =
      static_cast<int>(declare_parameter<int>("output_queue_depth", 200));
    // KEEP_ALL on the writer is the other half of a lossless link: the reader
    // asking for every sample cannot get them if the writer has already
    // dropped its own history. Offline only -- on the vehicle a writer that
    // holds frames for a slow reader is a writer that goes on holding them.
    const auto output_history =
      declare_parameter<std::string>("output_history", "keep_last");
    rclcpp::QoS output_qos = output_history == "keep_all"
      ? rclcpp::QoS(rclcpp::KeepAll())
      : rclcpp::QoS(static_cast<size_t>(std::max(output_depth, 1)));
    const auto output_durability =
      declare_parameter<std::string>("output_durability", "volatile");
    if (output_durability == "transient_local") {
      output_qos.transient_local();
    }
    if (output_reliability == "best_effort") {
      output_qos.best_effort();
    } else {
      output_qos.reliable();
    }

    const auto cameras = declare_parameter<std::vector<std::string>>(
      "cameras", std::vector<std::string>{"front", "rear"});
    const auto topics = declare_parameter<std::vector<std::string>>(
      "image_topics",
      std::vector<std::string>{
        "/sensing/camera/front/vio/image_raw",
        "/sensing/camera/rear/vio/image_raw"});

    if (cameras.size() != topics.size()) {
      RCLCPP_FATAL(get_logger(), "cameras and image_topics must be the same length");
      throw std::runtime_error("camera configuration mismatch");
    }

    for (size_t index = 0; index < cameras.size(); ++index) {
      const std::string & name = cameras[index];
      states_[name].target = max_features_;
      publishers_[name] = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/vision/tracks/" + name, output_qos);
      // Each camera needs its own callback group. Subscriptions left in the
      // node's default group are mutually exclusive with each other, so a
      // multi-threaded executor runs them one at a time no matter how many
      // threads it has -- the rear camera kept 36% of its frames against the
      // front's 97% because it was always the one waiting.
      groups_.push_back(
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive));
      rclcpp::SubscriptionOptions options;
      options.callback_group = groups_.back();
      subscriptions_.push_back(
        create_subscription<sensor_msgs::msg::Image>(
          topics[index], input_qos,
          [this, name](sensor_msgs::msg::Image::ConstSharedPtr message) {
            onImage(name, *message);
          },
          options));
      RCLCPP_INFO(
        get_logger(), "tracking %s on %s", name.c_str(), topics[index].c_str());
    }
  }

private:
  void onImage(const std::string & name, const sensor_msgs::msg::Image & message)
  {
    const auto entered = std::chrono::steady_clock::now();
    auto mark = entered;
    const auto lap = [&mark]() {
      const auto now = std::chrono::steady_clock::now();
      const double seconds = std::chrono::duration<double>(now - mark).count();
      mark = now;
      return seconds;
    };
    StageTimes stage;
    cv::Mat gray;
    if (!to_gray(message, gray)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "unsupported image encoding: %s",
        message.encoding.c_str());
      return;
    }
    if (processing_width_ > 0 && gray.cols > processing_width_) {
      const double scale = static_cast<double>(processing_width_) / gray.cols;
      // Rounded, not truncated: 540 * (640/960) lands on 359.99999999999994 in
      // double, so a cast gives 359 rows where the image path gives 360.
      cv::resize(
        gray, gray, cv::Size(processing_width_, cvRound(gray.rows * scale)),
        0, 0, cv::INTER_AREA);
    }

    TrackState & state = states_[name];
    std::vector<cv::Mat> pyramid;
#ifdef MONOSCALE_TRACKER_HAS_CUDA
    // Only declared where the type is complete: without cudaoptflow, OpenCV
    // forward declares GpuMat and nothing can be held in a vector of them.
    std::vector<cv::cuda::GpuMat> device_pyramid;
    if (cuda_active_) {
      build_device_pyramid(state, gray, device_pyramid);
    } else {
      cv::buildOpticalFlowPyramid(gray, pyramid, cv::Size(window_, window_), levels_);
    }
#else
    cv::buildOpticalFlowPyramid(gray, pyramid, cv::Size(window_, window_), levels_);
#endif
    stage.prep = lap();

    std::vector<cv::Point2f> previous_points;
    std::vector<cv::Point2f> current_points;
    std::vector<cv::Point2f> velocities;
    std::vector<int64_t> identities;
    FollowStats stats;

    // How much longer this hop is than the one the stored velocities measure.
    // Frames get dropped under load, and when one is missed the next hop
    // covers twice the ground. Predicting one frame of motion into a two
    // frame gap leaves the search half a hop short, which costs iterations,
    // fails the backward check, and drops more features -- so the next frame
    // is dearer still and the camera never catches back up. Scaling by the
    // gap that actually elapsed is what stops that from running away.
    const double stamp = static_cast<double>(message.header.stamp.sec) +
      static_cast<double>(message.header.stamp.nanosec) * 1e-9;
    const double elapsed = state.stamp > 0.0 ? stamp - state.stamp : 0.0;
    const double reach = state.interval > 0.0 && elapsed > 0.0
      ? std::clamp(elapsed / state.interval, 0.25, 4.0)
      : 1.0;

    const bool have_previous = !state.points.empty() &&
      state.previous_gray.size() == gray.size();
    if (cuda_active_) {
#ifdef MONOSCALE_TRACKER_HAS_CUDA
      if (have_previous && !state.previous_device_pyramid.empty()) {
        follow_cuda(
          state, gray, device_pyramid, previous_points, current_points, velocities,
          identities, stats, reach);
      }
#endif
    } else if (have_previous && !state.previous_pyramid.empty()) {
      follow(
        state, gray, pyramid, previous_points, current_points, velocities,
        identities, stats, reach);
    }
    if (elapsed > 0.0) {
      state.interval = elapsed;
    }
    state.stamp = stamp;

    stage.follow = lap();
    trim(
      gray, previous_points, current_points, velocities, identities,
      state.target);
    stage.trim = lap();

    // How far the surviving features moved on this hop. If the cost per
    // point differs between the cameras, this is what would explain it: a
    // longer jump makes the pyramid search descend further and iterate more.
    double shift = 0.0;
    for (const auto & step : velocities) {
      shift += std::hypot(step.x, step.y);
    }
    if (!velocities.empty()) {
      shift /= static_cast<double>(velocities.size());
    }

    // How different this frame is from the one it was just compared against.
    // The rear camera reported a 0.03 px hop where the very same recorded
    // frames give 15 px when the flow is run over them offline, and the only
    // way to get that is to compare a picture with itself.
    double change = 0.0;
    if (!state.previous_gray.empty() && state.previous_gray.size() == gray.size()) {
      change = cv::norm(state.previous_gray, gray, cv::NORM_L1) /
        static_cast<double>(gray.total());
    }

    if (predict_by_plane_ && previous_points.size() >= 12 &&
      previous_points.size() == current_points.size())
    {
      // RANSAC because the horizon and anything standing up are not on the
      // plane; they are the minority and the ground is the consensus.
      cv::Mat fitted = cv::findHomography(previous_points, current_points, cv::RANSAC, 3.0);
      if (!fitted.empty() && fitted.rows == 3 && fitted.cols == 3) {
        state.plane_motion = fitted;
      }
    }
    state.points = current_points;
    state.velocities = velocities;
    state.identities = identities;
    const size_t followed = state.points.size();
    detect(state, gray);
    stage.detect = lap();
    // Held before it is replaced: the road fit needs the frame the flow just
    // came from, and detection has already overwritten the state's copy.
    const cv::Mat road_previous = state.previous_gray;
    state.previous_gray = gray;
    if (cuda_active_) {
#ifdef MONOSCALE_TRACKER_HAS_CUDA
      state.previous_device_pyramid = device_pyramid;
#endif
    } else {
      state.previous_pyramid = pyramid;
    }

    // Features detected on this frame go out too, standing still: previous
    // position equal to current, because they have not moved yet. The image
    // path puts them into the tracked set the moment it finds them, so they
    // are in the snapshot a solve compares against and survive the whole of
    // the next interval. Holding them back for one frame left this path
    // solving over a pool that was always a generation older -- median
    // baseline 0.272 m against 0.389 -- and the shorter the baseline the
    // noisier the geometry.
    //
    // Appended to the followed set in place. Copying the three vectors whole
    // to do this was copying the entire frame's output for the sake of the
    // handful of points detection had just added.
    for (size_t i = followed; i < state.points.size(); ++i) {
      previous_points.push_back(state.points[i]);
      current_points.push_back(state.points[i]);
      identities.push_back(state.identities[i]);
    }

    if (road_alignment_ && !road_previous.empty() && road_previous.size() == gray.size()) {
      align_road(state, road_previous, gray, previous_points, current_points, identities);
    }

    if (!dump_dir_.empty() && ++dumped_[name] == 60) {
      cv::Mat canvas;
      cv::cvtColor(gray, canvas, cv::COLOR_GRAY2BGR);
      for (const auto & point : state.points) {
        cv::circle(canvas, point, 2, cv::Scalar(0, 0, 255), -1);
      }
      cv::imwrite(dump_dir_ + "/dump_" + name + ".png", canvas);
    }

    publish(
      name, message, gray.size(), previous_points, current_points, identities);

    stage.publish = lap();
    const double spent = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - entered).count() * 1000.0;
    if (frame_budget_ms_ > 0.0) {
      // Cut hard when over, creep back when there is room to spare. A frame
      // that overruns costs the next one too -- it arrives late, its hop is
      // twice as long and dearer still -- so the way out has to be quicker
      // than the way in.
      if (spent > frame_budget_ms_) {
        state.target = std::max(min_features_, state.target - state.target / 8);
      } else if (spent < 0.7 * frame_budget_ms_) {
        state.target = std::min(max_features_, state.target + state.target / 32 + 1);
      }
    }

    report(
      name,
      spent / 1000.0,
      state.points.size(),
      shift,
      velocities.size(),
      change,
      stats,
      stage);
  }

  void report(
    const std::string & name, double seconds, size_t held, double shift,
    size_t followed, double change, const FollowStats & stats,
    const StageTimes & stage)
  {
    std::lock_guard<std::mutex> guard(count_mutex_);
    ++received_[name];
    spent_[name] += seconds;
    held_[name] += static_cast<double>(held);
    shift_[name] += shift;
    followed_[name] += static_cast<double>(followed);
    change_[name] += change;
    prep_[name] += stage.prep;
    follow_[name] += stage.follow;
    trim_[name] += stage.trim;
    detect_[name] += stage.detect;
    pub_[name] += stage.publish;
    raw_[name] += stats.raw;
    lost_[name] += static_cast<double>(stats.lost);
    noisy_[name] += static_cast<double>(stats.noisy);
    drifted_[name] += static_cast<double>(stats.drifted);
    const rclcpp::Time now = get_clock()->now();
    if (!counting_) {
      counted_since_ = now;
      counting_ = true;
      return;
    }
    const double span = (now - counted_since_).seconds();
    if (span < 2.0) {
      return;
    }
    std::string line;
    for (const auto & entry : received_) {
      const int hz = static_cast<int>(entry.second / span + 0.5);
      const int ms = entry.second > 0
        ? static_cast<int>(1000.0 * spent_[entry.first] / entry.second + 0.5)
        : 0;
      const int points = entry.second > 0
        ? static_cast<int>(held_[entry.first] / entry.second + 0.5)
        : 0;
      line += entry.first + "=" + std::to_string(hz) + "Hz/" +
        std::to_string(ms) + "ms/" + std::to_string(points) + "pts ";
      const double px = entry.second > 0
        ? shift_[entry.first] / entry.second
        : 0.0;
      const int kept = entry.second > 0
        ? static_cast<int>(followed_[entry.first] / entry.second + 0.5)
        : 0;
      const double changed = entry.second > 0
        ? change_[entry.first] / entry.second
        : 0.0;
      const auto stage_ms = [&](std::map<std::string, double> & bucket) {
        return std::to_string(
          1000.0 * bucket[entry.first] / entry.second).substr(0, 4);
      };
      line += "[prep=" + stage_ms(prep_) + " follow=" + stage_ms(follow_) +
        " trim=" + stage_ms(trim_) + " detect=" + stage_ms(detect_) +
        " pub=" + stage_ms(pub_) + "] ";
      line += "(" + std::to_string(px).substr(0, 4) + "px, " +
        std::to_string(kept) + " followed, d=" +
        std::to_string(changed).substr(0, 5) + ", raw=" +
        std::to_string(raw_[entry.first] / entry.second).substr(0, 5) +
        "px, lost/noisy/drifted=" +
        std::to_string(static_cast<int>(lost_[entry.first] / entry.second + 0.5)) + "/" +
        std::to_string(static_cast<int>(noisy_[entry.first] / entry.second + 0.5)) + "/" +
        std::to_string(static_cast<int>(drifted_[entry.first] / entry.second + 0.5)) + ") ";
      followed_[entry.first] = 0.0;
      spent_[entry.first] = 0.0;
      held_[entry.first] = 0.0;
      shift_[entry.first] = 0.0;
      change_[entry.first] = 0.0;
      prep_[entry.first] = 0.0;
      follow_[entry.first] = 0.0;
      trim_[entry.first] = 0.0;
      detect_[entry.first] = 0.0;
      pub_[entry.first] = 0.0;
      raw_[entry.first] = 0.0;
      lost_[entry.first] = 0.0;
      noisy_[entry.first] = 0.0;
      drifted_[entry.first] = 0.0;
    }
    for (auto & entry : received_) {
      entry.second = 0;
    }
    counted_since_ = now;
    // The enclosing report() already holds count_mutex_; taking it again on
    // a non-recursive mutex froze the tracker outright, which is a reminder
    // that instrumentation is code and can break what it measures.
    std::string totals;
    for (const auto & entry : published_) {
      totals += " pub_" + entry.first + "=" + std::to_string(entry.second);
    }
    RCLCPP_INFO(
      get_logger(), "front end in: %s%s", line.c_str(), totals.c_str());
  }


  // Photometric alignment of the road region, emitted as virtual features.
  //
  // A grid of points is carried across frames by one affine warp fitted to the
  // whole road ROI rather than by tracking each of them. Everything downstream
  // sees ordinary features with ordinary identities; what is different is where
  // their positions come from. Per-corner flow on a low-textured road is
  // aperture limited and its errors are shared across the frame -- measured on
  // this rig, the hop those corners produce is eight to twenty times noisier
  // than independent point noise would allow. A warp fitted over the whole
  // region is not aperture limited.
  void align_road(
    TrackState & state, const cv::Mat & previous, const cv::Mat & current,
    std::vector<cv::Point2f> & previous_points, std::vector<cv::Point2f> & current_points,
    std::vector<int64_t> & identities)
  {
    const cv::Rect roi(
      cv::Point(
        cvRound(road_roi_[0] * current.cols), cvRound(road_roi_[1] * current.rows)),
      cv::Point(
        cvRound(road_roi_[2] * current.cols), cvRound(road_roi_[3] * current.rows)));
    const cv::Rect bounded = roi & cv::Rect(0, 0, current.cols, current.rows);
    if (bounded.width < 32 || bounded.height < 32) {
      return;
    }

    const int motion = road_homography_ ? cv::MOTION_HOMOGRAPHY : cv::MOTION_AFFINE;
    const int rows = road_homography_ ? 3 : 2;
    if (state.road_warp.empty() || state.road_warp.rows != rows) {
      state.road_warp = cv::Mat::eye(rows, 3, CV_32F);
    }
    // Warm started from the last frame's warp. Starting from identity every
    // time asks the iteration to cross the whole hop, which at road speed is
    // further than it can see.
    cv::Mat warp = state.road_warp.clone();
    try {
      cv::findTransformECC(
        previous(bounded), current(bounded), warp, motion,
        cv::TermCriteria(
          cv::TermCriteria::COUNT + cv::TermCriteria::EPS, road_iterations_, road_epsilon_),
        cv::noArray(), 5);
    } catch (const cv::Exception &) {
      // Did not converge. The grid is dropped rather than carried on a warp
      // nobody vouched for; it is reseeded next frame.
      state.road_points.clear();
      state.road_identities.clear();
      state.road_ages.clear();
      state.road_warp = cv::Mat::eye(rows, 3, CV_32F);
      return;
    }
    if (!warp.empty() && cv::checkRange(warp)) {
      state.road_warp = warp;
    } else {
      return;
    }

    const cv::Mat_<float> w = state.road_warp;
    const cv::Point2f origin(
      static_cast<float>(bounded.x), static_cast<float>(bounded.y));
    // Carry whoever is already there, then emit the pair. Coordinates are the
    // ROI's for the warp and the frame's for the message.
    std::vector<cv::Point2f> carried;
    std::vector<int64_t> kept;
    carried.reserve(state.road_points.size());
    kept.reserve(state.road_points.size());
    std::vector<int> ages;
    ages.reserve(state.road_points.size());
    for (size_t i = 0; i < state.road_points.size(); ++i) {
      if (road_max_age_ > 0 && i < state.road_ages.size() &&
        state.road_ages[i] >= road_max_age_)
      {
        continue;
      }
      const cv::Point2f & from = state.road_points[i];
      const float scale = road_homography_
        ? w(2, 0) * from.x + w(2, 1) * from.y + w(2, 2) : 1.0f;
      if (std::abs(scale) < 1e-6f) {
        continue;
      }
      const cv::Point2f to(
        (w(0, 0) * from.x + w(0, 1) * from.y + w(0, 2)) / scale,
        (w(1, 0) * from.x + w(1, 1) * from.y + w(1, 2)) / scale);
      if (to.x < 0.0f || to.y < 0.0f ||
        to.x > static_cast<float>(bounded.width - 1) ||
        to.y > static_cast<float>(bounded.height - 1))
      {
        continue;
      }
      previous_points.push_back(from + origin);
      current_points.push_back(to + origin);
      identities.push_back(state.road_identities[i]);
      carried.push_back(to);
      kept.push_back(state.road_identities[i]);
      ages.push_back(i < state.road_ages.size() ? state.road_ages[i] + 1 : 1);
    }

    // Refill the grid wherever it has emptied. A cell is free when nothing
    // carried lands within half a cell of its centre.
    const float step_x = static_cast<float>(bounded.width) / road_columns_;
    const float step_y = static_cast<float>(bounded.height) / road_rows_;
    for (int row = 0; row < road_rows_; ++row) {
      for (int column = 0; column < road_columns_; ++column) {
        const cv::Point2f centre(
          (column + 0.5f) * step_x, (row + 0.5f) * step_y);
        bool taken = false;
        for (const auto & point : carried) {
          if (std::abs(point.x - centre.x) < 0.5f * step_x &&
            std::abs(point.y - centre.y) < 0.5f * step_y)
          {
            taken = true;
            break;
          }
        }
        if (taken) {
          continue;
        }
        carried.push_back(centre);
        kept.push_back(road_next_identity_++);
        ages.push_back(0);
      }
    }
    state.road_points = carried;
    state.road_identities = kept;
    state.road_ages = ages;
  }

  void follow(
    TrackState & state, const cv::Mat & gray,
    const std::vector<cv::Mat> & pyramid,
    std::vector<cv::Point2f> & previous_points,
    std::vector<cv::Point2f> & current_points,
    std::vector<cv::Point2f> & velocities,
    std::vector<int64_t> & identities, FollowStats & stats,
    double reach) const
  {
    std::vector<cv::Point2f> forward;
    std::vector<uchar> status;
    std::vector<float> error;
    const cv::Size window(window_, window_);
    const cv::TermCriteria criteria(
      cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01);
    // Start each search where the feature was heading. The near ground sweeps
    // fastest and carries most of the metric information, so those are the
    // points that fall into a wrong local minimum without a guess -- which
    // biases displacement short and the whole trajectory with it. The image
    // path predicts from the estimated motion; the front end has no access to
    // that, so it carries each feature's own last hop forward instead.
    int flags = 0;
    if (predict_by_plane_ && !state.plane_motion.empty() && !state.points.empty()) {
      // One prediction for the whole frame, from the plane's last hop.
      std::vector<cv::Point2f> warped;
      cv::perspectiveTransform(state.points, warped, state.plane_motion);
      forward.resize(state.points.size());
      for (size_t i = 0; i < state.points.size(); ++i) {
        const cv::Point2f step = warped[i] - state.points[i];
        forward[i] = state.points[i] + step * static_cast<float>(reach);
      }
      flags = cv::OPTFLOW_USE_INITIAL_FLOW;
    } else if (warm_start_ && state.velocities.size() == state.points.size()) {
      forward.resize(state.points.size());
      for (size_t i = 0; i < state.points.size(); ++i) {
        forward[i] = state.points[i] +
          state.velocities[i] * static_cast<float>(reach);
      }
      flags = cv::OPTFLOW_USE_INITIAL_FLOW;
    }
    cv::calcOpticalFlowPyrLK(
      state.previous_pyramid, pyramid, state.points, forward, status, error,
      window, levels_, criteria, flags);

    // What the flow returned before anything was thrown away, and why each
    // rejection happened. The survivors' mean hop alone cannot tell a camera
    // that is not moving apart from one whose moving features are all being
    // filtered out.
    double raw = 0.0;
    size_t raw_count = 0;
    size_t lost = 0;
    size_t noisy = 0;
    size_t drifted = 0;

    // Settle everything the forward pass decides on its own, then run the
    // backward pass over the survivors alone.
    //
    // Both flow calls are charged per point and together they are most of the
    // frame. A point the forward pass lost, or whose patch error is already
    // over the threshold, or that landed off the image, is going to be dropped
    // whatever comes back -- so the second call was paying full price for
    // points on their way to the bin. The gates below are the same gates in
    // the same conjunction, and the flow treats each point independently of
    // the others in the call, so the set that survives is unchanged to the
    // bit. Only which counter a doomed point lands in can move: one that fails
    // two gates is now charged to the first gate tested rather than to
    // whichever the old order happened to reach first.
    std::vector<cv::Point2f> candidates;
    std::vector<size_t> source;
    candidates.reserve(forward.size());
    source.reserve(forward.size());
    for (size_t i = 0; i < forward.size(); ++i) {
      if (status[i]) {
        raw += std::hypot(
          forward[i].x - state.points[i].x, forward[i].y - state.points[i].y);
        ++raw_count;
      }
      if (!status[i]) {
        ++lost;
        continue;
      }
      if (error[i] > error_threshold_) {
        ++noisy;
        continue;
      }
      if (forward[i].x < 0 || forward[i].y < 0 ||
        forward[i].x >= gray.cols || forward[i].y >= gray.rows)
      {
        continue;
      }
      candidates.push_back(forward[i]);
      source.push_back(i);
    }

    std::vector<cv::Point2f> backward;
    std::vector<uchar> backward_status;
    if (!candidates.empty()) {
      // No error vector out of this one. Nothing ever read it, and OpenCV
      // skips the patch sum altogether when it is not asked for -- a window's
      // worth of arithmetic per point, computed and discarded on every hop.
      cv::calcOpticalFlowPyrLK(
        pyramid, state.previous_pyramid, candidates, backward, backward_status,
        cv::noArray(), window, levels_, criteria);
    }

    for (size_t k = 0; k < candidates.size(); ++k) {
      const size_t i = source[k];
      if (!backward_status[k]) {
        ++lost;
        continue;
      }
      const cv::Point2f drift = backward[k] - state.points[i];
      if (std::hypot(drift.x, drift.y) > backward_threshold_) {
        ++drifted;
        continue;
      }
      previous_points.push_back(state.points[i]);
      current_points.push_back(candidates[k]);
      velocities.push_back(candidates[k] - state.points[i]);
      identities.push_back(state.identities[i]);
    }
    stats.raw = raw_count > 0 ? raw / static_cast<double>(raw_count) : 0.0;
    stats.lost = lost;
    stats.noisy = noisy;
    stats.drifted = drifted;
  }

#ifdef MONOSCALE_TRACKER_HAS_CUDA
  void build_device_pyramid(
    TrackState & state, const cv::Mat & gray,
    std::vector<cv::cuda::GpuMat> & pyramid) const
  {
    // The same ladder the flow builds for itself, kept so that two calls a
    // frame do not build four pyramids where two are needed.
    //
    // Staged through page-locked memory and queued on the camera's stream. The
    // pageable upload it replaces blocked until the copy had landed and could
    // overlap with nothing, which was half the cost of building the ladder.
    pyramid.resize(static_cast<size_t>(levels_) + 1);
    if (state.host_image.empty() || state.host_image.size() != gray.size() ||
      state.host_image.type() != gray.type())
    {
      state.host_image.create(gray.rows, gray.cols, gray.type());
    }
    cv::Mat staged = state.host_image.createMatHeader();
    gray.copyTo(staged);
    pyramid[0].upload(staged, state.stream);
    for (int level = 1; level <= levels_; ++level) {
      cv::cuda::pyrDown(
        pyramid[static_cast<size_t>(level - 1)],
        pyramid[static_cast<size_t>(level)], state.stream);
    }
  }

  // The GPU twin of follow(). Every gate below is the one above it, applied to
  // the same quantities: OpenCV's CUDA kernel reports the patch error as the
  // mean absolute intensity difference over the window, which is what the CPU
  // reports and what `lk_error_threshold` was measured against.
  //
  // The two do not agree to the bit and cannot: the pyramids are built
  // differently -- cv::buildOpticalFlowPyramid pads each level by the window
  // where cuda::pyrDown does not -- and the flow itself iterates in floats
  // where the CPU uses fixed point. What has to hold is that they track the
  // same features to within a fraction of a pixel, which is what the
  // comparison in tools measures.
  void follow_cuda(
    TrackState & state, const cv::Mat & gray,
    const std::vector<cv::cuda::GpuMat> & pyramid,
    std::vector<cv::Point2f> & previous_points,
    std::vector<cv::Point2f> & current_points,
    std::vector<cv::Point2f> & velocities,
    std::vector<int64_t> & identities, FollowStats & stats, double reach)
  {
    const size_t count = state.points.size();
    // The points already sit in a contiguous buffer of exactly this layout, so
    // staging them is one memcpy rather than a loop of element writes through
    // a Mat accessor.
    cv::Mat source = pinned(state.host_source, static_cast<int>(count), CV_32FC2);
    std::memcpy(source.ptr(), state.points.data(), count * sizeof(cv::Point2f));

    // Start each search where the feature was heading, the same warm start the
    // CPU path uses: the near ground sweeps fastest and carries most of the
    // metric information, so those are the points that fall into a wrong local
    // minimum without a guess.
    cv::Mat guess = pinned(state.host_guess, static_cast<int>(count), CV_32FC2);
    const bool warm = warm_start_ && state.velocities.size() == count;
    cv::Point2f * heading = guess.ptr<cv::Point2f>();
    for (size_t i = 0; i < count; ++i) {
      heading[i] = warm
        ? state.points[i] + state.velocities[i] * static_cast<float>(reach)
        : state.points[i];
    }

    // The whole hop is queued on the camera's stream and waited on once. Every
    // upload, flow call and download used to synchronise on its own -- five
    // stalls a frame where one will do, and on a 640 px frame that plumbing
    // cost more than the flow it was carrying.
    state.device_source.upload(source, state.stream);
    state.device_forward.upload(guess, state.stream);
    (warm ? forward_flow_ : backward_flow_)
    ->calc(
      state.previous_device_pyramid, pyramid, state.device_source,
      state.device_forward, state.device_status, state.device_error,
      state.stream);

    backward_flow_->calc(
      pyramid, state.previous_device_pyramid, state.device_forward,
      state.device_backward, state.device_backward_status, cv::noArray(),
      state.stream);

    cv::Mat forward = pinned(state.host_forward, static_cast<int>(count), CV_32FC2);
    cv::Mat backward = pinned(state.host_backward, static_cast<int>(count), CV_32FC2);
    cv::Mat status = pinned(state.host_status, static_cast<int>(count), CV_8UC1);
    cv::Mat backward_status =
      pinned(state.host_backward_status, static_cast<int>(count), CV_8UC1);
    state.device_forward.download(forward, state.stream);
    state.device_backward.download(backward, state.stream);
    state.device_status.download(status, state.stream);
    state.device_backward_status.download(backward_status, state.stream);
    cv::Mat error;
    if (!state.device_error.empty()) {
      error = pinned(state.host_error, static_cast<int>(count), CV_32FC1);
      state.device_error.download(error, state.stream);
    }
    state.stream.waitForCompletion();

    double raw = 0.0;
    size_t raw_count = 0;
    size_t lost = 0;
    size_t noisy = 0;
    size_t drifted = 0;

    for (size_t i = 0; i < count; ++i) {
      const int at = static_cast<int>(i);
      const cv::Point2f moved = forward.at<cv::Point2f>(0, at);
      const uchar kept = status.at<uchar>(0, at);
      if (kept) {
        raw += std::hypot(moved.x - state.points[i].x, moved.y - state.points[i].y);
        ++raw_count;
      }
      // The same gates in the same order as the CPU arm, so a rejection is
      // charged to the same counter on both. Compacting before the backward
      // call is what the CPU does with this order; here the flow has already
      // run over every point, since compacting on the device would cost a
      // pass of its own to save a pass that is already cheap.
      if (!kept) {
        ++lost;
        continue;
      }
      if (!error.empty() && error.at<float>(0, at) > error_threshold_) {
        ++noisy;
        continue;
      }
      if (moved.x < 0 || moved.y < 0 || moved.x >= gray.cols || moved.y >= gray.rows) {
        continue;
      }
      if (!backward_status.at<uchar>(0, at)) {
        ++lost;
        continue;
      }
      const cv::Point2f drift = backward.at<cv::Point2f>(0, at) - state.points[i];
      if (std::hypot(drift.x, drift.y) > backward_threshold_) {
        ++drifted;
        continue;
      }
      previous_points.push_back(state.points[i]);
      current_points.push_back(moved);
      velocities.push_back(moved - state.points[i]);
      identities.push_back(state.identities[i]);
    }
    stats.raw = raw_count > 0 ? raw / static_cast<double>(raw_count) : 0.0;
    stats.lost = lost;
    stats.noisy = noisy;
    stats.drifted = drifted;
  }
#endif

  // The same quota, applied to what survives rather than to what is added.
  //
  // Detection alone only bounds a cell that is being filled. Nothing bounds
  // one that is merely holding: the horizon's features survive every hop, so
  // the cells that filled at the start stayed full for the rest of the drive
  // while the emptying ground cells were topped up every frame. The set grew
  // to 52949 points and 211 ms a frame with no cap in sight. Trimming keeps
  // the oldest track in each cell -- the one the anchor map has the longest
  // history for, since follow preserves order -- and drops the surplus.
  void trim(
    const cv::Mat & gray,
    std::vector<cv::Point2f> & previous_points,
    std::vector<cv::Point2f> & current_points,
    std::vector<cv::Point2f> & velocities,
    std::vector<int64_t> & identities, int target) const
  {
    const int columns = std::max(grid_columns_, 1);
    const int rows = std::max(grid_rows_, 1);
    const int quota = std::max(target, 1) / (columns * rows) + 1;
    const double cell_width = static_cast<double>(gray.cols) / columns;
    const double cell_height = static_cast<double>(gray.rows) / rows;

    std::vector<int> occupancy(static_cast<size_t>(columns * rows), 0);
    size_t kept = 0;
    for (size_t i = 0; i < current_points.size(); ++i) {
      const int cx = std::clamp(
        static_cast<int>(current_points[i].x / cell_width), 0, columns - 1);
      const int cy = std::clamp(
        static_cast<int>(current_points[i].y / cell_height), 0, rows - 1);
      if (++occupancy[static_cast<size_t>(cy * columns + cx)] > quota) {
        continue;
      }
      previous_points[kept] = previous_points[i];
      current_points[kept] = current_points[i];
      velocities[kept] = velocities[i];
      identities[kept] = identities[i];
      ++kept;
    }
    previous_points.resize(kept);
    current_points.resize(kept);
    velocities.resize(kept);
    identities.resize(kept);
  }

  // Detection has to be told where to look, not only how many to find.
  //
  // A single global count as the refill gate let the rear camera collapse.
  // Its near ground sweeps about 28 px a frame against the front's 15; the
  // warm start here carries each feature's own last hop forward, which does
  // not predict a jump that size, so those points fail the backward check and
  // die. The horizon's points do not move and never die. Once the surviving
  // horizon band alone stood above refill_ratio_ * max_features_ detection
  // never ran again: 1548 points, every one of them within 28 rows of the
  // top, mean hop 0.03 px, for the rest of the drive. The camera was reported
  // as keeping 60 Hz and holding a full feature set while contributing
  // nothing to the ground registration.
  //
  // So the quota is per cell, and so is the refill gate. A cell that has lost
  // its features is refilled whatever the rest of the frame is holding, and
  // the quality threshold is measured against its own cell rather than the
  // strongest corner in the image, which is what lets the near ground compete
  // with the horizon at all. This is what ov_core's Grider_GRID does.
  void detect(TrackState & state, const cv::Mat & gray) const
  {
    // Per-cell quota and a per-cell refill gate, which is ov_core's
    // Grider_GRID: keeping the strongest responses in each cell rather than
    // the strongest in the frame is what stops the near ground losing to the
    // horizon, and a global count as the gate let the rear camera freeze on
    // a stationary band for a whole drive.
    //
    // The detector itself is not ov_core's. Grider_FAST was ported and
    // measured: it is cheaper -- 2.2 ms a frame against 3.5, and 6 ms for the
    // whole frame against 9 -- and less accurate here, 0.036-0.094 m against
    // 0.038-0.042 over the same drive, with the scale reading long. FAST
    // takes corners above an absolute intensity threshold, and on a low
    // contrast road surface those are weaker and less repeatable than
    // minimum-eigenvalue corners; this estimate is a ground plane fitted to
    // exactly these pixels, so their placement is the measurement. Adding
    // cornerSubPix on top, as TrackKLT does, cost 5 ms and bought nothing.
    const int columns = std::max(grid_columns_, 1);
    const int rows = std::max(grid_rows_, 1);
    const int quota = std::max(state.target, 1) / (columns * rows) + 1;
    const double cell_width = static_cast<double>(gray.cols) / columns;
    const double cell_height = static_cast<double>(gray.rows) / rows;

    std::vector<int> occupancy(static_cast<size_t>(columns * rows), 0);
    for (const auto & point : state.points) {
      const int cx = std::clamp(static_cast<int>(point.x / cell_width), 0, columns - 1);
      const int cy = std::clamp(static_cast<int>(point.y / cell_height), 0, rows - 1);
      ++occupancy[static_cast<size_t>(cy * columns + cx)];
    }

    // Which cells are being refilled, settled before the mask is built.
    //
    // The mask is only ever read through a refilled cell's rectangle. Painting
    // it over the whole frame and stamping every held feature into it is work
    // the skipped cells never look at, and on a frame where nothing needs
    // refilling it is the entire cost of this stage, paid to be thrown away.
    // Once the grid is holding its quota -- which is the steady state, since
    // the point of the quota is to reach it -- that is every frame.
    std::vector<cv::Rect> refill;
    std::vector<int> room;
    for (int cy = 0; cy < rows; ++cy) {
      for (int cx = 0; cx < columns; ++cx) {
        const int held = occupancy[static_cast<size_t>(cy * columns + cx)];
        if (held > refill_ratio_ * quota) {
          continue;
        }
        const int x0 = static_cast<int>(cx * cell_width);
        const int y0 = static_cast<int>(cy * cell_height);
        const int x1 = cx + 1 == columns
          ? gray.cols : static_cast<int>((cx + 1) * cell_width);
        const int y1 = cy + 1 == rows
          ? gray.rows : static_cast<int>((cy + 1) * cell_height);
        const cv::Rect cell(x0, y0, x1 - x0, y1 - y0);
        if (cell.width < 7 || cell.height < 7) {
          continue;
        }
        refill.push_back(cell);
        room.push_back(quota - held);
      }
    }
    if (refill.empty()) {
      return;
    }

    // Left uninitialised on purpose: every pixel that will be read is written
    // below, first by the cell fill and then by the exclusion discs.
    cv::Mat mask(gray.size(), CV_8UC1);
    for (const auto & cell : refill) {
      mask(cell).setTo(255);
    }
    if (skip_top_ > 0.0) {
      const int cut = std::min(
        static_cast<int>(skip_top_ * gray.rows), gray.rows - 1);
      if (cut > 0) {
        const cv::Rect band(0, 0, gray.cols, cut);
        for (const auto & cell : refill) {
          const cv::Rect hidden = cell & band;
          if (hidden.area() > 0) {
            mask(hidden).setTo(0);
          }
        }
      }
    }
    // A held feature only has to be stamped where it can actually block a
    // detection, which is the refilled cells its exclusion disc reaches. The
    // horizon band is what makes this worth doing: those features survive
    // every hop, so their cells stop refilling, and they are exactly the ones
    // that were being drawn every frame into a mask nobody read.
    const int radius = static_cast<int>(min_distance_);
    for (const auto & point : state.points) {
      const cv::Rect disc(
        cvRound(point.x) - radius, cvRound(point.y) - radius,
        2 * radius + 1, 2 * radius + 1);
      for (const auto & cell : refill) {
        if ((disc & cell).area() > 0) {
          cv::circle(mask, point, radius, 0, -1);
          break;
        }
      }
    }

    // What the flow is doing where each refilled cell sits, taken from the
    // features that survived into it. A feature seen once has no hop of its
    // own, and at road speed most features have been seen once: the ground
    // sweeps 15 to 22 px a frame at 8 m/s and identities last one or two
    // frames, so the warm start it was meant to get never exists. Its
    // neighbours' does, and the ground flow field is smooth enough to borrow.
    // What the flow is doing where each refilled cell sits, taken from the
    // features that survived into it. A feature seen once has no hop of its
    // own, and at road speed most features have been seen once: the ground
    // sweeps 15 to 22 px a frame at 8 m/s and identities last one or two
    // frames, so the warm start it was meant to get never exists. Its
    // neighbours' does, and the ground flow field is smooth enough to borrow.
    std::vector<cv::Point2f> seed(refill.size(), cv::Point2f(0.0f, 0.0f));
    if (seed_new_from_cell_ && state.velocities.size() == state.points.size() &&
      !state.points.empty())
    {
      const auto middle = [](std::vector<float> & v) {
          if (v.empty()) {
            return 0.0f;
          }
          const size_t half = v.size() / 2;
          std::nth_element(v.begin(), v.begin() + half, v.end());
          return v[half];
        };
      std::vector<float> xs;
      std::vector<float> ys;
      xs.reserve(state.points.size());
      ys.reserve(state.points.size());
      for (const auto & step : state.velocities) {
        xs.push_back(std::hypot(step.x, step.y));
      }
      // Decided for the whole frame, not per cell. Seeding only the fast cells
      // was measured worse at 8 m/s than seeding all of them or none: what the
      // consensus behind the solve wants is one population, and half a frame
      // carrying a neighbour's hop while the other half starts from zero is
      // two. Below a few pixels the search finds the right minimum from where
      // the feature already is anyway.
      if (middle(xs) >= seed_min_flow_) {
        xs.clear();
        ys.clear();
        for (const auto & step : state.velocities) {
          xs.push_back(step.x);
          ys.push_back(step.y);
        }
        const cv::Point2f overall(middle(xs), middle(ys));
        for (size_t c = 0; c < refill.size(); ++c) {
          xs.clear();
          ys.clear();
          for (size_t i = 0; i < state.points.size(); ++i) {
            if (refill[c].contains(state.points[i])) {
              xs.push_back(state.velocities[i].x);
              ys.push_back(state.velocities[i].y);
            }
          }
          // A cell that kept nothing borrows the frame rather than guess zero.
          seed[c] = xs.empty() ? overall : cv::Point2f(middle(xs), middle(ys));
        }
      }
    }

    std::vector<cv::Point2f> found;
    for (size_t c = 0; c < refill.size(); ++c) {
      const cv::Rect & cell = refill[c];
      found.clear();
      cv::goodFeaturesToTrack(
        gray(cell), found, room[c], quality_level_, min_distance_,
        mask(cell), 7);
      for (const auto & point : found) {
        state.points.emplace_back(point.x + cell.x, point.y + cell.y);
        // The vectors have to stay the same length or the warm start silently
        // switches itself off.
        state.velocities.push_back(seed[c]);
        state.identities.push_back(state.next_identity++);
      }
    }
  }

  void publish(
    const std::string & name, const sensor_msgs::msg::Image & message,
    const cv::Size & frame,
    const std::vector<cv::Point2f> & previous_points,
    const std::vector<cv::Point2f> & current_points,
    const std::vector<int64_t> & identities)
  {
    std_msgs::msg::Float64MultiArray out;
    out.data.reserve(4 + identities.size() * 5);
    out.data.push_back(
      static_cast<float>(message.header.stamp.sec) +
      static_cast<float>(message.header.stamp.nanosec) * 1e-9f);
    out.data.push_back(static_cast<float>(identities.size()));
    out.data.push_back(static_cast<float>(frame.width));
    out.data.push_back(static_cast<float>(frame.height));
    for (size_t i = 0; i < identities.size(); ++i) {
      out.data.push_back(static_cast<float>(identities[i]));
      out.data.push_back(previous_points[i].x);
      out.data.push_back(previous_points[i].y);
      out.data.push_back(current_points[i].x);
      out.data.push_back(current_points[i].y);
    }
    publishers_[name]->publish(std::move(out));
    {
      // Counted at the writer itself, so the estimator's tally can be
      // compared against it without a third subscriber in the middle
      // changing what the middleware does.
      std::lock_guard<std::mutex> guard(count_mutex_);
      published_[name] += 1;
    }
  }

  int processing_width_;
  bool road_alignment_ = false;
  std::array<double, 4> road_roi_{{0.2, 0.5, 0.8, 1.0}};
  int road_columns_ = 3;
  int road_rows_ = 3;
  int road_iterations_ = 30;
  bool road_homography_ = false;
  int road_max_age_ = 0;
  double road_epsilon_ = 1.0e-4;
  // Virtual road features live in their own identity range so they can never
  // collide with a corner's. 2^40 is far above anything the corner counter can
  // reach -- it mints of the order of a million an hour -- and far below the
  // 2^53 the float64 message carries exactly.
  static constexpr int64_t kRoadIdentityBase = 1LL << 40;
  int64_t road_next_identity_ = kRoadIdentityBase;
  int max_features_;
  double quality_level_ = 0.01;
  double min_distance_;
  int window_;
  int levels_;
  double backward_threshold_;
  double error_threshold_;
  double refill_ratio_;
  bool warm_start_;
  bool predict_by_plane_ = false;
  bool seed_new_from_cell_ = false;
  float seed_min_flow_ = 8.0f;
  bool use_cuda_ = false;
  int cuda_iterations_ = 30;
  // Whether the GPU path is actually in use, which needs the parameter, an
  // OpenCV that has cudaoptflow, and a device to run on.
  bool cuda_active_ = false;
#ifdef MONOSCALE_TRACKER_HAS_CUDA
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> forward_flow_;
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> backward_flow_;
#endif
  double skip_top_;
  // How many frames each camera actually handed this node, and how many
  // tracked sets went back out. Counted here because a bag recorder running
  // beside a live stack competes for the same CPU and drops what it is meant
  // to be measuring.
  std::map<std::string, int> received_;
  std::map<std::string, double> spent_;
  std::map<std::string, double> held_;
  std::map<std::string, double> shift_;
  std::map<std::string, double> followed_;
  std::map<std::string, double> change_;
  std::map<std::string, double> prep_;
  std::map<std::string, double> follow_;
  std::map<std::string, double> trim_;
  std::map<std::string, double> detect_;
  std::map<std::string, double> pub_;
  std::map<std::string, double> raw_;
  std::map<std::string, double> lost_;
  std::map<std::string, double> noisy_;
  std::map<std::string, double> drifted_;
  double frame_budget_ms_ = 14.0;
  int min_features_ = 500;
  int grid_columns_ = 4;
  int grid_rows_ = 3;
  std::string dump_dir_;
  std::map<std::string, int> dumped_;
  rclcpp::Time counted_since_;
  bool counting_ = false;
  std::mutex count_mutex_;
  std::map<std::string, long> published_;
  std::vector<rclcpp::CallbackGroup::SharedPtr> groups_;
  std::map<std::string, TrackState> states_;
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr>
  publishers_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscriptions_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FeatureTracker>();
  // Settable so the count can follow the number of cameras. Four measured
  // the same as two against a 60 Hz replay (front 26-39 Hz, rear 57-60 either
  // way), so the asymmetry between the two cameras is not thread starvation.
  // Each camera already has its own callback group; more threads than that
  // buy nothing here.
  const int threads = static_cast<int>(
    node->declare_parameter<int>("executor_threads", 4));
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), static_cast<size_t>(std::max(threads, 1)));
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
