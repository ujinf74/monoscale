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
// Published layout, one message per camera, flat float32:
//   [stamp_sec, count, width, height, id0, prev_x0, prev_y0, cur_x0, cur_y0, ...]
// Identities persist for as long as a feature is followed, which is what lets
// the estimator match a sighting against the anchor it built earlier.
//
// The frame size is in the message because the pixels are measured in the
// downscaled frame, and the estimator has to scale the intrinsics to match.
// Leaving it to both sides reading the same processing_width parameter makes
// a silent geometry error out of a configuration mismatch.

#include <algorithm>
#include <chrono>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace
{

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
  std::vector<int64_t> identities;
  int64_t next_identity = 0;
  // When the last frame arrived and how long the hop before it took. A
  // dropped frame doubles the gap, and a prediction that does not know that
  // is wrong by a whole frame of motion.
  double stamp = 0.0;
  double interval = 0.0;
  // How many features this camera is currently trying to hold. Owned by the
  // one thread that runs this camera's callback, so it needs no guard.
  int target = 0;
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
      states_[name] = TrackState{};
      states_[name].target = max_features_;
      publishers_[name] = create_publisher<std_msgs::msg::Float32MultiArray>(
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
    try {
      gray = cv_bridge::toCvCopy(message, "mono8")->image;
    } catch (const cv_bridge::Exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "cv_bridge: %s", error.what());
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

    std::vector<cv::Mat> pyramid;
    cv::buildOpticalFlowPyramid(gray, pyramid, cv::Size(window_, window_), levels_);
    stage.prep = lap();

    TrackState & state = states_[name];
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

    if (!state.previous_pyramid.empty() && !state.points.empty() &&
      state.previous_gray.size() == gray.size())
    {
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

    state.points = current_points;
    state.velocities = velocities;
    state.identities = identities;
    const size_t followed = state.points.size();
    detect(state, gray);
    stage.detect = lap();
    state.previous_gray = gray;
    state.previous_pyramid = pyramid;

    // Features detected on this frame go out too, standing still: previous
    // position equal to current, because they have not moved yet. The image
    // path puts them into the tracked set the moment it finds them, so they
    // are in the snapshot a solve compares against and survive the whole of
    // the next interval. Holding them back for one frame left this path
    // solving over a pool that was always a generation older -- median
    // baseline 0.272 m against 0.389 -- and the shorter the baseline the
    // noisier the geometry.
    std::vector<cv::Point2f> published_previous = previous_points;
    std::vector<cv::Point2f> published_current = current_points;
    std::vector<int64_t> published_identities = identities;
    for (size_t i = followed; i < state.points.size(); ++i) {
      published_previous.push_back(state.points[i]);
      published_current.push_back(state.points[i]);
      published_identities.push_back(state.identities[i]);
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
      name, message, gray.size(), published_previous, published_current,
      published_identities);

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
    if (warm_start_ && state.velocities.size() == state.points.size()) {
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

    std::vector<cv::Point2f> backward;
    std::vector<uchar> backward_status;
    std::vector<float> backward_error;
    cv::calcOpticalFlowPyrLK(
      pyramid, state.previous_pyramid, forward, backward, backward_status,
      backward_error, window, levels_, criteria);

    // What the flow returned before anything was thrown away, and why each
    // rejection happened. The survivors' mean hop alone cannot tell a camera
    // that is not moving apart from one whose moving features are all being
    // filtered out.
    double raw = 0.0;
    size_t raw_count = 0;
    size_t lost = 0;
    size_t noisy = 0;
    size_t drifted = 0;

    for (size_t i = 0; i < forward.size(); ++i) {
      if (status[i]) {
        raw += std::hypot(
          forward[i].x - state.points[i].x, forward[i].y - state.points[i].y);
        ++raw_count;
      }
      if (!status[i] || !backward_status[i]) {
        ++lost;
        continue;
      }
      if (error[i] > error_threshold_) {
        ++noisy;
        continue;
      }
      const cv::Point2f drift = backward[i] - state.points[i];
      if (std::hypot(drift.x, drift.y) > backward_threshold_) {
        ++drifted;
        continue;
      }
      if (forward[i].x < 0 || forward[i].y < 0 ||
        forward[i].x >= gray.cols || forward[i].y >= gray.rows)
      {
        continue;
      }
      previous_points.push_back(state.points[i]);
      current_points.push_back(forward[i]);
      velocities.push_back(forward[i] - state.points[i]);
      identities.push_back(state.identities[i]);
    }
    stats.raw = raw_count > 0 ? raw / static_cast<double>(raw_count) : 0.0;
    stats.lost = lost;
    stats.noisy = noisy;
    stats.drifted = drifted;
  }

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

    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
    if (skip_top_ > 0.0) {
      const int cut = std::min(
        static_cast<int>(skip_top_ * gray.rows), gray.rows - 1);
      if (cut > 0) {
        mask(cv::Rect(0, 0, gray.cols, cut)).setTo(0);
      }
    }
    for (const auto & point : state.points) {
      cv::circle(mask, point, static_cast<int>(min_distance_), 0, -1);
    }

    std::vector<cv::Point2f> found;
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
        found.clear();
        cv::goodFeaturesToTrack(
          gray(cell), found, quota - held, quality_level_, min_distance_,
          mask(cell), 7);
        for (const auto & point : found) {
          state.points.emplace_back(point.x + x0, point.y + y0);
          // A feature seen once has no hop behind it; it starts its first
          // search where it is. The vectors have to stay the same length or
          // the warm start silently switches itself off.
          state.velocities.emplace_back(0.0f, 0.0f);
          state.identities.push_back(state.next_identity++);
        }
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
    std_msgs::msg::Float32MultiArray out;
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
    publishers_[name]->publish(out);
    {
      // Counted at the writer itself, so the estimator's tally can be
      // compared against it without a third subscriber in the middle
      // changing what the middleware does.
      std::lock_guard<std::mutex> guard(count_mutex_);
      published_[name] += 1;
    }
  }

  int processing_width_;
  int max_features_;
  double quality_level_ = 0.01;
  double min_distance_;
  int window_;
  int levels_;
  double backward_threshold_;
  double error_threshold_;
  double refill_ratio_;
  bool warm_start_;
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
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr>
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
