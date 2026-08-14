#include "monoscale_core/estimator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace monoscale
{

namespace
{

class Stopwatch
{
public:
  Stopwatch(Diagnostics & diagnostics, const char * stage)
  : total_(diagnostics.stage_seconds[stage]),
    began_(std::chrono::steady_clock::now()) {}

  ~Stopwatch()
  {
    total_ += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - began_).count();
  }

private:
  double & total_;
  std::chrono::steady_clock::time_point began_;
};

double percentile_90(const Points2 & from, const Points2 & to)
{
  const Eigen::Index count = from.rows();
  if (count == 0) {
    return 0.0;
  }
  std::vector<double> shift(static_cast<size_t>(count));
  for (Eigen::Index i = 0; i < count; ++i) {
    shift[static_cast<size_t>(i)] =
      std::hypot(to(i, 0) - from(i, 0), to(i, 1) - from(i, 1));
  }
  // numpy's linear interpolation between the two straddling samples, so the
  // trigger reads the same figure the Python did.
  const double position = 0.9 * static_cast<double>(count - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = std::min(lower + 1, shift.size() - 1);
  std::nth_element(shift.begin(), shift.begin() + lower, shift.end());
  const double low = shift[lower];
  const double high = upper == lower
    ? low
    : *std::min_element(shift.begin() + lower + 1, shift.end());
  return low + (position - static_cast<double>(lower)) * (high - low);
}

}  // namespace

FusionModel fusion_model_from_name(const std::string & name)
{
  if (name == "velocity") {
    return FusionModel::Velocity;
  }
  if (name == "msckf") {
    return FusionModel::Msckf;
  }
  if (name == "msckf6") {
    return FusionModel::Msckf6;
  }
  return FusionModel::Displacement;
}

struct Estimator::Frame
{
  double stamp = 0.0;
  // How far the ground moved on the hop that produced this frame. Held until
  // the frame is consumed so the solve trigger sees the same signal at the
  // same moment however the frames were queued.
  double disparity = 0.0;
  Points2 pixels;
  Identities ids;
};

struct Estimator::Camera
{
  explicit Camera(const CameraSettings & from, const AnchorSettings & anchor_settings)
  : settings(from), anchors(anchor_settings)
  {
    calibration = make_camera_model(
      from.k, from.rotation_base_from_camera, from.translation_base_from_camera,
      from.distortion, from.lens);
    calibration_width = from.calibration_width;
    calibration_height = from.calibration_height;
    model = calibration;
  }

  CameraSettings settings;
  // As the calibration reports it, at its own resolution.
  CameraModel calibration;
  int calibration_width = 0;
  int calibration_height = 0;
  // Brought to the frame the tracker actually measured pixels in.
  CameraModel model;

  std::deque<Frame> queue;
  std::optional<Frame> latest;

  Points2 track_pixels;
  Identities track_ids;

  // Where each surviving feature sat, and which frame that was, the last time
  // a pose was solved. Tracking runs every frame so each hop stays small, but
  // the solve compares against this instead of the frame before it.
  Identities solve_ids;
  Points2 solve_points;
  std::optional<Frame> solve_frame;

  // Pixel travel accumulated over the single-frame hops since the last solve,
  // where the near ground is still in view.
  double hop_disparity = 0.0;

  GroundAnchorMap anchors;

  // How this camera's ground projection has been leaning, summed over the
  // solves the map answered. Kept here rather than in the diagnostics because
  // the solve does not know which camera it is.
  double radial_linear_sum = 0.0;
  int64_t radial_samples = 0;

  // Per feature, the ground projection and camera position from when it was
  // first seen, so each earns its own baseline over as many frames as it
  // survives rather than sharing one keyframe. Sorted by identity.
  Identities slip_ids;
  Points2 slip_world;
  Points2 slip_camera;
  std::vector<int32_t> slip_misses;
};

struct Estimator::Solved
{
  Points2 previous_ground;
  Points2 current_ground;
  Points2 current_pixels;
  Mask ground_valid;
  Mask motion_inliers;
  Identities track_ids;
  std::optional<PlanarMotion> motion;
  bool anchored_from_map = false;
  double spread = 0.0;
  // How well this camera pinned the heading down, when it was asked to solve
  // for one. Infinite when it was not.
  double yaw_sigma = std::numeric_limits<double>::infinity();
};

Estimator::Estimator(const EstimatorSettings & settings)
: settings_(settings),
  heading_(
    settings.gyro_bias_sigma_rad_s, settings.gyro_bias_walk_sigma_rad_s,
    settings.gyro_noise_sigma_rad_s),
  inertial_(
    [&settings]() {
      PlanarInertialPropagator::Settings inertial;
      inertial.max_horizontal_acceleration = settings.inertial_max_acceleration_mps2;
      inertial.median_window = settings.inertial_acceleration_median_window;
      inertial.integration = settings.inertial_integration;
      return inertial;
    } ()),
  velocity_filter_(
    [&settings]() {
      PlanarVelocityFilter::Settings filter;
      filter.acceleration_noise = settings.filter_acceleration_noise;
      filter.vision_noise = settings.filter_vision_noise;
      filter.vision_reference_inliers = settings.filter_reference_inliers;
      filter.innovation_gate = settings.filter_innovation_gate;
      return filter;
    } ())
{
  AnchorSettings anchor_settings;
  anchor_settings.max_anchors = settings.max_ground_anchors;
  anchor_settings.max_age_frames = settings.anchor_max_age_frames;
  anchor_settings.update_gain = settings.anchor_update_gain;
  anchor_settings.max_observations = settings.anchor_max_observations;
  anchor_settings.select_by_consistency = settings.anchor_select_by_consistency;

  for (const auto & camera : settings.cameras) {
    cameras_.push_back(std::make_unique<Camera>(camera, anchor_settings));
  }

  if (settings.attitude_from_imu) {
    attitude_ = std::make_unique<AttitudeFilter>(
      settings.attitude_tau_sec, settings.attitude_gravity_tolerance);
  }
  if (settings.fusion_model == FusionModel::Displacement) {
    PlanarDisplacementFilter::Settings filter;
    filter.acceleration_noise = settings.filter_acceleration_noise;
    filter.bias_walk = settings.filter_bias_walk;
    filter.vision_noise_m = settings.filter_vision_noise_m;
    filter.vision_reference_inliers = settings.filter_reference_inliers;
    filter.innovation_gate = settings.filter_innovation_gate;
    displacement_filter_ = std::make_unique<PlanarDisplacementFilter>(filter);
  }
  if (settings.fusion_model == FusionModel::Msckf) {
    PlanarMsckfFilter::Settings filter;
    filter.acceleration_noise = settings.filter_acceleration_noise;
    filter.bias_walk = settings.filter_bias_walk;
    filter.vision_noise_m = settings.filter_vision_noise_m;
    filter.vision_reference_inliers = settings.filter_reference_inliers;
    filter.gyro_noise = settings.msckf_gyro_noise;
    filter.gyro_bias_walk = settings.msckf_gyro_bias_walk;
    filter.vision_yaw_noise = settings.msckf_vision_yaw_noise;
    filter.initial_gyro_bias_variance = settings.msckf_initial_gyro_bias_variance;
    filter.innovation_gate = settings.msckf_innovation_gate;
    filter.reject_beyond_m = settings.msckf_reject_beyond_m;
    filter.heading_adaptive_gain = settings.msckf_heading_adaptive_gain;
    filter.heading_adaptive_window = settings.msckf_heading_adaptive_window;
    msckf_filter_ = std::make_unique<PlanarMsckfFilter>(filter);
  }
  if (settings.fusion_model == FusionModel::Msckf6) {
    SpatialMsckfFilter::Settings filter;
    filter.acceleration_noise = settings.filter_acceleration_noise;
    filter.bias_walk = settings.filter_bias_walk;
    filter.vision_noise_m = settings.filter_vision_noise_m;
    filter.vision_reference_inliers = settings.filter_reference_inliers;
    filter.gyro_noise = settings.msckf_gyro_noise;
    filter.gyro_bias_walk = settings.msckf_gyro_bias_walk;
    filter.vision_yaw_noise = settings.msckf_vision_yaw_noise;
    filter.initial_gyro_bias_variance = settings.msckf_initial_gyro_bias_variance;
    filter.reject_beyond_m = settings.msckf_reject_beyond_m;
    filter.gravity_tolerance = settings.spatial_gravity_tolerance;
    filter.gravity_noise = settings.spatial_gravity_noise;
    filter.tilt_gyro_noise = settings.spatial_tilt_gyro_noise;
    filter.height_noise_m = settings.spatial_height_noise_m;
    filter.innovation_gate = settings.spatial_innovation_gate;
    filter.initial_scale_variance = settings.spatial_scale_variance;
    filter.initial_bias_variance = settings.spatial_bias_variance;
    filter.initial_tilt_variance = settings.spatial_tilt_variance;
    spatial_filter_ = std::make_unique<SpatialMsckfFilter>(filter);
  }
  map_ready_ = !settings.require_map_before_translating;
}

Estimator::~Estimator() = default;

void Estimator::set_calibration(
  size_t camera, const Eigen::Matrix3d & k, const Eigen::VectorXd & distortion,
  Lens lens, int width, int height)
{
  if (camera >= cameras_.size() || width <= 0 || height <= 0 || k(0, 0) <= 0.0) {
    return;
  }
  Camera & target = *cameras_[camera];
  target.calibration = make_camera_model(
    k, target.calibration.rotation_base_from_camera,
    target.calibration.translation_base_from_camera, distortion, lens);
  target.calibration_width = width;
  target.calibration_height = height;
}

void Estimator::set_mount(
  size_t camera, const Eigen::Matrix3d & rotation, const Eigen::Vector3d & translation)
{
  if (camera >= cameras_.size()) {
    return;
  }
  Camera & target = *cameras_[camera];
  target.calibration.rotation_base_from_camera = rotation;
  target.calibration.translation_base_from_camera = translation;
  target.model.rotation_base_from_camera = rotation;
  target.model.translation_base_from_camera = translation;
}

CameraModel Estimator::frame_model(const Camera & camera, int width, int height) const
{
  // The calibration describes the camera at its own resolution while the
  // estimator works on a downscaled frame. Pairing one with the other is not a
  // clean scale error -- the principal point moves as well, so the whole image
  // projects to a shifted patch of ground -- and it is the kind of mistake that
  // leaves the scale looking almost right.
  Eigen::Matrix3d k = camera.calibration.k;
  if (camera.calibration_width > 0 && camera.calibration_height > 0) {
    k.row(0) *= static_cast<double>(width) / camera.calibration_width;
    k.row(1) *= static_cast<double>(height) / camera.calibration_height;
  }
  return make_camera_model(
    k, camera.calibration.rotation_base_from_camera,
    camera.calibration.translation_base_from_camera, camera.calibration.distortion,
    camera.calibration.lens);
}

void Estimator::ingest_tracks(size_t index, const TrackFrame & incoming)
{
  if (index >= cameras_.size() || incoming.width <= 0 || incoming.height <= 0 ||
    incoming.stamp <= 0.0)
  {
    return;
  }
  Camera & camera = *cameras_[index];
  // The front end sends pixels in its own downscaled frame, so the intrinsics
  // have to be brought to that frame before anything is projected.
  camera.model = frame_model(camera, incoming.width, incoming.height);

  Frame frame;
  frame.stamp = incoming.stamp;
  frame.pixels = incoming.pixels;
  frame.ids = incoming.ids;
  frame.disparity = percentile_90(incoming.previous_pixels, incoming.pixels);

  camera.queue.push_back(std::move(frame));
  while (static_cast<int>(camera.queue.size()) > settings_.pair_queue_depth) {
    camera.queue.pop_front();
    ++diagnostics_.frames_evicted;
  }

  // Drain here as well as on the IMU. Waiting for the next inertial sample
  // costs whatever arrives after the last one -- two pairs at the end of a
  // recording, and on a vehicle whatever the IMU drops.
  try_process_pairs();
}

void Estimator::ingest_imu(const ImuSample & sample)
{
  if (sample.stamp <= 0.0) {
    return;
  }
  const Eigen::Vector4d & q = sample.orientation;
  const double yaw = std::atan2(
    2.0 * (q(3) * q(2) + q(0) * q(1)),
    1.0 - 2.0 * (q(1) * q(1) + q(2) * q(2)));
  if (!imu_yaw_datum_.has_value()) {
    imu_yaw_datum_ = yaw;
  }
  imu_yaw_samples_.emplace_back(sample.stamp, yaw);
  while (imu_yaw_samples_.size() > 400) {
    imu_yaw_samples_.pop_front();
  }

  if (attitude_) {
    const double step = imu_stamp_.has_value() ? sample.stamp - *imu_stamp_ : 0.0;
    if (step >= 0.0 && step <= 0.1) {
      attitude_->update(sample.angular_velocity, sample.linear_acceleration, step);
    }
    imu_stamp_ = sample.stamp;
  }

  if (settings_.zero_velocity_update) {
    imu_motion_samples_.push_back(
      MotionSample{
        sample.stamp, sample.angular_velocity.norm(), sample.linear_acceleration.norm()});
    while (imu_motion_samples_.size() > 4000) {
      imu_motion_samples_.pop_front();
    }
  }

  if (settings_.use_inertial_prediction) {
    const auto step = inertial_.add_sample(
      sample.stamp, sample.orientation, sample.linear_acceleration);
    Eigen::Vector2d acceleration = step.acceleration;
    if (!settings_.inertial_use_acceleration) {
      acceleration.setZero();
    }
    velocity_filter_.predict(acceleration, step.dt);
    if (displacement_filter_ || msckf_filter_ || spatial_filter_) {
      // The same acceleration seen from the body, for the filter that carries
      // its own heading and must not be handed a vector already rotated by the
      // instrument's idea of which way the vehicle points.
      const double c = std::cos(yaw);
      const double s = std::sin(yaw);
      const Eigen::Vector2d body(
        c * acceleration.x() + s * acceleration.y(),
        -s * acceleration.x() + c * acceleration.y());
      // The same screen the propagator applies, kept in the body frame so the
      // filter that owns its own attitude is not handed a vector somebody
      // else's attitude decided about. Gravity is along z on a road vehicle, so
      // what is left in x and y is what the vehicle is doing.
      Eigen::Vector3d force = sample.linear_acceleration;
      bool screened = false;
      if (settings_.inertial_max_acceleration_mps2 > 0.0 &&
        force.head<2>().norm() > settings_.inertial_max_acceleration_mps2)
      {
        screened = true;
        force = last_specific_force_.value_or(Eigen::Vector3d(0.0, 0.0, 9.80665));
      } else {
        last_specific_force_ = force;
      }
      imu_window_.push_back(
        AccelerationSample{
          sample.stamp + settings_.imu_acceleration_offset_sec, acceleration, body,
          sample.angular_velocity.z(), force, sample.angular_velocity, screened,
          step.dt, yaw});
      while (imu_window_.size() > 8000) {
        imu_window_.pop_front();
      }
    }
  }

  try_process_pairs();
}

std::vector<Update> Estimator::take_updates()
{
  std::vector<Update> updates;
  updates.swap(pending_updates_);
  return updates;
}

bool Estimator::imu_still_arriving(double stamp) const
{
  if (!settings_.use_imu_yaw || imu_yaw_samples_.empty()) {
    return false;
  }
  return imu_yaw_samples_.back().first < stamp;
}

std::optional<double> Estimator::imu_yaw_at(double stamp) const
{
  if (imu_yaw_samples_.empty()) {
    return std::nullopt;
  }
  // Taking the nearest sample only works while the IMU runs far faster than
  // the cameras. Attached to an externally driven session every sensor drops
  // to the server frame rate, and nearest sample threw away two thirds of the
  // frames.
  const std::pair<double, double> * before = nullptr;
  const std::pair<double, double> * after = nullptr;
  for (const auto & sample : imu_yaw_samples_) {
    if (sample.first <= stamp) {
      before = &sample;
    } else {
      after = &sample;
      break;
    }
  }
  if (before != nullptr && after != nullptr) {
    const double span = after->first - before->first;
    if (span > 0.0 && span <= settings_.imu_max_gap_sec) {
      const double ratio = (stamp - before->first) / span;
      const double step = std::remainder(after->second - before->second, 2.0 * M_PI);
      return before->second + ratio * step;
    }
  }
  const std::pair<double, double> * nearest = &imu_yaw_samples_.front();
  for (const auto & sample : imu_yaw_samples_) {
    if (std::abs(sample.first - stamp) < std::abs(nearest->first - stamp)) {
      nearest = &sample;
    }
  }
  if (std::abs(nearest->first - stamp) <= settings_.imu_max_age_sec) {
    return nearest->second;
  }
  return std::nullopt;
}

void Estimator::try_process_pairs()
{
  const size_t count = cameras_.size();
  if (count == 0) {
    return;
  }
  while (true) {
    for (const auto & camera : cameras_) {
      if (camera->queue.empty()) {
        return;
      }
    }

    // One frame per camera, picked so their stamps sit closest together. Every
    // queued frame of every camera gets a turn as the reference and the others
    // answer with whichever frame lands nearest it; the reference whose answers
    // spread least wins.
    double best_spread = std::numeric_limits<double>::infinity();
    std::vector<size_t> best_picks(count, 0);
    std::vector<double> best_stamps(count, 0.0);
    std::vector<size_t> picks(count, 0);
    std::vector<double> stamps(count, 0.0);
    for (size_t reference = 0; reference < count; ++reference) {
      for (const auto & frame : cameras_[reference]->queue) {
        for (size_t other = 0; other < count; ++other) {
          const auto & queue = cameras_[other]->queue;
          size_t chosen = 0;
          double closest = std::numeric_limits<double>::infinity();
          for (size_t i = 0; i < queue.size(); ++i) {
            const double distance = std::abs(queue[i].stamp - frame.stamp);
            if (distance < closest) {
              closest = distance;
              chosen = i;
            }
          }
          picks[other] = chosen;
          stamps[other] = queue[chosen].stamp;
        }
        const double spread =
          *std::max_element(stamps.begin(), stamps.end()) -
          *std::min_element(stamps.begin(), stamps.end());
        if (spread < best_spread) {
          best_spread = spread;
          best_picks = picks;
          best_stamps = stamps;
        }
      }
    }

    if (best_spread > settings_.sync_tolerance_sec) {
      // No alignment inside tolerance: the oldest head is the one with no peer
      // coming, so it goes and the rest get another chance.
      size_t oldest = 0;
      for (size_t i = 1; i < count; ++i) {
        if (cameras_[i]->queue.front().stamp < cameras_[oldest]->queue.front().stamp) {
          oldest = i;
        }
      }
      cameras_[oldest]->hop_disparity += cameras_[oldest]->queue.front().disparity;
      cameras_[oldest]->queue.pop_front();
      continue;
    }

    const double stamp =
      std::accumulate(best_stamps.begin(), best_stamps.end(), 0.0) /
      static_cast<double>(count);
    if (imu_still_arriving(stamp)) {
      // The bridge publishes this frame before the IMU sample that covers it.
      // Leave the pair queued; the IMU callback retries.
      return;
    }

    // Frames dropped for want of a peer still moved the ground, so their
    // disparity counts towards the next solve even though they are not
    // themselves compared.
    for (size_t i = 0; i < count; ++i) {
      Camera & camera = *cameras_[i];
      for (size_t n = 0; n <= best_picks[i]; ++n) {
        camera.latest = camera.queue.front();
        camera.hop_disparity += camera.queue.front().disparity;
        camera.queue.pop_front();
      }
      // Take the pixels that belong to the frame being processed, not whatever
      // the camera has received since.
      camera.track_pixels = camera.latest->pixels;
      camera.track_ids = camera.latest->ids;
    }

    ++diagnostics_.pairs_seen;
    ++frames_since_solve_;
    if (settings_.adaptive_solve_interval && !ready_to_solve()) {
      return;
    }
    if (!settings_.adaptive_solve_interval &&
      diagnostics_.pairs_seen % settings_.frame_decimation != 0)
    {
      return;
    }
    process_pair();
    return;
  }
}

bool Estimator::ready_to_solve() const
{
  // A fixed frame count cannot do this. What a solve needs is ground that has
  // visibly moved, and how many frames that takes depends on speed, on the rate
  // the camera actually delivers, and on how close the ground in view is.
  double travelled = 0.0;
  for (const auto & camera : cameras_) {
    travelled = std::max(travelled, camera->hop_disparity);
  }
  if (frames_since_solve_ < settings_.solve_min_frames) {
    return false;
  }
  return travelled >= settings_.solve_trigger_disparity_px ||
         frames_since_solve_ >= settings_.solve_max_frames;
}

std::optional<Eigen::Matrix3d> Estimator::body_tilt() const
{
  // The six degree of freedom filter carries roll and pitch as states, so it
  // answers this itself and the separate attitude filter is not asked. That is
  // the whole point of the container: one covariance rather than two estimators
  // that cannot tell each other how sure they are.
  if (spatial_filter_ && settings_.spatial_tilt_to_projection) {
    return spatial_filter_->body_tilt();
  }
  if (!attitude_ || !attitude_->started()) {
    return std::nullopt;
  }
  return attitude_->body_tilt();
}

// Propagate over exactly the interval a measurement spans, boundaries
// included. Each buffered sample carries the step since the one before it, so
// taking whole samples covers a window shifted early by up to one IMU period
// -- 17% of a 100 ms hop at 60 Hz. The first step is clipped to where the
// interval actually begins and a zero-order-hold tail closes it at the end.
void Estimator::replay_inertial(
  double from, double to,
  const std::function<void(const AccelerationSample &, double)> & step) const
{
  const AccelerationSample * last = nullptr;
  bool first = true;
  for (const auto & sample : imu_window_) {
    if (sample.stamp <= from || sample.stamp > to) {
      continue;
    }
    double span = std::min(sample.dt, 0.1);
    if (first) {
      span = std::min(span, sample.stamp - from);
      first = false;
    }
    if (span > 0.0) {
      step(sample, span);
    }
    last = &sample;
  }
  if (last != nullptr) {
    const double tail = to - last->stamp;
    if (tail > 0.0 && tail <= 0.1) {
      step(*last, tail);
    }
  }
}

void Estimator::remember_solve_pixels(Camera & camera)
{
  const Eigen::Index count = camera.track_ids.size();
  if (count == 0 || camera.track_pixels.rows() != count) {
    camera.solve_ids.resize(0);
    camera.solve_points.resize(0, 2);
  } else {
    // Sorted once here so that every lookup afterwards is a binary search over
    // an array rather than a walk over a table.
    std::vector<Eigen::Index> order(static_cast<size_t>(count));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
      order.begin(), order.end(),
      [&camera](Eigen::Index a, Eigen::Index b) {
        return camera.track_ids(a) < camera.track_ids(b);
      });
    camera.solve_ids.resize(count);
    camera.solve_points.resize(count, 2);
    for (Eigen::Index i = 0; i < count; ++i) {
      const Eigen::Index from = order[static_cast<size_t>(i)];
      camera.solve_ids(i) = camera.track_ids(from);
      camera.solve_points(i, 0) = camera.track_pixels(from, 0);
      camera.solve_points(i, 1) = camera.track_pixels(from, 1);
    }
  }
  camera.solve_frame = camera.latest;
  camera.hop_disparity = 0.0;
}

std::optional<Estimator::Solved> Estimator::solve_camera(
  Camera & camera, std::optional<double> yaw_delta, std::optional<double> yaw_guess)
{
  // Pixels at the last solve and now, for features that survived both.
  const Eigen::Index held = camera.solve_ids.size();
  const Eigen::Index current = camera.track_ids.size();
  if (held == 0 || current == 0) {
    remember_solve_pixels(camera);
    return std::nullopt;
  }

  std::vector<Eigen::Index> matched_now;
  std::vector<Eigen::Index> matched_then;
  matched_now.reserve(static_cast<size_t>(current));
  matched_then.reserve(static_cast<size_t>(current));
  {
    Stopwatch watch(diagnostics_, "pair");
    const int64_t * first = camera.solve_ids.data();
    const int64_t * last = first + held;
    for (Eigen::Index i = 0; i < current; ++i) {
      const int64_t identity = camera.track_ids(i);
      const int64_t * found = std::lower_bound(first, last, identity);
      if (found != last && *found == identity) {
        matched_now.push_back(i);
        matched_then.push_back(static_cast<Eigen::Index>(found - first));
      }
    }
  }
  if (static_cast<int>(matched_now.size()) < settings_.ground_min_inliers) {
    remember_solve_pixels(camera);
    return std::nullopt;
  }

  const Eigen::Index count = static_cast<Eigen::Index>(matched_now.size());
  Points2 previous_pixels(count, 2);
  Points2 current_pixels(count, 2);
  Identities track_ids(count);
  for (Eigen::Index i = 0; i < count; ++i) {
    const Eigen::Index now = matched_now[static_cast<size_t>(i)];
    const Eigen::Index then = matched_then[static_cast<size_t>(i)];
    previous_pixels(i, 0) = camera.solve_points(then, 0);
    previous_pixels(i, 1) = camera.solve_points(then, 1);
    current_pixels(i, 0) = camera.track_pixels(now, 0);
    current_pixels(i, 1) = camera.track_pixels(now, 1);
    track_ids(i) = camera.track_ids(now);
  }

  Solved solved;
  solved.track_ids = track_ids;
  solved.current_pixels = current_pixels;

  const auto tilt = body_tilt();
  const Eigen::Matrix3d * tilt_ptr = tilt.has_value() ? &tilt.value() : nullptr;
  // The band is fixed. An adaptive one keyed on speed was measured on the
  // original recordings and lost to it -- park came out at 2.518 m with the
  // scale at 0.776 against 0.110 m and 0.996 held fixed -- and it was left
  // switched off ever since, which is a way of carrying code that has never
  // run.
  const double band = settings_.ground_max_distance_m;

  // Both views, always. Projecting only the current one and leaving the earlier
  // frame for the paths that need it looks like a saving of half the work, and
  // is not: a feature counts only if BOTH of its positions land inside the
  // band, so the earlier projection is what decides the population the anchor
  // solve registers against, not merely what the fallback compares.
  Mask valid_current;
  Mask valid_previous;
  {
    Stopwatch watch(diagnostics_, "ground");
    pixels_to_ground(
      current_pixels, camera.model, band, camera.settings.ground_min_distance_m,
      tilt_ptr, camera.settings.range_scale, solved.current_ground, valid_current);
    pixels_to_ground(
      previous_pixels, camera.model, band, camera.settings.ground_min_distance_m,
      tilt_ptr, camera.settings.range_scale, solved.previous_ground, valid_previous);
  }
  solved.ground_valid = valid_previous && valid_current;

  solved.motion_inliers = Mask::Constant(count, false);
  if (!yaw_delta.has_value() || !solved.ground_valid.any()) {
    remember_solve_pixels(camera);
    return solved;
  }

  Stopwatch watch(diagnostics_, "solve");

  // Prefer the accumulated map; fall back to the previous frame alone. Matching
  // against anchors averaged over a feature's whole life is what stops a burst
  // of bad matches from carrying the estimate, which is how the two-frame solve
  // failed above walking pace.
  std::vector<Eigen::Index> usable;
  usable.reserve(static_cast<size_t>(count));
  for (Eigen::Index i = 0; i < count; ++i) {
    if (solved.ground_valid(i)) {
      usable.push_back(i);
    }
  }

  Identities usable_ids(static_cast<Eigen::Index>(usable.size()));
  for (size_t i = 0; i < usable.size(); ++i) {
    usable_ids(static_cast<Eigen::Index>(i)) = track_ids(usable[i]);
  }
  Mask anchored;
  {
    Stopwatch lookup(diagnostics_, "lookup");
    camera.anchors.anchored(usable_ids, anchored);
  }

  if (anchored.count() >= settings_.ground_min_inliers) {
    std::vector<Eigen::Index> selected;
    selected.reserve(static_cast<size_t>(anchored.count()));
    for (size_t i = 0; i < usable.size(); ++i) {
      if (anchored(static_cast<Eigen::Index>(i))) {
        selected.push_back(usable[i]);
      }
    }
    const Eigen::Index chosen = static_cast<Eigen::Index>(selected.size());
    Identities selected_ids(chosen);
    Points2 body(chosen, 2);
    for (Eigen::Index i = 0; i < chosen; ++i) {
      selected_ids(i) = track_ids(selected[static_cast<size_t>(i)]);
      body(i, 0) = solved.current_ground(selected[static_cast<size_t>(i)], 0);
      body(i, 1) = solved.current_ground(selected[static_cast<size_t>(i)], 1);
    }
    Points2 world;
    Weights weights;
    {
      Stopwatch lookup(diagnostics_, "lookup");
      camera.anchors.anchor_view(selected_ids, world, weights);
    }

    // Where the solve is told to start looking. With the MSCKF that is the
    // filter's own propagated heading rather than the instrument's, and the
    // solve is asked to improve on it -- the improvement is the measurement.
    const double handed = yaw_guess.has_value()
      ? *yaw_guess : wrap_pi(pose_.yaw + *yaw_delta);
    std::optional<AnchorAlignment> aligned;
    {
      Stopwatch align(diagnostics_, "align");
      const Eigen::Vector3d & mount = camera.model.translation_base_from_camera;
      aligned = align_to_anchors(
        body, world, weights, handed, settings_.ground_ransac_threshold_m,
        settings_.ground_min_inliers,
        heading_.enabled() || msckf_filter_ != nullptr || spatial_filter_ != nullptr,
        Eigen::Vector2d(mount.x(), mount.y()), settings_.radial_min_range_m);
    }
    if (aligned.has_value()) {
      if (aligned->radial_reference > 0.0) {
        camera.radial_linear_sum += aligned->radial_linear;
        ++camera.radial_samples;
      }
      // Recorded, not applied. Both cameras see the same heading and each has
      // its own opinion of how far it is out; folding them in one at a time
      // gives the filter two updates against one prediction.
      if (heading_.enabled() && std::isfinite(aligned->yaw_sigma) &&
        aligned->yaw_sigma > 0.0)
      {
        heading_observations_.emplace_back(
          wrap_pi(aligned->yaw - handed), aligned->yaw_sigma);
      }
      // With the MSCKF the heading the ground settled on is what the filter is
      // being told; without it the heading handed in stands, which is what
      // every measurement before this assumed.
      const double heading =
        (msckf_filter_ || spatial_filter_) ? aligned->yaw : handed;
      solved.yaw_sigma = aligned->yaw_sigma;
      const Pose2 placed{aligned->translation.x(), aligned->translation.y(), heading};
      PlanarMotion motion = relative_motion(pose_, placed);
      motion.inliers = static_cast<int>(aligned->inliers.count());
      motion.scale = 1.0;
      solved.motion = motion;
      solved.anchored_from_map = true;
      solved.spread = aligned->spread;
      for (Eigen::Index i = 0; i < chosen; ++i) {
        if (aligned->inliers(i)) {
          solved.motion_inliers(selected[static_cast<size_t>(i)]) = true;
        }
      }
      remember_solve_pixels(camera);
      return solved;
    }
  }

  // The map could not answer. Fall back to comparing the two frames directly.
  std::vector<Eigen::Index> pairs;
  pairs.reserve(static_cast<size_t>(count));
  for (Eigen::Index i = 0; i < count; ++i) {
    if (solved.ground_valid(i)) {
      pairs.push_back(i);
    }
  }
  const Eigen::Index paired = static_cast<Eigen::Index>(pairs.size());
  Points2 previous_ground(paired, 2);
  Points2 current_ground(paired, 2);
  for (Eigen::Index i = 0; i < paired; ++i) {
    previous_ground(i, 0) = solved.previous_ground(pairs[static_cast<size_t>(i)], 0);
    previous_ground(i, 1) = solved.previous_ground(pairs[static_cast<size_t>(i)], 1);
    current_ground(i, 0) = solved.current_ground(pairs[static_cast<size_t>(i)], 0);
    current_ground(i, 1) = solved.current_ground(pairs[static_cast<size_t>(i)], 1);
  }
  const auto estimate = estimate_planar_motion_with_yaw(
    previous_ground, current_ground, *yaw_delta, settings_.ground_ransac_threshold_m,
    settings_.ground_min_inliers);
  if (estimate.has_value()) {
    solved.motion = estimate->motion;
    for (Eigen::Index i = 0; i < paired; ++i) {
      solved.motion_inliers(pairs[static_cast<size_t>(i)]) = estimate->inliers(i);
    }
    // Spread of the pairwise solution, on the same footing as the map one.
    const double c = std::cos(*yaw_delta);
    const double s = std::sin(*yaw_delta);
    double squared = 0.0;
    int kept = 0;
    for (Eigen::Index i = 0; i < paired; ++i) {
      if (!estimate->inliers(i)) {
        continue;
      }
      const double ox = previous_ground(i, 0) -
        (c * current_ground(i, 0) - s * current_ground(i, 1));
      const double oy = previous_ground(i, 1) -
        (s * current_ground(i, 0) + c * current_ground(i, 1));
      squared += std::pow(ox - estimate->motion.x, 2) + std::pow(oy - estimate->motion.y, 2);
      ++kept;
    }
    solved.spread = kept > 0 ? std::sqrt(squared / kept) : 0.0;
  }

  remember_solve_pixels(camera);
  return solved;
}

bool Estimator::is_stationary(double start, double end, double speed) const
{
  // Gravity is the only specific force on a vehicle at rest, so the magnitude
  // sits at g and the rates sit at zero. Anything driving, however gently,
  // breaks one of those.
  if (!settings_.zero_velocity_update || speed > settings_.zero_velocity_speed_mps) {
    return false;
  }
  std::vector<MotionSample> window;
  for (const auto & sample : imu_motion_samples_) {
    if (sample.stamp >= start && sample.stamp <= end) {
      window.push_back(sample);
    }
  }
  if (window.size() < 2) {
    return false;
  }
  // A fraction, not every sample. CARLA injects contact impulses, and all()
  // hands any one of them a veto over the whole window.
  std::vector<double> forces;
  std::vector<double> rates;
  forces.reserve(window.size());
  rates.reserve(window.size());
  for (const auto & sample : window) {
    forces.push_back(sample.force);
    rates.push_back(std::abs(sample.rate));
  }
  std::sort(forces.begin(), forces.end());
  std::sort(rates.begin(), rates.end());
  const double gravity = forces[forces.size() / 2];
  const double median_rate = rates[rates.size() / 2];
  const double gyro_limit = settings_.zero_velocity_gyro_dps * M_PI / 180.0;

  if (settings_.zero_velocity_dropout_mps2 > 0.0 &&
    gravity <= settings_.zero_velocity_dropout_mps2 && median_rate <= gyro_limit)
  {
    return true;
  }
  int quiet = 0;
  for (const auto & sample : window) {
    if (sample.rate <= gyro_limit &&
      std::abs(sample.force - gravity) <= settings_.zero_velocity_accel_mps2)
    {
      ++quiet;
    }
  }
  return quiet >= settings_.zero_velocity_quiet_fraction * window.size() &&
         std::abs(gravity - 9.81) <= 0.5;
}

void Estimator::process_pair()
{
  const size_t count = cameras_.size();
  // The frames are aligned to within the sync tolerance, so their mean is the
  // time the set describes.
  double current_stamp = 0.0;
  bool seeded = true;
  for (const auto & camera : cameras_) {
    current_stamp += camera->latest->stamp;
    if (!camera->solve_frame.has_value()) {
      seeded = false;
    }
  }
  current_stamp /= static_cast<double>(count);

  if (!seeded) {
    for (auto & camera : cameras_) {
      remember_solve_pixels(*camera);
    }
    Update update;
    update.stamp = current_stamp;
    update.pose = pose_;
    update.pose_valid = !settings_.suppress_pose_until_map_ready || map_ready_;
    pending_updates_.push_back(std::move(update));
    return;
  }

  // The interval a solve spans, not the gap to the frame before it: tracking
  // runs every frame while the solve stands further back.
  double previous_stamp = 0.0;
  for (const auto & camera : cameras_) {
    previous_stamp += camera->solve_frame->stamp;
  }
  previous_stamp /= static_cast<double>(count);
  const double dt = current_stamp - previous_stamp;

  std::optional<double> yaw_delta;
  bool imu_available = !settings_.use_imu_yaw;
  if (settings_.use_imu_yaw) {
    const auto previous_yaw = imu_yaw_at(previous_stamp);
    const auto current_yaw = imu_yaw_at(current_stamp);
    if (previous_yaw.has_value() && current_yaw.has_value()) {
      yaw_delta = std::remainder(*current_yaw - *previous_yaw, 2.0 * M_PI);
      imu_available = true;
    } else {
      ++diagnostics_.imu_yaw_misses;
    }
  }

  // Per pair, not per camera. Both cameras used to reset this, so whichever
  // finished first decided what the other one saw.
  frames_since_solve_ = 0;

  // Before the cameras speak, not after: they are what fills this.
  heading_.predict(std::max(dt, 0.0));
  heading_observations_.clear();

  // The MSCKF propagates before it measures, which is the order an MSCKF runs
  // in and the reason the heading it hands the solve is worth more than the
  // instrument's: it has already carried the gyro across this interval with
  // the bias it has learned taken out.
  std::optional<double> yaw_guess;
  if (msckf_filter_ && dt > 1e-4) {
    msckf_filter_->open_hop();
    replay_inertial(
      previous_stamp, current_stamp,
      [this](const AccelerationSample & sample, double step) {
        msckf_filter_->predict(sample.body_acceleration, sample.rate, step);
      });
    yaw_guess = msckf_filter_->yaw();
  }
  if (spatial_filter_ && dt > 1e-4) {
    spatial_filter_->open_hop();
    std::optional<Eigen::Vector3d> levelling;
    replay_inertial(
      previous_stamp, current_stamp,
      [this, &levelling](const AccelerationSample & sample, double step) {
        Eigen::Vector3d force = sample.specific_force;
        if (settings_.spatial_screen_impulses && sample.force_screened) {
          // Held rather than believed. See AccelerationSample.
        } else if (!settings_.spatial_screen_impulses) {
          force = sample.specific_force;
        }
        // Nothing sideways until vision has supplied the integral's constant.
        // The planar propagator withholds the spawn and drop transient for the
        // same reason, and a filter that integrates it starts the drive with a
        // velocity nobody asked for.
        if (settings_.spatial_wait_for_vision && !spatial_filter_->settled()) {
          force.head<2>().setZero();
        }
        spatial_filter_->predict(force, sample.angular_velocity, step);
        // The gate inside decides whether the reading is gravity at all, and a
        // vehicle doing anything interesting fails it exactly when believing
        // the accelerometer would be wrong.
        if (settings_.spatial_level_every_sample) {
          spatial_filter_->update_gravity(sample.specific_force);
        } else {
          levelling = sample.specific_force;
        }
      });
    if (levelling.has_value()) {
      spatial_filter_->update_gravity(*levelling);
    }
    yaw_guess = spatial_filter_->yaw();
  }

  std::vector<std::optional<Solved>> solved(count);
  if (imu_available) {
    for (size_t i = 0; i < count; ++i) {
      solved[i] = solve_camera(*cameras_[i], yaw_delta, yaw_guess);
    }
  } else {
    for (auto & camera : cameras_) {
      remember_solve_pixels(*camera);
    }
  }

  // Carried out to the diagnostics here, where the cameras are indexed: the
  // solve holds a reference and does not know which one it was handed.
  diagnostics_.radial_linear.assign(count, 0.0);
  diagnostics_.radial_samples.assign(count, 0);
  for (size_t i = 0; i < count; ++i) {
    const Camera & camera = *cameras_[i];
    diagnostics_.radial_samples[i] = camera.radial_samples;
    if (camera.radial_samples > 0) {
      const double n = static_cast<double>(camera.radial_samples);
      diagnostics_.radial_linear[i] = camera.radial_linear_sum / n;
    }
  }

  std::vector<PlanarMotion> motions;
  std::vector<CameraTranslation> precision_inputs;
  bool all_have_spread = true;
  for (size_t i = 0; i < count; ++i) {
    if (!solved[i].has_value() || !solved[i]->motion.has_value()) {
      continue;
    }
    Camera & camera = *cameras_[i];
    PlanarMotion measured = *solved[i]->motion;
    // Each camera reports where it is at its own frame's time, but the pair is
    // stamped at the midpoint and dt is measured between midpoints. While both
    // cameras run at the same rate those are the same instant. When one starts
    // dropping frames they are not: the sync tolerance allows 20 ms, which at
    // 8 m/s is 16 cm, more than the 13 cm the pair is trying to measure.
    if (camera.solve_frame.has_value()) {
      const double own = camera.latest->stamp - camera.solve_frame->stamp;
      if (own > 1e-4) {
        const double ratio = dt / own;
        if (ratio >= 0.5 && ratio <= 2.0) {
          // Along the arc the vehicle is on, not along a stretched chord.
          measured = rescale_motion(measured, ratio);
        }
      }
    }
    solved[i]->motion = measured;
    motions.push_back(measured);
    if (solved[i]->spread > 0.0) {
      precision_inputs.push_back(
        CameraTranslation{measured.x, measured.y, measured.inliers, solved[i]->spread});
    } else {
      all_have_spread = false;
    }
  }

  // Two cameras looking at different ground disagreeing is the one signal
  // neither can produce alone.
  double disagreement = 0.0;
  if (motions.size() >= 2) {
    for (size_t a = 0; a < motions.size(); ++a) {
      for (size_t b = a + 1; b < motions.size(); ++b) {
        disagreement = std::max(
          disagreement, std::hypot(motions[a].x - motions[b].x, motions[a].y - motions[b].y));
      }
    }
  }
  diagnostics_.camera_disagreement = disagreement;

  for (const auto & entry : solved) {
    if (entry.has_value() && entry->anchored_from_map) {
      ++diagnostics_.map_aligned_frames;
      map_ready_ = true;
      break;
    }
  }

  auto motion = fuse_planar_motions(motions);
  if (settings_.fuse_cameras_by_spread && all_have_spread &&
    precision_inputs.size() == motions.size() && motion.has_value())
  {
    const auto combined = fuse_by_precision(precision_inputs);
    if (combined.has_value()) {
      motion->x = combined->x;
      motion->y = combined->y;
      motion->inliers = combined->count;
    }
  }

  const double vision_speed = (motion.has_value() && dt > 1e-4)
    ? std::hypot(motion->x, motion->y) / dt
    : std::numeric_limits<double>::infinity();

  double disparity = -1.0;
  bool have_disparity = false;
  for (const auto & camera : cameras_) {
    if (camera->latest.has_value()) {
      disparity = std::max(disparity, camera->hop_disparity);
      have_disparity = true;
    }
  }
  diagnostics_.last_disparity = have_disparity ? disparity : -1.0;

  // Decided before the fusion, not after it. Standing still used to be noticed
  // only once the vision displacement had already gone into the filter, and
  // those are the hops it most needed protecting from.
  const bool standing = settings_.zero_velocity_update && have_disparity &&
    disparity <= settings_.zero_velocity_disparity_px &&
    is_stationary(previous_stamp, current_stamp, vision_speed);
  if (standing) {
    ++diagnostics_.zupt_holds;
  }

  const Pose2 previous_pose = pose_;

  if (msckf_filter_ && motion.has_value() && dt > 1e-4) {
    // The measurement is the hop as vision saw it, in the frame the anchor was
    // cloned in: how far, and how much the heading turned. Both halves are the
    // ground solve's own answer -- the turn is not the instrument's, which is
    // the point of carrying the heading as a state.
    Eigen::Vector2d hop(motion->x, motion->y);
    double turn = motion->yaw;
    double extra = motions.size() >= 2
      ? std::pow(settings_.camera_disagreement_weight * disagreement, 2)
      : settings_.single_camera_variance * dt * dt;
    double spread = 0.0;
    {
      double weighted = 0.0;
      double total = 0.0;
      for (const auto & input : precision_inputs) {
        weighted += input.spread * input.count;
        total += input.count;
      }
      spread = total > 0.0 ? weighted / total : 0.0;
    }
    // Both cameras see the same heading, so their opinions of it combine the
    // way two measurements of one quantity do.
    double precision = 0.0;
    for (const auto & entry : solved) {
      if (entry.has_value() && entry->motion.has_value() &&
        std::isfinite(entry->yaw_sigma) && entry->yaw_sigma > 0.0)
      {
        precision += 1.0 / (entry->yaw_sigma * entry->yaw_sigma);
      }
    }
    const double yaw_sigma = precision > 0.0 ? std::sqrt(1.0 / precision) : 0.0;

    if (standing) {
      hop.setZero();
      turn = 0.0;
      extra = std::pow(settings_.zupt_velocity_sigma * dt, 2);
      spread = 0.0;
    }
    if (!msckf_filter_->update(hop, turn, motion->inliers, extra, spread, yaw_sigma)) {
      ++diagnostics_.filter_rejections;
    }
    if (standing) {
      msckf_filter_->update_zero_velocity(settings_.zupt_velocity_sigma);
    }
    // And what the instrument says the heading is, weighed rather than
    // believed. This is the reference that makes the gyro bias observable;
    // without it a systematic error in the ground solve's heading is
    // indistinguishable from one.
    if (settings_.msckf_heading_noise > 0.0 && imu_yaw_datum_.has_value()) {
      const auto reported = imu_yaw_at(current_stamp);
      if (reported.has_value()) {
        msckf_filter_->update_heading(
          wrap_pi(*reported - *imu_yaw_datum_), settings_.msckf_heading_noise);
      }
    }
    inertial_.correct_velocity(msckf_filter_->velocity());
    diagnostics_.gyro_bias = msckf_filter_->gyro_bias();
    diagnostics_.heading_drift = msckf_filter_->heading_drift();
    diagnostics_.filter_dropped = msckf_filter_->dropped();
    if (msckf_filter_->last_update().has_value()) {
      diagnostics_.last_nis = msckf_filter_->last_update()->nis;
    }
    const Eigen::Vector2d fused = msckf_filter_->body_translation();
    if (fused.norm() <= settings_.max_translation_per_frame_m) {
      motion->x = fused.x();
      motion->y = fused.y();
      motion->yaw = msckf_filter_->hop_yaw();
      motion->scale = 1.0;
    }
  } else if (spatial_filter_ && motion.has_value() && dt > 1e-4) {
    // The same measurement the planar MSCKF is given. What differs is what
    // receives it: a hop into a filter that also knows which way the vehicle is
    // leaning, and can therefore be told that it did not climb.
    Eigen::Vector2d hop(motion->x, motion->y);
    double turn = motion->yaw;
    double extra = motions.size() >= 2
      ? std::pow(settings_.camera_disagreement_weight * disagreement, 2)
      : settings_.single_camera_variance * dt * dt;
    double spread = 0.0;
    {
      double weighted = 0.0;
      double total = 0.0;
      for (const auto & input : precision_inputs) {
        weighted += input.spread * input.count;
        total += input.count;
      }
      spread = total > 0.0 ? weighted / total : 0.0;
    }
    double precision = 0.0;
    for (const auto & entry : solved) {
      if (entry.has_value() && entry->motion.has_value() &&
        std::isfinite(entry->yaw_sigma) && entry->yaw_sigma > 0.0)
      {
        precision += 1.0 / (entry->yaw_sigma * entry->yaw_sigma);
      }
    }
    const double yaw_sigma = precision > 0.0 ? std::sqrt(1.0 / precision) : 0.0;

    if (standing) {
      hop.setZero();
      turn = 0.0;
      extra = std::pow(settings_.zupt_velocity_sigma * dt, 2);
      spread = 0.0;
    }
    if (!spatial_filter_->update(hop, turn, motion->inliers, extra, spread, yaw_sigma)) {
      ++diagnostics_.filter_rejections;
    }
    {
      // Straight after the hop and before anything else touches the state:
      // whether the filter took the measurement it was given.
      const Eigen::Vector2d taken = spatial_filter_->body_translation();
      hop_taken_squared_ += (taken - hop).squaredNorm();
    }
    // Separately, and ungated: the vehicle did not climb over this hop.
    spatial_filter_->update_height();
    if (standing) {
      spatial_filter_->update_zero_velocity(settings_.zupt_velocity_sigma);
    }
    if (settings_.msckf_heading_noise > 0.0 && imu_yaw_datum_.has_value()) {
      const auto reported = imu_yaw_at(current_stamp);
      if (reported.has_value()) {
        spatial_filter_->update_heading(
          wrap_pi(*reported - *imu_yaw_datum_), settings_.msckf_heading_noise);
      }
    }
    inertial_.correct_velocity(spatial_filter_->velocity().head<2>());
    diagnostics_.gyro_bias = spatial_filter_->gyro_bias().z();
    diagnostics_.filter_dropped = spatial_filter_->dropped();
    diagnostics_.roll = spatial_filter_->roll();
    diagnostics_.pitch = spatial_filter_->pitch();
    diagnostics_.height = spatial_filter_->position().z();
    diagnostics_.levelled = spatial_filter_->levelled();
    diagnostics_.range_scale = spatial_filter_->range_scale();
    {
      const Eigen::Vector2d after = spatial_filter_->body_translation();
      hop_residual_squared_ += (after - hop).squaredNorm();
      ++hop_residual_count_;
      diagnostics_.hop_residual =
        std::sqrt(hop_residual_squared_ / static_cast<double>(hop_residual_count_));
      diagnostics_.hop_taken =
        std::sqrt(hop_taken_squared_ / static_cast<double>(hop_residual_count_));
    }
    if (spatial_filter_->last_update().has_value()) {
      diagnostics_.last_nis = spatial_filter_->last_update()->nis;
    }
    const Eigen::Vector2d fused = spatial_filter_->body_translation();
    if (fused.norm() <= settings_.max_translation_per_frame_m) {
      motion->x = fused.x();
      motion->y = fused.y();
      motion->yaw = spatial_filter_->hop_yaw();
      motion->scale = 1.0;
    }
  } else if (displacement_filter_ && motion.has_value() && dt > 1e-4) {
    // Propagate over exactly the interval this measurement spans, boundaries
    // included. Each buffered sample carries the step since the one before it,
    // so taking whole samples covers a window shifted early by up to one IMU
    // period -- 17% of a 100 ms hop at 60 Hz.
    displacement_filter_->open_hop();
    replay_inertial(
      previous_stamp, current_stamp,
      [this](const AccelerationSample & sample, double step) {
        displacement_filter_->predict(sample.acceleration, step, sample.yaw);
      });

    const double c = std::cos(previous_pose.yaw);
    const double s = std::sin(previous_pose.yaw);
    Eigen::Vector2d world(
      c * motion->x - s * motion->y, s * motion->x + c * motion->y);
    // The disagreement between the two cameras is already a distance, so here
    // it needs no division to become a variance.
    double extra = motions.size() >= 2
      ? std::pow(settings_.camera_disagreement_weight * disagreement, 2)
      : settings_.single_camera_variance * dt * dt;
    // Inlier-weighted spread across whichever cameras reported one.
    double spread = 0.0;
    {
      double weighted = 0.0;
      double total = 0.0;
      for (const auto & input : precision_inputs) {
        weighted += input.spread * input.count;
        total += input.count;
      }
      spread = total > 0.0 ? weighted / total : 0.0;
    }
    if (standing) {
      // The measurement is that nothing moved, and it is a better measurement
      // than the solve.
      world.setZero();
      extra = std::pow(settings_.zupt_velocity_sigma * dt, 2);
      spread = 0.0;
    }
    if (!displacement_filter_->update(world, motion->inliers, extra, spread)) {
      ++diagnostics_.filter_rejections;
    }
    if (standing) {
      displacement_filter_->update_zero_velocity(settings_.zupt_velocity_sigma);
    }
    inertial_.correct_velocity(displacement_filter_->velocity());
    const Eigen::Vector2d fused = displacement_filter_->body_translation(previous_pose.yaw);
    if (fused.norm() <= settings_.max_translation_per_frame_m) {
      motion->x = fused.x();
      motion->y = fused.y();
      motion->scale = 1.0;
    }
  } else if (settings_.use_inertial_prediction && motion.has_value() && dt > 1e-4) {
    const auto measured =
      world_velocity_from_motion(motion->x, motion->y, dt, previous_pose.yaw);
    if (measured.has_value()) {
      // Disagreement is a translation over dt, so as a velocity variance it is
      // (d/dt)^2.
      const double extra = motions.size() >= 2
        ? std::pow(settings_.camera_disagreement_weight * disagreement / dt, 2)
        : settings_.single_camera_variance;
      if (!velocity_filter_.update(*measured, motion->inliers, extra)) {
        ++diagnostics_.filter_rejections;
      }
      // The propagator withholds spawn/drop acceleration until a vision
      // velocity has fixed its integration constant.
      inertial_.correct_velocity(velocity_filter_.velocity());
    }
    const Eigen::Vector2d fused =
      velocity_filter_.body_translation(dt, previous_pose.yaw);
    if (fused.norm() <= settings_.max_translation_per_frame_m) {
      motion->x = fused.x();
      motion->y = fused.y();
      motion->scale = 1.0;
    }
  }

  if (motion.has_value() && standing) {
    // Hold the position, keep the heading. A parking manoeuvre stops several
    // times, so every stop is a chance to stop accumulating.
    motion->x = 0.0;
    motion->y = 0.0;
  }

  const bool warming_up = !map_ready_;
  // The cap is on how far the vehicle can plausibly have moved, so it has to
  // cover the whole interval the update is closing. Against the map -- which
  // returns an absolute position, not a hop -- a rejected frame leaves the pose
  // behind, and the next solve reports the arrears along with the new motion.
  double allowance = settings_.max_translation_per_frame_m;
  if (dt > 1e-4 && last_accept_stamp_.has_value()) {
    const double since = current_stamp - *last_accept_stamp_;
    allowance *= std::max(1.0, std::min(since / dt, settings_.max_arrears_frames));
  }

  Eigen::Vector3d twist = Eigen::Vector3d::Zero();
  const bool rejected = !motion.has_value() || warming_up || dt <= 1e-4 ||
    std::hypot(motion->x, motion->y) > allowance ||
    std::abs(motion->yaw) > settings_.max_yaw_per_frame_rad;

  Update update;
  update.stamp = current_stamp;

  if (rejected) {
    ++diagnostics_.motion_failures;
    if (!motion.has_value()) {
      ++diagnostics_.fail_no_solve;
    } else if (std::hypot(motion->x, motion->y) > allowance) {
      ++diagnostics_.fail_translation;
      diagnostics_.worst_rejected =
        std::max(diagnostics_.worst_rejected, std::hypot(motion->x, motion->y));
      diagnostics_.worst_allowance = std::max(diagnostics_.worst_allowance, allowance);
    } else if (std::abs(motion->yaw) > settings_.max_yaw_per_frame_rad) {
      ++diagnostics_.fail_yaw;
    }

    // Rejecting the vision translation is no reason to throw away the IMU
    // rotation over the same interval. Nor is it a reason to stand still:
    // holding the position on a rejected pair is a claim that the vehicle did
    // not move, which at road speed is the one thing that is certainly false.
    if (yaw_delta.has_value() &&
      std::abs(*yaw_delta) <= settings_.max_yaw_per_frame_rad)
    {
      std::optional<PlanarMotion> carried;
      // Not while the map is still being built. There the pose is deliberately
      // pinned at the origin, because the anchors are written from it.
      if (dt > 1e-4 && settings_.coast_on_reject && !warming_up) {
        const Eigen::Vector2d predicted =
          velocity_filter_.body_translation(dt, pose_.yaw);
        if (predicted.norm() <= allowance) {
          // The IMU already says how fast the heading is turning over this same
          // gap, so the dead reckoning can follow the arc rather than shoot off
          // along the tangent.
          carried = motion_from_twist(
            predicted.x() / dt, predicted.y() / dt, *yaw_delta / dt, dt);
          ++diagnostics_.coasted;
        }
      }
      PlanarMotion applied{0.0, 0.0, *yaw_delta, 0, 1.0};
      if (carried.has_value()) {
        applied.x = carried->x;
        applied.y = carried->y;
      }
      pose_ = pose_.compose(applied);
      ++diagnostics_.yaw_only_updates;
    }
    if (warming_up) {
      // The map has to come from somewhere, and while it is missing the pose is
      // held at the origin, which for a vehicle that has not moved is the right
      // place to write these from.
      update_anchors(solved);
    }
  } else {
    pose_ = pose_.compose(*motion);
    // One update against one prediction, from both cameras at once, weighted
    // the way two measurements of the same quantity are.
    if (heading_.enabled() && !heading_observations_.empty()) {
      double precision = 0.0;
      double weighted = 0.0;
      for (const auto & [residual, sigma] : heading_observations_) {
        precision += 1.0 / (sigma * sigma);
        weighted += residual / (sigma * sigma);
      }
      if (precision > 0.0) {
        const double offset =
          heading_.update(weighted / precision, std::sqrt(1.0 / precision));
        pose_.yaw = wrap_pi(pose_.yaw + offset);
      }
    }
    last_accept_stamp_ = current_stamp;

    const double c = std::cos(motion->yaw);
    const double s = std::sin(motion->yaw);
    const Eigen::Vector3d raw(
      (c * motion->x + s * motion->y) / dt,
      (-s * motion->x + c * motion->y) / dt,
      motion->yaw / dt);
    const double alpha = settings_.twist_lowpass_tau == 0.0
      ? 1.0
      : dt / (settings_.twist_lowpass_tau + dt);
    filtered_twist_ += alpha * (raw - filtered_twist_);
    twist = filtered_twist_;

    {
      Stopwatch watch(diagnostics_, "anchor");
      update_anchors(solved);
    }
    if (!last_mapping_stamp_.has_value() ||
      current_stamp - *last_mapping_stamp_ >= settings_.mapping_min_period_sec)
    {
      last_mapping_stamp_ = current_stamp;
      Stopwatch watch(diagnostics_, "map");
      integrate_points(solved, previous_pose, update);
    }
  }

  if (msckf_filter_) {
    // Put the filter where the estimator ended up. On an accepted solve the
    // two already agree -- the pose came out of the filter. On a rejected one
    // the estimator coasted or held, and during the warm-up it is pinned at
    // the origin on purpose; without this the filter would carry on
    // integrating away from it and hand back the difference as travel the
    // moment a solve is finally accepted.
    msckf_filter_->set_pose(Eigen::Vector2d(pose_.x, pose_.y), pose_.yaw);
  }
  if (spatial_filter_) {
    spatial_filter_->set_pose(Eigen::Vector2d(pose_.x, pose_.y), pose_.yaw);
  }

  ++diagnostics_.frames_processed;
  int anchors = 0;
  for (const auto & camera : cameras_) {
    anchors += camera->anchors.size();
  }
  diagnostics_.anchors = anchors;

  update.pose = pose_;
  update.twist = twist;
  // Nothing goes out while the map is still being built. The pose is pinned to
  // the origin over that stretch, and a vehicle already at 8 m/s covers 0.375 m
  // before the first map-anchored solve lands -- published as (0, 0) it becomes
  // a constant offset the estimate then carries for the rest of the run.
  update.pose_valid = !settings_.suppress_pose_until_map_ready || map_ready_;
  pending_updates_.push_back(std::move(update));
}

void Estimator::update_anchors(const std::vector<std::optional<Solved>> & solved)
{
  bool allow_new = true;
  if (settings_.anchor_seed_travel_m > 0.0) {
    const Eigen::Vector2d here(pose_.x, pose_.y);
    if (!last_seed_position_.has_value()) {
      last_seed_position_ = here;
    } else if ((here - *last_seed_position_).norm() >= settings_.anchor_seed_travel_m) {
      last_seed_position_ = here;
    } else {
      allow_new = false;
    }
  }

  for (size_t i = 0; i < solved.size(); ++i) {
    if (!solved[i].has_value()) {
      continue;
    }
    const Solved & entry = *solved[i];
    Camera & camera = *cameras_[i];
    std::vector<Eigen::Index> usable;
    for (Eigen::Index n = 0; n < entry.ground_valid.size(); ++n) {
      if (entry.ground_valid(n)) {
        usable.push_back(n);
      }
    }
    if (usable.empty()) {
      continue;
    }
    const Eigen::Index count = static_cast<Eigen::Index>(usable.size());
    Points3 body(count, 3);
    Identities ids(count);
    Weights information(count);
    const Eigen::Vector2d mount =
      camera.model.translation_base_from_camera.head<2>();
    for (Eigen::Index n = 0; n < count; ++n) {
      const Eigen::Index at = usable[static_cast<size_t>(n)];
      body(n, 0) = entry.current_ground(at, 0);
      body(n, 1) = entry.current_ground(at, 1);
      body(n, 2) = 0.0;
      ids(n) = entry.track_ids(at);
      // How much this sighting is worth. A ground point's position error grows
      // with its range, so the precision goes as the inverse square of it. The
      // offset is the camera, not the vehicle: range is measured from the lens
      // that saw the point.
      const double range = std::hypot(body(n, 0) - mount.x(), body(n, 1) - mount.y());
      information(n) = 1.0 / std::pow(std::max(range, 0.1), 2);
    }
    const Points3 world = transform_points_to_world(body, pose_);
    Points2 flat(count, 2);
    flat.col(0) = world.col(0);
    flat.col(1) = world.col(1);
    camera.anchors.update(ids, flat, allow_new, information);
  }
}

void Estimator::integrate_points(
  const std::vector<std::optional<Solved>> & solved, const Pose2 & previous_pose,
  Update & update)
{
  for (size_t i = 0; i < solved.size(); ++i) {
    if (!solved[i].has_value()) {
      continue;
    }
    const Solved & entry = *solved[i];
    Camera & camera = *cameras_[i];
    const Eigen::Vector3d & mount = camera.model.translation_base_from_camera;
    Points3 origin_base(1, 3);
    origin_base.row(0) = mount.transpose();
    const Points3 origin_world = transform_points_to_world(origin_base, previous_pose);

    // Free space comes from ground inliers, which are reliable. Obstacle rays
    // do not carve: depth from a short baseline is not good enough to declare
    // everything in front of an obstacle empty, and the carving was erasing
    // real obstacles.
    std::vector<Eigen::Index> inliers;
    for (Eigen::Index n = 0; n < entry.motion_inliers.size(); ++n) {
      if (entry.motion_inliers(n) && n < entry.previous_ground.rows()) {
        inliers.push_back(n);
      }
    }
    if (!inliers.empty()) {
      Points3 free_points(static_cast<Eigen::Index>(inliers.size()), 3);
      for (size_t n = 0; n < inliers.size(); ++n) {
        free_points(static_cast<Eigen::Index>(n), 0) = entry.previous_ground(inliers[n], 0);
        free_points(static_cast<Eigen::Index>(n), 1) = entry.previous_ground(inliers[n], 1);
        free_points(static_cast<Eigen::Index>(n), 2) = 0.0;
      }
      const Points3 free_world = transform_points_to_world(free_points, previous_pose);
      for (Eigen::Index n = 0; n < free_world.rows(); ++n) {
        LabelledPoint point;
        point.x = static_cast<float>(free_world(n, 0));
        point.y = static_cast<float>(free_world(n, 1));
        point.z = 0.0f;
        point.label = kGroundLabel;
        point.origin_x = static_cast<float>(origin_world(0, 0));
        point.origin_y = static_cast<float>(origin_world(0, 1));
        update.points.push_back(point);
      }
    }

    if (settings_.obstacle_slip_baseline_m > 0.0) {
      integrate_obstacle_slip(camera, entry, update);
    } else {
      // The other way of placing an obstacle is to triangulate the current view
      // against a held keyframe, and that needs the images this path does not
      // receive. The Python reached for it here and dereferenced a frame that
      // has no picture in it. Say nothing rather than crash, and count it.
      ++diagnostics_.obstacles_unavailable;
    }
  }
}

void Estimator::integrate_obstacle_slip(
  Camera & camera, const Solved & entry, Update & update)
{
  // Read each feature's height off how far its ground projection slides.
  //
  // A feature really on the road projects to a fixed world point: the ray from
  // wherever the camera has moved to meets the plane in the same place. One
  // standing h above the road does not. With the camera at height H its ray
  // crosses the plane H/(H-h) times as far out, so moving the camera by D
  // slides that crossing by -D*h/(H-h), against the travel:
  //
  //     r = -(slip . D) / |D|^2,    h = H * r / (1 + r)
  //
  // and the feature sits on the ray at (H-h)/H of the way to the crossing.
  // Nothing is triangulated; this is the residual the ground registration
  // already computes and discards as an outlier.
  //
  // What makes it work is the length of D. The projection is taken against an
  // assumed pitch, and that assumption wobbles: a fifth of a degree slides a
  // crossing ten metres out by 0.37 m. The slip grows with the travel and the
  // wobble does not, so each feature keeps the camera pose and projection from
  // when it was first seen and is only read out once it has earned a baseline.
  std::vector<Eigen::Index> usable;
  for (Eigen::Index n = 0; n < entry.ground_valid.size(); ++n) {
    if (entry.ground_valid(n)) {
      usable.push_back(n);
    }
  }
  if (usable.empty()) {
    return;
  }

  const double height = camera.model.translation_base_from_camera.z();
  const Eigen::Index count = static_cast<Eigen::Index>(usable.size());

  Points3 ground(count, 3);
  Identities ids(count);
  for (Eigen::Index n = 0; n < count; ++n) {
    const Eigen::Index at = usable[static_cast<size_t>(n)];
    ground(n, 0) = entry.current_ground(at, 0);
    ground(n, 1) = entry.current_ground(at, 1);
    ground(n, 2) = 0.0;
    ids(n) = entry.track_ids(at);
  }
  const Points3 world3 = transform_points_to_world(ground, pose_);
  Points3 mount(1, 3);
  mount.row(0) = camera.model.translation_base_from_camera.transpose();
  const Points3 camera_world = transform_points_to_world(mount, pose_);
  const Eigen::Vector2d here(camera_world(0, 0), camera_world(0, 1));

  // Sorted by identity so the held table can be searched rather than walked.
  std::vector<Eigen::Index> order(static_cast<size_t>(count));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
    order.begin(), order.end(),
    [&ids](Eigen::Index a, Eigen::Index b) {return ids(a) < ids(b);});

  Identities sorted_ids(count);
  Points2 sorted_world(count, 2);
  for (Eigen::Index n = 0; n < count; ++n) {
    const Eigen::Index from = order[static_cast<size_t>(n)];
    sorted_ids(n) = ids(from);
    sorted_world(n, 0) = world3(from, 0);
    sorted_world(n, 1) = world3(from, 1);
  }

  if (camera.slip_ids.size() == 0) {
    camera.slip_ids = sorted_ids;
    camera.slip_world = sorted_world;
    camera.slip_camera = Points2(count, 2);
    camera.slip_camera.col(0).setConstant(here.x());
    camera.slip_camera.col(1).setConstant(here.y());
    camera.slip_misses.assign(static_cast<size_t>(count), 0);
    return;
  }

  // The common part of the slip -- the wobble in the assumed pitch -- has to be
  // read off everything in view, not off the handful of features that happen to
  // have earned their baseline on this step.
  std::vector<double> ratios;
  struct Ready
  {
    Eigen::Index row;
    Eigen::Index held;
    double travel_x;
    double travel_y;
    double reach;
  };
  std::vector<Ready> ready;
  std::vector<char> seen(static_cast<size_t>(count), 0);
  std::vector<Eigen::Index> where(static_cast<size_t>(count), -1);

  const int64_t * first = camera.slip_ids.data();
  const int64_t * last = first + camera.slip_ids.size();
  for (Eigen::Index n = 0; n < count; ++n) {
    const int64_t * found = std::lower_bound(first, last, sorted_ids(n));
    if (found == last || *found != sorted_ids(n)) {
      continue;
    }
    const Eigen::Index held = static_cast<Eigen::Index>(found - first);
    seen[static_cast<size_t>(n)] = 1;
    where[static_cast<size_t>(n)] = held;
    const double travel_x = here.x() - camera.slip_camera(held, 0);
    const double travel_y = here.y() - camera.slip_camera(held, 1);
    const double reach = std::hypot(travel_x, travel_y);
    if (reach > 0.1) {
      const double slip_x = sorted_world(n, 0) - camera.slip_world(held, 0);
      const double slip_y = sorted_world(n, 1) - camera.slip_world(held, 1);
      ratios.push_back(-(slip_x * travel_x + slip_y * travel_y) / (reach * reach));
    }
    if (reach >= settings_.obstacle_slip_baseline_m) {
      ready.push_back(Ready{n, held, travel_x, travel_y, reach});
    }
  }

  double common = 0.0;
  if (ratios.size() >= 8) {
    std::nth_element(ratios.begin(), ratios.begin() + ratios.size() / 2, ratios.end());
    common = ratios[ratios.size() / 2];
  }

  const double ceiling =
    std::min(settings_.obstacle_max_height_m, height * settings_.obstacle_height_margin);
  for (const auto & item : ready) {
    const double slip_x = sorted_world(item.row, 0) - camera.slip_world(item.held, 0);
    const double slip_y = sorted_world(item.row, 1) - camera.slip_world(item.held, 1);
    const double ratio =
      -(slip_x * item.travel_x + slip_y * item.travel_y) / (item.reach * item.reach) - common;
    if (!(ratio > 0.0)) {
      continue;
    }
    const double obstacle_height = height * ratio / (1.0 + ratio);
    if (!std::isfinite(obstacle_height) ||
      obstacle_height < settings_.obstacle_min_height_m || obstacle_height > ceiling)
    {
      continue;
    }
    // The crossing overshoots the feature by the same factor that gave the
    // height, so walking back along the ray puts it where it stands.
    const double share = (height - obstacle_height) / height;
    LabelledPoint point;
    point.x = static_cast<float>(here.x() + (sorted_world(item.row, 0) - here.x()) * share);
    point.y = static_cast<float>(here.y() + (sorted_world(item.row, 1) - here.y()) * share);
    point.z = static_cast<float>(obstacle_height);
    point.label = kObstacleLabel;
    point.origin_x = static_cast<float>(here.x());
    point.origin_y = static_cast<float>(here.y());
    update.points.push_back(point);
  }

  // Entries persist until they are either read out or have gone missing for
  // long enough to be gone for good. A feature that has just been read starts a
  // fresh baseline where it stands rather than being dropped: one that lives
  // ten metres should be heard from ten times over a one metre baseline.
  for (auto & miss : camera.slip_misses) {
    ++miss;
  }
  for (Eigen::Index n = 0; n < count; ++n) {
    if (seen[static_cast<size_t>(n)]) {
      camera.slip_misses[static_cast<size_t>(where[static_cast<size_t>(n)])] = 0;
    }
  }
  for (const auto & item : ready) {
    camera.slip_world(item.held, 0) = sorted_world(item.row, 0);
    camera.slip_world(item.held, 1) = sorted_world(item.row, 1);
    camera.slip_camera(item.held, 0) = here.x();
    camera.slip_camera(item.held, 1) = here.y();
  }

  std::vector<Eigen::Index> alive;
  for (Eigen::Index n = 0; n < camera.slip_ids.size(); ++n) {
    if (camera.slip_misses[static_cast<size_t>(n)] <= settings_.obstacle_slip_patience) {
      alive.push_back(n);
    }
  }
  std::vector<Eigen::Index> fresh;
  for (Eigen::Index n = 0; n < count; ++n) {
    if (!seen[static_cast<size_t>(n)]) {
      fresh.push_back(n);
    }
  }

  const Eigen::Index total =
    static_cast<Eigen::Index>(alive.size() + fresh.size());
  Identities merged_ids(total);
  Points2 merged_world(total, 2);
  Points2 merged_camera(total, 2);
  std::vector<int32_t> merged_misses(static_cast<size_t>(total));
  // Both inputs are sorted by identity, so one merge keeps the table searchable.
  size_t a = 0;
  size_t b = 0;
  for (Eigen::Index n = 0; n < total; ++n) {
    const bool take_alive = b >= fresh.size() ||
      (a < alive.size() && camera.slip_ids(alive[a]) <= sorted_ids(fresh[b]));
    if (take_alive) {
      const Eigen::Index at = alive[a++];
      merged_ids(n) = camera.slip_ids(at);
      merged_world.row(n) = camera.slip_world.row(at);
      merged_camera.row(n) = camera.slip_camera.row(at);
      merged_misses[static_cast<size_t>(n)] = camera.slip_misses[static_cast<size_t>(at)];
    } else {
      const Eigen::Index at = fresh[b++];
      merged_ids(n) = sorted_ids(at);
      merged_world.row(n) = sorted_world.row(at);
      merged_camera(n, 0) = here.x();
      merged_camera(n, 1) = here.y();
      merged_misses[static_cast<size_t>(n)] = 0;
    }
  }
  camera.slip_ids = std::move(merged_ids);
  camera.slip_world = std::move(merged_world);
  camera.slip_camera = std::move(merged_camera);
  camera.slip_misses = std::move(merged_misses);
}

}  // namespace monoscale
