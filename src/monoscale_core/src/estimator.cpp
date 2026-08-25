#include "monoscale_core/estimator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

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
  Camera(const CameraSettings & from, int source_index)
  : settings(from), source(source_index)
  {
    calibration = make_camera_model(
      from.k, from.rotation_base_from_camera, from.translation_base_from_camera,
      from.distortion, from.lens);
    calibration_width = from.calibration_width;
    calibration_height = from.calibration_height;
    model = calibration;
  }

  CameraSettings settings;
  // A track that is not on the road, kept until its own bearings have spread
  // far enough to intersect. The rays are world-frame, so the camera centre
  // travels with them and the pose that recorded each one is already folded in.
  struct Offground
  {
    std::vector<Eigen::Vector3d> centres;
    std::vector<Eigen::Vector3d> bearings;
    Eigen::Vector3d world = Eigen::Vector3d::Zero();
    bool solved = false;
    int64_t seen = 0;
    int64_t fixed = 0;
  };
  std::unordered_map<int64_t, Offground> offground;
  int64_t offground_frame = 0;

  // What the ground residual has actually been running at, for the paths that
  // weight by it. Zero until the first solve reports one, so the configured
  // constant covers the warm-up.
  double residual_scale = 0.0;
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

  // Which column of the shared anchor map this camera writes into.
  int source = 0;

  // How this camera's ground projection has been leaning, summed over the
  // solves the map answered. Kept here rather than in the diagnostics because
  // the solve does not know which camera it is.
  double radial_linear_sum = 0.0;
  int64_t radial_samples = 0;
  // Learned correction on top of the configured range scale. The radial
  // residual regressed against range is exactly dh/h -- a ground point read
  // through a camera height that is wrong by dh lands wrong by range*dh/h --
  // so the solve already measures the very quantity ground_range_scale was
  // hand-tuned to supply, and this folds that measurement back in.
  double range_scale_learned = 1.0;
  // The last hop this camera solved, in world coordinates, as a seed.
  std::optional<Eigen::Vector2d> last_translation;
  // Where this camera's own map last said the vehicle was, and whether that
  // was the immediately preceding solve. The hop the map path reports is
  // `placed - pose_`, which is a pose *correction*, not a displacement: any
  // standing disagreement between this camera's map and the fused pose rides
  // out on every hop it reports.
  std::optional<Pose2> last_placed;
  bool placed_fresh = false;
  std::optional<Eigen::Vector2d> previous_translation;

  // How far this camera's mounting pitch is believed to be out, in radians, and
  // how sure of that. A scalar filter and not a state on the pose filter: the
  // lean is what is left after a rigid alignment has already solved the pose
  // out, so it is not a function of any pose state and nothing would be gained
  // by carrying it beside them.
  double mounting_pitch = 0.0;
  double mounting_variance = 0.0;
  // The mount the correction is measured from, kept apart from `calibration`
  // because that is what the correction writes into.
  Eigen::Matrix3d mount_rotation = Eigen::Matrix3d::Identity();
  // The frame the calibration was last brought to, so a mounting correction can
  // be folded back into it without asking the caller for the size again.
  int frame_width = 0;
  int frame_height = 0;

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
  // The banded, paired ground points this camera contributed, in base_link at
  // each of the two frames. Both cameras' points live in the same frame and
  // describe the same hop, so they can be solved together rather than solved
  // apart and averaged. Only filled when that is asked for.
  Points2 pair_previous;
  Points2 pair_current;
  Identities pair_ids;
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
  anchor_settings.initial_variance = settings.anchor_initial_variance;
  anchor_settings.max_variance = settings.anchor_max_variance;
  anchor_settings.trial_observations = settings.anchor_trial_observations;
  anchor_settings.min_update_gain = settings.anchor_min_update_gain;
  anchor_settings.link_radius_m = settings.anchor_link_radius_m;
  anchor_settings.link_measure_only = settings.anchor_link_measure_only;
  anchor_settings.weight_by_information = settings.anchor_weight_by_information;
  anchor_settings.link_adopter_writes = settings.anchor_link_adopter_writes;
  anchor_settings.link_rebind_grace_frames = settings.anchor_link_rebind_grace;
  anchor_settings.evict_by_age = settings.anchor_evict_by_age;
  anchor_settings.evict_for_new = settings.anchor_evict_for_new;
  anchor_settings.drift_variance_per_m = settings.anchor_drift_variance_per_m;

  // One map, every camera. The capacity parameter stays per camera so the
  // configured number keeps its meaning.
  anchor_settings.max_anchors =
    settings.max_ground_anchors * std::max<int>(1, static_cast<int>(settings.cameras.size()));
  anchors_ = std::make_unique<GroundAnchorMap>(
    anchor_settings, std::max<int>(1, static_cast<int>(settings.cameras.size())));
  camera_travel_.assign(settings.cameras.size(), 0.0);
  camera_solves_.assign(settings.cameras.size(), 0);
  camera_inliers_.assign(settings.cameras.size(), 0.0);
  camera_spread_.assign(settings.cameras.size(), 0.0);
  camera_usable_.assign(settings.cameras.size(), 0.0);
  camera_known_.assign(settings.cameras.size(), 0.0);
  camera_offground_.assign(settings.cameras.size(), 0.0);
  camera_bearing_.assign(settings.cameras.size(), 0.0);
  camera_projected_.assign(settings.cameras.size(), 0.0);
  camera_bearings_.assign(settings.cameras.size(), 0);
  camera_looks_.assign(settings.cameras.size(), 0);
  camera_anchored_.assign(settings.cameras.size(), 0);
  int source_index = 0;
  for (const auto & camera : settings.cameras) {
    cameras_.push_back(std::make_unique<Camera>(camera, source_index++));
    cameras_.back()->mounting_variance = settings.mounting_pitch_variance;
    cameras_.back()->mount_rotation = camera.rotation_base_from_camera;
    // On the model rather than in a global, so two estimators in one process
    // cannot overwrite each other's.
    cameras_.back()->calibration.level_frame_origin = settings.level_frame_origin;
    cameras_.back()->model.level_frame_origin = settings.level_frame_origin;
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
  target.mount_rotation = rotation;
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
  CameraModel model = make_camera_model(
    k, camera.calibration.rotation_base_from_camera,
    camera.calibration.translation_base_from_camera, camera.calibration.distortion,
    camera.calibration.lens);
  model.level_frame_origin = settings_.level_frame_origin;
  return model;
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
  camera.frame_width = incoming.width;
  camera.frame_height = incoming.height;
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

ImuSample Estimator::shift_imu_to_base(const ImuSample & measured)
{
  ImuSample shifted = measured;
  const Eigen::Vector3d & arm = settings_.imu_translation_base_from_imu;
  if (arm.squaredNorm() <= 0.0) {
    return shifted;
  }
  const Eigen::Vector3d & rate = measured.angular_velocity;
  // The gyro is the only source of angular acceleration there is, so this is
  // a difference of two noisy rates divided by a small number. The tau exists
  // because of that, not as a tuning knob.
  if (imu_rate_stamp_.has_value()) {
    const double step = measured.stamp - *imu_rate_stamp_;
    if (step > 1e-6 && step <= 0.1) {
      const Eigen::Vector3d fresh = (rate - imu_rate_) / step;
      const double blend = settings_.imu_angular_accel_tau_sec > 0.0
        ? step / (settings_.imu_angular_accel_tau_sec + step) : 1.0;
      imu_angular_ += blend * (fresh - imu_angular_);
    }
  }
  imu_rate_ = rate;
  imu_rate_stamp_ = measured.stamp;
  // a_imu = a_base + alpha x r + omega x (omega x r). Gravity is common to
  // both and drops out, so this holds for the specific force as it stands.
  shifted.linear_acceleration = measured.linear_acceleration -
    imu_angular_.cross(arm) - rate.cross(rate.cross(arm));
  return shifted;
}

void Estimator::ingest_imu(const ImuSample & measured)
{
  if (measured.stamp <= 0.0) {
    return;
  }
  const ImuSample sample = shift_imu_to_base(measured);
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
      const Eigen::Vector3d raw = sample.linear_acceleration;
      Eigen::Vector3d held = raw;
      if (settings_.inertial_max_acceleration_mps2 > 0.0 &&
        raw.head<2>().norm() > settings_.inertial_max_acceleration_mps2)
      {
        held = last_specific_force_.value_or(Eigen::Vector3d(0.0, 0.0, 9.80665));
      } else {
        last_specific_force_ = raw;
      }
      imu_window_.push_back(
        AccelerationSample{
          sample.stamp + settings_.imu_acceleration_offset_sec, acceleration, body,
          sample.angular_velocity.z(), raw, held, sample.angular_velocity,
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
      continue;
    }
    if (!settings_.adaptive_solve_interval &&
      diagnostics_.pairs_seen % settings_.frame_decimation != 0)
    {
      continue;
    }
    process_pair();
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

// The propagator integrates against the orientation the instrument reports, so
// its world frame is the instrument's. The MSCKF and the six degree of freedom
// filter carry their own heading, whose zero is wherever the drive started.
Eigen::Vector2d Estimator::imu_world_velocity(const Eigen::Vector2d & velocity) const
{
  if (!imu_yaw_datum_.has_value()) {
    return velocity;
  }
  const double c = std::cos(*imu_yaw_datum_);
  const double s = std::sin(*imu_yaw_datum_);
  return Eigen::Vector2d(c * velocity.x() - s * velocity.y(), s * velocity.x() + c * velocity.y());
}

std::optional<Eigen::Vector2d> Estimator::fused_world_velocity() const
{
  // The order is the order the constructor builds them in, so whichever filter
  // this run owns is the one that answers. Unsettled means it has not seen a
  // vision update yet, and an accelerometer-only velocity is not worth
  // reckoning on.
  //
  // The two that carry their own heading answer in the instrument's frame, and
  // are turned into the estimator's the same way their velocity corrections
  // are -- the estimator's frame starts at zero yaw, the instrument's does not.
  if (spatial_filter_) {
    return spatial_filter_->settled()
           ? std::optional<Eigen::Vector2d>(
      imu_world_velocity(spatial_filter_->velocity().head<2>()))
           : std::nullopt;
  }
  if (msckf_filter_) {
    return msckf_filter_->settled()
           ? std::optional<Eigen::Vector2d>(imu_world_velocity(msckf_filter_->velocity()))
           : std::nullopt;
  }
  if (displacement_filter_) {
    return displacement_filter_->settled()
           ? std::optional<Eigen::Vector2d>(displacement_filter_->velocity())
           : std::nullopt;
  }
  return velocity_filter_.settled()
         ? std::optional<Eigen::Vector2d>(velocity_filter_.velocity())
         : std::nullopt;
}

// Buffer this frame's off-ground bearings and triangulate whatever has spread
// far enough. Called after the pose is settled, so every observation carries
// the pose it was actually taken from.
void Estimator::record_offground(Camera & camera)
{
  if (!settings_.offground_anchors || !camera.latest.has_value()) {
    return;
  }
  const Eigen::Index count = camera.track_ids.size();
  if (count == 0 || camera.track_pixels.rows() != count) {
    return;
  }
  ++camera.offground_frame;

  const auto tilt = body_tilt();
  Eigen::Matrix3d level = Eigen::Matrix3d::Identity();
  if (tilt.has_value()) {
    level = *tilt;
  }
  const Eigen::Vector3d mount = level * camera.model.translation_base_from_camera;
  const double c = std::cos(pose_.yaw);
  const double s = std::sin(pose_.yaw);
  Eigen::Matrix3d turn = Eigen::Matrix3d::Identity();
  turn(0, 0) = c; turn(0, 1) = -s; turn(1, 0) = s; turn(1, 1) = c;
  const Eigen::Vector3d centre =
    Eigen::Vector3d(pose_.x, pose_.y, 0.0) + turn * mount;

  const auto rays = pixels_to_bearings(camera.track_pixels, camera.model);
  const double parallax_limit =
    std::cos(settings_.offground_min_parallax_deg * M_PI / 180.0);
  const size_t cap = static_cast<size_t>(std::max(settings_.offground_max_observations, 3));

  for (Eigen::Index i = 0; i < count; ++i) {
    const int64_t identity = camera.track_ids(i);
    const Eigen::Vector3d bearing = turn * (level * rays.row(i).transpose());
    if (!bearing.allFinite()) {
      continue;
    }
    auto & track = camera.offground[identity];
    track.seen = camera.offground_frame;
    if (track.solved) {
      // A point fixed from one stretch of pose and read against another is an
      // anchor that carries the drift between them. The ground anchors do not
      // have the problem because they are averaged over every sighting; these
      // are triangulated once, so they have to be triangulated again.
      if (settings_.offground_refresh_frames <= 0 ||
        camera.offground_frame - track.fixed < settings_.offground_refresh_frames)
      {
        continue;
      }
      track.solved = false;
      track.centres.clear();
      track.bearings.clear();
    }
    // Keep the oldest -- it is half the baseline -- and drop from just after it.
    if (track.centres.size() >= cap) {
      track.centres.erase(track.centres.begin() + 1);
      track.bearings.erase(track.bearings.begin() + 1);
    }
    track.centres.push_back(centre);
    track.bearings.push_back(bearing);
    if (static_cast<int>(track.centres.size()) < settings_.offground_min_views) {
      continue;
    }
    double widest = 1.0;
    for (size_t a = 0; a < track.bearings.size(); ++a) {
      for (size_t b = a + 1; b < track.bearings.size(); ++b) {
        widest = std::min(widest, track.bearings[a].dot(track.bearings[b]));
      }
    }
    if (widest > parallax_limit) {
      continue;
    }
    // Midpoint of the rays: the point closest to all of them at once.
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d moment = Eigen::Vector3d::Zero();
    for (size_t n = 0; n < track.centres.size(); ++n) {
      const Eigen::Matrix3d block =
        Eigen::Matrix3d::Identity() - track.bearings[n] * track.bearings[n].transpose();
      normal += block;
      moment += block * track.centres[n];
    }
    Eigen::Matrix3d inverse;
    double determinant = 0.0;
    bool invertible = false;
    normal.computeInverseAndDetWithCheck(inverse, determinant, invertible);
    if (!invertible || determinant < 1e-6) {
      continue;
    }
    const Eigen::Vector3d point = inverse * moment;
    if (!point.allFinite()) {
      continue;
    }
    double residual = 0.0;
    for (size_t n = 0; n < track.centres.size(); ++n) {
      const Eigen::Vector3d off = point - track.centres[n];
      residual += (off - off.dot(track.bearings[n]) * track.bearings[n]).norm();
    }
    residual /= static_cast<double>(track.centres.size());
    const double range = (point - centre).norm();
    if (residual > settings_.offground_max_residual_m ||
      point.z() < settings_.offground_min_height_m ||
      range > settings_.offground_max_range_m || (point - centre).dot(bearing) <= 0.0)
    {
      continue;
    }
    track.world = point;
    track.solved = true;
    track.fixed = camera.offground_frame;
    track.centres.clear();
    track.bearings.clear();
    ++diagnostics_.offground_anchors;
  }

  // Anything not seen this frame for a while is gone; the tracker will not
  // bring that identity back.
  if (camera.offground_frame % 60 == 0) {
    for (auto it = camera.offground.begin(); it != camera.offground.end(); ) {
      it = camera.offground_frame - it->second.seen > 250
        ? camera.offground.erase(it) : std::next(it);
    }
  }
  diagnostics_.offground_live = 0;
  for (const auto & entry : camera.offground) {
    diagnostics_.offground_live += entry.second.solved ? 1 : 0;
  }
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

// The lean the alignment left, turned into a mounting pitch.
//
// One scalar measurement of one scalar state, so the whole filter is three
// lines. What it is worth is the measured gain -- 0.0056 of lean per degree --
// against a noise that is mostly the road rather than the instrument: a real
// grade tilts the ground under the camera exactly the way a mounting error
// does, and nothing here can tell them apart within one solve.
void Estimator::learn_mounting_pitch(Camera & camera, double lean)
{
  if (!camera.settings.learn_mounting_pitch || settings_.mounting_pitch_gain == 0.0) {
    return;
  }
  const double noise = settings_.mounting_pitch_noise * settings_.mounting_pitch_noise;
  const double gain = settings_.mounting_pitch_gain;
  const double innovation = lean - gain * camera.mounting_pitch;
  const double block = gain * gain * camera.mounting_variance + noise;
  if (!(block > 0.0)) {
    return;
  }
  const double k = camera.mounting_variance * gain / block;
  camera.mounting_pitch += k * innovation;
  camera.mounting_variance = std::max((1.0 - k * gain) * camera.mounting_variance, 0.0) +
    settings_.mounting_pitch_walk * settings_.mounting_pitch_walk;
  if (!settings_.mounting_pitch_apply) {
    return;
  }
  // Onto the mount as it stands, which is the transform tree's when there is
  // one -- composing onto the parameter default would quietly discard it.
  camera.calibration.rotation_base_from_camera =
    Eigen::AngleAxisd(-camera.mounting_pitch, Eigen::Vector3d::UnitY()).toRotationMatrix() *
    camera.mount_rotation;
  if (camera.frame_width > 0 && camera.frame_height > 0) {
    camera.model = frame_model(camera, camera.frame_width, camera.frame_height);
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

void Estimator::steer_by_epipolar(
  const Camera & camera, Solved & solved, const Points2 & previous_pixels,
  const Points2 & current_pixels)
{
  if ((settings_.epipolar_weight <= 0.0 && settings_.epipolar_reject_deg <= 0.0 &&
    settings_.epipolar_trust_deg <= 0.0) ||
    !solved.motion.has_value())
  {
    return;
  }
  PlanarMotion & motion = *solved.motion;
  const Eigen::Vector2d hop(motion.x, motion.y);
  if (hop.norm() < settings_.epipolar_min_hop_m) {
    return;
  }
  // Off the ground by default. A point the projection rejected is either too
  // far, too near, or above the horizon, and the last of those is the whole
  // point -- it is structure the plane cannot explain, so its parallax says
  // something the ground solve has not already used.
  const Mask use = settings_.epipolar_non_ground_only
    ? Mask(!solved.ground_valid) : Mask::Constant(previous_pixels.rows(), true);
  const double yaw = motion.yaw;
  const auto bearing = epipolar_bearing(
    previous_pixels, current_pixels, use, camera.model, yaw,
    settings_.epipolar_softness_rad, settings_.ground_min_inliers);
  if (!bearing.has_value()) {
    return;
  }
  // The constraint sees the camera move, not the axle. Under yaw the mount
  // swings through an arc of its own, and that part of the bearing is nothing
  // to do with where the vehicle went.
  Eigen::Matrix2d turned;
  turned << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
  const Eigen::Vector2d lever = (turned - Eigen::Matrix2d::Identity()) *
    camera.model.translation_base_from_camera.head<2>();
  const Eigen::Vector2d travelled = hop + lever;
  const double length = travelled.norm();
  if (length < 1e-6) {
    return;
  }
  // Sign is the caller's to settle: x2'[t]xR x1 = 0 holds for -t as readily
  // as for t, so the epipolar geometry cannot tell forward from reverse.
  const Eigen::Vector2d direction =
    bearing->dot(travelled) < 0.0 ? Eigen::Vector2d(-*bearing) : *bearing;
  const double disagreement = std::abs(
    wrap_pi(std::atan2(direction.y(), direction.x()) -
    std::atan2(travelled.y(), travelled.x()))) * 180.0 / M_PI;
  camera_bearing_[camera.source] += disagreement;
  ++camera_bearings_[camera.source];
  if (settings_.epipolar_reject_deg > 0.0 && disagreement > settings_.epipolar_reject_deg) {
    solved.motion.reset();
    return;
  }
  if (settings_.epipolar_trust_deg > 0.0) {
    const double relative = disagreement / settings_.epipolar_trust_deg;
    motion.inliers = std::max(
      1, static_cast<int>(std::lround(motion.inliers * std::exp(-0.5 * relative * relative))));
  }
  const Eigen::Vector2d steered = length * direction - lever;
  const double share = std::clamp(settings_.epipolar_weight, 0.0, 1.0);
  motion.x += share * (steered.x() - motion.x);
  motion.y += share * (steered.y() - motion.y);
}

// The Gaussian weight's width is a residual scale, so it can be measured
// rather than configured. What the constant cannot be is right for every
// drive: the residual runs 0.012 m on a clean straight and 0.089 m through a
// parking manoeuvre, and it differs by two between the two cameras of one rig.
double Estimator::softness_for(const Camera & camera, double configured) const
{
  if (settings_.softness_from_residual <= 0.0 || camera.residual_scale <= 0.0) {
    return configured;
  }
  return settings_.softness_from_residual * camera.residual_scale;
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

  // Cleared here and set only where the map answers, so a hop that fell back
  // in between cannot be mistaken for the previous registration.
  const bool had_placed = camera.placed_fresh;
  camera.placed_fresh = false;

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
      tilt_ptr, camera.settings.range_scale * camera.range_scale_learned * imu_scale_,
      solved.current_ground, valid_current);
    pixels_to_ground(
      previous_pixels, camera.model, band, camera.settings.ground_min_distance_m,
      tilt_ptr, camera.settings.range_scale * camera.range_scale_learned * imu_scale_,
      solved.previous_ground, valid_previous);
  }
  solved.ground_valid = valid_previous && valid_current;

  solved.motion_inliers = Mask::Constant(count, false);
  if (!yaw_delta.has_value() || !solved.ground_valid.any()) {
    remember_solve_pixels(camera);
    return solved;
  }

  Stopwatch watch(diagnostics_, "solve");

  // Widened by whatever this hop turns through, for both solve paths.
  const double gate = settings_.ground_ransac_threshold_m +
    settings_.ground_rotation_threshold_m * std::abs(*yaw_delta);

  // Prefer the accumulated map; fall back to the previous frame alone. Matching
  // against anchors averaged over a feature's whole life is what stops a burst
  // of bad matches from carrying the estimate, which is how the two-frame solve
  // failed above walking pace.
  std::vector<Eigen::Index> usable;
  usable.reserve(static_cast<size_t>(count));
  // The pose may want a tighter band than the map does. A ground point's
  // position error grows with its range -- the same angular error is worth
  // proportionally more centimetres further out -- so the far half of the band
  // registers worse than it maps. Bounding the solve alone keeps the grid's
  // reach, which is what ground_max_distance_m is for.
  const double solve_band = settings_.solve_max_distance_m > 0.0
    ? settings_.solve_max_distance_m : std::numeric_limits<double>::infinity();
  Eigen::Vector3d mount_in_frame = camera.model.translation_base_from_camera;
  if (settings_.level_frame_origin && tilt.has_value()) {
    mount_in_frame = tilt.value() * mount_in_frame;
  }
  const Eigen::Vector2d lens = mount_in_frame.head<2>();
  // What a ground point at this position is worth. Its bearing error is
  // multiplied by (R^2 + h^2) / h on the way to a position, so the inverse
  // variance goes as (R^2 + h^2)^-2. Power 0 leaves every point equal, which
  // is what this solve did before.
  const double lens_height = camera.model.translation_base_from_camera.z();
  // A road-warp point is worth more than a corner because it is placed by a
  // fit over the whole region rather than by one patch of flow.
  const auto identity_weight = [&](Eigen::Index i) {
      return track_ids(i) >= kRoadIdentity ? settings_.road_point_weight : 1.0;
    };
  const auto range_weight = [&](double x, double y) {
      if (settings_.range_weight_power <= 0.0) {
        return 1.0;
      }
      const double dx = x - lens.x();
      const double dy = y - lens.y();
      const double squared = dx * dx + dy * dy + lens_height * lens_height;
      return std::pow(squared, -settings_.range_weight_power);
    };
  // Where in the frame a feature is allowed to be. Off unless x1 > x0.
  const bool region = settings_.pixel_region_x1 > settings_.pixel_region_x0 &&
    camera.frame_width > 0 && camera.frame_height > 0;
  const auto outside = [&](Eigen::Index i) {
      if (!region) {
        return false;
      }
      const double fx = current_pixels(i, 0) / camera.frame_width;
      const double fy = current_pixels(i, 1) / camera.frame_height;
      return fx < settings_.pixel_region_x0 || fx > settings_.pixel_region_x1 ||
             fy < settings_.pixel_region_y0 || fy > settings_.pixel_region_y1;
    };
  // How far a feature swept across the frame over this solve, and whether that
  // is more than the flow could have followed.
  //
  // The band above is in metres and what decides whether a feature is
  // trackable is pixels. A ground point's angular rate goes as h/(R^2+h^2), so
  // the near field accelerates as it approaches: at 0.89 m of camera height and
  // 0.53 m of travel -- 8 m/s over the two frames a solve spans -- a point
  // 0.6 m ahead sweeps 135 px against the 21 px window, while at 2 m/s the same
  // point sweeps 29. The same metre gate admits a good feature at one speed and
  // an untrackable one at another, and the rear camera does not have the
  // problem at all because its ground recedes and decelerates.
  const double disparity_limit = settings_.solve_max_pixel_flow > 0.0
    ? settings_.solve_max_pixel_flow : std::numeric_limits<double>::infinity();
  const auto sweep_of = [&](Eigen::Index i) {
      return std::hypot(
        current_pixels(i, 0) - previous_pixels(i, 0),
        current_pixels(i, 1) - previous_pixels(i, 1));
    };
  // And a floor, relative to what the rest of the frame did.
  //
  // A flow that cannot find its feature returns the point it started from, and
  // a point that did not move passes the forward-backward check perfectly:
  // zero out, zero back. It is the one failure that gate cannot see. At 8 m/s
  // the front camera's median published flow is 0.19 px against 5.24 at 4 m/s
  // and 4.05 at 2, with only 42% of features moving more than a pixel -- more
  // than half the frame is a feature that stayed where it was while the ground
  // swept past it. Those vote "did not move", which is the local optimum the
  // alignment's multi-start exists to escape.
  double floor_flow = 0.0;
  if (settings_.solve_min_pixel_flow > 0.0 && previous_pixels.rows() == count &&
    count > 0)
  {
    std::vector<double> sweeps;
    sweeps.reserve(static_cast<size_t>(count));
    for (Eigen::Index i = 0; i < count; ++i) {
      if (solved.ground_valid(i)) {
        sweeps.push_back(sweep_of(i));
      }
    }
    if (!sweeps.empty()) {
      const size_t half = sweeps.size() / 2;
      std::nth_element(sweeps.begin(), sweeps.begin() + half, sweeps.end());
      floor_flow = settings_.solve_min_pixel_flow * sweeps[half];
    }
  }
  const auto swept = [&](Eigen::Index i) {
      if (previous_pixels.rows() != count) {
        return false;
      }
      const double moved = sweep_of(i);
      return moved > disparity_limit || moved < floor_flow;
    };
  for (Eigen::Index i = 0; i < count; ++i) {
    if (!solved.ground_valid(i) || outside(i) || swept(i)) {
      continue;
    }
    if (std::isfinite(solve_band)) {
      // Both sightings, not just the current one. `ground_valid` is already
      // the AND of the two frames' band tests, and that conjunction is what
      // narrowing the band actually tightens -- a point that was far last
      // frame carried its error into the hop whether or not it is near now.
      const double dx = solved.current_ground(i, 0) - lens.x();
      const double dy = solved.current_ground(i, 1) - lens.y();
      const double px = solved.previous_ground(i, 0) - lens.x();
      const double py = solved.previous_ground(i, 1) - lens.y();
      if (std::hypot(dx, dy) > solve_band || std::hypot(px, py) > solve_band) {
        continue;
      }
    }
    usable.push_back(i);
  }

  if (settings_.fuse_camera_points || settings_.unified_solve) {
    const Eigen::Index kept = static_cast<Eigen::Index>(usable.size());
    solved.pair_previous.resize(kept, 2);
    solved.pair_current.resize(kept, 2);
    solved.pair_ids.resize(kept);
    for (Eigen::Index i = 0; i < kept; ++i) {
      const Eigen::Index at = usable[static_cast<size_t>(i)];
      solved.pair_previous(i, 0) = solved.previous_ground(at, 0);
      solved.pair_previous(i, 1) = solved.previous_ground(at, 1);
      solved.pair_current(i, 0) = solved.current_ground(at, 0);
      solved.pair_current(i, 1) = solved.current_ground(at, 1);
      solved.pair_ids(i) = track_ids(at);
    }
  }

  Identities usable_ids(static_cast<Eigen::Index>(usable.size()));
  for (size_t i = 0; i < usable.size(); ++i) {
    usable_ids(static_cast<Eigen::Index>(i)) = track_ids(usable[i]);
  }
  Mask anchored;
  {
    Stopwatch lookup(diagnostics_, "lookup");
    anchors_->anchored(camera.source, usable_ids, anchored);
  }
  // How much of what the solve could use the map actually knows. The map path
  // needs ground_min_inliers of these; below that the whole frame falls back
  // to comparing two views, which is the thing the anchor map exists to beat.
  camera_usable_[camera.source] += static_cast<double>(usable.size());
  camera_known_[camera.source] += static_cast<double>(anchored.count());
  camera_looks_[camera.source] += 1;

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
    Weights scale;
    {
      Stopwatch lookup(diagnostics_, "lookup");
      anchors_->anchor_view(camera.source, selected_ids, world, weights);
    }
    // Range is deliberately not folded in here. An anchor already carries a
    // weight -- how many sightings agree on it -- and multiplying range on top
    // counts the same geometry twice. The two-frame fallback below has no such
    // weight, and that is where range belongs.
    if (weights.size() == chosen) {
      for (Eigen::Index i = 0; i < chosen; ++i) {
        weights(i) *= identity_weight(selected[static_cast<size_t>(i)]);
      }
    }

    // And whatever is not on the road. A triangulated point has no plane to
    // read a range off, so its body position comes from the bearing seen now
    // and the range predicted from where the pose was last put. The range
    // error that carries lies *along* the ray, which is the direction the
    // alignment learns nothing from; what it uses is where the ray points, and
    // that is measured.
    if (settings_.offground_anchors && !camera.offground.empty()) {
      const auto tilt = body_tilt();
      Eigen::Matrix3d level = Eigen::Matrix3d::Identity();
      if (tilt.has_value()) {
        level = *tilt;
      }
      const Eigen::Vector3d mount = level * camera.model.translation_base_from_camera;
      // Carried onto where the vehicle is expected to be, not where it was
      // last put. The range is predicted from this, and a hop's worth of lag
      // in it displaces every point along its own ray -- 0.5 m of hop against
      // a 5 m range is a tenth of the way there.
      const double yaw_now = wrap_pi(pose_.yaw + *yaw_delta);
      Eigen::Vector2d where(pose_.x, pose_.y);
      if (expected_hop_.has_value()) {
        where += *expected_hop_;
      }
      const double cy = std::cos(yaw_now);
      const double sy = std::sin(yaw_now);
      const Eigen::Vector3d centre(
        where.x() + cy * mount.x() - sy * mount.y(),
        where.y() + sy * mount.x() + cy * mount.y(), mount.z());
      const auto rays = pixels_to_bearings(current_pixels, camera.model);
      std::vector<Eigen::Index> extra;
      std::vector<Eigen::Vector2d> extra_body;
      std::vector<Eigen::Vector2d> extra_world;
      std::vector<double> extra_range;
      extra.reserve(static_cast<size_t>(count));
      for (Eigen::Index i = 0; i < count; ++i) {
        const auto found = camera.offground.find(track_ids(i));
        if (found == camera.offground.end() || !found->second.solved) {
          continue;
        }
        const Eigen::Vector3d & point = found->second.world;
        const double range = (point - centre).norm();
        if (!(range > 0.5) || range > settings_.offground_max_range_m) {
          continue;
        }
        const Eigen::Vector3d bearing = level * rays.row(i).transpose();
        const Eigen::Vector3d placed = mount + range * bearing;
        if (!placed.allFinite()) {
          continue;
        }
        extra.push_back(i);
        extra_body.emplace_back(placed.x(), placed.y());
        extra_world.emplace_back(point.x(), point.y());
        extra_range.push_back(range);
      }
      const Eigen::Index added = static_cast<Eigen::Index>(extra.size());
      if (added > 0) {
        camera_offground_[camera.source] += static_cast<double>(added);
        body.conservativeResize(chosen + added, 2);
        world.conservativeResize(chosen + added, 2);
        // The ground points keep the configured gate; these are judged on the
        // same angle instead, which at their range is a wider distance.
        scale.setOnes(chosen + added);
        const bool carry = weights.size() == chosen;
        if (carry) {
          weights.conservativeResize(chosen + added);
        }
        for (Eigen::Index i = 0; i < added; ++i) {
          body(chosen + i, 0) = extra_body[static_cast<size_t>(i)].x();
          body(chosen + i, 1) = extra_body[static_cast<size_t>(i)].y();
          world(chosen + i, 0) = extra_world[static_cast<size_t>(i)].x();
          world(chosen + i, 1) = extra_world[static_cast<size_t>(i)].y();
          scale(chosen + i) = std::max(
            extra_range[static_cast<size_t>(i)] / settings_.offground_residual_range_m,
            1.0);
          if (carry) {
            weights(chosen + i) = settings_.offground_weight;
          }
        }
      }
    }

    // Where the solve is told to start looking. With the MSCKF that is the
    // filter's own propagated heading rather than the instrument's, and the
    // solve is asked to improve on it -- the improvement is the measurement.
    const double handed = yaw_guess.has_value()
      ? *yaw_guess : wrap_pi(pose_.yaw + *yaw_delta);
    // The alignment solves for where the camera *is*, not for how far it
    // moved, so the inertial expectation has to be carried onto the last
    // solved position before it can gate anything. Comparing a hop against an
    // absolute translation was the first version of this and rejected almost
    // every mode.
    std::optional<Eigen::Vector2d> gate_centre;
    if (settings_.inertial_gate_m > 0.0 && expected_hop_.has_value()) {
      // From the pose actually held, not from the last alignment: the map path
      // answers only a third of the time, so its record goes stale.
      gate_centre = Eigen::Vector2d(pose_.x, pose_.y) + *expected_hop_;
    }
    std::optional<AnchorAlignment> aligned;
    {
      Stopwatch align(diagnostics_, "align");
      aligned = align_to_anchors(
        body, world, weights, handed, gate,
        settings_.ground_min_inliers,
        settings_.align_solves_yaw &&
        (heading_.enabled() || msckf_filter_ != nullptr || spatial_filter_ != nullptr),
        lens, settings_.radial_min_range_m,
        softness_for(camera, settings_.ground_align_softness_m),
        settings_.align_seed_from_last_hop && camera.last_translation.has_value()
        ? &*camera.last_translation : nullptr,
        settings_.align_restarts, settings_.align_ambiguity_ratio,
        gate_centre.has_value() ? &*gate_centre : nullptr,
        settings_.inertial_gate_m, scale);
    }
    if (aligned.has_value()) {
      camera.last_translation = aligned->translation;
      if (aligned->radial_reference > 0.0) {
        camera.radial_linear_sum += aligned->radial_linear;
        ++camera.radial_samples;
        // Positive residual means the ground read further away than the map
        // says, which is a camera believed to sit higher than it does.
        if (settings_.range_scale_gain != 0.0) {
          const double relative = aligned->radial_linear / aligned->radial_reference;
          if (std::isfinite(relative)) {
            const double step = std::clamp(
              settings_.range_scale_gain * relative, -0.01, 0.01);
            camera.range_scale_learned = std::clamp(
              camera.range_scale_learned * (1.0 + step), 0.9, 1.1);
          }
        }
        learn_mounting_pitch(camera, aligned->radial_linear);
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
      // Take back a share of where this camera's map stood last time. The
      // difference between `placed - pose_` and `placed - last_placed` is
      // exactly that standing offset, so a gain of 1 turns the correction into
      // a plain displacement and 0 leaves it as it was. Measured, the two
      // cameras' maps walk apart over a drive -- the front/rear ratio runs
      // 1.09 to 1.95 across the fifths of str_v2 while str_v3 holds 0.98 --
      // and this is the only path by which that walk reaches the pose.
      if (settings_.anchor_divergence_gain > 0.0 && had_placed &&
        camera.last_placed.has_value())
      {
        const PlanarMotion standing = relative_motion(pose_, *camera.last_placed);
        motion.x -= settings_.anchor_divergence_gain * standing.x;
        motion.y -= settings_.anchor_divergence_gain * standing.y;
      }
      camera.last_placed = placed;
      camera.placed_fresh = true;
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
      steer_by_epipolar(camera, solved, previous_pixels, current_pixels);
      remember_solve_pixels(camera);
      return solved;
    }
  }

  // The map could not answer. Fall back to comparing the two frames directly.
  std::vector<Eigen::Index> pairs;
  pairs.reserve(static_cast<size_t>(count));
  for (Eigen::Index i = 0; i < count; ++i) {
    if (!solved.ground_valid(i) || outside(i)) {
      continue;
    }
    // The same band the map path uses. Most solves come through here whenever
    // the map cannot answer, so leaving it unbounded left the far half of the
    // ground in the estimate by the back door.
    if (std::isfinite(solve_band)) {
      const double dx = solved.current_ground(i, 0) - lens.x();
      const double dy = solved.current_ground(i, 1) - lens.y();
      const double px = solved.previous_ground(i, 0) - lens.x();
      const double py = solved.previous_ground(i, 1) - lens.y();
      if (std::hypot(dx, dy) > solve_band || std::hypot(px, py) > solve_band) {
        continue;
      }
    }
    pairs.push_back(i);
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
  Weights pair_weights;
  if (settings_.range_weight_power > 0.0 || settings_.road_point_weight != 1.0) {
    pair_weights.resize(paired);
    for (Eigen::Index i = 0; i < paired; ++i) {
      pair_weights(i) = range_weight(current_ground(i, 0), current_ground(i, 1)) *
        identity_weight(pairs[static_cast<size_t>(i)]);
    }
  }
  const auto estimate = estimate_planar_motion_with_yaw(
    previous_ground, current_ground, *yaw_delta, gate,
    settings_.ground_min_inliers, softness_for(camera, settings_.ground_pair_softness_m),
    pair_weights,
    settings_.ground_pair_passes);
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
    steer_by_epipolar(camera, solved, previous_pixels, current_pixels);
  }

  // A running mean, not the last value: solves come in at ten to fifty a
  // second and one bad hop should not resize the weight. The inliers this is
  // measured over are chosen by the hard gate, not by the softness, so
  // narrowing the weight cannot narrow its own evidence.
  if (solved.spread > 0.0) {
    camera.residual_scale = camera.residual_scale > 0.0
      ? camera.residual_scale + (solved.spread - camera.residual_scale) / 32.0
      : solved.spread;
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
        Eigen::Vector3d force =
          settings_.spatial_screen_impulses ? sample.held_force : sample.specific_force;
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

  // What inertial propagation expects this hop to be, in the world frame. It
  // is not accurate enough to steer the solve -- feeding it in as a starting
  // point was measured and made things worse -- but it is far more than
  // accurate enough to rule out "the vehicle did not move", which is the
  // failure that swallows the slow drives. Used as a gate, never as a seed.
  expected_hop_.reset();
  if (displacement_filter_ && dt > 1e-4 && displacement_filter_->settled()) {
    expected_hop_ = displacement_filter_->velocity() * dt;
  }

  // Read here, before the solves, because remembering a solve frame resets it.
  // What it holds at this point is the disparity accumulated since the last
  // solve, which is exactly what the zero-velocity test downstream wants -- and
  // reading it there gets zero from every camera, so the gate always passed.
  double hop_disparity = -1.0;
  bool have_disparity = false;
  for (const auto & camera : cameras_) {
    if (camera->latest.has_value()) {
      hop_disparity = std::max(hop_disparity, camera->hop_disparity);
      have_disparity = true;
    }
  }

  std::vector<std::optional<Solved>> solved(count);
  if (imu_available) {
    for (size_t i = 0; i < count; ++i) {
      solved[i] = solve_camera(*cameras_[i], yaw_delta, yaw_guess);
    }
  }
  // Nothing else. The solve frame stays where it is, deliberately: the vision
  // half of this interval is sound and only the heading prior is missing, so
  // remembering the current frame here would drop the motion since the last
  // solve out of the trajectory for good. Held, the next pair spans the gap and
  // measures it over a longer baseline. Nothing has to bound the hold -- every
  // path through solve_camera ends by remembering, so the first pair after the
  // instrument comes back resyncs whether or not it finds a motion.

  // Carried out to the diagnostics here, where the cameras are indexed: the
  // solve holds a reference and does not know which one it was handed.
  diagnostics_.radial_linear.assign(count, 0.0);
  diagnostics_.radial_samples.assign(count, 0);
  diagnostics_.mounting_pitch.assign(count, 0.0);
  for (size_t i = 0; i < count; ++i) {
    const Camera & camera = *cameras_[i];
    diagnostics_.radial_samples[i] = camera.radial_samples;
    diagnostics_.mounting_pitch[i] = camera.mounting_pitch * 180.0 / M_PI;
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
    // A car cannot slide sideways. Measured against truth at the rear axle,
    // the lateral part of a hop is 0.02-0.66 mm where the hop itself is
    // 22-62 mm -- under 1% even through the parking manoeuvre, and exactly
    // zero on a straight. The registration solves that component anyway, so
    // whatever noise lands there is taken for motion. Shrinking it towards the
    // constraint keeps the arc (a finite hop along a curve does carry a little
    // lateral, which is why this is a gain and not a projection to zero).
    if (settings_.nonholonomic_lateral > 0.0) {
      // Towards the arc's own lateral, not towards zero. A hop along a curve
      // is a chord, and a chord sits at half the hop's yaw from the body x
      // axis, so its lateral part is x*tan(dyaw/2) -- squeezing that to zero
      // sends the vehicle down the tangent instead and buys a systematic error
      // for a random one. Measured on curve_s05, damping to zero stops helping
      // at 0.99 with 0.273 m of cross error left, which is what the tangent
      // costs over 832 hops.
      const double arc = measured.x * std::tan(0.5 * measured.yaw);
      measured.y += settings_.nonholonomic_lateral * (arc - measured.y);
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

  // How far each camera thought the vehicle went, summed. The body is rigid,
  // so two cameras measuring the same hop must agree; a persistent ratio
  // between them is a difference in the scale each one derives from its own
  // ground projection, and unlike the radial residual it does not vanish just
  // because the map was built with the same scale.
  for (size_t i = 0; i < solved.size(); ++i) {
    if (!solved[i].has_value()) {
      continue;
    }
    const auto & moved = solved[i]->motion;
    if (moved.has_value()) {
      camera_travel_[i] += std::hypot(moved->x, moved->y);
      camera_solves_[i] += 1;
      camera_inliers_[i] += static_cast<double>(moved->inliers);
      camera_spread_[i] += solved[i]->spread;
      camera_anchored_[i] += solved[i]->anchored_from_map ? 1 : 0;
    }
  }

  bool aligned_from_map = false;
  for (const auto & entry : solved) {
    if (entry.has_value() && entry->anchored_from_map) {
      ++diagnostics_.map_aligned_frames;
      aligned_from_map = true;
      break;
    }
  }

  std::vector<double> fusion_weights;
  fusion_weights.reserve(motions.size());
  bool weighted = settings_.map_solve_weight != 1.0;
  for (size_t i = 0; i < solved.size(); ++i) {
    if (solved[i].has_value() && solved[i]->motion.has_value()) {
      const double own = cameras_[i]->settings.fusion_weight;
      weighted = weighted || own > 0.0;
      // Falls back to the inlier count, which is what carried the weighting
      // before any of this existed.
      double weight = own > 0.0
        ? own : static_cast<double>(std::max(solved[i]->motion->inliers, 1));
      if (solved[i]->anchored_from_map) {
        weight *= settings_.map_solve_weight;
      }
      fusion_weights.push_back(weight);
    }
  }
  if (!weighted || fusion_weights.size() != motions.size() ||
    std::any_of(
      fusion_weights.begin(), fusion_weights.end(), [](double w) {return !(w > 0.0);}))
  {
    fusion_weights.clear();
  }
  auto motion = fuse_planar_motions(motions, fusion_weights);

  // One solve over every reference there is.
  //
  // Each feature votes h = reference - R(dyaw) q. The reference is the anchor
  // map's averaged world position where the map knows the feature, and the
  // feature's own previous ground position where it does not -- the same
  // equation either way, so both go in one list and one consensus decides.
  // Both cameras are in that list too: the body is rigid and the hop is one
  // hop, so there is nothing to fuse afterwards.
  if (settings_.unified_solve && yaw_delta.has_value() && dt > 1e-4) {
    Eigen::Index total = 0;
    for (const auto & entry : solved) {
      if (entry.has_value()) {
        total += entry->pair_ids.size();
      }
    }
    if (total >= settings_.ground_min_inliers) {
      // One reference type for the whole hop, decided before any vote is cast.
      // A pair says what the hop was; an anchor says what the hop plus the
      // accumulated drift was. Two cameras can disagree about which they have,
      // but the consensus cannot hold both, so the choice belongs to the hop.
      bool take_map = false;
      if (settings_.unified_exclusive) {
        int64_t anchored_total = 0;
        for (size_t c = 0; c < solved.size(); ++c) {
          if (!solved[c].has_value() || solved[c]->pair_ids.size() == 0) {
            continue;
          }
          Points2 peek;
          Weights seen;
          anchors_->anchor_view(cameras_[c]->source, solved[c]->pair_ids, peek, seen);
          if (seen.size() == solved[c]->pair_ids.size()) {
            for (Eigen::Index i = 0; i < seen.size(); ++i) {
              anchored_total += seen(i) > 0.0 ? 1 : 0;
            }
          }
        }
        take_map = anchored_total >= settings_.ground_min_inliers;
      }
      Points2 reference(total, 2);
      Points2 observed(total, 2);
      Weights vote_weights(total);
      const double cp = std::cos(pose_.yaw);
      const double sp = std::sin(pose_.yaw);
      Eigen::Index at = 0;
      int64_t from_map = 0;
      for (size_t c = 0; c < solved.size(); ++c) {
        if (!solved[c].has_value() || solved[c]->pair_ids.size() == 0) {
          continue;
        }
        const auto & entry = *solved[c];
        const Eigen::Index rows = entry.pair_ids.size();
        Points2 world;
        Weights known;
        {
          Stopwatch lookup(diagnostics_, "lookup");
          anchors_->anchor_view(cameras_[c]->source, entry.pair_ids, world, known);
        }
        bool have = world.rows() == rows && known.size() == rows;
        // A pair's reference is a direct observation in the previous body
        // frame, so it carries no accumulated error. An anchor's is a world
        // position brought into that frame through the previous pose, so it
        // carries whatever that pose has drifted. The two measure different
        // things -- a hop, and a hop plus the correction for the drift.
        if (settings_.unified_exclusive && !take_map) {
          have = false;
        }
        for (Eigen::Index i = 0; i < rows; ++i) {
          // An anchor answers with a world position; bring it into the frame
          // the previous sighting would have been expressed in, and the two
          // references become interchangeable.
          const bool anchored_here = have && known(i) > 0.0;
          if (settings_.unified_exclusive && have && !anchored_here) {
            continue;   // this camera is answering from the map this hop
          }
          if (anchored_here) {
            const double wx = world(i, 0) - pose_.x;
            const double wy = world(i, 1) - pose_.y;
            reference(at, 0) = cp * wx + sp * wy;
            reference(at, 1) = -sp * wx + cp * wy;
            vote_weights(at) = settings_.unified_anchor_weight * known(i);
            ++from_map;
          } else {
            reference(at, 0) = entry.pair_previous(i, 0);
            reference(at, 1) = entry.pair_previous(i, 1);
            vote_weights(at) = 1.0;
          }
          observed(at, 0) = entry.pair_current(i, 0);
          observed(at, 1) = entry.pair_current(i, 1);
          ++at;
        }
      }
      if (at < total) {
        reference.conservativeResize(at, 2);
        observed.conservativeResize(at, 2);
        vote_weights.conservativeResize(at);
        total = at;
      }
      const double unified_gate = settings_.ground_ransac_threshold_m +
        settings_.ground_rotation_threshold_m * std::abs(*yaw_delta);
      const auto pooled = estimate_planar_motion_with_yaw(
        reference, observed, *yaw_delta, unified_gate,
        settings_.ground_min_inliers, settings_.ground_pair_softness_m,
        vote_weights, settings_.ground_pair_passes);
      if (pooled.has_value()) {
        if (!motion.has_value()) {
          motion = pooled->motion;
        } else {
          motion->x = pooled->motion.x;
          motion->y = pooled->motion.y;
          motion->inliers = pooled->motion.inliers;
        }
        diagnostics_.unified_votes += total;
        diagnostics_.unified_from_map += from_map;
        aligned_from_map = aligned_from_map || from_map > 0;
      }
    }
  }
  // Or one solve over everything both cameras saw. The average above treats
  // each camera's answer as a measurement and weighs them; this treats the
  // ground itself as the measurement, which is what it is. The body is rigid,
  // the points are already in base_link, and the hop is the same hop.
  // Only where the map could not answer. Overwriting a map-anchored hop with a
  // two-frame one throws away the thing that bounds the random walk, which is
  // a different experiment from the one this switch is for.
  bool any_anchored = false;
  for (const auto & entry : solved) {
    any_anchored = any_anchored || (entry.has_value() && entry->anchored_from_map);
  }
  if (settings_.fuse_camera_points && !any_anchored &&
    yaw_delta.has_value() && motion.has_value())
  {
    Eigen::Index total = 0;
    for (const auto & entry : solved) {
      if (entry.has_value()) {
        total += entry->pair_previous.rows();
      }
    }
    if (total >= settings_.ground_min_inliers) {
      Points2 pooled_previous(total, 2);
      Points2 pooled_current(total, 2);
      Eigen::Index at = 0;
      for (const auto & entry : solved) {
        if (!entry.has_value() || entry->pair_previous.rows() == 0) {
          continue;
        }
        const Eigen::Index rows = entry->pair_previous.rows();
        pooled_previous.block(at, 0, rows, 2) = entry->pair_previous;
        pooled_current.block(at, 0, rows, 2) = entry->pair_current;
        at += rows;
      }
      const double pooled_gate = settings_.ground_ransac_threshold_m +
        settings_.ground_rotation_threshold_m * std::abs(*yaw_delta);
      const auto pooled = estimate_planar_motion_with_yaw(
        pooled_previous, pooled_current, *yaw_delta, pooled_gate,
        settings_.ground_min_inliers, settings_.ground_pair_softness_m, Weights(),
        settings_.ground_pair_passes);
      if (pooled.has_value()) {
        motion->x = pooled->motion.x;
        motion->y = pooled->motion.y;
        motion->inliers = pooled->motion.inliers;
      }
    }
  }
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

  // The lateral each camera's own scale error contributes through the turn,
  // summed with the weights the fusion used. Taken off before anything else
  // touches the hop, because it is an error in the measurement rather than in
  // what the measurement is compared against.
  if (settings_.camera_split_lever != 0.0 && motion.has_value() && motions.size() >= 2) {
    const double reach = std::hypot(motion->x, motion->y);
    double total = 0.0;
    double lever = 0.0;
    size_t seen = 0;
    for (size_t i = 0; i < solved.size(); ++i) {
      if (!solved[i].has_value() || !solved[i]->motion.has_value()) {
        continue;
      }
      const double weight = seen < fusion_weights.size()
        ? fusion_weights[seen] : static_cast<double>(std::max(motions[seen].inliers, 1));
      ++seen;
      if (reach > 1e-9) {
        const double along =
          (solved[i]->motion->x * motion->x + solved[i]->motion->y * motion->y) / reach;
        lever += weight * ((along - reach) / reach) *
          cameras_[i]->model.translation_base_from_camera.x();
      }
      total += weight;
    }
    if (total > 0.0) {
      motion->y -= settings_.camera_split_lever * motion->yaw * lever / total;
    }
  }

  // Curvature carries a scale error of its own, shared by both cameras and so
  // invisible to their disagreement. Taken off the fused hop rather than each
  // camera's, because it is not a property of either.
  if (motion.has_value() &&
    (settings_.curvature_scale_gain != 0.0 || settings_.vision_scale != 1.0))
  {
    const double reach = std::hypot(motion->x, motion->y);
    if (reach > 1e-6) {
      const double correction = std::clamp(
        settings_.vision_scale -
        settings_.curvature_scale_gain * std::abs(motion->yaw) / reach, 0.5, 1.5);
      motion->x *= correction;
      motion->y *= correction;
    }
  }

  // Stashed before the filters touch `motion`, so what is reported is what the
  // cameras actually said.
  last_hops_valid_ = motion.has_value();
  last_fused_hop_ = motion.has_value()
    ? Eigen::Vector2d(motion->x, motion->y) : Eigen::Vector2d::Zero();
  last_camera_hops_.assign(
    solved.size(), Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN()));
  last_from_map_.assign(solved.size(), 0);
  for (size_t i = 0; i < solved.size(); ++i) {
    if (solved[i].has_value() && solved[i]->motion.has_value()) {
      last_camera_hops_[i] = Eigen::Vector2d(solved[i]->motion->x, solved[i]->motion->y);
      last_from_map_[i] = solved[i]->anchored_from_map ? 1 : 0;
    }
  }

  if (motion.has_value()) {
    const double fused = std::hypot(motion->x, motion->y);
    if (fused > 1e-9) {
      fused_path_ += fused;
      constexpr double kBinMetres = 10.0;
      constexpr size_t kBinLimit = 2048;
      auto & bins = diagnostics_.travel_bins;
      if (bins.empty() || (bins.back()(0) >= kBinMetres && bins.size() < kBinLimit)) {
        bins.emplace_back(Eigen::Vector3d::Zero());
      }
      bins.back()(0) += fused;
      for (size_t i = 0; i < solved.size(); ++i) {
        if (solved[i].has_value() && solved[i]->motion.has_value()) {
          const double along =
            (solved[i]->motion->x * motion->x + solved[i]->motion->y * motion->y) / fused;
          camera_projected_[i] += along;
          if (i < 2) {
            bins.back()(static_cast<Eigen::Index>(i) + 1) += along;
          }
        }
      }
    }
  }

  const double vision_speed = (motion.has_value() && dt > 1e-4)
    ? std::hypot(motion->x, motion->y) / dt
    : std::numeric_limits<double>::infinity();

  diagnostics_.last_disparity = have_disparity ? hop_disparity : -1.0;

  // Decided before the fusion, not after it. Standing still used to be noticed
  // only once the vision displacement had already gone into the filter, and
  // those are the hops it most needed protecting from.
  const bool standing = settings_.zero_velocity_update && have_disparity &&
    hop_disparity <= settings_.zero_velocity_disparity_px &&
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
    inertial_.correct_velocity(imu_world_velocity(msckf_filter_->velocity()));
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
      const Eigen::Vector2d taken =
        spatial_filter_->range_scale() * spatial_filter_->body_translation();
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
    inertial_.correct_velocity(imu_world_velocity(spatial_filter_->velocity().head<2>()));
    diagnostics_.gyro_bias = spatial_filter_->gyro_bias().z();
    diagnostics_.filter_dropped = spatial_filter_->dropped();
    diagnostics_.roll = spatial_filter_->roll();
    diagnostics_.pitch = spatial_filter_->pitch();
    diagnostics_.height = spatial_filter_->position().z();
    diagnostics_.levelled = spatial_filter_->levelled();
    diagnostics_.range_scale = spatial_filter_->range_scale();
    {
      const Eigen::Vector2d after =
        spatial_filter_->range_scale() * spatial_filter_->body_translation();
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
    // The filter's own innovation is the one scale reference that does not
    // come from the ground plane. `predicted` is where inertial propagation
    // says the vehicle went since the last vision update; `measured` is where
    // the ground solve says it went. Their ratio along the predicted direction
    // is a relative scale error, and unlike the radial residual it cannot be
    // satisfied by moving the map -- the accelerometer does not care what the
    // anchors think. This is the observability Ground-VIO gets from its
    // non-ground features, taken from the only other metric source we have.
    const bool learn_scale = settings_.imu_scale_gain != 0.0;
    if (!displacement_filter_->update(world, motion->inliers, extra, spread)) {
      ++diagnostics_.filter_rejections;
    }
    if (standing) {
      displacement_filter_->update_zero_velocity(settings_.zupt_velocity_sigma);
    } else if (learn_scale) {
      const auto & record = displacement_filter_->last_update();
      if (record.has_value() && record->accepted) {
        const double reach = record->predicted.norm();
        if (reach > settings_.imu_scale_min_hop_m) {
          const double relative =
            record->innovation.dot(record->predicted) / (reach * reach);
          if (std::isfinite(relative)) {
            const double step =
              std::clamp(settings_.imu_scale_gain * relative, -0.005, 0.005);
            imu_scale_ = std::clamp(imu_scale_ * (1.0 + step), 0.9, 1.1);
          }
        }
      }
    }
    inertial_.correct_velocity(displacement_filter_->velocity());
    const Eigen::Vector2d fused = displacement_filter_->body_translation(previous_pose.yaw);
    if (!settings_.vision_only_pose &&
      fused.norm() <= settings_.max_translation_per_frame_m)
    {
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

  const bool warming_up = !map_ready_ && !aligned_from_map;
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
  update.previous_stamp = previous_stamp;
  update.hops_valid = last_hops_valid_;
  update.fused_hop = last_fused_hop_;
  update.camera_hops = last_camera_hops_;
  update.camera_from_map = last_from_map_;

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
      const auto coast_velocity = fused_world_velocity();
      if (dt > 1e-4 && settings_.coast_on_reject && !warming_up &&
        coast_velocity.has_value())
      {
        const Eigen::Vector2d delta = *coast_velocity * dt;
        const double c = std::cos(pose_.yaw);
        const double s = std::sin(pose_.yaw);
        const Eigen::Vector2d predicted(
          c * delta.x() + s * delta.y(), -s * delta.x() + c * delta.y());
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
      update.applied_hop = Eigen::Vector2d(applied.x, applied.y);
      update.applied_valid = true;
      update.coasted = true;
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
    if (aligned_from_map) {
      map_ready_ = true;
    }
    update.applied_hop = Eigen::Vector2d(motion->x, motion->y);
    update.applied_valid = true;
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

    double vx = 0.0;
    double vy = 0.0;
    double omega = 0.0;
    twist_from_motion(*motion, vx, vy, omega);
    const Eigen::Vector3d raw(vx / dt, vy / dt, omega / dt);
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
    //
    // The heading is a different matter. On a rejected solve the pose carries
    // the instrument's heading, so handing it back makes the filter inherit
    // the AHRS it exists to improve on -- and the gyro bias it has learned is
    // what makes that improvement. The position is pinned; the heading is left
    // where the filter's own propagation put it.
    const Eigen::Vector2d position(pose_.x, pose_.y);
    if (rejected) {
      msckf_filter_->set_position(position);
    } else {
      msckf_filter_->set_pose(position, pose_.yaw);
    }
  }
  if (spatial_filter_) {
    const Eigen::Vector2d position(pose_.x, pose_.y);
    if (rejected) {
      spatial_filter_->set_position(position);
    } else {
      spatial_filter_->set_pose(position, pose_.yaw);
    }
  }

  // After the pose is settled, so each bearing is filed under where the camera
  // actually was when it saw it.
  for (auto & camera : cameras_) {
    record_offground(*camera);
  }

  ++diagnostics_.frames_processed;
  diagnostics_.camera_travel = camera_travel_;
  diagnostics_.camera_solves = camera_solves_;
  diagnostics_.camera_inliers = camera_inliers_;
  diagnostics_.camera_spread = camera_spread_;
  diagnostics_.camera_usable = camera_usable_;
  diagnostics_.camera_known = camera_known_;
  diagnostics_.camera_scale = camera_projected_;
  for (auto & value : diagnostics_.camera_scale) {
    value = fused_path_ > 1e-9 ? value / fused_path_ : -1.0;
  }
  diagnostics_.crossings = anchors_->crossings();
  diagnostics_.crossing_along_m = anchors_->crossing_along_m();
  diagnostics_.crossing_travel_m = anchors_->crossing_travel_m();
  diagnostics_.crossing_scale = 1.0 + anchors_->crossing_slope();
  diagnostics_.crossing_offset_m = anchors_->crossing_intercept();
  diagnostics_.camera_bearing = camera_bearing_;
  for (size_t i = 0; i < camera_bearing_.size(); ++i) {
    diagnostics_.camera_bearing[i] = camera_bearings_[i] > 0
      ? camera_bearing_[i] / static_cast<double>(camera_bearings_[i]) : -1.0;
  }
  diagnostics_.camera_looks = camera_looks_;
  diagnostics_.camera_anchored = camera_anchored_;
  diagnostics_.anchors = anchors_ ? anchors_->size() : 0;
  if (anchors_) {
    diagnostics_.anchors_adopted = anchors_->adopted();
    diagnostics_.link_gap_m = anchors_->link_gap_m();
    diagnostics_.link_gap_per_m = anchors_->link_gap_per_m();
    diagnostics_.link_range_m = anchors_->link_range_m();
  }

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

  // One tick for the frame, before any camera writes into the map. Ageing is a
  // property of the frame; with a map per camera it did not matter who counted
  // it, and with one shared map counting it twice halves every lifetime.
  anchors_->advance();
  anchors_->set_frame_pose(fused_path_, pose_.yaw);

  for (size_t i = 0; i < solved.size(); ++i) {
    if (!solved[i].has_value()) {
      continue;
    }
    const Solved & entry = *solved[i];
    Camera & camera = *cameras_[i];
    // Which sightings are allowed to define an anchor. A point's position
    // error grows with its range, so a far one founds a landmark that is
    // already wrong and then anchors the pose to it. Bounding births alone --
    // not the projection, not the map, not the registration -- is what the
    // measurement says matters.
    const Eigen::Vector2d anchor_lens =
      cameras_[i]->model.translation_base_from_camera.head<2>();
    const double birth_band = settings_.anchor_max_range_m > 0.0
      ? settings_.anchor_max_range_m : std::numeric_limits<double>::infinity();
    std::vector<Eigen::Index> usable;
    for (Eigen::Index n = 0; n < entry.ground_valid.size(); ++n) {
      if (!entry.ground_valid(n)) {
        continue;
      }
      if (std::isfinite(birth_band)) {
        const double dx = entry.current_ground(n, 0) - anchor_lens.x();
        const double dy = entry.current_ground(n, 1) - anchor_lens.y();
        const double px = entry.previous_ground(n, 0) - anchor_lens.x();
        const double py = entry.previous_ground(n, 1) - anchor_lens.y();
        if (std::hypot(dx, dy) > birth_band || std::hypot(px, py) > birth_band) {
          continue;
        }
      }
      usable.push_back(n);
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
    const double height = std::max(camera.model.translation_base_from_camera.z(), 0.05);
    for (Eigen::Index n = 0; n < count; ++n) {
      const Eigen::Index at = usable[static_cast<size_t>(n)];
      body(n, 0) = entry.current_ground(at, 0);
      body(n, 1) = entry.current_ground(at, 1);
      body(n, 2) = 0.0;
      ids(n) = entry.track_ids(at);
      // How much this sighting is worth. A ground point is a bearing crossed
      // with a plane, so its position error is the bearing error multiplied by
      // (R^2 + h^2) / h and the information is the square of the inverse:
      //
      //     information = (h / (R^2 + h^2))^2
      //
      // This used to be 1 / R^2, justified as "the precision goes as the
      // inverse square of the range". That is the wrong power. The measurement
      // that settles it: restricting the solve to one-metre rings, the
      // registration residual runs 0.0053 m at 1-2 m and 0.0379 at 4-5 m -- a
      // factor of seven, where an inverse square predicts three and this
      // expression predicts eight. Its effective slope near this camera's
      // height is 3.2, and sweeping the old power form finds its optimum at 3.
      // The offset is the camera, not the vehicle: range is measured from the
      // lens that saw the point.
      const double range = std::hypot(body(n, 0) - mount.x(), body(n, 1) - mount.y());
      if (settings_.anchor_information_power > 0.0) {
        information(n) =
          1.0 / std::pow(std::max(range, 0.1), settings_.anchor_information_power);
      } else {
        const double reach = std::max(range, 0.1);
        const double scale = height / (reach * reach + height * height);
        information(n) = scale * scale;
      }
    }
    const Points3 world = transform_points_to_world(body, pose_);
    Points2 flat(count, 2);
    flat.col(0) = world.col(0);
    flat.col(1) = world.col(1);
    anchors_->update(camera.source, ids, flat, allow_new, information);
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

  diagnostics_.obstacle_usable += static_cast<int64_t>(usable.size());
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
      ++diagnostics_.obstacle_ready;
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
      ++diagnostics_.obstacle_no_slip;
      continue;
    }
    const double obstacle_height = height * ratio / (1.0 + ratio);
    if (!std::isfinite(obstacle_height) ||
      obstacle_height < settings_.obstacle_min_height_m || obstacle_height > ceiling)
    {
      ++diagnostics_.obstacle_out_of_band;
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
    ++diagnostics_.obstacle_points;
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
