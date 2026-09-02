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
  // How distinct each feature's patch is. Empty unless the tracker publishes it.
  Weights clarity;
  // (x, y, dx, dy) per grid cell: the road warp's leftover parallax.
  Eigen::MatrixXd parallax;
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
  // Per track, how distinct its patch is. Empty unless the tracker publishes it.
  Weights track_clarity;
  Eigen::MatrixXd track_parallax;

  // Where each surviving feature sat, and which frame that was, the last time
  // a pose was solved. Tracking runs every frame so each hop stays small, but
  // the solve compares against this instead of the frame before it.
  Identities solve_ids;
  Points2 solve_points;
  // The body tilt that held when those pixels were taken. Projecting them
  // later with the tilt of the *current* frame puts the whole pitch change of
  // the hop into the earlier frame's ranges, and dR/dpitch is (R^2+h^2)/h --
  // metres per radian, not a fraction, so it lands in the hop as a constant.
  std::optional<Eigen::Matrix3d> solve_tilt;
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
  // The same quantity the anchor path reports, measured where the signal
  // actually is: on the two-frame pairs, which answer most solves. A point
  // read through a camera height wrong by dh lands wrong by range*dh/h, so the
  // radial residual regressed against range IS dh/h. The anchor path's version
  // is taken against accumulated map points on a fraction of the solves and
  // does not separate two road surfaces that differ by 5-7 mm; this does.
  double pair_n = 0.0;
  double pair_sr = 0.0;
  double pair_srr = 0.0;
  double pair_se = 0.0;
  double pair_sre = 0.0;
  // Learned correction on top of the configured range scale. The radial
  // residual regressed against range is exactly dh/h -- a ground point read
  // through a camera height that is wrong by dh lands wrong by range*dh/h --
  // so the solve already measures the very quantity ground_range_scale was
  // hand-tuned to supply, and this folds that measurement back in.
  double range_scale_learned = 1.0;
  // The last hop this camera solved, in world coordinates, as a seed.
  std::optional<Eigen::Vector2d> last_translation;
  // The road's own answer for how far this camera has moved since the last
  // solve, summed frame by frame. A solve spans about three frames, so the
  // per-frame measurement has to be accumulated before it can be laid beside
  // a hop.
  double radial_height_sum = 0.0;
  double radial_pitch_sum = 0.0;
  int64_t radial_terms = 0;
  double photometric_since_solve = 0.0;
  bool photometric_valid = false;
  bool photometric_broken = false;
  // This camera's tilt against the road, from the split bands. Held as running
  // means because the measurement's noise is per frame and independent while
  // what it measures is not.
  double band_pitch = 0.0;
  double band_roll = 0.0;
  double band_stamp = 0.0;
  bool band_ready = false;
  // The same two angles from the anchor alignment's bearing residuals.
  double anchor_roll = 0.0;
  double anchor_pitch = 0.0;
  // The photometric solve's rotation, summed over the frames a solve spans --
  // the same accumulation the step needs, for the same reason.
  double esm_yaw_since_solve = 0.0;
  double esm_pitch_since_solve = 0.0;
  double esm_roll_since_solve = 0.0;
  bool esm_valid = false;
  // The tilt those increments integrate to, leaked toward the absolute source.
  double esm_tilt_pitch = 0.0;
  double esm_tilt_roll = 0.0;
  double anchor_yaw_last = 0.0;
  bool anchor_yaw_fresh = false;
  double anchor_roll_last = 0.0;
  double anchor_pitch_last = 0.0;
  double anchor_tx_last = 0.0;
  double anchor_ty_last = 0.0;
  bool anchor_ready = false;
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
  // Which entry of the *current frame's* arrays each pair came from.
  //
  // `track_ids` names the feature; this names where it sits in
  // `camera.track_pixels` / `track_clarity`, which are indexed by the frame and
  // not by the pair. Without it anything the tracker publishes per feature can
  // only be read by re-matching identities -- and `admit_by_clarity` was
  // indexing `track_clarity` with a pair index, guarded by a size comparison
  // against a different population, so it silently never ran.
  std::vector<Eigen::Index> track_slot;
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
  // Condition number of the information the usable ground points carry, and
  // the body-frame bearing of its weakest direction. See the note on
  // `camera_condition` in the header.
  double condition = std::numeric_limits<double>::quiet_NaN();
  double weak_bearing = std::numeric_limits<double>::quiet_NaN();
  // Translation this camera's point set gains per metre of height error and
  // per radian of pitch error, in the body frame.
  Eigen::Vector2d height_gain = Eigen::Vector2d::Zero();
  Eigen::Vector2d pitch_gain = Eigen::Vector2d::Zero();
  double mean_range = std::numeric_limits<double>::quiet_NaN();
  double point_count = 0.0;
  // The rotation this hop turned through, whether it was handed in or solved
  // from the road. NaN where neither was available.
  double solved_yaw = std::numeric_limits<double>::quiet_NaN();
};

// The information the ground points carry about a translation, as a 2x2.
//
// A ground point is fixed by a bearing. One radian of bearing error moves it
// (R^2+h^2)/h along the line of sight and R across it, so its vote is an
// ellipse elongated radially, and its information is the inverse of that.
// The common pixel noise cancels out of the ratio, so it is left out.
static void ground_information(
  const Points2 & points, const std::vector<Eigen::Index> & used,
  const Eigen::Vector2d & lens, double height, double & condition,
  double & weak_bearing)
{
  if (used.size() < 2 || !(height > 1e-6)) {
    return;
  }
  Eigen::Matrix2d information = Eigen::Matrix2d::Zero();
  for (const Eigen::Index at : used) {
    const Eigen::Vector2d ray(points(at, 0) - lens.x(), points(at, 1) - lens.y());
    const double range = ray.norm();
    if (!(range > 1e-6)) {
      continue;
    }
    const Eigen::Vector2d radial = ray / range;
    const Eigen::Vector2d across(-radial.y(), radial.x());
    const double along_sigma = (range * range + height * height) / height;
    information += radial * radial.transpose() / (along_sigma * along_sigma);
    information += across * across.transpose() / (range * range);
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(information);
  if (solver.info() != Eigen::Success) {
    return;
  }
  const double low = solver.eigenvalues()(0);
  const double high = solver.eigenvalues()(1);
  if (!(low > 0.0)) {
    return;
  }
  condition = high / low;
  const Eigen::Vector2d weak = solver.eigenvectors().col(0);
  weak_bearing = std::atan2(weak.y(), weak.x());
}

// Where a common-mode error pushes the solve.
//
// The information above assumes each point's bearing errs on its own, and
// measures nearly isotropic because the fisheye's fan of bearings cancels the
// per-point elongation. But the errors that survive here are not independent:
// a height error or a pitch error moves every point at once, and those do not
// cancel -- they add. This returns the translation each unit of common-mode
// error buys, as a vector in the body frame.
//
//   height  dh: every range scales, so the point moves radially by R * dh/h
//   pitch   dp: every elevation shifts, so it moves radially by (R^2+h^2)/h * dp
//
// Equal weights, because the map path deliberately does not range-weight.
static void common_mode_gain(
  const Points2 & points, const std::vector<Eigen::Index> & used,
  const Eigen::Vector2d & lens, double height, Eigen::Vector2d & from_height,
  Eigen::Vector2d & from_pitch)
{
  if (used.empty() || !(height > 1e-6)) {
    return;
  }
  Eigen::Vector2d height_sum = Eigen::Vector2d::Zero();
  Eigen::Vector2d pitch_sum = Eigen::Vector2d::Zero();
  double total = 0.0;
  for (const Eigen::Index at : used) {
    const Eigen::Vector2d ray(points(at, 0) - lens.x(), points(at, 1) - lens.y());
    const double range = ray.norm();
    if (!(range > 1e-6)) {
      continue;
    }
    const Eigen::Vector2d radial = ray / range;
    height_sum += radial * (range / height);
    pitch_sum += radial * ((range * range + height * height) / height);
    total += 1.0;
  }
  if (total > 0.0) {
    from_height = height_sum / total;
    from_pitch = pitch_sum / total;
  }
}

// Mean range of the points a camera solved from, and how many there were.
static void ground_reach(
  const Points2 & points, const std::vector<Eigen::Index> & used,
  const Eigen::Vector2d & lens, double & mean_range, double & count)
{
  double sum = 0.0;
  double n = 0.0;
  for (const Eigen::Index at : used) {
    sum += std::hypot(points(at, 0) - lens.x(), points(at, 1) - lens.y());
    n += 1.0;
  }
  count = n;
  mean_range = n > 0.0 ? sum / n : std::numeric_limits<double>::quiet_NaN();
}

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
  anchor_settings.rebuild_measure_only = settings.rebuild_measure_only;
  anchor_settings.link_radius_m = settings.anchor_link_radius_m;
  anchor_settings.link_cross_source_only = settings.anchor_link_cross_source_only;
  anchor_settings.density_replace_margin = settings.anchor_density_replace_margin;
  anchor_settings.link_measure_only = settings.anchor_link_measure_only;
  anchor_settings.link_adopter_writes = settings.anchor_link_adopter_writes;
  anchor_settings.link_rebind_grace_frames = settings.anchor_link_rebind_grace;
  anchor_settings.max_age_frames = settings.anchor_max_age_frames;
  anchor_settings.update_gain = settings.anchor_update_gain;
  anchor_settings.max_observations = settings.anchor_max_observations;
  anchor_settings.select_by_consistency = settings.anchor_select_by_consistency;
  anchor_settings.initial_variance = settings.anchor_initial_variance;
  anchor_settings.max_variance = settings.anchor_max_variance;
  anchor_settings.trial_observations = settings.anchor_trial_observations;
  anchor_settings.min_update_gain = settings.anchor_min_update_gain;
  anchor_settings.weight_by_information = settings.anchor_weight_by_information;
  anchor_settings.lookahead_m = settings.anchor_lookahead_m;
  anchor_settings.lookahead_sec = settings.anchor_lookahead_sec;
  anchor_settings.geometry_power = settings.anchor_geometry_power;
  anchor_settings.weight_by_variance = settings.anchor_weight_by_variance;
  anchor_settings.bearing_variance = settings.anchor_bearing_variance;
  anchor_settings.weight_by_trend = settings.anchor_weight_by_trend;
  anchor_settings.trend_gain = settings.anchor_trend_gain;
  anchor_settings.trend_power = settings.anchor_trend_power;
  anchor_settings.trend_evict_variance = settings.anchor_trend_evict_variance;
  anchor_settings.admit_by_information = settings.anchor_admit_by_information;
  anchor_settings.admit_by_clarity = settings.anchor_admit_by_clarity;
  anchor_settings.found_after_observations = settings.anchor_found_after_observations;
  anchor_settings.priority_identity_floor =
    settings.anchor_road_priority ? kRoadIdentity : 0;
  anchor_settings.evict_by_age = settings.anchor_evict_by_age;
  anchor_settings.evict_for_new = settings.anchor_evict_for_new;
  anchor_settings.evict_by_weight = settings.anchor_evict_by_weight;
  anchor_settings.evict_unseen_solves = settings.anchor_evict_unseen_solves;
  anchor_settings.evict_by_information = settings.anchor_evict_by_information;
  anchor_settings.admit_per_update = settings.anchor_admit_per_update;
  anchor_settings.anchored_min_observations =
    settings.anchored_min_observations;
  anchor_settings.forget_beyond_bearing_deg =
    settings.anchor_forget_beyond_bearing_deg;
  anchor_settings.density_cell_m = settings.anchor_density_cell_m;
  anchor_settings.density_quota = settings.anchor_density_quota;
  anchor_settings.polar_sector_deg = settings.anchor_polar_sector_deg;
  anchor_settings.polar_ring_m = settings.anchor_polar_ring_m;
  anchor_settings.polar_rings = settings.anchor_polar_rings;
  anchor_settings.polar_quota = settings.anchor_polar_quota;
  anchor_settings.forget_beyond_range_m =
    settings.anchor_forget_beyond_range_m > 0.0
    ? settings.anchor_forget_beyond_range_m : settings.solve_max_distance_m;
  anchor_settings.drift_variance_per_m = settings.anchor_drift_variance_per_m;

  // One map, every camera. The capacity parameter stays per camera so the
  // configured number keeps its meaning.
  anchor_settings.max_anchors =
    settings.max_ground_anchors * std::max<int>(1, static_cast<int>(settings.cameras.size()));
  anchors_ = std::make_unique<GroundAnchorMap>(
    anchor_settings, std::max<int>(1, static_cast<int>(settings.cameras.size())));
  last_radial_height_.assign(settings.cameras.size(), 0.0);
  last_radial_pitch_.assign(settings.cameras.size(), 0.0);
  camera_travel_.assign(settings.cameras.size(), 0.0);
  camera_solves_.assign(settings.cameras.size(), 0);
  camera_inliers_.assign(settings.cameras.size(), 0.0);
  camera_spread_.assign(settings.cameras.size(), 0.0);
  camera_usable_.assign(settings.cameras.size(), 0.0);
  camera_known_.assign(settings.cameras.size(), 0.0);
  camera_bearing_.assign(settings.cameras.size(), 0.0);
  camera_projected_.assign(settings.cameras.size(), 0.0);
  camera_bearings_.assign(settings.cameras.size(), 0);
  camera_looks_.assign(settings.cameras.size(), 0);
  camera_anchored_.assign(settings.cameras.size(), 0);
  int source_index = 0;
  for (const auto & camera : settings.cameras) {
    cameras_.push_back(std::make_unique<Camera>(camera, source_index++));
    cameras_.back()->mount_rotation = camera.rotation_base_from_camera;
    // On the model rather than in a global, so two estimators in one process
    // cannot overwrite each other's.
    cameras_.back()->calibration.level_frame_origin = settings.level_frame_origin;
    cameras_.back()->model.level_frame_origin = settings.level_frame_origin;
  }

  if (settings.attitude_from_imu) {
    attitude_ = std::make_unique<AttitudeFilter>(
      settings.attitude_tau_sec, settings.attitude_gravity_tolerance, 9.80665,
      settings.attitude_bias_tau_sec, settings.attitude_bias_limit_rps,
      settings.attitude_bias_gate_rad);
    attitude_->set_slope_tau(settings.attitude_slope_tau_sec);
    attitude_->set_horizontal_tolerance(settings.attitude_horizontal_tolerance);
  }
  // Only the gyro path corrects its own source; a quaternion heading is read
  // as it comes and the filter's prediction is the whole of what it knows.
  heading_.set_source_corrected(
    settings.imu_yaw_from_gyro ? settings.gyro_bias_apply : 0.0);
  if (settings.fusion_model == FusionModel::Displacement) {
    PlanarDisplacementFilter::Settings filter;
    filter.acceleration_noise = settings.filter_acceleration_noise;
    filter.bias_walk = settings.filter_bias_walk;
    filter.vision_noise_m = settings.filter_vision_noise_m;
    filter.vision_reference_inliers = settings.filter_reference_inliers;
    filter.innovation_gate = settings.filter_innovation_gate;
    displacement_filter_ = std::make_unique<PlanarDisplacementFilter>(filter);
  }
  if (settings.ground_plane_offset_m != 0.0) {
    for (auto & camera : cameras_) {
      const double height = camera->model.translation_base_from_camera.z();
      const double left = height - settings.ground_plane_offset_m;
      if (left > 0.1) {
        camera->settings.range_scale = height / left;
      }
    }
  }  if (settings.ground_common_scale != 1.0) {
    // A ratio, not a length. The plane offset above is a common *distance* --
    // the road sits that far above the datum the mounts are measured from --
    // and it divides by the height, so it gives the two cameras a bias in the
    // ratio 1/0.89 to 1/1.26, which is 1.42. What is actually left over is in
    // the ratio 1.10, measured on three straights: 0.262/0.237, 0.271/0.255,
    // 0.325/0.289. That is a common *fraction*, which no offset can express and
    // which the same three candidates separate cleanly -- a common length
    // predicts 1.42, a body pitch predicts -1.00, and a fraction predicts 1.00.
    //
    // Its likely name is the lens: the rotation channel, which does not depend
    // on depth and so cannot confuse a focal length with a plane, measures the
    // model's focal length 0.42% large. Applied here rather than to `k` because
    // the tracks are already extracted against the recorded camera_info, and
    // because a scale is what the projection wants -- the bearing correction
    // would have to be re-run through the tracker.
    for (auto & camera : cameras_) {
      camera->settings.range_scale *= settings.ground_common_scale;
    }
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

  // Summed over the solve interval, and abandoned whole if any frame in it is
  // missing or did not land. Dropping one frame and keeping the rest does not
  // give a worse distance -- it gives one that is short by exactly that
  // frame's travel, which is a bias, not noise. Measured: gating 3.5% of the
  // frames on one drive took its ATE from 0.0568 to 0.1377.
  if (std::isfinite(incoming.photometric_step) &&
    incoming.photometric_score >= settings_.photometric_min_score &&
    (settings_.photometric_max_spread <= 0.0 ||
    incoming.photometric_spread <= settings_.photometric_max_spread))
  {
    camera.photometric_since_solve +=
      settings_.photometric_scale * incoming.photometric_step;
    camera.photometric_valid = true;
  } else {
    camera.photometric_broken = true;
  }

  if (settings_.band_attitude) {
    ingest_bands(camera, incoming);
  }

  // The photometric rotation, gathered the same way. A frame that did not solve
  // voids the interval rather than contributing nothing: a missing rotation is
  // not a zero rotation, and summing it as one bends the heading.
  if (std::isfinite(incoming.esm_yaw) && std::isfinite(incoming.esm_pitch) &&
    std::isfinite(incoming.esm_roll))
  {
    camera.esm_yaw_since_solve += incoming.esm_yaw;
    camera.esm_pitch_since_solve += incoming.esm_pitch;
    camera.esm_roll_since_solve += incoming.esm_roll;
    camera.esm_valid = true;
  } else {
    camera.esm_valid = false;
  }

  Frame frame;
  frame.stamp = incoming.stamp;
  frame.pixels = incoming.pixels;
  frame.ids = incoming.ids;
  frame.clarity = incoming.clarity;
  frame.parallax = incoming.parallax;
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
    // The gyro's integral starts where the instrument says the vehicle is
    // pointing, which is the one moment a real rig also reads an absolute
    // heading. After this it never looks at the orientation again.
    gyro_yaw_ = yaw;
  }
  double reported = yaw;
  if (settings_.imu_yaw_from_gyro) {
    if (gyro_yaw_stamp_.has_value()) {
      const double step = sample.stamp - *gyro_yaw_stamp_;
      // The same bound the attitude filter uses: a longer gap is a dropout,
      // and integrating across one invents rotation that was never measured.
      if (step > 0.0 && step <= 0.1) {
        // The learned bias taken out at the source, which is where it belongs:
        // the hop rotation handed to the ground solve is then right, not only
        // the heading the pose accumulates. `set_source_corrected` below is
        // what keeps this from being counted twice.
        //
        // Subtracting it was the obvious thing and it is a double count. The
        // filter is an error state on the pose's heading: `predict` already
        // adds `rate_ dt` as the drift it expects this source to have, and
        // `update` injects the correction into `pose_.yaw`. Removing the bias
        // here too makes the residual carry `-(b + r) dt` against a prediction
        // of `r dt`, so the innovation vanishes at **r = -b/2** and the filter
        // settles on half the bias. Measured at 47-58% recovery on every drive
        // that has a bias, which is what pointed at it.
        gyro_yaw_ = wrap_pi(
          gyro_yaw_ +
          (sample.angular_velocity.z() + settings_.gyro_bias_apply * heading_.rate()) * step);
      }
    }
    gyro_yaw_stamp_ = sample.stamp;
    reported = gyro_yaw_;
  }
  imu_yaw_samples_.emplace_back(sample.stamp, reported);
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


  if (settings_.use_inertial_prediction) {
    const auto step = inertial_.add_sample(
      sample.stamp, sample.orientation, sample.linear_acceleration);
    Eigen::Vector2d acceleration = step.acceleration;
    if (!settings_.inertial_use_acceleration) {
      acceleration.setZero();
    }
    velocity_filter_.predict(acceleration, step.dt);
    if (displacement_filter_) {
      // Into the frame the filter's state actually lives in. The measurement
      // side hands it `R(pose.yaw) * motion_body`, and the pose starts at zero
      // on the vehicle's initial heading, so that frame is the world turned by
      // -yaw0. Rotating by the *current* yaw instead -- which is what this did
      // -- puts the prediction in the body frame while the state is in the
      // world one, and the two then disagree by however far the vehicle has
      // turned. On a straight they coincide, which is why it went unnoticed.
      const double datum = imu_yaw_datum_.value_or(yaw);
      const double c = std::cos(yaw);
      const double s = std::sin(yaw);
      const Eigen::Vector2d body(
        c * acceleration.x() + s * acceleration.y(),
        -s * acceleration.x() + c * acceleration.y());
      // What `predict` is actually handed. `acceleration` comes out of the
      // propagator in the simulator's world frame, while the filter's state
      // lives in the estimator's -- the same world turned by -yaw0, because the
      // pose starts at zero on the vehicle's initial heading and the
      // measurement side hands it `R(pose.yaw) * motion_body`. The two differ
      // by a fixed rotation of yaw0, which is 89.8 deg on the Town10HD spawn
      // and 0.0 on the Town01 one.
      const double cd = std::cos(datum);
      const double sd = std::sin(datum);
      acceleration = Eigen::Vector2d(
        cd * acceleration.x() + sd * acceleration.y(),
        -sd * acceleration.x() + cd * acceleration.y());
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
          step.dt,
          // The estimator's heading, not the simulator's. `predict` uses this
          // only to rotate the accelerometer bias out of the body frame, and
          // the acceleration beside it is now in the estimator's world -- so a
          // bias rotated by the absolute yaw is wrong by yaw0 and the bias
          // state cannot converge. Same error as the acceleration had, one
          // line down.
          std::remainder(yaw - datum, 2.0 * M_PI)});
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
      camera.track_clarity = camera.latest->clarity;
      camera.track_parallax = camera.latest->parallax;
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
  if (displacement_filter_) {
    return displacement_filter_->settled()
           ? std::optional<Eigen::Vector2d>(displacement_filter_->velocity())
           : std::nullopt;
  }
  return velocity_filter_.settled()
         ? std::optional<Eigen::Vector2d>(velocity_filter_.velocity())
         : std::nullopt;
}


void Estimator::override_tilt(double roll, double pitch)
{
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  Eigen::Matrix3d level;
  level << cp, sp * sr, sp * cr,
    0.0, cr, -sr,
    -sp, cp * sr, cp * cr;
  tilt_override_ = level;
}

// What a focal length error costs a step measured over a band at this range.
// A model focal length larger than the truth by `scale` reads every bearing as
// that much shallower, and a ground range h cot(theta) grows by (R^2 + h^2)/h
// per radian of that. Carried through a step the way the pitch is --
// `bias = g + R g'` for a range distortion g -- the arctangent survives and
// the rest collapses:
//
//     bias = scale (2 R atan(h/R) / h - 1)
//
// It is not linear in R, which is exactly why a two-band difference cannot
// tell it from a pitch on its own and needs the rotation channel to price it.
double lens_step_bias(double range, double height, double scale)
{
  if (!(range > 1e-6) || !(height > 1e-6)) {
    return 0.0;
  }
  return scale * (2.0 * range * std::atan2(height, range) / height - 1.0);
}

void Estimator::ingest_bands(Camera & camera, const TrackFrame & incoming)
{
  const double height = std::abs(camera.model.translation_base_from_camera.z());
  if (!(height > 1e-6)) {
    return;
  }
  // Both halves are of one region and one motion, so the whole region's own
  // answer is what says whether a half landed on the road at all.
  const double whole = incoming.photometric_step;
  if (!std::isfinite(whole) || std::abs(whole) < 1e-3) {
    return;
  }
  const auto landed = [&](double half) {
      return std::isfinite(half) &&
             std::abs(half / whole - 1.0) <= settings_.band_max_disagreement;
    };
  const double dt = camera.band_stamp > 0.0 ? incoming.stamp - camera.band_stamp : 0.0;
  const double gain = (dt > 0.0 && settings_.band_attitude_tau_sec > 0.0)
    ? std::min(1.0, dt / settings_.band_attitude_tau_sec) : 1.0;

  // Near against far is the pitch. A pitch moves a ground point at range R by
  // (R^2 + h^2)/h and a height error by R; carried through the step both keep
  // only their leading behaviour -- `2 R p / h` against a constant -- because
  // the h/R terms cancel exactly. So the difference between two bands is the
  // pitch and cannot be a height, whatever the height is.
  //
  // Sign: nose-down is positive here, as it is in the inertial filter. A body
  // pitched nose-down meets the ground nearer than a level projection computes,
  // so the level assumption reads long and `bias = +2 R p / h`.
  // Divided by how far apart the bands are *along* the vehicle, not by how far
  // apart in range. The two are the same number on a forward mount and
  // opposite on a rear one, which is exactly the asymmetry the pitch has.
  const double reach = incoming.band_far_forward - incoming.band_near_forward;
  if (landed(incoming.band_near) && landed(incoming.band_far) &&
    std::isfinite(reach) && std::abs(reach) > 1e-3)
  {
    double difference = incoming.band_far / incoming.band_near - 1.0;
    difference -=
      lens_step_bias(incoming.band_far_range, height, settings_.band_lens_scale) -
      lens_step_bias(incoming.band_near_range, height, settings_.band_lens_scale);
    const double pitch = height * difference / (2.0 * reach);
    if (std::isfinite(pitch)) {
      camera.band_pitch += camera.band_ready ? gain * (pitch - camera.band_pitch)
        : pitch - camera.band_pitch;
      camera.band_ready = true;
    }
  }

  // Left against right is the roll, on the same footing: a roll puts the road
  // at depth `h + y phi` at lateral offset y, so every range there scales by
  // `h / (h + y phi)` and the step with it. The offsets come in body
  // coordinates, so the rear mount's yaw of 180 degrees is already in their
  // signs and the same expression serves both cameras.
  const double across = incoming.band_right_lateral - incoming.band_left_lateral;
  if (landed(incoming.band_left) && landed(incoming.band_right) &&
    std::isfinite(across) && std::abs(across) > 1e-3)
  {
    const double difference = incoming.band_right / incoming.band_left - 1.0;
    const double roll = -height * difference / across;
    if (std::isfinite(roll)) {
      camera.band_roll += camera.band_ready ? gain * (roll - camera.band_roll) : roll;
    }
  }
  camera.band_stamp = incoming.stamp;
}

std::optional<Eigen::Matrix3d> Estimator::camera_tilt(const Camera & camera) const
{
  if (settings_.esm_attitude) {
    return Eigen::Matrix3d(
      Eigen::AngleAxisd(camera.esm_tilt_roll, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(camera.esm_tilt_pitch, Eigen::Vector3d::UnitY()));
  }
  if (settings_.anchor_attitude) {
    if (!camera.anchor_ready) {
      return std::nullopt;
    }
    return Eigen::Matrix3d(
      Eigen::AngleAxisd(camera.anchor_roll, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(camera.anchor_pitch, Eigen::Vector3d::UnitY()));
  }
  if (!settings_.band_attitude) {
    return body_tilt();
  }
  if (!camera.band_ready) {
    // Level until the road has said otherwise. Falling back to the inertial
    // attitude here would put the very term this replaces back into the first
    // seconds of every drive, which is where it peaks.
    return std::nullopt;
  }
  return Eigen::Matrix3d(
    Eigen::AngleAxisd(camera.band_roll, Eigen::Vector3d::UnitX()) *
    Eigen::AngleAxisd(camera.band_pitch, Eigen::Vector3d::UnitY()));
}

std::optional<Eigen::Matrix3d> Estimator::body_tilt() const
{
  if (tilt_override_.has_value()) {
    return tilt_override_;
  }
  // The six degree of freedom filter carries roll and pitch as states, so it
  // answers this itself and the separate attitude filter is not asked. That is
  // the whole point of the container: one covariance rather than two estimators
  // that cannot tell each other how sure they are.
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
  // The tilt this frame was captured under, for the next solve to project its
  // earlier frame with. It has to be the same tilt the projection will use --
  // taking the inertial one here while the projection takes the road's puts a
  // different attitude on each frame of every hop, which is the frame-mixing
  // defect this field was added to remove.
  camera.solve_tilt = camera_tilt(camera);
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
  solved.track_slot = matched_now;
  solved.current_pixels = current_pixels;

  const auto tilt = camera_tilt(camera);
  const Eigen::Matrix3d * tilt_ptr = tilt.has_value() ? &tilt.value() : nullptr;
  // A band or anchor tilt is the camera's angle against the road, not the
  // body's attitude against gravity. A mounting error tilts the camera without
  // moving it, and a road that is not level does not move it either, so
  // swinging the mount on a lever about the pitch centre here would invent a
  // motion that never happened -- a height change, which is the larger of the
  // two ways a tilt reaches a range, and a displacement of the lens with it.
  //
  // Both have to follow this one switch. A rigid body rotating about a point
  // cannot move its camera along the road while holding its height: the swing
  // is x_p*(1 - cos t) horizontally against x_p*sin t vertically, so taking
  // the second-order term while dropping the first-order one is not a model of
  // anything. The lens displacement was being applied while the height was
  // not, which is that split exactly.
  const bool tilt_moves_camera = settings_.ground_height_from_tilt &&
    !settings_.band_attitude && !settings_.anchor_attitude;
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
      solved.current_ground, valid_current, settings_.pitch_centre_x_m,
      tilt_moves_camera);
    // The earlier frame gets the tilt it was taken under, not this one's.
    const Eigen::Matrix3d * then_ptr = tilt_ptr;
    if (settings_.tilt_at_capture && camera.solve_tilt.has_value()) {
      then_ptr = &camera.solve_tilt.value();
    }
    pixels_to_ground(
      previous_pixels, camera.model, band, camera.settings.ground_min_distance_m,
      then_ptr, camera.settings.range_scale * camera.range_scale_learned * imu_scale_,
      solved.previous_ground, valid_previous, settings_.pitch_centre_x_m,
      tilt_moves_camera);
  }
  solved.ground_valid = valid_previous && valid_current;

  solved.motion_inliers = Mask::Constant(count, false);

  // The heading, from the road if it was not handed in. The similarity fit
  // below recovers a rotation as well as a translation and its rotation has
  // always been discarded, because a heading was always supplied; when it is
  // not, this is where it comes from.
  std::optional<double> yaw = yaw_delta;
  // The photometric rotation first where it is being solved: the two-frame fit
  // below reads the same motion off a correspondence set that goes asymmetric
  // as the patch overlap shrinks, and pays for it with a bias that grows with
  // the step.
  if (!yaw.has_value() && settings_.esm_yaw_source && camera.esm_valid &&
    std::isfinite(camera.esm_yaw_since_solve) &&
    std::abs(camera.esm_yaw_since_solve) <= settings_.max_yaw_per_frame_rad)
  {
    yaw = camera.esm_yaw_since_solve;
  }
  // The ESM's turn as an *observation* of the heading that was handed in,
  // rather than as a replacement for it.
  //
  // The gyro's error over a hop is its bias times the interval, and a bias is
  // a constant the filter already carries as a state. What it lacked was
  // anything that could see it: the anchor map is built in the estimator's own
  // frame, so a slow heading drift turns the map and the vehicle together and
  // the alignment residual barely moves. Measured on curve_s05, where the gyro
  // bias is +0.903 deg/s: 261 updates recovered 21% of it. The ESM reads its
  // rotation off the image and does not turn with that frame.
  if (settings_.esm_yaw_sigma_rad > 0.0 && heading_.enabled() &&
    (settings_.esm_yaw_camera.empty() ||
    settings_.esm_yaw_camera == camera.settings.name) &&
    yaw_delta.has_value() && camera.esm_valid &&
    std::isfinite(camera.esm_yaw_since_solve) &&
    std::abs(camera.esm_yaw_since_solve) <= settings_.max_yaw_per_frame_rad)
  {
    // Sign: `error_` is the correction the filter adds to the pose, so an
    // instrument that over-reads the turn by b dt needs -b dt applied, and the
    // ESM minus the instrument is that difference already.
    const double innovation = wrap_pi(camera.esm_yaw_since_solve - *yaw_delta);
    // The sigma this observation deserves is the spread of this residual, which
    // is the ESM's own error plus the instrument's over the same interval. It
    // has to be measured rather than assumed, so it can be written out.
    if (const char * path = std::getenv("MONOSCALE_HEADING_INNOVATION")) {
      if (heading_innovation_file_ == nullptr) {
        heading_innovation_file_ = std::fopen(path, "w");
        if (heading_innovation_file_ != nullptr) {
          std::fprintf(heading_innovation_file_, "stamp,innovation,esm_yaw,instrument\n");
        }
      }
      if (heading_innovation_file_ != nullptr) {
        std::fprintf(
          heading_innovation_file_, "%.6f,%.9f,%.9f,%.9f\n", camera.band_stamp, innovation,
          camera.esm_yaw_since_solve, *yaw_delta);
      }
    }
    heading_observations_.emplace_back(
      innovation,
      settings_.esm_yaw_sigma_rad +
      settings_.esm_yaw_sigma_rate * std::abs(camera.esm_yaw_since_solve));
  }
  if (!yaw.has_value() && settings_.vision_yaw && solved.ground_valid.any()) {
    Eigen::Index usable_pairs = 0;
    for (Eigen::Index i = 0; i < count; ++i) {
      if (solved.ground_valid(i)) {
        ++usable_pairs;
      }
    }
    if (usable_pairs >= settings_.ground_min_inliers) {
      Points2 before(usable_pairs, 2);
      Points2 after(usable_pairs, 2);
      Eigen::Index at = 0;
      for (Eigen::Index i = 0; i < count; ++i) {
        if (!solved.ground_valid(i)) {
          continue;
        }
        before.row(at) = solved.previous_ground.row(i);
        after.row(at) = solved.current_ground.row(i);
        ++at;
      }
      const auto free_fit = estimate_planar_motion(
        before, after, settings_.ground_ransac_threshold_m,
        settings_.ground_min_inliers, settings_.max_scale_error);
      if (free_fit.has_value() && std::isfinite(free_fit->motion.yaw) &&
        std::abs(free_fit->motion.yaw) <= settings_.max_yaw_per_frame_rad)
      {
        yaw = free_fit->motion.yaw;
        // The similarity fit carries a scale as well, and the ground's scale is
        // set by the camera height rather than by this hop. With a patch held
        // four metres off the rotation centre a yaw and a sideways slide are
        // already nearly the same thing -- conditioning 45.5 against 2.4 for a
        // patch about the origin -- and a free scale is a third direction for
        // them to trade against. So the rotation is taken again rigidly, over
        // the correspondences that fit agreed on, with the scale held at one.
        if (settings_.vision_yaw_rigid && free_fit->inliers.size() == before.rows()) {
          Eigen::Vector2d mean_before = Eigen::Vector2d::Zero();
          Eigen::Vector2d mean_after = Eigen::Vector2d::Zero();
          double kept_terms = 0.0;
          for (Eigen::Index i = 0; i < before.rows(); ++i) {
            if (!free_fit->inliers(i)) {
              continue;
            }
            mean_before += before.row(i).transpose();
            mean_after += after.row(i).transpose();
            kept_terms += 1.0;
          }
          if (kept_terms >= 3.0) {
            mean_before /= kept_terms;
            mean_after /= kept_terms;
            double cross = 0.0;
            double dot = 0.0;
            for (Eigen::Index i = 0; i < before.rows(); ++i) {
              if (!free_fit->inliers(i)) {
                continue;
              }
              const double bx = before(i, 0) - mean_before.x();
              const double by = before(i, 1) - mean_before.y();
              const double ax = after(i, 0) - mean_after.x();
              const double ay = after(i, 1) - mean_after.y();
              // The similarity fit above maps the current cloud onto the
              // earlier one, so this has to turn the same way round or the two
              // disagree by exactly a sign.
              cross += ax * by - ay * bx;
              dot += bx * ax + by * ay;
            }
                const double rigid = std::atan2(cross, dot);
            if (std::isfinite(rigid) &&
              std::abs(rigid) <= settings_.max_yaw_per_frame_rad)
            {
              yaw = rigid;
            }
          }
          if (settings_.vision_yaw_vehicle) {
            // The vehicle's own two freedoms, over the same inliers. To first
            // order a step s and a yaw d move a ground point at (x, y) by
            // (-s + d*y, -d*x), which is linear in both, so this is two normal
            // equations and no search.
            double a11 = 0.0;
            double a12 = 0.0;
            double a22 = 0.0;
            double b1 = 0.0;
            double b2 = 0.0;
            for (Eigen::Index i = 0; i < before.rows(); ++i) {
              if (!free_fit->inliers(i)) {
                continue;
              }
              const double px = before(i, 0);
              const double py = before(i, 1);
              const double dx = after(i, 0) - px;
              const double dy = after(i, 1) - py;
              // Row for dx: -1 * s + py * d.  Row for dy: 0 * s + (-px) * d.
              a11 += 1.0;
              a12 += -py;
              a22 += py * py + px * px;
              b1 += -dx;
              b2 += py * dx - px * dy;
            }
            const double det = a11 * a22 - a12 * a12;
            if (std::abs(det) > 1e-12) {
              const double vehicle = (a11 * b2 - a12 * b1) / det;
              if (std::isfinite(vehicle) &&
                std::abs(vehicle) <= settings_.max_yaw_per_frame_rad)
              {
                yaw = vehicle;
              }
            }
          }
        }
      }
    }
  }
  solved.solved_yaw = yaw.value_or(std::numeric_limits<double>::quiet_NaN());

  if (!yaw.has_value() || !solved.ground_valid.any()) {
    remember_solve_pixels(camera);
    return solved;
  }
  const std::optional<double> & yaw_for_solve = yaw;

  Stopwatch watch(diagnostics_, "solve");

  // Widened by whatever this hop turns through, for both solve paths.
  const double gate = settings_.ground_ransac_threshold_m +
    settings_.ground_rotation_threshold_m * std::abs(*yaw_for_solve);

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
  // Held when the map answered and its correction is to be scaled rather than
  // taken whole. See `map_correction_gain`.
  std::optional<PlanarMotion> map_motion;
  Eigen::Vector3d mount_in_frame = camera.model.translation_base_from_camera;
  if (settings_.level_frame_origin && tilt_moves_camera && tilt.has_value()) {
    const Eigen::Vector3d centre(settings_.pitch_centre_x_m, 0.0, 0.0);
    mount_in_frame = tilt.value() * (mount_in_frame - centre) + centre;
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

  if (settings_.fuse_camera_points) {
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

  if (settings_.equalise_reach && std::isfinite(reach_target_) && usable.size() > 2) {
    // Farthest first, dropped until the mean reach matches. Exact, and it needs
    // no threshold of its own -- the target is what the other camera measured.
    double sum = 0.0;
    for (const Eigen::Index at : usable) {
      sum += std::hypot(
        solved.current_ground(at, 0) - lens.x(), solved.current_ground(at, 1) - lens.y());
    }
    if (sum / static_cast<double>(usable.size()) > reach_target_) {
      std::sort(
        usable.begin(), usable.end(), [&](Eigen::Index a, Eigen::Index b) {
          return std::hypot(
            solved.current_ground(a, 0) - lens.x(), solved.current_ground(a, 1) - lens.y()) <
          std::hypot(
            solved.current_ground(b, 0) - lens.x(), solved.current_ground(b, 1) - lens.y());
        });
      while (usable.size() > 2 &&
        sum / static_cast<double>(usable.size()) > reach_target_)
      {
        const Eigen::Index at = usable.back();
        sum -= std::hypot(
          solved.current_ground(at, 0) - lens.x(), solved.current_ground(at, 1) - lens.y());
        usable.pop_back();
      }
    }
  }

  ground_information(
    solved.current_ground, usable, lens, lens_height, solved.condition,
    solved.weak_bearing);
  common_mode_gain(
    solved.current_ground, usable, lens, lens_height, solved.height_gain,
    solved.pitch_gain);
  ground_reach(
    solved.current_ground, usable, lens, solved.mean_range, solved.point_count);

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


    // Where the solve is told to start looking. With the MSCKF that is the
    // filter's own propagated heading rather than the instrument's, and the
    // solve is asked to improve on it -- the improvement is the measurement.
    const double handed = yaw_guess.has_value()
      ? *yaw_guess : wrap_pi(pose_.yaw + *yaw_for_solve);
    // The alignment solves for where the camera *is*, not for how far it
    // moved, so the inertial expectation has to be carried onto the last
    // solved position before it can gate anything. Comparing a hop against an
    // absolute translation was the first version of this and rejected almost
    // every mode.
    // Where the road says this camera has got to, if it said anything.
    std::optional<Eigen::Vector2d> road_centre;
    if (settings_.photometric_align_prior && camera.photometric_valid &&
      !camera.photometric_broken && camera.photometric_since_solve > 0.0)
    {
      road_centre = Eigen::Vector2d(pose_.x, pose_.y) +
        camera.photometric_since_solve *
        Eigen::Vector2d(std::cos(handed), std::sin(handed));
    }
    std::optional<Eigen::Vector2d> gate_centre;
    if (settings_.inertial_gate_m > 0.0 && expected_hop_.has_value()) {
      // From the pose actually held, not from the last alignment: the map path
      // answers only a third of the time, so its record goes stale.
      gate_centre = Eigen::Vector2d(pose_.x, pose_.y) + *expected_hop_;
    }
    if (road_centre.has_value()) {
      gate_centre = road_centre;
    }
    std::optional<AnchorAlignment> aligned;
    {
      Stopwatch align(diagnostics_, "align");
      aligned = align_to_anchors(
        body, world, weights, handed, gate,
        settings_.ground_min_inliers,
        settings_.align_solves_yaw &&
        heading_.enabled(),
        lens, settings_.radial_min_range_m, lens_height,
        softness_for(camera, settings_.ground_align_softness_m),
        road_centre.has_value() ? &*road_centre
        : (settings_.align_seed_from_last_hop && camera.last_translation.has_value()
        ? &*camera.last_translation : nullptr),
        settings_.align_restarts, settings_.align_ambiguity_ratio,
        gate_centre.has_value() ? &*gate_centre : nullptr,
        settings_.inertial_gate_m, scale,
        settings_.anchor_bearing_nonholonomic,
        settings_.anchor_bearing_cell_rad, settings_.anchor_bearing_cell_rho);
    }
    if (aligned.has_value()) {
      camera.last_translation = aligned->translation;
      if (aligned->bearing_terms > 0) {
        camera.anchor_yaw_last = aligned->bearing_yaw;
        camera.anchor_yaw_fresh = true;
        camera.anchor_roll_last = aligned->bearing_roll;
        camera.anchor_pitch_last = aligned->bearing_pitch;
        camera.anchor_tx_last = aligned->bearing_tx;
        camera.anchor_ty_last = aligned->bearing_ty;
      }
      if (settings_.anchor_attitude && aligned->bearing_terms > 0) {
        const double gain = settings_.anchor_attitude_solves > 1.0
          ? 1.0 / settings_.anchor_attitude_solves : 1.0;
        const double roll = settings_.anchor_attitude_gain * aligned->bearing_roll;
        const double pitch = settings_.anchor_attitude_gain * aligned->bearing_pitch;
        if (std::isfinite(roll) && std::isfinite(pitch)) {
          if (camera.anchor_ready) {
            camera.anchor_roll += gain * (roll - camera.anchor_roll);
            camera.anchor_pitch += gain * (pitch - camera.anchor_pitch);
          } else {
            camera.anchor_roll = roll;
            camera.anchor_pitch = pitch;
            camera.anchor_ready = true;
          }
        }
      }
      camera.radial_height_sum += aligned->radial_height;
      camera.radial_pitch_sum += aligned->radial_pitch;
      ++camera.radial_terms;
      last_radial_height_[camera.source] = aligned->radial_height;
      last_radial_pitch_[camera.source] = aligned->radial_pitch;
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
      const double heading = handed;
      solved.yaw_sigma = aligned->yaw_sigma;
      const Pose2 placed{aligned->translation.x(), aligned->translation.y(), heading};
      // From the fused pose, deliberately. This is a pose *correction*, not a
      // displacement: it carries this map's standing disagreement with the
      // fused pose into every hop it reports, and since the fused pose is the
      // mean of the two cameras those disagreements come out equal and
      // opposite -- a front/rear split of 1.1 to 2.4 per cent where the
      // observations themselves carry 0.04.
      //
      // That split is the binding, not a defect. Taking the hop between this
      // camera's own two placements instead removes it and **doubles the
      // error** (composite 2.04, all eleven metrics worse): the correction is
      // what bounds the random walk, and without it the map path is just
      // another odometry increment. It is also why no weighting but 50:50
      // works -- the two maps' disagreement is accumulated drift, not noise,
      // and only the midpoint cancels it.
      PlanarMotion motion = relative_motion(pose_, placed);
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
      remember_solve_pixels(camera);
      // At unit gain this is the whole answer and the pair solve is not worth
      // its cost. Below or above it, the pair solve is needed as the other end
      // of the blend, so fall through and take it.
      if (settings_.map_correction_gain == 1.0) {
        return solved;
      }
      map_motion = motion;
    }
  }

  // The map could not answer -- or it did and the correction is being scaled
  // against this, which is the same two-frame solve either way.
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
    previous_ground, current_ground, *yaw_for_solve, gate,
    settings_.ground_min_inliers, softness_for(camera, settings_.ground_pair_softness_m),
    pair_weights,
    settings_.ground_pair_passes);
  if (estimate.has_value() && map_motion.has_value()) {
    // What the map path actually reports is a displacement with a correction
    // folded into it: `relative_motion(pose_, placed)` is where this camera
    // has moved *plus* this map's standing disagreement with the fused pose.
    // The two-frame solve is the same displacement with no correction at all,
    // so their difference is the correction alone and this scales that and
    // nothing else. Unit gain is what the map path did before this existed;
    // zero leaves the displacement with the correction removed.
    //
    // This is the knob `map_solve_weight` was mistaken for. That one moved how
    // much of the *camera* the fusion listened to, which conflates correction
    // strength with throwing away the other camera's measurement -- and it was
    // being discarded anyway.
    const double gain = settings_.map_correction_gain;
    PlanarMotion blended = *map_motion;
    blended.x = estimate->motion.x + gain * (map_motion->x - estimate->motion.x);
    blended.y = estimate->motion.y + gain * (map_motion->y - estimate->motion.y);
    solved.motion = blended;
    // The map branch's spread and inlier mask stand: they describe the answer
    // this hop is anchored on.
    return solved;
  }
  if (estimate.has_value()) {
    solved.motion = estimate->motion;
    // The road's own length for this interval, where the hop really is one.
    if (settings_.photometric_on_pairs && settings_.photometric_step_gain > 0.0 &&
      camera.photometric_valid && !camera.photometric_broken &&
      camera.photometric_since_solve > 0.0)
    {
      const double length = std::hypot(solved.motion->x, solved.motion->y);
      if (length > 1e-6) {
        const double blended = length + settings_.photometric_step_gain *
          (camera.photometric_since_solve - length);
        const double ratio = blended / length;
        solved.motion->x *= ratio;
        solved.motion->y *= ratio;
      }
    }
    for (Eigen::Index i = 0; i < paired; ++i) {
      solved.motion_inliers(pairs[static_cast<size_t>(i)]) = estimate->inliers(i);
    }
    // Spread of the pairwise solution, on the same footing as the map one.
    const double c = std::cos(*yaw_for_solve);
    const double s = std::sin(*yaw_for_solve);
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
      const double ex = ox - estimate->motion.x;
      const double ey = oy - estimate->motion.y;
      squared += ex * ex + ey * ey;
      ++kept;
      const double rx = previous_ground(i, 0) - lens.x();
      const double ry = previous_ground(i, 1) - lens.y();
      const double range = std::hypot(rx, ry);
      if (range > settings_.radial_min_range_m) {
        const double radial = (ex * rx + ey * ry) / range;
        camera.pair_n += 1.0;
        camera.pair_sr += range;
        camera.pair_srr += range * range;
        camera.pair_se += radial;
        camera.pair_sre += range * radial;
        // The projection can read its own height back out of the residual, so
        // let it. Applied as a direct map from the accumulated regression, not
        // as an integrator: the integrator form was what went unstable when
        // this was tried before, and with a hundred thousand points behind it
        // the slope needs no smoothing of its own.
        if (settings_.pair_scale_gain != 0.0 && camera.pair_n > 2000.0) {
          const double d = camera.pair_n * camera.pair_srr - camera.pair_sr * camera.pair_sr;
          if (std::abs(d) > 1e-9) {
            const double slope =
              (camera.pair_n * camera.pair_sre - camera.pair_sr * camera.pair_se) / d;
            camera.range_scale_learned =
              std::clamp(1.0 / (1.0 + settings_.pair_scale_gain * slope), 0.98, 1.02);
          }
        }
      }
    }
    solved.spread = kept > 0 ? std::sqrt(squared / kept) : 0.0;
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
  if (!yaw_delta.has_value()) {
    // What the road said the hop turned through, averaged over whichever
    // cameras found it. They are measuring one body rotation, so a
    // disagreement between them is noise and not two different answers.
    double total = 0.0;
    int terms = 0;
    for (const auto & entry : solved) {
      if (entry.has_value() && std::isfinite(entry->solved_yaw)) {
        total += entry->solved_yaw;
        ++terms;
      }
    }
    if (terms > 0) {
      yaw_delta = total / static_cast<double>(terms);
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
  diagnostics_.pair_radial.assign(count, 0.0);
  if (settings_.remember_sighting_poses) {
    diagnostics_.remembered_sightings = anchors_->remembered();
    diagnostics_.pose_history = static_cast<int64_t>(pose_history_.size());
    const auto start = std::chrono::steady_clock::now();
    anchors_->clear_rebuild_shift();
    anchors_->rebuild(pose_history_);
    diagnostics_.rebuild_shift_m = anchors_->rebuild_shift();
    diagnostics_.sighting_span = anchors_->sighting_span();
    diagnostics_.rebuild_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  }
  diagnostics_.pair_radial_samples.assign(count, 0);
  diagnostics_.radial_samples.assign(count, 0);
  for (size_t i = 0; i < count; ++i) {
    const Camera & camera = *cameras_[i];
    diagnostics_.radial_samples[i] = camera.radial_samples;
    if (camera.pair_n > 8.0) {
      const double d = camera.pair_n * camera.pair_srr - camera.pair_sr * camera.pair_sr;
      diagnostics_.pair_radial[i] = std::abs(d) > 1e-9
        ? (camera.pair_n * camera.pair_sre - camera.pair_sr * camera.pair_se) / d : 0.0;
      diagnostics_.pair_radial_samples[i] = static_cast<int64_t>(camera.pair_n);
    }
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
      double own = cameras_[i]->settings.fusion_weight;
      if (settings_.fusion_gain_mode != 0) {
        const Eigen::Vector2d & gain = settings_.fusion_gain_mode == 1
          ? solved[i]->pitch_gain : solved[i]->height_gain;
        const double size = gain.norm();
        if (size > 1e-9) {
          own = 1.0 / size;
        }
      }
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

  // The road's length, over the hop the cameras just described.
  //
  // Taken as a length and nothing else. The cameras set the direction and,
  // through the map path's front/rear correction, the drift binding; the road
  // fit has no map and cannot replace either. What it has is scale.
  double road_distance = 0.0;
  int road_cameras = 0;
  for (const auto & held : cameras_) {
    if (held->photometric_valid && !held->photometric_broken &&
      held->photometric_since_solve > 0.0)
    {
      road_distance += held->photometric_since_solve;
      ++road_cameras;
    }
  }
  bool any_from_map = false;
  ++diagnostics_.photometric_chances;
  for (const auto & entry : solved) {
    if (entry.has_value() && entry->motion.has_value() && entry->anchored_from_map) {
      any_from_map = true;
    }
  }
  if (!any_from_map) {
    ++diagnostics_.photometric_mapless;
  }
  last_photometric_distance_ = road_cameras > 0
    ? road_distance / road_cameras : std::numeric_limits<double>::quiet_NaN();
  last_fused_length_ = motion.has_value()
    ? std::hypot(motion->x, motion->y) : std::numeric_limits<double>::quiet_NaN();
  if (settings_.photometric_step_gain > 0.0 && !settings_.photometric_on_pairs &&
    road_cameras > 0 &&
    motion.has_value() && !(settings_.photometric_when_mapless && any_from_map))
  {
    const double measured = road_distance / road_cameras;
    const double length = std::hypot(motion->x, motion->y);
    if (length > 1e-6 && std::isfinite(measured)) {
      // These are two independent measurements of the same displacement, and
      // the photometric one is the more precise: its error against truth is
      // 0.08% of the hop, against 0.3-1.2% for the pair solve. So a
      // disagreement of several per cent is not the pair solve being wrong --
      // it is the photometric search having failed to find its peak, and the
      // scale it then reports is unbounded. Measured over ten drives the
      // disagreement is 0.32% at the median and 1.9% at worst on a drive where
      // nothing goes wrong, while the failures reach **3.8x and 32x**; two
      // consecutive frames of the latter took str_4.0 from 0.079 to 1.347.
      //
      // `max_scale_error` is already the stack's bound on exactly this
      // quantity -- how far a scale may be off before it stops being a
      // measurement -- so it gates here too rather than introducing a second
      // number to tune.
      const double disagreement = std::abs(measured / length - 1.0);
      if (settings_.max_scale_error > 0.0 && disagreement > settings_.max_scale_error) {
        ++diagnostics_.photometric_rejected;
      } else {
        const double blended = length +
          settings_.photometric_step_gain * (measured - length);
        const double ratio = blended / length;
        motion->x *= ratio;
        motion->y *= ratio;
        diagnostics_.photometric_ratio = ratio;
        ++diagnostics_.photometric_uses;
      }
    }
  }
  // Integrate the photometric tilt increments, and leak them back toward
  // whatever absolute source is running. The leak is not smoothing: over a
  // drive whose true attitude never moves these increments still read a
  // hundredth of a degree a frame, so integrating them alone walks off, and
  // what stops it has to be a measurement of the angle itself rather than of
  // its rate. Where no absolute source is enabled the leak goes to level,
  // which for a road vehicle is the next best statement available.
  if (settings_.esm_attitude) {
    const double leak = settings_.esm_attitude_leak_sec > 0.0 && dt > 0.0
      ? std::min(1.0, dt / settings_.esm_attitude_leak_sec) : 1.0;
    for (auto & held : cameras_) {
      double target_pitch = 0.0;
      double target_roll = 0.0;
      if (settings_.anchor_attitude && held->anchor_ready) {
        target_pitch = held->anchor_pitch;
        target_roll = held->anchor_roll;
      } else if (settings_.band_attitude && held->band_ready) {
        target_pitch = held->band_pitch;
        target_roll = held->band_roll;
      }
      if (held->esm_valid) {
        held->esm_tilt_pitch += held->esm_pitch_since_solve;
        held->esm_tilt_roll += held->esm_roll_since_solve;
      }
      held->esm_tilt_pitch += leak * (target_pitch - held->esm_tilt_pitch);
      held->esm_tilt_roll += leak * (target_roll - held->esm_tilt_roll);
    }
  }
  for (auto & held : cameras_) {
    held->photometric_since_solve = 0.0;
    held->photometric_valid = false;
    held->photometric_broken = false;
    held->esm_yaw_since_solve = 0.0;
    held->esm_pitch_since_solve = 0.0;
    held->esm_roll_since_solve = 0.0;
    held->esm_valid = false;
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
    settings_.vision_scale != 1.0)
  {
    motion->x *= settings_.vision_scale;
    motion->y *= settings_.vision_scale;
  }

  // Stashed before the filters touch `motion`, so what is reported is what the
  // cameras actually said.
  last_hops_valid_ = motion.has_value();
  last_fused_hop_ = motion.has_value()
    ? Eigen::Vector2d(motion->x, motion->y) : Eigen::Vector2d::Zero();
  last_camera_hops_.assign(
    solved.size(), Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN()));
  last_from_map_.assign(solved.size(), 0);
  last_condition_.assign(
    solved.size(), std::numeric_limits<double>::quiet_NaN());
  last_weak_bearing_.assign(
    solved.size(), std::numeric_limits<double>::quiet_NaN());
  last_height_gain_.assign(solved.size(), Eigen::Vector2d::Zero());
  last_pitch_gain_.assign(solved.size(), Eigen::Vector2d::Zero());
  last_mean_range_.assign(
    solved.size(), std::numeric_limits<double>::quiet_NaN());
  last_point_count_.assign(solved.size(), 0.0);
  reach_target_ = std::numeric_limits<double>::quiet_NaN();
  for (size_t i = 0; i < solved.size(); ++i) {
    if (solved[i].has_value() && solved[i]->motion.has_value()) {
      last_camera_hops_[i] = Eigen::Vector2d(solved[i]->motion->x, solved[i]->motion->y);
      last_from_map_[i] = solved[i]->anchored_from_map ? 1 : 0;
    }
    if (solved[i].has_value()) {
      last_condition_[i] = solved[i]->condition;
      last_weak_bearing_[i] = solved[i]->weak_bearing;
      last_height_gain_[i] = solved[i]->height_gain;
      last_pitch_gain_[i] = solved[i]->pitch_gain;
      last_mean_range_[i] = solved[i]->mean_range;
      last_point_count_[i] = solved[i]->point_count;
      if (std::isfinite(solved[i]->mean_range)) {
        reach_target_ = std::isfinite(reach_target_)
          ? std::min(reach_target_, solved[i]->mean_range) : solved[i]->mean_range;
      }
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

  const Pose2 previous_pose = pose_;
  // The scatter of the inlying votes, kept past the block that computes it so
  // the published covariance can be built from what the solve measured.
  double solve_spread = 0.0;

  if (displacement_filter_ && motion.has_value() && dt > 1e-4) {
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
    solve_spread = spread;
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
    if (learn_scale) {
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
    if (const auto & record = displacement_filter_->last_update(); record.has_value()) {
      // Two degrees of freedom, so an honest covariance averages 2. Below that
      // the filter is claiming less certainty than it has.
      diagnostics_.nis_total += record->nis;
      ++diagnostics_.nis_samples;
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
  update.camera_condition = last_condition_;
  update.camera_weak_bearing = last_weak_bearing_;
  update.camera_height_gain = last_height_gain_;
  update.camera_pitch_gain = last_pitch_gain_;
  update.camera_mean_range = last_mean_range_;
  update.camera_point_count = last_point_count_;
  update.radial_height = last_radial_height_;
  update.radial_pitch = last_radial_pitch_;
  update.photometric_distance = last_photometric_distance_;
  update.fused_length = last_fused_length_;

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
        // Reported so a run can be asked whether this loop ran at all. Both
        // fields were declared and never assigned, so the diagnostic line they
        // gate was unreachable and their absence proved nothing.
        diagnostics_.gyro_bias = heading_.rate();
        diagnostics_.heading_drift = offset;
        ++diagnostics_.heading_updates;
      }
    }
    // The map's own reading of the heading, applied where there is no
    // instrument to hold it. Taken only from cameras whose anchors answered
    // this solve -- a carried-over value is the same measurement applied twice.
    if (settings_.anchor_heading_gain != 0.0) {
      double total = 0.0;
      int terms = 0;
      for (auto & camera : cameras_) {
        if (camera->anchor_yaw_fresh) {
          total += camera->anchor_yaw_last;
          ++terms;
          camera->anchor_yaw_fresh = false;
        }
      }
      if (terms > 0 && std::isfinite(total)) {
        pose_.yaw = wrap_pi(
          pose_.yaw + settings_.anchor_heading_gain * total / static_cast<double>(terms));
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

    // The heading the anchor weights are judged from, filtered. See
    // `GroundAnchorMap::set_frame_pose`.
    if (!anchor_weight_yaw_.has_value()) {
      anchor_weight_yaw_ = pose_.yaw;
    } else {
      const double yaw_alpha = settings_.anchor_weight_yaw_tau_sec <= 0.0
        ? 1.0
        : dt / (settings_.anchor_weight_yaw_tau_sec + dt);
      anchor_weight_yaw_ = wrap_pi(
        *anchor_weight_yaw_ + yaw_alpha * wrap_pi(pose_.yaw - *anchor_weight_yaw_));
    }

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


  // After the pose is settled, so each bearing is filed under where the camera
  // actually was when it saw it.
  for (auto & camera : cameras_) {
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
  if (anchors_ != nullptr) {
    std::vector<std::array<double, 7>> polar;
    anchors_->polar(pose_.x, pose_.y, pose_.yaw, polar);
    diagnostics_.anchor_sector_count.fill(0);
    diagnostics_.anchor_sector_weight.fill(0.0);
    diagnostics_.anchor_sector_range.fill(0.0);
    diagnostics_.anchor_sightings.fill(0);
    diagnostics_.anchor_road = 0;
    diagnostics_.anchor_usable = 0;
    diagnostics_.anchor_scatter.fill(0);
    for (const auto & entry : polar) {
      // `polar` reports the bearing in radians.
      const int sector = std::min(
        5, static_cast<int>(std::abs(entry[1]) * 180.0 / M_PI / 30.0));
      diagnostics_.anchor_sector_count[static_cast<size_t>(sector)] += 1;
      diagnostics_.anchor_sector_weight[static_cast<size_t>(sector)] += entry[2];
      diagnostics_.anchor_sector_range[static_cast<size_t>(sector)] += entry[0];
      const int seen = static_cast<int>(entry[3]);
      const int bucket = seen <= 1 ? 0 : (seen == 2 ? 1 : (seen <= 4 ? 2
        : (seen <= 8 ? 3 : (seen <= 16 ? 4 : 5))));
      diagnostics_.anchor_sightings[static_cast<size_t>(bucket)] += 1;
      if (entry[4] >= static_cast<double>(kRoadIdentity)) {
        ++diagnostics_.anchor_road;
      }
      if (entry[6] > 0.5) {
        ++diagnostics_.anchor_usable;
      }
      const double v = entry[5];
      const int band = v < 1e-6 ? 0 : (v < 1e-5 ? 1 : (v < 1e-4 ? 2
        : (v < 1e-3 ? 3 : (v < 1e-2 ? 4 : 5))));
      diagnostics_.anchor_scatter[static_cast<size_t>(band)] += 1;
    }
  }
  if (anchors_) {
    anchors_->extent(
      pose_.x, pose_.y, pose_.yaw, settings_.solve_max_distance_m,
      diagnostics_.anchors_within, diagnostics_.anchors_along,
      diagnostics_.anchors_across);
  }
  if (anchors_) {
    diagnostics_.anchors_adopted = anchors_->adopted();
    diagnostics_.link_gap_m = anchors_->link_gap_m();
    diagnostics_.link_gap_per_m = anchors_->link_gap_per_m();
    diagnostics_.link_range_m = anchors_->link_range_m();
  }

  update.pose = pose_;
  update.twist = twist;
  if (!cameras_.empty()) {
    update.bearing_yaw = cameras_.front()->anchor_yaw_last;
    update.bearing_roll_raw = cameras_.front()->anchor_roll_last;
    update.bearing_pitch_raw = cameras_.front()->anchor_pitch_last;
    update.bearing_tx = cameras_.front()->anchor_tx_last;
    update.bearing_ty = cameras_.front()->anchor_ty_last;
  }
  if (attitude_ && attitude_->started()) {
    update.roll = attitude_->roll();
    update.pitch = attitude_->pitch();
    update.tilt_valid = true;
  }
  if (settings_.esm_attitude && !cameras_.empty()) {
    // What the leaked integrator is actually holding, so a steady-state offset
    // shows up as a number rather than as a score.
    update.roll = cameras_.front()->esm_tilt_roll;
    update.pitch = cameras_.front()->esm_tilt_pitch;
    update.tilt_valid = true;
  } else if (settings_.anchor_attitude && !cameras_.empty() &&
    cameras_.front()->anchor_ready)
  {
    update.roll = cameras_.front()->anchor_roll;
    update.pitch = cameras_.front()->anchor_pitch;
    update.tilt_valid = true;
  } else if (settings_.band_attitude && !cameras_.empty() &&
    cameras_.front()->band_ready)
  {
    // The attitude actually in use, which is the first camera's own reading of
    // the road rather than anything inertial.
    update.roll = cameras_.front()->band_roll;
    update.pitch = cameras_.front()->band_pitch;
    update.tilt_valid = true;
  }
  if (displacement_filter_ && dt > 1e-4) {
    // The hop's own covariance, turned into the world frame the pose lives in,
    // and added to what has accumulated. Dead reckoning has no other answer:
    // there is no absolute reference to bound it.
    const double c = std::cos(previous_pose.yaw);
    const double s = std::sin(previous_pose.yaw);
    Eigen::Matrix2d turn;
    turn << c, -s, s, c;
    // Not the filter's own covariance. That is dominated by
    // `filter_acceleration_noise`, which is set to 8.0 because it makes the
    // filter defer to vision and not because the instrument is that bad -- the
    // measured value is 0.006 to 0.823. The filter's NIS reads 0.001 to 0.05
    // where an honest covariance gives 2.0, so it is claiming a thousandfold
    // less certainty than it has, and a pose covariance built on it inherits
    // that.
    //
    // What is defensible is what the solve itself measured: the scatter of the
    // inlying votes over their count, which is the same quantity the filter
    // takes as R.
    const double hop_variance =
      displacement_filter_->measurement_variance(motion->inliers, solve_spread);
    pose_covariance_.topLeftCorner<2, 2>() +=
      turn * (Eigen::Matrix2d::Identity() * hop_variance) * turn.transpose();
    // Heading is carried by the gyro between solves and trimmed by the ground
    // where it answers, so its variance grows at the instrument's noise.
    const double gyro = settings_.gyro_noise_sigma_rad_s;
    pose_covariance_(2, 2) += gyro * gyro * dt;
    update.pose_covariance = pose_covariance_;
    // Velocity is a hop over a time, so its variance is the hop's over dt^2.
    update.twist_covariance.topLeftCorner<2, 2>() =
      Eigen::Matrix2d::Identity() * (hop_variance / (dt * dt));
    update.twist_covariance(2, 2) = gyro * gyro;
    update.covariance_valid = true;
  }
  // Nothing goes out while the map is still being built. The pose is pinned to
  // the origin over that stretch, and a vehicle already at 8 m/s covers 0.375 m
  // before the first map-anchored solve lands -- published as (0, 0) it becomes
  // a constant offset the estimate then carries for the rest of the run.
  update.pose_valid = !settings_.suppress_pose_until_map_ready || map_ready_;
  pending_updates_.push_back(std::move(update));
}


// A pose graph over the recent trajectory, and the one thing on this rig that
// can drive it.
//
// Nodes are the poses each anchor sighting was taken from. Odometry edges hold
// consecutive nodes at the separation dead reckoning measured. Revisit edges
// come from anchors that two cameras have both bound: the rear drives over
// ground the front mapped 4.5 m earlier, and both know where the point sat in
// their own body frame, so their separation is fixed. The oldest node in the
// window is held, which is the gauge -- without it every earlier attempt at
// this slid, because moving the poses moved the map they were being corrected
// against and nothing was pinned.
//
// Yaw is not a variable. The IMU's heading is exact on this rig
// (correlation 0.985-0.9997 against truth), so the graph is two independent
// linear problems in x and y.
void Estimator::anchor_polar(std::vector<std::array<double, 7>> & out) const
{
  out.clear();
  if (anchors_) {
    anchors_->polar(pose_.x, pose_.y, pose_.yaw, out);
  }
}

std::vector<Estimator::RevisitAudit> Estimator::revisit_audit() const
{
  std::vector<RevisitAudit> out;
  for (const auto & loop : anchors_->revisits()) {
    if (loop.from < 0 || loop.to < 0 ||
      static_cast<size_t>(loop.to) >= pose_history_.size() ||
      static_cast<size_t>(loop.from) >= pose_history_.size())
    {
      continue;
    }
    const auto & a = pose_history_[static_cast<size_t>(loop.from)];
    const auto & b = pose_history_[static_cast<size_t>(loop.to)];
    const double ca = std::cos(a[2]);
    const double sa = std::sin(a[2]);
    const double cb = std::cos(b[2]);
    const double sb = std::sin(b[2]);
    RevisitAudit r;
    r.time_from = pose_time_[static_cast<size_t>(loop.from)];
    r.time_to = pose_time_[static_cast<size_t>(loop.to)];
    r.edge_dx = (ca * loop.bx_from - sa * loop.by_from) - (cb * loop.bx_to - sb * loop.by_to);
    r.edge_dy = (sa * loop.bx_from + ca * loop.by_from) - (sb * loop.bx_to + cb * loop.by_to);
    r.odometry_dx = b[0] - a[0];
    r.odometry_dy = b[1] - a[1];
    r.weight = loop.weight;
    out.push_back(r);
  }
  return out;
}

void Estimator::solve_pose_graph()
{
  const int64_t held = static_cast<int64_t>(pose_history_.size());
  const int64_t window = std::min<int64_t>(settings_.pose_graph_window, held);
  if (window < 8) {
    return;
  }
  const int64_t first = held - window;
  const auto loops = anchors_->revisits();
  std::vector<std::array<double, 7>> edges;   // from, to, dx, dy (relative index)
  std::vector<double> weight;
  for (const auto & loop : loops) {
    if (loop.from < first || loop.to < first || loop.from == loop.to) {
      continue;
    }
    const auto & a = pose_history_[static_cast<size_t>(loop.from)];
    const auto & b = pose_history_[static_cast<size_t>(loop.to)];
    const double ca = std::cos(a[2]);
    const double sa = std::sin(a[2]);
    const double cb = std::cos(b[2]);
    const double sb = std::sin(b[2]);
    // Both saw the same world point, so the separation the two poses must have
    // is the difference of where each put it in its own body frame.
    edges.push_back(
      {static_cast<double>(loop.from - first), static_cast<double>(loop.to - first),
        (ca * loop.bx_from - sa * loop.by_from) - (cb * loop.bx_to - sb * loop.by_to),
        (sa * loop.bx_from + ca * loop.by_from) - (sb * loop.bx_to + cb * loop.by_to)});
    weight.push_back(std::max(loop.weight, 1e-9) * settings_.pose_graph_loop_weight);
  }
  diagnostics_.pose_graph_loops = static_cast<int64_t>(edges.size());
  if (edges.empty()) {
    return;
  }
  // Normal equations for a chain plus a few loops, solved by Gauss-Seidel. The
  // matrix is diagonally dominant and the chain is stiff, so it converges in a
  // handful of sweeps and needs no factorisation.
  const int n = static_cast<int>(window);
  std::vector<double> x(n);
  std::vector<double> y(n);
  for (int k = 0; k < n; ++k) {
    x[static_cast<size_t>(k)] = pose_history_[static_cast<size_t>(first + k)][0];
    y[static_cast<size_t>(k)] = pose_history_[static_cast<size_t>(first + k)][1];
  }
  const std::vector<double> x0 = x;
  const std::vector<double> y0 = y;
  for (int sweep = 0; sweep < settings_.pose_graph_sweeps; ++sweep) {
    for (int k = 1; k < n; ++k) {
      double wx = 0.0;
      double wy = 0.0;
      double total = 0.0;
      // Odometry, both sides.
      wx += (x[static_cast<size_t>(k - 1)] + (x0[static_cast<size_t>(k)] -
        x0[static_cast<size_t>(k - 1)]));
      wy += (y[static_cast<size_t>(k - 1)] + (y0[static_cast<size_t>(k)] -
        y0[static_cast<size_t>(k - 1)]));
      total += 1.0;
      if (k + 1 < n) {
        wx += (x[static_cast<size_t>(k + 1)] - (x0[static_cast<size_t>(k + 1)] -
          x0[static_cast<size_t>(k)]));
        wy += (y[static_cast<size_t>(k + 1)] - (y0[static_cast<size_t>(k + 1)] -
          y0[static_cast<size_t>(k)]));
        total += 1.0;
      }
      for (size_t e = 0; e < edges.size(); ++e) {
        const int from = static_cast<int>(edges[e][0]);
        const int to = static_cast<int>(edges[e][1]);
        const double w = weight[e];
        if (to == k) {
          wx += w * (x[static_cast<size_t>(from)] + edges[e][2]);
          wy += w * (y[static_cast<size_t>(from)] + edges[e][3]);
          total += w;
        } else if (from == k) {
          wx += w * (x[static_cast<size_t>(to)] - edges[e][2]);
          wy += w * (y[static_cast<size_t>(to)] - edges[e][3]);
          total += w;
        }
      }
      if (total > 0.0) {
        x[static_cast<size_t>(k)] = wx / total;
        y[static_cast<size_t>(k)] = wy / total;
      }
    }
  }
  double moved = 0.0;
  for (int k = 0; k < n; ++k) {
    moved += std::hypot(
      x[static_cast<size_t>(k)] - x0[static_cast<size_t>(k)],
      y[static_cast<size_t>(k)] - y0[static_cast<size_t>(k)]);
    pose_history_[static_cast<size_t>(first + k)][0] = x[static_cast<size_t>(k)];
    pose_history_[static_cast<size_t>(first + k)][1] = y[static_cast<size_t>(k)];
  }
  diagnostics_.pose_graph_shift_m = moved / static_cast<double>(n);
  anchors_->rebuild(pose_history_);
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
  anchors_->set_frame_pose(
    fused_path_, pose_.yaw, anchor_weight_yaw_.value_or(pose_.yaw));
  {
    const auto velocity = fused_world_velocity();
    // The filtered yaw rate, not the raw one. `filtered_twist_` already carries
    // it -- it was only ever used to smooth the twist this node publishes, so
    // `twist_lowpass_tau` has had no effect on the trajectory at all. Here it
    // does: it is the rate the lookahead's arc is predicted with.
    anchors_->set_lookahead(
      velocity.has_value() ? velocity->norm() : 0.0, filtered_twist_.z());
  }
  anchors_->set_frame_position(
    pose_.x, pose_.y,
    cameras_.empty() ? 0.0
    : cameras_.front()->model.translation_base_from_camera.z());
  if (settings_.pose_graph_window > 0) {
    solve_pose_graph();
  }

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
    // The same candidates' clarity, in the same order, so the map can spend its
    // free slots on landmarks rather than on whichever the frame listed first.
    Weights clarity;
    if (camera.track_clarity.size() == camera.track_ids.size() &&
      entry.track_slot.size() == static_cast<size_t>(entry.ground_valid.size()))
    {
      clarity.resize(count);
      for (Eigen::Index n = 0; n < count; ++n) {
        clarity(n) =
          camera.track_clarity(entry.track_slot[static_cast<size_t>(usable[
            static_cast<size_t>(n)])]);
      }
    }
    const Points3 world = transform_points_to_world(body, pose_);
    Points2 flat(count, 2);
    flat.col(0) = world.col(0);
    flat.col(1) = world.col(1);
    if (settings_.remember_sighting_poses) {
      Points2 body_flat(count, 2);
      body_flat.col(0) = body.col(0);
      body_flat.col(1) = body.col(1);
      pose_history_.push_back({pose_.x, pose_.y, pose_.yaw});
      pose_time_.push_back(last_accept_stamp_.value_or(0.0));
      anchors_->update(
        camera.source, ids, flat, allow_new, information, body_flat,
        static_cast<int32_t>(pose_history_.size() - 1), clarity);
    } else {
      anchors_->update(camera.source, ids, flat, allow_new, information,
        Points2(), -1, clarity);
    }
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
    }
    if (settings_.parallax_height) {
      integrate_parallax(camera, entry, update);
    } else {
      // The other way of placing an obstacle is to triangulate the current view
      // against a held keyframe, and that needs the images this path does not
      // receive. The Python reached for it here and dereferenced a frame that
      // has no picture in it. Say nothing rather than crash, and count it.
      ++diagnostics_.obstacles_unavailable;
    }
  }
}

// The road warp's leftover parallax, turned into height.
//
// The sparse path holds each feature's first sighting and reads the slip
// between then and now. This does not have to: the warp has already put every
// point that lies on the plane back where it came from, so what dense flow
// still finds *is* the slip, in one frame, everywhere at once.
//
// The relation is the same one. A point at height z, seen from a lens at
// height h, projects onto the plane further out than it is, and that error
// walks with the vehicle:
//
//     ratio = -(slip . travel) / reach^2 ,    z = h * ratio / (1 + ratio)
//
// then the point is walked back along its own ray by (h - z) / h.
void Estimator::integrate_parallax(Camera & camera, const Solved & entry, Update & update)
{
  const Eigen::Index cells = camera.track_parallax.rows();
  if (cells == 0 || camera.track_parallax.cols() != 4 || !entry.motion.has_value()) {
    return;
  }
  const double height = camera.model.translation_base_from_camera.z();
  if (!(height > 1e-6)) {
    return;
  }
  // Where the vehicle went over this hop, in the body frame it started in.
  const Eigen::Vector2d travel(entry.motion->x, entry.motion->y);
  const double travelled = travel.norm();
  if (!(travelled > 1e-4)) {
    return;
  }
  // Both ends of each cell's residual, projected onto the plane. The pair is
  // taken through the same projection so anything the projection gets wrong
  // cancels in their difference.
  Points2 here_pixels(cells, 2);
  Points2 there_pixels(cells, 2);
  for (Eigen::Index i = 0; i < cells; ++i) {
    here_pixels(i, 0) = camera.track_parallax(i, 0);
    here_pixels(i, 1) = camera.track_parallax(i, 1);
    there_pixels(i, 0) = camera.track_parallax(i, 0) - camera.track_parallax(i, 2);
    there_pixels(i, 1) = camera.track_parallax(i, 1) - camera.track_parallax(i, 3);
  }
  const auto tilt = camera_tilt(camera);
  const Eigen::Matrix3d * tilt_ptr = tilt.has_value() ? &tilt.value() : nullptr;
  const bool moves = settings_.ground_height_from_tilt &&
    !settings_.band_attitude && !settings_.anchor_attitude;
  const double scale =
    camera.settings.range_scale * camera.range_scale_learned * imu_scale_;
  Points2 here_ground;
  Points2 there_ground;
  Mask here_valid;
  Mask there_valid;
  pixels_to_ground(
    here_pixels, camera.model, settings_.ground_max_distance_m,
    camera.settings.ground_min_distance_m, tilt_ptr, scale, here_ground, here_valid,
    settings_.pitch_centre_x_m, moves);
  pixels_to_ground(
    there_pixels, camera.model, settings_.ground_max_distance_m,
    camera.settings.ground_min_distance_m, tilt_ptr, scale, there_ground, there_valid,
    settings_.pitch_centre_x_m, moves);

  const Eigen::Vector2d lens = camera.model.translation_base_from_camera.head<2>();
  Points3 mount(1, 3);
  mount.row(0) = camera.model.translation_base_from_camera.transpose();
  const Points3 mount_world = transform_points_to_world(mount, pose_);
  const Eigen::Vector2d origin(mount_world(0, 0), mount_world(0, 1));
  const double ceiling = std::min(
    settings_.obstacle_max_height_m, height * settings_.obstacle_height_margin);

  Points3 body(cells, 3);
  std::vector<double> heights;
  std::vector<Eigen::Index> kept;
  for (Eigen::Index i = 0; i < cells; ++i) {
    if (!here_valid(i) || !there_valid(i)) {
      continue;
    }
    const double residual = std::hypot(
      camera.track_parallax(i, 2), camera.track_parallax(i, 3));
    // `!(x >= t)` rather than `x < t`, so a NaN is refused rather than passed.
    if (!std::isfinite(residual) || !(residual >= settings_.parallax_min_pixels)) {
      continue;
    }
    const double range = std::hypot(
      here_ground(i, 0) - lens.x(), here_ground(i, 1) - lens.y());
    if (!(range > 0.1)) {
      continue;
    }
    // The residual displacement this cell shows against what the plane
    // predicted, resolved along the direction the vehicle went.
    //
    // A point at height z projects onto the plane at range R*h/(h-z), further
    // out than it is, and over a hop d that projection runs on by d*z/(h-z).
    // So the ratio is the slip *per metre travelled* -- not per metre of range,
    // which is what the sparse path's `reach` is (it names the travel, and its
    // gate is `obstacle_slip_baseline_m`, a distance driven).
    const Eigen::Vector2d slip(
      here_ground(i, 0) - there_ground(i, 0), here_ground(i, 1) - there_ground(i, 1));
    const double ratio = settings_.parallax_sign * slip.dot(travel) /
      (travelled * travelled);
    if (!(ratio > 0.0)) {
      continue;
    }
    const double z = height * ratio / (1.0 + ratio);
    if (!std::isfinite(z) || z < settings_.obstacle_min_height_m || z > ceiling) {
      continue;
    }
    body(static_cast<Eigen::Index>(kept.size()), 0) = here_ground(i, 0);
    body(static_cast<Eigen::Index>(kept.size()), 1) = here_ground(i, 1);
    body(static_cast<Eigen::Index>(kept.size()), 2) = 0.0;
    heights.push_back(z);
    kept.push_back(i);
  }
  if (kept.empty()) {
    return;
  }
  const Points3 world = transform_points_to_world(
    body.topRows(static_cast<Eigen::Index>(kept.size())), pose_);
  for (size_t n = 0; n < kept.size(); ++n) {
    const double z = heights[n];
    const double share = (height - z) / height;
    LabelledPoint point;
    point.x = static_cast<float>(
      origin.x() + share * (world(static_cast<Eigen::Index>(n), 0) - origin.x()));
    point.y = static_cast<float>(
      origin.y() + share * (world(static_cast<Eigen::Index>(n), 1) - origin.y()));
    point.z = static_cast<float>(z);
    point.label = kObstacleLabel;
    point.origin_x = static_cast<float>(origin.x());
    point.origin_y = static_cast<float>(origin.y());
    update.points.push_back(point);
    ++diagnostics_.parallax_points;
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
