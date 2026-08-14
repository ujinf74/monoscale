// The estimator, with the ROS taken out of it.
//
// What arrives is feature tracks and IMU samples; what leaves is a pose, a
// twist, and the labelled ground points the solve already produced. Nothing
// here knows about topics, QoS or transform trees, which is what lets it be
// tested without a graph and later be called directly by the tracker in one
// process.
//
// This is the deployed path only. The Python it replaces also carries an
// optical-flow front end for when no tracker node is running; that is the same
// work monoscale_tracker already does in C++, and duplicating it here would
// mean two implementations of the one stage whose cost actually matters.

#ifndef MONOSCALE_CORE__ESTIMATOR_HPP_
#define MONOSCALE_CORE__ESTIMATOR_HPP_

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "monoscale_core/anchors.hpp"
#include "monoscale_core/attitude.hpp"
#include "monoscale_core/fusion.hpp"
#include "monoscale_core/geometry.hpp"
#include "monoscale_core/inertial.hpp"
#include "monoscale_core/strapdown.hpp"

namespace monoscale
{

enum class FusionModel
{
  // Takes the vision translation divided by the interval as a velocity.
  Velocity,
  // Fuses what the two instruments actually measure -- acceleration, and
  // metres between two image times -- and carries an accelerometer bias state.
  Displacement,
  // The same, with the heading and the gyro's bias in the state as well, so
  // the vision residual reaches the one quantity that decides which way the
  // other three point. Propagates on the gyro's rate rather than on the
  // orientation the instrument reports.
  Msckf,
  // The same again with the road taken off the page: position, velocity and a
  // full attitude, so the roll and pitch the ground projection needs come out
  // of the filter's own covariance rather than from a filter standing beside it
  // with none. This is the container the camera-ground parameters go in; it
  // does not hold them yet.
  Msckf6,
};

FusionModel fusion_model_from_name(const std::string & name);

struct CameraSettings
{
  std::string name;
  // Where the camera is bolted, in base_link.
  Eigen::Matrix3d rotation_base_from_camera = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation_base_from_camera = Eigen::Vector3d(0.0, 0.0, 1.0);
  // The nearest ground worth reading. Pixel motion goes as the inverse of
  // range, so the closest ground sweeps the image fastest and is the first
  // thing optical flow loses.
  double ground_min_distance_m = 0.0;
  // How much further than the truth this camera measures the ground to have
  // moved, as a factor to divide out. Not an extrinsic.
  double range_scale = 1.0;
  // Root mean square of how far the filter's own hop ends up from the one
  // vision handed it. Near zero means the filter is reproducing the
  // measurement and any error is upstream of it.
  double hop_residual = 0.0;
  double hop_taken = 0.0;
  // Calibration, as CameraInfo would report it, at the resolution it describes.
  Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
  Eigen::VectorXd distortion;
  Lens lens = Lens::Pinhole;
  int calibration_width = 0;
  int calibration_height = 0;
};

struct EstimatorSettings
{
  std::vector<CameraSettings> cameras;

  double sync_tolerance_sec = 0.02;
  int pair_queue_depth = 12;

  bool use_imu_yaw = true;
  double imu_max_age_sec = 0.02;
  double imu_max_gap_sec = 0.12;

  double ground_max_distance_m = 25.0;
  double ground_ransac_threshold_m = 0.12;
  int ground_min_inliers = 24;
  double max_scale_error = 0.08;
  double max_translation_per_frame_m = 1.0;
  double max_yaw_per_frame_rad = 0.35;
  // How many frames of arrears a single update may close. It has to be finite:
  // without a bound, a solve that has genuinely gone wrong is accepted simply
  // because enough frames were rejected first.
  double max_arrears_frames = 8.0;

  bool attitude_from_imu = true;
  double attitude_tau_sec = 60.0;
  double attitude_gravity_tolerance = 0.3;

  // What the heading source is expected to do wrong, as the part's own numbers
  // rather than as tuning. Setting the first to 0 believes the reported heading
  // outright, which is what every measurement before this assumed.
  double gyro_bias_sigma_rad_s = 0.0;
  double gyro_bias_walk_sigma_rad_s = 1.0e-5;
  double gyro_noise_sigma_rad_s = 1.0e-3;

  bool coast_on_reject = true;
  double twist_lowpass_tau = 0.12;

  double mapping_baseline_m = 0.35;
  double mapping_min_period_sec = 0.04;
  double obstacle_min_height_m = 0.2;
  double obstacle_max_height_m = 2.5;
  // Travel a feature must accumulate before its ground projection's slip is
  // read as a height. 0 means no obstacles at all here: the other way of
  // placing one is to triangulate against a held keyframe, and that needs the
  // images this path does not receive.
  //
  // The Python shipped this at 0 and reached for the keyframe path, which on
  // the track front end dereferences a frame with no picture in it -- so the
  // deployed default produced obstacles by crashing. On is the only setting
  // that means anything here.
  //
  // 0.35 is inherited from `mapping_baseline_m` and is not measured for this
  // method. The slip's own note reports that baseline as marginal: a 0.15 m
  // obstacle slides 68 mm over it, which the pitch wobble buries, and that
  // scored precision 0.455 at recall 0.016. What fixed it was letting each
  // feature accumulate its own baseline over as many frames as it survives,
  // which is what this threshold now gates -- so the right value is probably
  // longer, and wants scoring against the LiDAR reference grid before anyone
  // trusts the map it builds.
  double obstacle_slip_baseline_m = 0.35;
  // A feature at the camera's own height projects to infinity, so the band
  // stops short of it.
  double obstacle_height_margin = 0.7;
  int obstacle_slip_patience = 30;

  // What the six degree of freedom filter is told about the two things only it
  // can carry: how much a levelling against gravity is worth, and how far the
  // vehicle may climb over one hop. The second is a hop-length number and not a
  // drive-length one, so a graded road is left alone.
  double spatial_gravity_noise = 2.0;
  // What the gyro is worth on roll and pitch, apart from the heading. See
  // SpatialMsckfFilter::Settings: this is the stiffness the ground projection
  // needs, and it is not the stiffness the heading needs.
  double spatial_tilt_gyro_noise = 4.4e-4;
  // How much acceleration the vehicle may be under and still be levelled, on
  // the world-frame residual rather than on the reading's magnitude. Its own
  // parameter and not attitude_gravity_tolerance: the two gate different
  // quantities and only look alike.
  double spatial_gravity_tolerance = 0.15;
  double spatial_height_noise_m = 0.01;
  // How far the ground projection's scale is allowed to move. Zero holds it at
  // one; SpatialMsckfFilter::Settings carries what it costs and buys, measured
  // on three drives and on the same drives with the scale injected wrong.
  double spatial_scale_variance = 0.0;
  // The six degree of freedom filter's own priors on the accelerometer bias and
  // on how level the vehicle starts. Both differ from what the planar filters
  // carry, and SpatialMsckfFilter::Settings says why.
  double spatial_bias_variance = 0.01;
  double spatial_tilt_variance = 7.6e-3;
  // How often the attitude is levelled against gravity: on every instrument
  // sample, or once for the whole interval a hop spans.
  bool spatial_level_every_sample = true;
  // Screen the instrument the way the planar path's propagator does before
  // handing it to the six degree of freedom filter: hold the last believable
  // reading through an impulse, and integrate nothing sideways until vision has
  // supplied the constant the integral is missing.
  // Whether the six degree of freedom filter's own roll and pitch are what the
  // ground projection is given. Off, the separate attitude filter answers, as
  // it does for every other path, and off is measured.
  //
  // Closing that loop was the point of carrying an attitude at all, and it does
  // not work yet. One degree of tilt moves every range in the frame by two and
  // a half to five per cent, so an attitude that jitters is a set of ranges
  // that jitter, and that lands in the hop vision reports before any filter
  // sees it. Relative error over three drives: 4.4, 3.3 and 4.4 per cent with
  // the filter's own tilt, against 2.1, 2.3 and 2.3 with the trim's -- and the
  // planar filters' own 2.0, 2.2 and 2.3.
  //
  // The separate filter is not more accurate. It is stiffer: a sixty second
  // trim against this one's effective two and a half, and what the projection
  // needs is stiffness while what the filter needs is response. Tuning did not
  // reconcile them -- the tilt's process noise was taken to zero and the
  // relative error stayed at 3.3 to 5.4 per cent, because what moves the tilt
  // is the levelling measurement rather than the propagation. So the projection
  // is given the trim and the filter keeps its state, until something smooths
  // the one without slowing the other.
  bool spatial_tilt_to_projection = false;
  bool spatial_screen_impulses = true;
  bool spatial_wait_for_vision = true;
  // The same three degrees of freedom the planar filter's hop has, and the same
  // gate. The vehicle staying on the road is not one of them: it is an
  // assumption rather than a measurement, it gets its own update, and gating
  // the two together threw away 57% of the vision the assumption was strained
  // against.
  double spatial_innovation_gate = 11.3;

  // The nearest ground the mounting-pitch lean is read from. It is not the
  // band the solve uses -- those points still vote for the pose -- because the
  // near ground is where flow is worst and precision weighting would hand it
  // the regression. AnchorAlignment carries what leaving it at zero costs.
  double radial_min_range_m = 1.5;

  int max_ground_anchors = 4000;
  int anchor_max_age_frames = 120;
  double anchor_update_gain = 0.0;
  int anchor_max_observations = 20;
  bool anchor_select_by_consistency = true;
  double anchor_seed_travel_m = 0.0;

  int frame_decimation = 1;
  bool adaptive_solve_interval = true;
  double solve_trigger_disparity_px = 3.0;
  int solve_min_frames = 2;
  int solve_max_frames = 40;

  // Hold the pose until the anchor map can answer for itself. The first frames
  // have no map, so they fall back to the two-frame solve, and on a vehicle
  // that has not moved yet that solve reads ground texture noise as travel.
  bool require_map_before_translating = true;
  bool suppress_pose_until_map_ready = true;

  bool zero_velocity_update = false;
  double zero_velocity_gyro_dps = 0.5;
  double zero_velocity_accel_mps2 = 0.15;
  double zero_velocity_speed_mps = 0.8;
  double zero_velocity_disparity_px = 0.5;
  double zero_velocity_dropout_mps2 = 0.0;
  double zero_velocity_quiet_fraction = 1.0;
  double zupt_velocity_sigma = 0.01;

  bool use_inertial_prediction = true;
  bool inertial_use_acceleration = true;
  double inertial_velocity_gain = 0.6;
  double inertial_max_acceleration_mps2 = 12.0;
  int inertial_acceleration_median_window = 1;
  Integration inertial_integration = Integration::Rk4;
  // How far before its own stamp an accelerometer sample describes. CARLA fits
  // a quadratic to the sensor's last three positions and reports the second
  // derivative, which is valid one sensor period before the stamp it carries. A
  // real instrument does measure specific force, so this stays zero unless the
  // recording is CARLA's.
  double imu_acceleration_offset_sec = 0.0;

  FusionModel fusion_model = FusionModel::Displacement;
  double filter_acceleration_noise = 1.55;
  double filter_vision_noise = 0.25;
  double filter_vision_noise_m = 0.005;
  double filter_bias_walk = 0.01;
  double filter_reference_inliers = 300.0;
  double filter_innovation_gate = 9.0;

  // The MSCKF's own numbers. The gate is larger because the measurement has
  // three degrees of freedom rather than two, and the yaw noise is the floor
  // under whatever the ground solve reports for its own heading.
  double msckf_gyro_noise = 0.01;
  double msckf_gyro_bias_walk = 0.0005;
  double msckf_vision_yaw_noise = 0.02;
  double msckf_initial_gyro_bias_variance = 1.0e-4;
  double msckf_innovation_gate = 11.3;
  double msckf_reject_beyond_m = 1.5;
  // How far the instrument's reported heading is allowed to be wrong, in
  // radians. Nothing else in the filter says where north is -- the anchors
  // turn with the estimate, so they cannot argue with a heading that drifts --
  // which makes this the term that pins the gyro bias down.
  //
  // It buys accuracy against a good instrument and costs it against a bad one,
  // and the trade is monotone, so no value wins everywhere. Measured over one
  // recording replayed with a gyro bias injected into it, ATE in metres:
  //
  //   bias deg/s  displacement   0.02    0.05     0.1     off
  //          0.0        0.111   0.136   0.171   0.178   0.206
  //          0.1        0.327   0.183   0.205   0.226   0.240
  //          0.3        0.438   0.338   0.329   0.319   0.309
  //          1.0        1.129   1.038   1.017   0.815   0.567
  //
  // 0.1 is the hedge that never loses badly at either end. Tighten it towards
  // 0.02 for a rig whose AHRS is trusted, and set it to 0 -- which switches
  // the measurement off entirely and leaves the heading to the gyro and the
  // ground solve -- for one whose AHRS is not.
  double msckf_heading_noise = 0.1;
  // Or let the drive decide which of those it is: the gain is how hard a
  // one-sided vision yaw residual counts against the reported heading, and 0
  // switches the whole mechanism off. Off is the measured answer as well as
  // the default -- PlanarMsckfFilter::Settings carries the numbers and why.
  double msckf_heading_adaptive_gain = 0.0;
  double msckf_heading_adaptive_window = 50.0;

  // How hard front/rear disagreement counts against a solve, and the penalty
  // when only one camera produced an answer at all.
  double camera_disagreement_weight = 1.0;
  double single_camera_variance = 0.5;
  bool fuse_cameras_by_spread = false;
};

// One hop as the front end followed it. Pixels are in the frame the tracker
// processed, whose size travels with them so the intrinsics can be brought to
// the same coordinates.
struct TrackFrame
{
  double stamp = 0.0;
  int width = 0;
  int height = 0;
  Identities ids;
  Points2 previous_pixels;
  Points2 pixels;
};

struct ImuSample
{
  double stamp = 0.0;
  // (x, y, z, w)
  Eigen::Vector4d orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
};

// What a published point is claimed to be. The solve already knows: a point
// that survived the ground registration is road, one whose projection slid
// against the travel is standing on it.
constexpr float kGroundLabel = 0.0f;
constexpr float kObstacleLabel = 1.0f;

struct LabelledPoint
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float label = kGroundLabel;
  // Where the camera stood when it saw this, so a consumer can carve free
  // space along the ray without knowing the mounts or the pose it was solved
  // against.
  float origin_x = 0.0f;
  float origin_y = 0.0f;
};

struct Update
{
  double stamp = 0.0;
  Pose2 pose;
  // Body-frame vx, vy and yaw rate.
  Eigen::Vector3d twist = Eigen::Vector3d::Zero();
  // False while the map is still being built and the pose is pinned at the
  // origin. Whoever consumes this is better served by no pose than by one that
  // is knowingly standing still while the car is not.
  bool pose_valid = false;
  std::vector<LabelledPoint> points;
};

struct Diagnostics
{
  int64_t pairs_seen = 0;
  int64_t frames_processed = 0;
  int64_t frames_evicted = 0;
  int64_t motion_failures = 0;
  int64_t fail_no_solve = 0;
  int64_t fail_translation = 0;
  int64_t fail_yaw = 0;
  int64_t yaw_only_updates = 0;
  int64_t coasted = 0;
  int64_t filter_rejections = 0;
  int64_t imu_yaw_misses = 0;
  int64_t map_aligned_frames = 0;
  int64_t zupt_holds = 0;
  int64_t obstacles_unavailable = 0;
  // How each camera's ground projection leans, averaged over the solves it
  // answered. This is the mounting pitch and nothing else -- AnchorAlignment
  // carries why the height cannot appear here. Per camera because they are
  // mounted at different heights and opposite pitches, and a mean across them
  // would cancel the very thing being measured.
  std::vector<double> radial_linear;
  std::vector<int64_t> radial_samples;

  // What the MSCKF learned and what it thought of the last measurement. The
  // gyro bias is the state the older filters had no place for, so its value is
  // the whole claim; the NIS says whether the covariance is honest.
  double gyro_bias = 0.0;
  double last_nis = 0.0;
  // The one-sided part of vision's yaw residual, in radians per hop: how far
  // the reported heading is pulling the estimate away from the ground.
  double heading_drift = 0.0;
  int64_t filter_dropped = 0;
  // What only the six degree of freedom filter has to report: how it thinks the
  // vehicle is leaning, how high it thinks it is, and how many of the
  // accelerometer's samples it was willing to level against.
  double roll = 0.0;
  double pitch = 0.0;
  double height = 0.0;
  int64_t levelled = 0;
  // How much further than the truth vision is measuring the ground to have
  // moved, as the six degree of freedom filter has it. 1 is the ground
  // projection's scale being right.
  double range_scale = 1.0;
  // Root mean square of how far the filter's own hop ends up from the one
  // vision handed it. Near zero means the filter is reproducing the
  // measurement and any error is upstream of it.
  double hop_residual = 0.0;
  double hop_taken = 0.0;
  double worst_rejected = 0.0;
  double worst_allowance = 0.0;
  double camera_disagreement = 0.0;
  // How far the most-moved ground points shifted on the last solve, in pixels.
  // The mean is useless here: ground features sit anywhere from half a metre to
  // twelve away and pixel shift falls with the square of that, so the far ones
  // drag the average under any sensible threshold. This is the figure the solve
  // trigger reads.
  double last_disparity = -1.0;
  int anchors = 0;
  std::map<std::string, double> stage_seconds;
};

class Estimator
{
public:
  explicit Estimator(const EstimatorSettings & settings);
  ~Estimator();

  // Replace a camera's calibration, as a CameraInfo would. Safe to call while
  // running: where a camera is bolted does not change, but what it reports its
  // intrinsics to be can arrive late.
  void set_calibration(
    size_t camera, const Eigen::Matrix3d & k, const Eigen::VectorXd & distortion,
    Lens lens, int width, int height);

  // Take the mount off a transform tree instead of the settings.
  void set_mount(
    size_t camera, const Eigen::Matrix3d & rotation, const Eigen::Vector3d & translation);

  void ingest_tracks(size_t camera, const TrackFrame & frame);
  void ingest_imu(const ImuSample & sample);

  // Drain whatever the ingests produced. One ingest can close several pairs.
  std::vector<Update> take_updates();

  const Diagnostics & diagnostics() const {return diagnostics_;}
  const Pose2 & pose() const {return pose_;}
  size_t camera_count() const {return cameras_.size();}

private:
  struct Camera;
  struct Frame;
  struct Solved;

  void try_process_pairs();
  bool ready_to_solve() const;
  void process_pair();
  std::optional<Solved> solve_camera(
    Camera & camera, std::optional<double> yaw_delta, std::optional<double> yaw_guess);
  void remember_solve_pixels(Camera & camera);
  std::optional<Eigen::Matrix3d> body_tilt() const;
  std::optional<double> imu_yaw_at(double stamp) const;
  bool imu_still_arriving(double stamp) const;
  bool is_stationary(double start, double end, double speed) const;
  void update_anchors(const std::vector<std::optional<Solved>> & solved);
  void integrate_points(
    const std::vector<std::optional<Solved>> & solved, const Pose2 & previous_pose,
    Update & update);
  void integrate_obstacle_slip(Camera & camera, const Solved & solved, Update & update);
  CameraModel frame_model(const Camera & camera, int width, int height) const;

  EstimatorSettings settings_;
  std::vector<std::unique_ptr<Camera>> cameras_;

  Pose2 pose_;
  int frames_since_solve_ = 0;
  std::optional<double> last_accept_stamp_;
  std::optional<double> last_mapping_stamp_;
  std::optional<Eigen::Vector2d> last_seed_position_;
  Eigen::Vector3d filtered_twist_ = Eigen::Vector3d::Zero();
  bool map_ready_ = false;

  std::deque<std::pair<double, double>> imu_yaw_samples_;
  struct MotionSample
  {
    double stamp;
    double rate;
    double force;
  };
  std::deque<MotionSample> imu_motion_samples_;
  struct AccelerationSample
  {
    double stamp;
    // World frame, gravity removed: what the displacement filter propagates on.
    Eigen::Vector2d acceleration;
    // The same acceleration in the body frame, and the yaw rate beside it:
    // what the MSCKF propagates on, because it carries the heading itself and
    // must not be handed a vector already rotated by somebody else's.
    Eigen::Vector2d body_acceleration;
    double rate;
    // What the part actually reported, gravity and all. The six degree of
    // freedom filter puts gravity back itself, because where gravity points is
    // one of the things it is estimating.
    Eigen::Vector3d specific_force;
    Eigen::Vector3d angular_velocity;
    // Whether that reading was believable, or is the last one that was. The
    // simulator reports suspension contacts as hundreds of m/s2, and a single
    // 17 ms step of one leaves the velocity wrong for many frames.
    bool force_screened = false;
    double dt;
    double yaw;
  };
  std::deque<AccelerationSample> imu_window_;

  // Walk the buffered IMU samples covering an interval, boundaries included.
  void replay_inertial(
    double from, double to,
    const std::function<void(const AccelerationSample &, double)> & step) const;
  std::optional<double> imu_stamp_;
  // The instrument's heading when the run started. The filter works in the
  // estimator's frame, which begins at zero, so the reported heading only
  // means anything as a difference from where it began.
  std::optional<double> imu_yaw_datum_;

  std::unique_ptr<AttitudeFilter> attitude_;
  HeadingBiasFilter heading_;
  std::vector<std::pair<double, double>> heading_observations_;
  PlanarInertialPropagator inertial_;
  PlanarVelocityFilter velocity_filter_;
  std::unique_ptr<PlanarDisplacementFilter> displacement_filter_;
  std::unique_ptr<PlanarMsckfFilter> msckf_filter_;
  std::unique_ptr<SpatialMsckfFilter> spatial_filter_;
  // The last accelerometer reading believable enough to propagate on.
  std::optional<Eigen::Vector3d> last_specific_force_;
  double hop_residual_squared_ = 0.0;
  double hop_taken_squared_ = 0.0;
  int64_t hop_residual_count_ = 0;

  std::vector<Update> pending_updates_;
  Diagnostics diagnostics_;
};

}  // namespace monoscale

#endif  // MONOSCALE_CORE__ESTIMATOR_HPP_
