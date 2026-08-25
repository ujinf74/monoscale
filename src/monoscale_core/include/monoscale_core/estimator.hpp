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
  // Learn how far this camera's mounting pitch is out, from the lean the
  // alignment residual leaves along each point's own bearing.
  //
  // Off by default and per camera, because the observable is not there for
  // every camera. Measured over mounting errors of -1 to +1 degree, the rear
  // camera's lean runs monotone at 0.0056 per degree and through zero where the
  // mounting is right; the front camera's says nothing, because its ground band
  // starts at 0.6 m and the near ground -- weighted by precision, and the first
  // thing optical flow loses -- dominates the alignment those residuals are
  // measured about. Excluding it from the regression is not enough; it would
  // have to leave the solve, and the band it uses was chosen on drift.
  bool learn_mounting_pitch = false;
  // What this camera counts for when the cameras are combined. 0 on any of
  // them falls back to inlier counts for all.
  double fusion_weight = 0.0;
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
  double ground_align_softness_m = 0.0;
  // Far limit for points entering the pose solve. 0 uses ground_max_distance_m.
  double solve_max_distance_m = 0.0;
  // Far limit for sightings that may enter the anchor map. 0 means no limit
  // beyond the ground band itself.
  double anchor_max_range_m = 0.0;
  // How hard to pull a hop's lateral component towards the non-holonomic
  // constraint. 0 leaves it free, 1 removes it entirely.
  double nonholonomic_lateral = 0.0;
  // How fast the ground scale follows the inertial filter's innovation.
  double imu_scale_gain = 0.0;
  // Hops shorter than this carry more noise than signal in that ratio.
  double imu_scale_min_hop_m = 0.05;
  // How far a translation may sit from what inertial propagation expects.
  double inertial_gate_m = 0.0;
  // How fast the per-camera range scale follows the measured radial residual.
  double range_scale_gain = 0.0;
  // Seed the anchor alignment with the previous hop instead of the median vote.
  bool align_seed_from_last_hop = false;
  int align_restarts = 1;
  double align_ambiguity_ratio = 0.0;
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

  // What a mounting pitch error is worth knowing to before the drive starts,
  // and how much of the lean is that error rather than the road. The gain is
  // measured: 0.0056 of lean per degree, which is 0.321 per radian.
  double mounting_pitch_variance = 3.0e-4;
  double mounting_pitch_gain = 0.321;
  double mounting_pitch_noise = 0.002;
  // How fast a mounting can drift, per root second. Without it the estimate
  // converges on the first few hundred solves -- taken while the anchor map is
  // still filling -- and never reconsiders.
  double mounting_pitch_walk = 1.0e-4;
  // Whether what is learned is fed back into the ground projection.
  //
  // Off, and measured. The anchor map is built from whatever projection was in
  // force when each sighting went into it, so changing the projection mid-drive
  // leaves the current view registering against a history that disagrees with
  // it. Closing the loop that way was unstable: over a mounting deliberately
  // turned by -0.5 degrees the two drives learned +0.53 and -0.40, opposite
  // signs from the same error, and ATE went from 0.18 m to 1.54 on a rig whose
  // mounting was right to begin with.
  //
  // Open, the estimate is a calibration finding rather than a correction --
  // which is most of what online calibration is for, minus the online part.
  bool mounting_pitch_apply = false;

  // The nearest ground the mounting-pitch lean is read from. It is not the
  // band the solve uses -- those points still vote for the pose -- because the
  // near ground is where flow is worst and precision weighting would hand it
  // the regression. AnchorAlignment carries what leaving it at zero costs.
  double radial_min_range_m = 1.5;

  int max_ground_anchors = 4000;
  int anchor_max_age_frames = 120;
  double anchor_update_gain = 0.0;
  int anchor_max_observations = 20;
  // The anchor map's own quality gates. These existed in AnchorSettings from
  // the start but were never forwarded, so no configuration could reach them
  // and the hardcoded defaults were the only values ever measured.
  double anchor_initial_variance = 0.04;
  double anchor_max_variance = 0.09;
  int anchor_trial_observations = 4;
  double anchor_min_update_gain = 0.0;
  double anchor_link_radius_m = 0.0;
  bool anchor_link_adopter_writes = false;
  int anchor_link_rebind_grace = 1;
  bool anchor_evict_by_age = false;
  bool anchor_evict_for_new = false;
  bool anchor_link_measure_only = false;
  // Steering the hop with off-ground structure. The ground solve fixes how far
  // the vehicle went; this says which way, from features that carry no height.
  // Weight is the fraction of the correction taken, 0 leaves the solve alone.
  double epipolar_weight = 0.0;
  double epipolar_softness_rad = 0.02;
  double epipolar_min_hop_m = 0.0;
  // Where the IMU is bolted, from base_link, in the base frame. An IMU off the
  // origin does not measure what the origin does: it also picks up alpha x r
  // as the vehicle's yaw rate changes and omega x (omega x r) as it turns.
  // Zero leaves the samples exactly as they arrive. The instrument's axes are
  // taken to be the base frame's -- the rig mounts it with no rotation, and a
  // rotated one would have to be turned before any of this applies.
  Eigen::Vector3d imu_translation_base_from_imu = Eigen::Vector3d::Zero();
  // Angular acceleration has to be differentiated from the gyro, and
  // differentiating amplifies exactly the noise the gyro has most of. This
  // low-passes it; 0 takes the raw difference.
  double imu_angular_accel_tau_sec = 0.0;
  bool epipolar_non_ground_only = true;
  // Throw the solve away when the two disagree by more than this many degrees.
  // The bearing is too coarse to steer by, but a solve the off-ground
  // structure flatly contradicts is a solve worth not having.
  // How much of a camera's standing disagreement with the fused pose to take
  // back out of the hop its map reports. 0 leaves the correction whole.
  // What a camera counts for when its hop came off the anchor map, against
  // one that fell back to comparing two frames.
  //
  // These are not the same quantity. The map path reports `placed - pose_`,
  // which carries a correction for whatever error the pose had already
  // accumulated; the fallback reports a plain displacement, which carries
  // none. Averaging them as equals applies the correction at half strength and
  // drags the camera that had no map. Measured, the drives split exactly in
  // the stretches where one camera has the map and the other does not.
  // Metres of hop to give back per radian of turn, and a flat scale on top.
  //
  // Measured per solve against truth: hops on a straight read 0.1 to 0.5% short
  // while hops through a manoeuvre read 0.7 to 0.8% long, and the sign flips
  // inside a single drive. That is about a percentage point per 0.15 rad/m of
  // curvature. It is a common mode error -- both cameras share it -- which is
  // the one kind this two-camera structure cannot cancel for itself.
  // The same Gaussian the anchor alignment uses, for the two frame fallback.
  // Extra metres of inlier gate per radian the vehicle turns over a hop.
  //
  // A ground point at range R sweeps R*dyaw when the vehicle rotates, so any
  // error in its range or in the yaw is multiplied by the turn -- the votes
  // spread wider through a manoeuvre than they ever do on a straight. A gate
  // sized for the straight then throws away good points exactly when there are
  // fewest of them. Measured, the curved drives want 0.16-0.24 m where the
  // straight ones want 0.06-0.10.
  // Move the pose by what the cameras solved, leaving the filter to run
  // alongside without writing the hop. Diagnostic: it answers whether a drive's
  // error is the vision's or the filter's, which nothing else can.
  // How much of the lateral error the camera split predicts to take back out.
  //
  // Each camera's range scale error acts about its own lens, not about
  // base_link, so a scale error eps on a camera mounted at x contributes
  // eps * dyaw * x SIDEWAYS whenever the vehicle turns. The front lens sits at
  // +3.694 and the rear at -0.82, and the two cameras' scale errors are equal
  // and opposite -- so the along-track parts cancel, as they always have, and
  // the lateral parts ADD. It is invisible on a straight and accumulates
  // through a manoeuvre, which is exactly where this estimator is worst.
  //
  // Everything in it is observable: eps_i is how far camera i's hop reads
  // against the fused one, x_i is the mount, and the weights are the fusion's
  // own. 1.0 subtracts the whole prediction.
  double camera_split_lever = 0.0;
  // Restrict the solve to a rectangle of the frame, in fractions of width and
  // height. Diagnostic: if the projection from pixels to metres were exact,
  // every region of the image would produce the same hop. Where they differ,
  // the difference is a map of the model's error. x1 <= x0 leaves it off.
  // How hard to discount a ground point for being far away.
  //
  // Measured on this rig: restricting the solve to a 1 m ring, the registration
  // residual runs 0.0053 m at 1-2 m and 0.0379 m at 4-5 m -- a factor of seven
  // -- and all four rings imply the same 1.5-2.2 mrad of bearing error, which
  // at 640 px is half a pixel. So the residual is one pixel-level error
  // amplified by range, exactly as (R^2 + h^2) / h says it should be, and the
  // solve has been averaging near and far points alike. Weight is
  // (R^2 + h^2)^-power: 0 is the old uniform behaviour, 2 is the inverse
  // variance the geometry implies.
  // What one road-warp point counts for against one corner.
  //
  // The tracker can carry a handful of points on a photometric fit of the road
  // region instead of on per-corner flow, and marks them with identities at or
  // above `kRoadIdentity`. Compared like for like -- 150 corners against 150
  // corners plus the warp -- the warp's points register with a spread of
  // 0.0026 m against the corners' 0.0083, so their inverse variance is about
  // ten times higher. That ratio is the weight; the number of points is only
  // what it takes to define the warp. Replicating a grid to buy weight instead
  // would be forging it.
  double road_point_weight = 1.0;
  // Restore the old inverse-square weighting of anchor sightings, as a power
  // on range. 0 uses the geometry instead, which is the default and the
  // correct model -- see `update_anchors`.
  double anchor_information_power = 0.0;
  // Weigh anchors in the registration by information rather than sightings.
  bool anchor_weight_by_information = false;

  // Measure a ground point's distance from the camera, and hand the anchor
  // alignment the mount, in the same frame the points are in.
  //
  // `pixels_to_ground` rotates the intersection into the level frame and then
  // subtracted the camera's position in the BODY frame from it. That is only
  // harmless while the body is level, and it is not: curve_s10 rolls 5.4
  // degrees at p95, where 0.89 m of camera height is 78 mm against a 114 mm
  // hop. The claim in vision_fisheye.param.yaml that this vehicle "barely
  // tilts -- 0.1 degrees" is wrong by fifty times.
  //
  // Set false to restore the mixed frames. The eleven drive set cannot judge
  // this on its own: seven drives move by under a millimetre because they have
  // no roll to speak of, and the mean is decided by curve_s05, which moves
  // under every perturbation tried. Applied because it is right.
  bool level_frame_origin = true;
  double range_weight_power = 0.0;
  double pixel_region_x0 = 0.0;
  double pixel_region_y0 = 0.0;
  double pixel_region_x1 = 0.0;
  double pixel_region_y1 = 0.0;
  bool vision_only_pose = false;
  double ground_rotation_threshold_m = 0.0;
  // How many reweighting passes the two-frame fallback runs. The anchor
  // alignment takes three; this path took one.
  int ground_pair_passes = 1;
  double ground_pair_softness_m = 0.0;
  // Multiplier on the measured ground residual, replacing the two constants
  // above when non-zero. The reasoning was that they are widths of a Gaussian
  // over a residual, so the residual is the thing to set them from, and it
  // varies seven times across these drives -- 0.012 m on a clean straight,
  // 0.089 through a parking manoeuvre -- which no constant can follow.
  //
  // Measured, and it loses. Multipliers 1.0 / 1.5 / 2.0 / 3.0 score 0.2477 /
  // 0.2201 / 0.2074 / 0.2044 against 0.1951 for the tuned constants, and the
  // repeat spread it was meant to bring down goes to 2.05-4.70x against 2.17.
  // Sharing one scale between the cameras instead of one each is worse again
  // (0.2135 at 2.0), so it is not the two-camera weighting. Left at 0.
  double softness_from_residual = 0.0;
  // Solve both cameras' ground points together instead of solving each camera
  // apart and averaging the two answers. Their points are already in base_link
  // and describe the same hop, so this is the fusion the geometry allows.
  bool fuse_camera_points = false;
  double curvature_scale_gain = 0.0;
  double vision_scale = 1.0;
  double map_solve_weight = 1.0;
  double anchor_divergence_gain = 0.0;
  double epipolar_reject_deg = 0.0;
  // Or keep the solve and trust it less: a camera whose ground answer the
  // off-ground structure half-agrees with weighs proportionally less when the
  // two cameras are combined. Soft has beaten hard everywhere else here.
  double epipolar_trust_deg = 0.0;
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
  // A floor under what the ground solve claims about its own heading, not a
  // fallback for when it claims nothing. The fit reports fit over lever over
  // the root of the inlier count, which on a good frame is 2e-4 rad -- and that
  // root assumes the residuals are independent, which a mounting or tilt error
  // is not: it leans the whole frame one way and more points do not average it
  // out. Measured over three drives, mean ATE in metres: 0.248 with no floor,
  // 0.165 at 0.01, 0.223 at 0.02. Better on every drive at 0.01, and the cost
  // is two tenths of a point of relative error.
  double msckf_vision_yaw_noise = 0.01;
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
// Identities at or above this are carried by the tracker's photometric road
// fit rather than by per-corner flow. 2^40 sits far above what the corner
// counter reaches in any run and far below the 2^53 a float64 message carries
// exactly, so the two ranges cannot meet.
constexpr int64_t kRoadIdentity = 1LL << 40;

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
  // What each camera made of this hop, in the body frame of the pose held at
  // `previous_stamp`, and what they fused to before any filter touched it.
  //
  // This is the only per-solve quantity that can be laid beside ground truth.
  // Everything else the estimator says about a camera is measured against the
  // other camera or against itself, and a self-referential measurement cannot
  // tell a bias from a disagreement.
  double previous_stamp = 0.0;
  bool hops_valid = false;
  Eigen::Vector2d fused_hop = Eigen::Vector2d::Zero();
  // Not finite where that camera did not answer this pair.
  std::vector<Eigen::Vector2d> camera_hops;
  // 1 where that camera's hop came from the anchor map rather than the two
  // frame fallback.
  std::vector<uint8_t> camera_from_map;
  // What the pose was actually moved by, after the filters and the rejection
  // gate have had it. `fused_hop` is what the cameras said; this is what the
  // map got. They are not the same quantity and only this one moves the pose.
  Eigen::Vector2d applied_hop = Eigen::Vector2d::Zero();
  bool applied_valid = false;
  bool coasted = false;
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
  // Temporary instrumentation: where the slip obstacle path loses its
  // candidates. Live runs produce no obstacle points at all and the gates are
  // only distinguishable by counting them.
  int64_t obstacle_usable = 0;
  int64_t obstacle_ready = 0;
  int64_t obstacle_no_slip = 0;
  int64_t obstacle_out_of_band = 0;
  int64_t obstacle_points = 0;
  int64_t anchors_adopted = 0;
  double link_gap_m = 0.0;
  double link_gap_per_m = 0.0;
  double link_range_m = 0.0;
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
  // What each camera has learned about its own mounting pitch, in degrees.
  std::vector<double> mounting_pitch;
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
  // Distance each camera believes the vehicle has travelled, in order.
  std::vector<double> camera_travel;
  std::vector<int64_t> camera_solves;
  std::vector<double> camera_inliers;
  std::vector<double> camera_spread;
  std::vector<double> camera_usable;
  std::vector<double> camera_known;
  std::vector<int64_t> camera_looks;
  // Mean angle, in degrees, between the hop the ground solved and the bearing
  // off-ground features voted for.
  std::vector<double> camera_bearing;
  int64_t crossings = 0;
  double crossing_along_m = 0.0;
  double crossing_travel_m = 0.0;
  double crossing_scale = 0.0;
  double crossing_offset_m = 0.0;
  // Each camera's hop projected on the fused hop, over the fused path length.
  std::vector<double> camera_scale;
  // The same, but per ten metres of path rather than per solve: the fused
  // distance and what each camera contributed along it. The running totals
  // cannot say whether the split is there from the first metre or grows over
  // the drive, and those are different faults. Binned by distance because
  // `report()` copies this whole struct under the estimator's lock, so a
  // per-solve trace would grow without bound behind that lock.
  std::vector<Eigen::Vector3d> travel_bins;
  std::vector<int64_t> camera_anchored;
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
  // Rotate a solved hop onto the bearing off-ground features agree with,
  // keeping the length the ground gave it.
  void steer_by_epipolar(
    const Camera & camera, Solved & solved, const Points2 & previous_pixels,
    const Points2 & current_pixels);

  // The acceleration base_link would have felt, given what the IMU felt.
  ImuSample shift_imu_to_base(const ImuSample & measured);

  std::optional<Solved> solve_camera(
    Camera & camera, std::optional<double> yaw_delta, std::optional<double> yaw_guess);
  void remember_solve_pixels(Camera & camera);
  void learn_mounting_pitch(Camera & camera, double lean);
  std::optional<Eigen::Matrix3d> body_tilt() const;
  Eigen::Vector2d imu_world_velocity(const Eigen::Vector2d & velocity) const;
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
  // One ground map for every camera. See GroundAnchorMap's note on sources.
  std::unique_ptr<GroundAnchorMap> anchors_;
  std::vector<double> camera_travel_;
  // Ground scale correction learned against inertial propagation.
  double imu_scale_ = 1.0;
  std::optional<Eigen::Vector2d> expected_hop_;
  std::vector<int64_t> camera_solves_;
  std::vector<double> camera_inliers_;
  std::vector<double> camera_spread_;
  std::vector<double> camera_usable_;
  std::vector<double> camera_known_;
  std::vector<int64_t> camera_looks_;
  std::vector<double> camera_bearing_;
  std::vector<double> camera_projected_;
  std::vector<Eigen::Vector2d> last_camera_hops_;
  std::vector<uint8_t> last_from_map_;
  Eigen::Vector2d last_fused_hop_ = Eigen::Vector2d::Zero();
  bool last_hops_valid_ = false;
  double fused_path_ = 0.0;
  std::vector<int64_t> camera_bearings_;
  std::vector<int64_t> camera_anchored_;

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
    // What the part reported, and what to propagate on if impulses are being
    // screened -- the same reading, or the last believable one when this is not.
    // The simulator reports suspension contacts as hundreds of m/s2, and a
    // single 17 ms step of one leaves the velocity wrong for many frames.
    Eigen::Vector3d specific_force;
    Eigen::Vector3d held_force;
    Eigen::Vector3d angular_velocity;
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
  Eigen::Vector3d imu_rate_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d imu_angular_ = Eigen::Vector3d::Zero();
  std::optional<double> imu_rate_stamp_;
  std::optional<double> imu_yaw_datum_;

  std::unique_ptr<AttitudeFilter> attitude_;
  HeadingBiasFilter heading_;
  std::vector<std::pair<double, double>> heading_observations_;
  PlanarInertialPropagator inertial_;
  PlanarVelocityFilter velocity_filter_;
  std::unique_ptr<PlanarDisplacementFilter> displacement_filter_;
  std::unique_ptr<PlanarMsckfFilter> msckf_filter_;
  std::unique_ptr<SpatialMsckfFilter> spatial_filter_;
  // World frame velocity from whichever filter this run is actually running.
  // Only one of the four is fed vision; the other three are propagated on the
  // accelerometer alone and drift without bound, so asking the wrong one is
  // asking an open integrator.
  std::optional<Eigen::Vector2d> fused_world_velocity() const;
  double softness_for(const Camera & camera, double configured) const;
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
