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

#include <array>
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
  // moved, as a factor to divide out. Not an extrinsic, and not free: it is
  // derived from `ground_plane_offset_m`, which both cameras share.
  double range_scale = 1.0;
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
  // m/s^2 of the vehicle's own horizontal acceleration admitted before the
  // accelerometer stops being trusted about which way is down. The false
  // tilt it lets through is atan(a/g), so 0.17 is one degree.
  double attitude_horizontal_tolerance = 0.0;
  // The attitude trim's integral term, and the bound on what it may claim the
  // gyro's bias to be, in radians per second. See attitude.hpp: without it the
  // loop settles at bias*tau, and that offset is the estimator's common mode.
  double attitude_bias_tau_sec = 0.0;
  double attitude_bias_limit_rps = 0.001;
  double attitude_bias_gate_rad = 0.0;

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
  bool anchor_evict_by_age = false;
  bool anchor_evict_for_new = false;
  // Evict the least trusted rather than the longest unseen.
  bool anchor_evict_by_weight = false;
  // Solves an anchor must go unseen before it may be evicted.
  int anchor_evict_unseen_solves = 1;
  // Rank eviction by history times the axis the anchor measures.
  bool anchor_evict_by_information = false;
  // Cap on new anchors per update; 0 leaves it unbounded.
  int anchor_admit_per_update = 0;
  // Sightings before the solve will register against an anchor.
  int anchored_min_observations = 0;
  // Forget anchors past this bearing off the heading, in degrees.
  double anchor_forget_beyond_bearing_deg = 0.0;
  // Range past which an astern anchor may be forgotten; 0 uses the
  // solve band, which is the range at which it stops being reachable.
  double anchor_forget_beyond_range_m = 0.0;
  // Ground cell size in metres and how many anchors one cell may hold.
  double anchor_density_cell_m = 0.0;
  int anchor_density_quota = 0;
  // Admission quota on bearing-by-range cells in the vehicle's frame.
  double anchor_polar_sector_deg = 0.0;
  double anchor_polar_ring_m = 0.0;
  int anchor_polar_rings = 0;
  int anchor_polar_quota = 0;
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


  // How much an anchor's weight decays per metre driven since it was founded.
  // An anchor is a world position and the pose that wrote it has drifted since;
  // this is that drift priced in, instead of the hard age cutoff pretending it
  // is worth the same until the moment it is thrown away.
  double anchor_drift_variance_per_m = 0.0;

  // Everything the cameras see, in one filter, with one covariance.
  //
  // The ground and the structure above it stop being different code and become
  // different priors on a range. See landmark.hpp: this is the switch that
  // hands the hop to that filter instead of to the two solvers.
  // Whether the anchor alignment estimates the heading itself when a filter
  // exists that could consume one. It only ever did because such a filter
  // implies a gyro bias state, and a bias is only observable against an
  // independent heading. On a rig whose gyro is already exact that trade is
  // all cost: turning it on under the displacement model takes mean ATE from
  // 0.1703 to 0.8056.
  bool align_solves_yaw = true;
  // Largest sweep across the frame, in pixels over the whole solve, that a
  // ground feature may have and still be used. Zero is no limit. The band this
  // sits beside is in metres, and metres do not say whether the flow could
  // follow it.
  double solve_max_pixel_flow = 0.0;
  // Smallest sweep a ground feature may have, as a fraction of the frame's
  // median. Zero is no floor. This is the gate for a flow that never found its
  // feature and handed back the point it started from -- which the
  // forward-backward check passes, because zero out is zero back.
  double solve_min_pixel_flow = 0.0;

  // Project each frame with the tilt it was captured under. False replays the
  // original behaviour, where both frames took the current one.
  bool tilt_at_capture = false;
  // Where along base_link's x the body is taken to pitch about. Zero is the
  // rear axle at road level, which is what the projection assumed.
  // How much of the two-frame dh/h to fold back into the camera height. Zero
  // leaves the height at whatever ground_range_scale was calibrated to.
  // Where the ground the cameras see actually is, relative to the datum the
  // extrinsics are measured from. Measured, not fitted: triangulating the
  // tracked features from the truth poses and fitting a plane puts the
  // rendered road 2.4-8.1 mm above z=0, the same figure for both mounts.
  //
  // It replaces the free pair `ground_range_scale_front/rear`. The physics has
  // one degree of freedom here, not two: a road `d` above the datum leaves each
  // camera at `h - d`, so `range_scale = h / (h - d)` -- which makes the FRONT
  // scale the larger one, because it is the lower mount. The fitted pair had
  // them the other way round, which is how you can tell it was absorbing
  // something else.
  // Anchor sharing, restored 2026-08-27 from 8da9f5f. The algorithm never
  // left `anchors.hpp` -- only the wiring did.
  // Keep, per sighting, which pose it was taken from -- the precondition for
  // ever pushing a corrected trajectory back into the map.
  bool remember_sighting_poses = false;
  // Nodes in the pose graph's window, how hard a revisit pulls against the
  // odometry chain, and how many Gauss-Seidel sweeps to spend. Zero window
  // leaves the trajectory alone.
  // Weight the cameras by the inverse of their common-mode sensitivity.
  //
  // Both cameras sit on one body, so a height or pitch error is the same error
  // for both. Their point sets turn it into a translation pointing opposite
  // ways -- front 0.6 degrees, rear 179.8 -- so averaging cancels it. But only
  // if the magnitudes match, and they do not: the rear's ground recedes and it
  // keeps points further out, where the pitch gain (R^2+h^2)/h is larger. The
  // rear reads 1.5x the front, which leaves 16-31% of the common-mode pitch
  // error standing after a 50:50 average.
  //
  // Weighting each camera by 1/|gain| cancels it exactly, and derives the
  // weight from the geometry each frame rather than fitting one.
  //
  // 0 off, 1 from the pitch gain, 2 from the height gain.
  int fusion_gain_mode = 0;
  // Trim each camera's point set back to the shortest reach any camera had, so
  // that a 50:50 average cancels the common-mode error exactly.
  //
  // The alternative -- weighting by 1/|gain| -- cancels it too, and costs 28%:
  // the equal weighting is already load-bearing for the map path's front/rear
  // correction split. This leaves the weights alone and equalises the geometry
  // instead, at the price of the rear's furthest points.
  bool equalise_reach = false;
  // How far to move the fused hop's length toward the distance the road's own
  // photometric fit measured, 0 for not at all and 1 for all the way.
  //
  // Only the length. The direction and the drift binding come from the map
  // path's front/rear correction, which is what stops the walk; the road fit
  // carries no map and cannot replace that. What it does carry is scale, at
  // 0.011-0.077% per hop against the corner path's 1.06-1.62% -- so this
  // splits the two along the axis where each is better.
  double photometric_step_gain = 0.0;
  // Frames whose road did not land on itself this well are not used. The
  // alignment reads 0.93 to 0.99 when it has the surface; the tail events that
  // wreck a hop at full gain are the frames where it does not.
  double photometric_min_score = 0.0;
  // Void the solve interval when the tiles disagree by more than this fraction
  // of the step. A kerb or any other thing that is not the plane contaminates
  // the tile it sits in and not the others; the alignment score cannot see it,
  // because the intruder scores higher than the road it displaces. 0 is off.
  double photometric_max_spread = 0.0;
  // Blend the road's length into each camera's own two-frame solve, before the
  // two are fused, instead of into the fused hop.
  //
  // The fused hop is not a displacement. Where the map answers it is a
  // *correction* of that camera's pose against the map -- its length against
  // truth reads +0.05% on the 8 m/s straight and +0.27% on a curve, so it is
  // not the distance travelled -- and the disagreement it carries is what bounds
  // the drift. Overwriting that length throws the binding away: at full gain the
  // walk falls to 0.64 while ATE rises to 0.0762 and the worst drive doubles.
  // The two-frame solve *is* a displacement, so a measured distance belongs
  // there and the map correction is left alone.
  bool photometric_on_pairs = false;
  // Give the map alignment the road's distance as a prior and a gate centre,
  // instead of overwriting what it returns.
  //
  // The map path is the *less* accurate half per hop -- 0.52-0.65 against the
  // two-frame path's 0.31-0.50 -- and it is also the only thing that
  // decorrelates: lag-one autocorrelation of the hop error runs 0.86-0.92 where
  // the map is silent and 0.49-0.52 where it answers, and the two paths' biases
  // carry opposite signs. So it cannot simply be replaced by the road's length;
  // what it can be given is a better place to start and a tighter gate, leaving
  // the correction itself the map's to make.
  bool photometric_align_prior = false;
  // Use the road's length only on the frames where no camera answered from the
  // anchor map.
  //
  // Where the map answers, the hop is not a displacement at all -- it is a
  // correction of this camera's pose against the map, and that correction is
  // what bounds the drift. Rescaling it makes the map pull back next frame and
  // the disagreement carries: the lag-1 autocorrelation of the hop error goes
  // from 0.66 to 0.90 while the bias falls, and the trajectory error does not
  // move. Where the map is silent the hop really is a displacement, and a
  // length is exactly what it needs.
  bool photometric_when_mapless = false;
  int pose_graph_window = 0;
  double pose_graph_loop_weight = 1.0;
  int pose_graph_sweeps = 8;
  bool rebuild_measure_only = false;
  double anchor_link_radius_m = 0.0;
  bool anchor_link_adopter_writes = false;
  int anchor_link_rebind_grace = 1;
  bool anchor_link_measure_only = false;
  double ground_plane_offset_m = 0.0;
  double pair_scale_gain = 0.0;
  double pitch_centre_x_m = 0.0;
  // Let the body tilt move the camera's height over the road. True is what the
  // projection has always done; false holds the height at its nominal value and
  // leaves the tilt to set only the plane's direction.
  bool ground_height_from_tilt = true;
  double attitude_slope_tau_sec = 0.0;
  double vision_scale = 1.0;
  double map_solve_weight = 1.0;
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


  bool use_inertial_prediction = true;
  bool inertial_use_acceleration = true;
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
  // How far the road says this camera moved on this frame, in metres. Not
  // finite where the tracker did not measure it. It is a displacement, so the
  // estimator takes it as one rather than as points.
  double photometric_step = std::numeric_limits<double>::quiet_NaN();
  // How well the region landed on itself, 0 where unknown.
  double photometric_score = 0.0;
  // Spread of the across-track tiles' answers, relative to the step.
  double photometric_spread = 0.0;
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
  // Roll and pitch as the attitude filter holds them. The planar solve owns
  // yaw, so it is in `pose`; these two are estimated and were, until now, only
  // ever used internally -- the projection they steer is the reason this stack
  // works at all, and a consumer reading the orientation was being told the
  // vehicle is always level.
  double roll = 0.0;
  double pitch = 0.0;
  bool tilt_valid = false;
  // What the estimator claims to know, so that a consumer can weigh it.
  //
  // The pose is dead reckoned, so its covariance grows without bound: this is
  // the accumulated per-hop covariance, in the world frame, plus the heading's
  // own accumulation. That accumulation is only honest because the hops turned
  // out to be very nearly independent -- the benchmark's `walk` reads 0.97
  // against 1.0 for a true random walk -- and it would understate the growth
  // on a stack whose hops were correlated.
  Eigen::Matrix3d pose_covariance = Eigen::Matrix3d::Zero();
  // Body frame, per sample: vx, vy from the filter and the yaw rate from the
  // instrument.
  Eigen::Matrix3d twist_covariance = Eigen::Matrix3d::Zero();
  bool covariance_valid = false;
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
  // The information ellipse of the ground points this camera solved from.
  //
  // The alignment reduces every point to one scalar weight and takes a
  // weighted mean of their votes, which makes the solve isotropic by
  // construction. The measurement is not. A ground point is located by a
  // bearing, and a bearing error of one pixel moves it (R^2+h^2)/h along the
  // line of sight but only R across it -- a ratio of R/h + h/R, which is 2.7
  // at 2 m and 6.7 at 5.8. So each vote is an ellipse, and where the points
  // sit decides whether those ellipses cancel or stack.
  //
  // Recorded, not applied. This is the instrument that says whether the
  // solve's weak direction is where its error actually goes.
  std::vector<double> camera_condition;
  // Bearing of the weakest direction in the body frame, radians.
  std::vector<double> camera_weak_bearing;
  // What a common-mode error costs this camera: translation per metre of
  // height error, and per radian of pitch error, in the body frame. Unlike the
  // condition number these do not cancel over the fan of bearings -- every
  // point moves the same way -- which is why they are the ones worth watching.
  std::vector<Eigen::Vector2d> camera_height_gain;
  std::vector<Eigen::Vector2d> camera_pitch_gain;
  // Mean range of the points each camera solved from, and how many.
  std::vector<double> camera_mean_range;
  std::vector<double> camera_point_count;
  // Distance the road measured over this hop's interval, and the length
  // the cameras' fused hop carried. Not the same quantity: the fused hop
  // is a correction against the map, so laying the two side by side is
  // what says whether a length can be blended into it at all.
  // Per camera, the radial residual split into a height term and a pitch
  // term. Diagnostic; nothing acts on them.
  std::vector<double> radial_height;
  std::vector<double> radial_pitch;
  double photometric_distance = 0.0;
  double fused_length = 0.0;
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
  // Live anchors the next solve could reach, and the spread of all of them
  // along and across the heading.
  int anchors_within = 0;
  // Mean offset along the heading, and the spread about it.
  double anchors_along = 0.0;
  double anchors_across = 0.0;
  double link_gap_m = 0.0;
  double link_gap_per_m = 0.0;
  double link_range_m = 0.0;
  // How each camera's ground projection leans, averaged over the solves it
  // answered. This is the mounting pitch and nothing else -- AnchorAlignment
  // carries why the height cannot appear here. Per camera because they are
  // mounted at different heights and opposite pitches, and a mean across them
  // would cancel the very thing being measured.
  std::vector<double> radial_linear;
  // dh/h read off the two-frame pairs -- see the comment on Camera::pair_n.
  int64_t remembered_sightings = 0;
  int64_t pose_history = 0;
  double rebuild_ms = 0.0;
  double rebuild_shift_m = 0.0;
  double sighting_span = 0.0;
  int64_t pose_graph_loops = 0;
  double pose_graph_shift_m = 0.0;
  // What the road's length did to the fused hop, 1 for nothing.
  double photometric_ratio = 1.0;
  int64_t photometric_uses = 0;
  int64_t photometric_chances = 0;
  int64_t photometric_mapless = 0;
  std::vector<double> pair_radial;
  std::vector<int64_t> pair_radial_samples;
  std::vector<int64_t> radial_samples;

  // What the MSCKF learned and what it thought of the last measurement. The
  // gyro bias is the state the older filters had no place for, so its value is
  // the whole claim; the NIS says whether the covariance is honest.
  double gyro_bias = 0.0;
  double last_nis = 0.0;
  double nis_total = 0.0;
  int64_t nis_samples = 0;
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

  // Hand the estimator the attitude instead of letting it estimate one.
  //
  // Only the offline harness calls this, and only to measure a ceiling: the
  // pitch reaches the ground projection through the camera's lever arm, which
  // is 3.694 m on the front mount against 0.89 m of height, so a tenth of a
  // degree of pitch error is 0.4% of that camera's scale. Substituting the
  // truth says how much of the error is that and how much is everything else.
  void override_tilt(double roll, double pitch);
  const Pose2 & pose() const {return pose_;}
  size_t camera_count() const {return cameras_.size();}

private:
  struct Camera;
  struct Frame;
  struct Solved;

  void try_process_pairs();
  bool ready_to_solve() const;
  void process_pair();

  // The acceleration base_link would have felt, given what the IMU felt.
  ImuSample shift_imu_to_base(const ImuSample & measured);

  std::optional<Solved> solve_camera(
    Camera & camera, std::optional<double> yaw_delta, std::optional<double> yaw_guess);
  void remember_solve_pixels(Camera & camera);
  std::optional<Eigen::Matrix3d> body_tilt() const;

  Eigen::Vector2d imu_world_velocity(const Eigen::Vector2d & velocity) const;
  std::optional<double> imu_yaw_at(double stamp) const;
  bool imu_still_arriving(double stamp) const;
  void update_anchors(const std::vector<std::optional<Solved>> & solved);
  void solve_pose_graph();

public:
  // The revisit constraints as they stand, each paired with the odometry the
  // trajectory already holds between the same two poses, and the times of
  // both. Lets an offline check ask the only question that matters: does the
  // constraint know the separation better than dead reckoning does?
  struct RevisitAudit
  {
    double time_from, time_to;
    double edge_dx, edge_dy;
    double odometry_dx, odometry_dy;
    double weight;
  };
  std::vector<RevisitAudit> revisit_audit() const;
  // Live anchors as range, bearing off the heading, weight, solves unseen.
  void anchor_polar(std::vector<std::array<double, 4>> & out) const;

private:
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
  std::vector<double> last_condition_;
  std::vector<double> last_weak_bearing_;
  std::vector<Eigen::Vector2d> last_height_gain_;
  std::vector<Eigen::Vector2d> last_pitch_gain_;
  std::vector<double> last_mean_range_;
  std::vector<double> last_point_count_;
  std::vector<double> last_radial_height_;
  std::vector<double> last_radial_pitch_;
  double last_photometric_distance_ = 0.0;
  double last_fused_length_ = 0.0;
  // Shortest mean reach any camera had last frame, which is what
  // `equalise_reach` trims the others back to.
  double reach_target_ = std::numeric_limits<double>::quiet_NaN();
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
  // Every pose an anchor sighting was taken from, in order.
  std::vector<std::array<double, 3>> pose_history_;
  std::vector<double> pose_time_;
  Eigen::Vector3d filtered_twist_ = Eigen::Vector3d::Zero();
  // Accumulated dead-reckoning covariance for the absolute pose.
  Eigen::Matrix3d pose_covariance_ = Eigen::Matrix3d::Zero();
  bool map_ready_ = false;

  std::deque<std::pair<double, double>> imu_yaw_samples_;
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
  std::optional<Eigen::Matrix3d> tilt_override_;
  HeadingBiasFilter heading_;
  std::vector<std::pair<double, double>> heading_observations_;
  PlanarInertialPropagator inertial_;
  PlanarVelocityFilter velocity_filter_;
  std::unique_ptr<PlanarDisplacementFilter> displacement_filter_;
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
