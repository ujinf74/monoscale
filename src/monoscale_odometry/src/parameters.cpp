#include "monoscale_odometry/parameters.hpp"

#include <map>

namespace monoscale_ros
{

namespace
{

// Where a camera sits on the vehicle, for a rig that publishes no transform
// tree at all. Any name not listed here starts at identity, which a real mount
// will overwrite.
const std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> kMounts = {
  {"front",
    {{0.0, -0.5, 0.8660254, -1.0, 0.0, 0.0, 0.0, -0.8660254, -0.5}, {3.5, 0.0, 0.89}}},
  {"rear",
    {{0.0, 0.5, -0.8660254, 1.0, 0.0, 0.0, 0.0, -0.8660254, -0.5}, {-0.82, 0.0, 1.26}}},
};

Eigen::Matrix3d matrix_of(const std::vector<double> & flat)
{
  Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();
  if (flat.size() != 9) {
    return matrix;
  }
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      matrix(r, c) = flat[static_cast<size_t>(3 * r + c)];
    }
  }
  return matrix;
}

Eigen::Vector3d vector_of(const std::vector<double> & flat)
{
  Eigen::Vector3d vector = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < std::min<size_t>(3, flat.size()); ++i) {
    vector(static_cast<Eigen::Index>(i)) = flat[i];
  }
  return vector;
}

Eigen::VectorXd coefficients_of(const std::vector<double> & flat)
{
  Eigen::VectorXd out(static_cast<Eigen::Index>(flat.size()));
  for (size_t i = 0; i < flat.size(); ++i) {
    out(static_cast<Eigen::Index>(i)) = flat[i];
  }
  return out;
}

void validate_rotation(const std::string & name, const Eigen::Matrix3d & rotation)
{
  const double error =
    (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm();
  if (error > 1e-3 || rotation.determinant() < 0.99) {
    throw std::runtime_error(name + " camera rotation is not a proper rotation matrix");
  }
}

}  // namespace

Configuration declare_and_read(rclcpp::Node & node)
{
  Configuration configuration;
  monoscale::EstimatorSettings & settings = configuration.estimator;
  Topics & topics = configuration.topics;

  const auto declare_double = [&node](const std::string & name, double value) {
      return node.declare_parameter<double>(name, value);
    };
  // ROS integers are 64 bit; every count here fits in an int and the cast keeps
  // the call sites from narrowing silently.
  const auto declare_int = [&node](const std::string & name, int value) {
      return static_cast<int>(node.declare_parameter<int>(name, value));
    };
  const auto declare_bool = [&node](const std::string & name, bool value) {
      return node.declare_parameter<bool>(name, value);
    };
  const auto declare_string = [&node](const std::string & name, const std::string & value) {
      return node.declare_parameter<std::string>(name, value);
    };

  topics.imu = declare_string("imu_topic", topics.imu);
  settings.use_imu_yaw = declare_bool("use_imu_yaw", true);
  // The CARLA bridge publishes the IMU against simulation time.
  settings.imu_max_age_sec = declare_double("imu_max_age_sec", 0.02);
  // Largest gap between IMU samples still worth interpolating across.
  settings.imu_max_gap_sec = declare_double("imu_max_gap_sec", 0.12);
  // CARLA reports a valid pinhole CameraInfo derived from the blueprint FOV.
  topics.use_camera_info = declare_bool("use_camera_info", true);
  topics.odometry = declare_string("odometry_topic", topics.odometry);
  topics.pose = declare_string("pose_topic", topics.pose);
  // What the solve saw on the road, labelled and already in the map frame.
  // Each point carries the camera position it was seen from, so the consumer
  // can carve free space along the ray without knowing where the cameras are
  // mounted or which pose the hop was solved against.
  topics.ground_points = declare_string("ground_points_topic", topics.ground_points);
  topics.map_frame = declare_string("map_frame", topics.map_frame);
  topics.odom_frame = declare_string("odom_frame", topics.odom_frame);
  topics.base_frame = declare_string("base_frame", topics.base_frame);
  settings.sync_tolerance_sec = declare_double("sync_tolerance_sec", 0.02);
  // How many frames may wait for a partner before the oldest is thrown away.
  // A shallow queue is right online, because a stale frame is worse than no
  // frame; offline it makes a measurement irreproducible. So the depth is
  // settable and every eviction is counted.
  settings.pair_queue_depth = declare_int("pair_queue_depth", 12);
  topics.publish_tf = declare_bool("publish_tf", true);
  // Identity, and published only so the tree is whole. There is no absolute
  // localisation in this stack -- the anchor map binds drift, it does not
  // remove it -- so map and odom coincide. The moment something can actually
  // localise, it owns this transform and this node must stop publishing it.
  topics.publish_map_to_odom = declare_bool("publish_map_to_odom", true);
  // Where a camera is bolted. The transform tree is the one place that already
  // has to be right for anything else to work, so it is asked first and the
  // parameters below are only the answer for a rig that publishes no tree.
  topics.extrinsics_from_tf = declare_bool("extrinsics_from_tf", true);
  topics.tf_lookup_timeout_sec = declare_double("tf_lookup_timeout_sec", 5.0);

  // Which cameras this node fuses. One is enough -- the filter then carries
  // `single_camera_variance` where two would have carried their disagreement --
  // and there is no upper bound: the solve already fuses a list, and the frames
  // are aligned by their stamps rather than by being a pair.
  const auto names = node.declare_parameter<std::vector<std::string>>(
    "camera_names", {"front", "rear"});

  for (const auto & name : names) {
    monoscale::CameraSettings camera;
    camera.name = name;
    const auto found = kMounts.find(name);
    const std::vector<double> default_rotation = found != kMounts.end()
      ? found->second.first
      : std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const std::vector<double> default_translation = found != kMounts.end()
      ? found->second.second
      : std::vector<double>{0.0, 0.0, 1.0};

    node.declare_parameter<double>(name + ".horizontal_fov_deg", 90.0);
    const auto k = node.declare_parameter<std::vector<double>>(
      name + ".k", {900.0, 0.0, 960.0, 0.0, 900.0, 540.0, 0.0, 0.0, 1.0});
    const auto model = node.declare_parameter<std::string>(
      name + ".distortion_model", "plumb_bob");
    const auto d = node.declare_parameter<std::vector<double>>(
      name + ".d", std::vector<double>(5, 0.0));
    camera.ground_min_distance_m =
      node.declare_parameter<double>(name + ".ground_min_distance_m", 0.0);
    camera.calibration_width = node.declare_parameter<int>(name + ".calibration_width", 1920);
    camera.calibration_height =
      node.declare_parameter<int>(name + ".calibration_height", 1080);
    const auto rotation = node.declare_parameter<std::vector<double>>(
      name + ".rotation_base_from_camera", default_rotation);
    const auto translation = node.declare_parameter<std::vector<double>>(
      name + ".translation_base_from_camera", default_translation);
    topics.image_topics.push_back(
      node.declare_parameter<std::string>(
        name + "_image_topic", "/sensing/camera/" + name + "/image_raw"));
    topics.camera_info_topics.push_back(
      node.declare_parameter<std::string>(name + "_camera_info_topic", "/camera_info"));
    // How much further than the truth this camera measures the ground to have
    // moved, as a factor to divide out. Not an extrinsic: the camera is where
    // the kit says it is, and this is measured downstream of that.
    camera.range_scale = node.declare_parameter<double>("ground_range_scale_" + name, 1.0);
    // Whether this camera's mounting pitch is learned from the lean its
    // alignment residual leaves. Off unless the observable is there for it --
    // CameraSettings carries which cameras that was measured to be.
    camera.fusion_weight = node.declare_parameter<double>(name + ".fusion_weight", 0.0);

    camera.k = matrix_of(k);
    camera.distortion = coefficients_of(d);
    camera.lens = monoscale::lens_from_name(model);
    camera.rotation_base_from_camera = matrix_of(rotation);
    camera.translation_base_from_camera = vector_of(translation);
    validate_rotation(name, camera.rotation_base_from_camera);
    settings.cameras.push_back(camera);
  }

  // Declared for compatibility with the Python parameter files. The optical
  // flow front end they belong to is not part of this path: the C++ tracker is
  // the same work, and two implementations of the one stage whose cost matters
  // is one too many.
  declare_int("processing_width", 640);
  declare_int("max_features_per_camera", 600);
  declare_double("quality_level", 0.01);
  declare_double("min_feature_distance_px", 8.0);
  declare_double("lk_error_threshold", 30.0);
  declare_int("lk_window_px", 21);
  declare_int("lk_pyramid_levels", 3);
  declare_double("lk_forward_backward_threshold_px", 1.0);
  declare_double("feature_refill_ratio", 0.7);
  declare_int("detection_grid_columns", 0);
  declare_int("detection_grid_rows", 0);
  declare_int("detection_oversample", 3);
  declare_bool("forward_backward_on_hops", true);
  declare_int("backward_check_stride", 1);
  declare_int("camera_workers", 2);
  declare_bool("allow_fov_fallback", false);
  declare_bool("anchor_reassociation", false);
  declare_double("anchor_reassociation_radius_m", 0.5);
  declare_int("anchor_reassociation_max_hamming", 40);
  declare_int("anchor_reassociation_min_observations", 3);

  settings.ground_max_distance_m = declare_double("ground_max_distance_m", 25.0);
  // `min_ground_flow_px` is not declared. The adaptive ground band it keyed
  // was measured against the original recordings and lost to the fixed one, so
  // it shipped switched off and stayed that way. Parameter files that still
  // carry the key load unchanged; the value is simply not read.
  settings.ground_ransac_threshold_m = declare_double("ground_ransac_threshold_m", 0.12);
  settings.ground_align_softness_m = declare_double("ground_align_softness_m", 0.0);
  settings.solve_max_distance_m = declare_double("solve_max_distance_m", 0.0);
  settings.anchor_max_range_m = declare_double("anchor_max_range_m", 0.0);
  settings.imu_scale_gain = declare_double("imu_scale_gain", 0.0);
  settings.imu_scale_min_hop_m = declare_double("imu_scale_min_hop_m", 0.05);
  settings.inertial_gate_m = declare_double("inertial_gate_m", 0.0);
  settings.range_scale_gain = declare_double("range_scale_gain", 0.0);
  settings.align_seed_from_last_hop = declare_bool("align_seed_from_last_hop", false);
  settings.align_restarts = declare_int("align_restarts", 1);
  settings.align_ambiguity_ratio = declare_double("align_ambiguity_ratio", 0.0);
  settings.ground_min_inliers = declare_int("ground_min_inliers", 24);
  settings.max_scale_error = declare_double("max_scale_error", 0.08);
  settings.max_translation_per_frame_m = declare_double("max_translation_per_frame_m", 1.0);
  settings.max_yaw_per_frame_rad = declare_double("max_yaw_per_frame_rad", 0.35);
  settings.max_arrears_frames = declare_double("max_arrears_frames", 8.0);

  // Roll and pitch from the IMU. See AttitudeFilter for what the two numbers
  // cost when set wrongly: a one second constant left pitch out by 4.3 degrees
  // against a signal of 0.04.
  settings.attitude_from_imu = declare_bool("attitude_from_imu", true);
  settings.attitude_tau_sec = declare_double("attitude_tau_sec", 60.0);
  settings.attitude_gravity_tolerance = declare_double("attitude_gravity_tolerance", 0.3);
  settings.attitude_horizontal_tolerance =
    declare_double("attitude_horizontal_tolerance", 0.0);
  settings.attitude_bias_tau_sec = declare_double("attitude_bias_tau_sec", 0.0);
  settings.attitude_bias_limit_rps =
    declare_double("attitude_bias_limit_rps", 0.001);
  settings.attitude_bias_gate_rad = declare_double("attitude_bias_gate_rad", 0.0);

  // What the heading source is expected to do wrong, as the part's own numbers
  // rather than as tuning. Setting the first to 0 believes the reported heading
  // outright, which is what every measurement before this assumed -- and that
  // assumption is stronger than it looks, because the simulator provides an
  // absolute heading that never drifts and a MEMS part on a vehicle does not.
  // Rerecording the same drive with a 0.3 deg/s bias took position RMSE from
  // 0.038 m to 0.169 and drift from 0.24% to 1.17%.
  settings.gyro_bias_sigma_rad_s = declare_double("gyro_bias_sigma_rad_s", 0.0);
  settings.gyro_bias_walk_sigma_rad_s = declare_double("gyro_bias_walk_sigma_rad_s", 1.0e-5);
  settings.gyro_noise_sigma_rad_s = declare_double("gyro_noise_sigma_rad_s", 1.0e-3);

  settings.coast_on_reject = declare_bool("coast_on_reject", true);
  settings.twist_lowpass_tau = declare_double("twist_lowpass_tau", 0.12);

  // Triangulation settings, kept so the parameter files load unchanged. The
  // keyframe path they serve needs images; on the track path obstacle height
  // comes from `obstacle_slip_baseline_m` instead.
  declare_double("triangulation_min_parallax_deg", 1.0);
  declare_double("triangulation_reprojection_error_px", 2.5);
  declare_double("triangulation_min_distance_m", 0.5);
  declare_double("triangulation_max_distance_m", 30.0);

  settings.obstacle_min_height_m = declare_double("obstacle_min_height_m", 0.2);
  settings.obstacle_max_height_m = declare_double("obstacle_max_height_m", 2.5);
  // Travel a feature must accumulate before its ground projection's slip is
  // read as a height. On by default here, because it is the only obstacle
  // method this path can run: the keyframe triangulation the Python defaulted
  // to needs images the track front end does not carry, and reaching for it
  // there is a crash rather than a fallback. The value itself is inherited
  // rather than measured -- see EstimatorSettings.
  settings.obstacle_slip_baseline_m = declare_double("obstacle_slip_baseline_m", 0.35);
  // A feature at the camera's own height projects to infinity, so the band
  // stops short of it.
  settings.obstacle_height_margin = declare_double("obstacle_height_margin", 0.7);
  settings.obstacle_slip_patience = declare_int("obstacle_slip_patience", 30);
  settings.mapping_baseline_m = declare_double("mapping_baseline_m", 0.35);
  // The occupancy grid is the deliverable, not part of the pose loop, and it
  // cost more per frame than the odometry it rode along with. A parking grid
  // does not improve by being rebuilt at the camera rate.
  settings.mapping_min_period_sec = declare_double("mapping_min_period_sec", 0.04);

  settings.max_ground_anchors = declare_int("max_ground_anchors", 4000);
  settings.anchor_max_age_frames = declare_int("anchor_max_age_frames", 120);
  // How fast an anchor follows its latest sighting. Sightings are written in
  // the world frame the estimate just settled on, so an anchor that follows too
  // eagerly absorbs the estimate's own error and stops being able to correct
  // it. Unused while each sighting is weighted by its own precision.
  settings.anchor_update_gain = declare_double("anchor_update_gain", 0.0);
  // Where an anchor's weight stops growing. Reached in a third of a second at
  // 60 Hz, it stops telling a long-lived landmark apart from a fresh one.
  settings.anchor_max_observations = declare_int("anchor_max_observations", 20);
  settings.anchor_select_by_consistency = declare_bool("anchor_select_by_consistency", true);
  settings.anchor_initial_variance = declare_double("anchor_initial_variance", 0.04);
  settings.anchor_max_variance = declare_double("anchor_max_variance", 0.09);
  settings.anchor_trial_observations = declare_int("anchor_trial_observations", 4);
  settings.anchor_min_update_gain = declare_double("anchor_min_update_gain", 0.0);
  settings.anchor_evict_by_age = declare_bool("anchor_evict_by_age", false);
  settings.anchor_evict_for_new = declare_bool("anchor_evict_for_new", false);
  settings.anchor_evict_by_weight =
    declare_bool("anchor_evict_by_weight", false);
  settings.anchor_evict_unseen_solves =
    declare_int("anchor_evict_unseen_solves", 1);
  settings.anchor_evict_by_information =
    declare_bool("anchor_evict_by_information", false);
  settings.anchor_admit_per_update =
    declare_int("anchor_admit_per_update", 0);
  settings.anchored_min_observations =
    declare_int("anchored_min_observations", 0);
  settings.anchor_forget_beyond_bearing_deg =
    declare_double("anchor_forget_beyond_bearing_deg", 0.0);
  settings.anchor_forget_beyond_range_m =
    declare_double("anchor_forget_beyond_range_m", 0.0);
  settings.anchor_density_cell_m =
    declare_double("anchor_density_cell_m", 0.0);
  settings.anchor_density_quota =
    declare_int("anchor_density_quota", 0);
  settings.anchor_polar_sector_deg =
    declare_double("anchor_polar_sector_deg", 15.0);
  settings.anchor_polar_ring_m =
    declare_double("anchor_polar_ring_m", 2.0);
  settings.anchor_polar_rings =
    declare_int("anchor_polar_rings", 4);
  settings.anchor_polar_quota =
    declare_int("anchor_polar_quota", 0);
  settings.camera_split_lever = declare_double("camera_split_lever", 0.0);
  settings.road_point_weight = declare_double("road_point_weight", 1.0);
  settings.anchor_information_power = declare_double("anchor_information_power", 0.0);
  settings.anchor_weight_by_information = declare_bool("anchor_weight_by_information", false);
  settings.level_frame_origin = declare_bool("level_frame_origin", true);
  settings.range_weight_power = declare_double("range_weight_power", 0.0);
  settings.pixel_region_x0 = declare_double("pixel_region_x0", 0.0);
  settings.pixel_region_y0 = declare_double("pixel_region_y0", 0.0);
  settings.pixel_region_x1 = declare_double("pixel_region_x1", 0.0);
  settings.pixel_region_y1 = declare_double("pixel_region_y1", 0.0);
  settings.vision_only_pose = declare_bool("vision_only_pose", false);
  settings.ground_rotation_threshold_m = declare_double("ground_rotation_threshold_m", 0.0);
  settings.ground_pair_passes = declare_int("ground_pair_passes", 1);
  settings.ground_pair_softness_m = declare_double("ground_pair_softness_m", 0.0);
  settings.softness_from_residual = declare_double("softness_from_residual", 0.0);
  settings.fuse_camera_points = declare_bool("fuse_camera_points", false);
  settings.anchor_drift_variance_per_m =
    declare_double("anchor_drift_variance_per_m", 0.0);
  settings.align_solves_yaw = declare_bool("align_solves_yaw", true);
  settings.solve_max_pixel_flow = declare_double("solve_max_pixel_flow", 0.0);
  settings.solve_min_pixel_flow = declare_double("solve_min_pixel_flow", 0.0);
  settings.tilt_at_capture = declare_bool("tilt_at_capture", false);
  settings.remember_sighting_poses =
    declare_bool("remember_sighting_poses", false);
  settings.rebuild_measure_only = declare_bool("rebuild_measure_only", false);
  settings.fusion_gain_mode = declare_int("fusion_gain_mode", 0);
  settings.equalise_reach = declare_bool("equalise_reach", false);
  settings.photometric_step_gain =
    declare_double("photometric_step_gain", 0.0);
  settings.photometric_min_score =
    declare_double("photometric_min_score", 0.0);
  settings.photometric_max_spread =
    declare_double("photometric_max_spread", 0.0);
  settings.photometric_on_pairs =
    declare_bool("photometric_on_pairs", false);
  settings.photometric_align_prior =
    declare_bool("photometric_align_prior", false);
  settings.photometric_when_mapless =
    declare_bool("photometric_when_mapless", false);
  settings.pose_graph_window = declare_int("pose_graph_window", 0);
  settings.pose_graph_loop_weight =
    declare_double("pose_graph_loop_weight", 1.0);
  settings.pose_graph_sweeps = declare_int("pose_graph_sweeps", 8);
  settings.anchor_link_radius_m = declare_double("anchor_link_radius_m", 0.0);
  settings.anchor_link_adopter_writes =
    declare_bool("anchor_link_adopter_writes", false);
  settings.anchor_link_rebind_grace = declare_int("anchor_link_rebind_grace", 1);
  settings.anchor_link_measure_only =
    declare_bool("anchor_link_measure_only", false);
  settings.ground_plane_offset_m =
    declare_double("ground_plane_offset_m", 0.0);
  settings.pair_scale_gain = declare_double("pair_scale_gain", 0.0);
  settings.pitch_centre_x_m = declare_double("pitch_centre_x_m", 0.0);
  settings.ground_height_from_tilt =
    declare_bool("ground_height_from_tilt", true);
  settings.attitude_slope_tau_sec = declare_double("attitude_slope_tau_sec", 0.0);
  settings.vision_scale = declare_double("vision_scale", 1.0);
  settings.map_solve_weight = declare_double("map_solve_weight", 1.0);
  settings.imu_translation_base_from_imu = vector_of(
    node.declare_parameter<std::vector<double>>(
      "imu_translation_base_from_imu", std::vector<double>{0.0, 0.0, 0.0}));
  settings.imu_angular_accel_tau_sec = declare_double("imu_angular_accel_tau_sec", 0.0);
  settings.anchor_seed_travel_m = declare_double("anchor_seed_travel_m", 0.0);

  settings.frame_decimation = std::max(declare_int("frame_decimation", 1), 1);
  // Trigger a solve on accumulated image motion instead of a frame count. A
  // faster camera shortens the baseline each solve gets, and the ground
  // displacement it has to measure falls with it.
  settings.adaptive_solve_interval = declare_bool("adaptive_solve_interval", true);
  settings.solve_trigger_disparity_px = declare_double("solve_trigger_disparity_px", 3.0);
  settings.solve_min_frames = declare_int("solve_min_frames", 2);
  settings.solve_max_frames = declare_int("solve_max_frames", 40);

  // Hold the pose until the anchor map can answer for itself. The first frames
  // have no map, so they fall back to the two-frame solve, and on a vehicle
  // that has not moved yet that solve reads ground texture noise as travel:
  // 38 cm of it inside the first 0.7 s, which then rides along as a constant
  // offset for the rest of the run.
  settings.require_map_before_translating =
    declare_bool("require_map_before_translating", true);
  settings.suppress_pose_until_map_ready =
    declare_bool("suppress_pose_until_map_ready", true);

  // A standing vehicle still produces a vision solve, because tracking a
  // texture is never exact and the residual reads as travel. Off: the gain it
  // once appeared to buy was an earlier version suppressing motion that was
  // real.
  // An accelerometer cannot tell rest from constant velocity: both read gravity
  // and nothing else. Asked on its own it called 144 of 495 solves stationary
  // on a run held at 8 m/s. Vision has to agree.
  // Borrowed from OpenVINS, which asks the images rather than the accelerometer
  // whether the platform is moving.
  // How much of the window has to be still. Relaxing this to 0.8 let the check
  // fire five or six times a run instead of one to three, and the extra
  // firings were wrong: ATE went from 0.046-0.055 to 0.067-0.070.

  settings.use_inertial_prediction = declare_bool("use_inertial_prediction", true);
  settings.inertial_use_acceleration = declare_bool("inertial_use_acceleration", true);
  declare_double("inertial_gate_absolute_m", 0.08);
  declare_double("inertial_gate_relative", 0.35);
  declare_double("inertial_gate_minimum_m", 0.05);
  settings.inertial_max_acceleration_mps2 =
    declare_double("inertial_max_acceleration_mps2", 12.0);
  settings.inertial_acceleration_median_window =
    declare_int("inertial_acceleration_median_window", 1);
  settings.inertial_integration = monoscale::integration_from_name(
    declare_string("inertial_integration_method", "rk4"));
  // How far before its own stamp an accelerometer sample describes. CARLA does
  // not measure specific force; it fits a quadratic to the sensor's last three
  // positions and reports the second derivative, which is valid one sensor
  // period before the stamp it is published with. A real instrument does
  // measure specific force, so this stays zero unless the recording is CARLA's.
  settings.imu_acceleration_offset_sec = declare_double("imu_acceleration_offset_sec", 0.0);

  // 'velocity' takes the vision translation divided by the interval as a
  // velocity measurement; 'displacement' fuses what the two instruments
  // actually measure -- acceleration and metres -- and carries an accelerometer
  // bias state. Measured on both recordings whose accelerometer is a signal,
  // displacement is 28-30% better.
  settings.fusion_model =
    monoscale::fusion_model_from_name(declare_string("fusion_model", "displacement"));
  // q^2*dt, so this is a spectral density in m/s^2/sqrt(Hz). 1.55 is the same
  // process noise per step that 12.0 gave under the old (q*dt)^2 form at 60 Hz,
  // and unlike 12.0 it keeps meaning the same thing when the IMU rate changes.
  settings.filter_acceleration_noise = declare_double("filter_acceleration_noise", 1.55);
  settings.filter_vision_noise = declare_double("filter_vision_noise", 0.25);
  settings.filter_vision_noise_m = declare_double("filter_vision_noise_m", 0.005);
  settings.filter_bias_walk = declare_double("filter_bias_walk", 0.01);
  settings.filter_reference_inliers = declare_double("filter_reference_inliers", 300.0);
  settings.filter_innovation_gate = declare_double("filter_innovation_gate", 9.0);

  // The MSCKF's own numbers, read only when `fusion_model` asks for it.
  //
  // What it changes is where the heading comes from. The other two models take
  // it from the instrument's orientation and hand it to the solve, so the
  // vision residual corrects position, velocity and the accelerometer bias and
  // can never correct the one state that decides which way all three point.
  // Here the gyro's rate drives the heading, its bias is a state, and the
  // ground solve's own heading is the measurement that observes both.
  // Three degrees of freedom, so the chi-square gate moves with them: 11.3 is
  // the 99th percentile for three where 9.0 was for two.
  // Beyond this the innovation is not an outlier to be down-weighted, it is a
  // different vehicle, and it is dropped rather than inflated.
  // How far the instrument's reported heading is allowed to be wrong. Tight and
  // this is the older behaviour with a filter wrapped round it; loose and the
  // gyro and the ground solve carry the heading between fixes. 0 switches the
  // measurement off, which leaves the gyro bias with nothing to observe it.
  // EstimatorSettings carries the measured trade behind this default.
  // Read the mounting-pitch lean from beyond this range only. Not the solve's
  // ground band: the near ground still votes for the pose, it just cannot carry
  // a regression it is the noisiest part of.
  settings.radial_min_range_m = declare_double("radial_min_range_m", 1.5);
  // What a mounting pitch error is worth knowing to, how much lean a radian of
  // it produces, and how much of the lean is the road rather than the mounting.
  // The six degree of freedom filter's own two numbers, and its gate.
  // Rather than choosing that trade once, watch the ground disagree with the
  // reported heading and loosen it by however one-sided the disagreement is.
  // Off by default and off is what the measurement says: it only slides
  // between the fixed settings, and worse than choosing one of them does.

  // How hard front/rear disagreement counts against a solve, and the penalty
  // when only one camera produced an answer at all.
  settings.camera_disagreement_weight = declare_double("camera_disagreement_weight", 1.0);
  settings.single_camera_variance = declare_double("single_camera_variance", 0.5);
  settings.fuse_cameras_by_spread = declare_bool("fuse_cameras_by_spread", false);

  // Take tracks from the C++ front end. Empty is the Python's "do the optical
  // flow here" setting, which this path does not offer.
  topics.track_prefix = declare_string("track_topic_prefix", "");
  // Camera input is best effort by default, which is right on a vehicle: a late
  // frame is worth less than the one behind it. For measurement it is wrong,
  // because the frames DDS drops under load differ from run to run.
  topics.input_reliability = declare_string("input_reliability", "best_effort");
  topics.input_queue_depth = declare_int("input_queue_depth", 5);
  // KEEP_LAST or KEEP_ALL. RELIABLE does not mean lossless: a KEEP_LAST reader
  // that falls more than `depth` messages behind overwrites its own backlog.
  topics.input_history = declare_string("input_history", "keep_last");
  topics.input_durability = declare_string("input_durability", "volatile");
  // The IMU link. Best effort is right on the vehicle -- a late angular rate is
  // worse than none -- but it is not a measurement.
  topics.imu_reliability = declare_string("imu_reliability", "best_effort");
  topics.imu_queue_depth = declare_int("imu_queue_depth", 200);
  topics.odometry_queue_depth = declare_int("odometry_queue_depth", 10);
  // Hz at which queued pairs are solved, decoupled from arrival.
  topics.solve_timer_hz = declare_double("solve_timer_hz", 200.0);
  topics.executor_threads = declare_int("executor_threads", 1);
  declare_string("filter_debug_csv", "");

  return configuration;
}

}  // namespace monoscale_ros
