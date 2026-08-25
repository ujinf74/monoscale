// The estimator driven end to end on a synthetic drive.
//
// The Python equivalent (test_estimator_integration.py) renders images and
// feeds the optical-flow front end. This path takes tracks, so the drive is
// built the same way but stops one step earlier: ground points are placed in
// the world, projected through the shipped calibration into both cameras, and
// handed over as the tracker would hand them over. What is being checked is
// that the extrinsics, the intrinsics, the lens model, the anchor map and the
// motion solve agree with each other -- each of which is tested alone
// elsewhere, and none of which proves the assembly.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "monoscale_core/estimator.hpp"

using monoscale::CameraSettings;
using monoscale::Estimator;
using monoscale::EstimatorSettings;
using monoscale::Identities;
using monoscale::ImuSample;
using monoscale::Lens;
using monoscale::Points2;
using monoscale::Pose2;
using monoscale::TrackFrame;

namespace
{

constexpr int kTrackWidth = 1280;
constexpr int kTrackHeight = 720;
constexpr double kGravity = 9.81;

Eigen::Matrix3d rotation_of(const std::vector<double> & flat)
{
  Eigen::Matrix3d rotation;
  rotation << flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], flat[6], flat[7],
    flat[8];
  return rotation;
}

// The assembled fisheye pair, as vision_fisheye.param.yaml describes it.
CameraSettings front_camera()
{
  CameraSettings camera;
  camera.name = "front";
  camera.rotation_base_from_camera =
    rotation_of({0.0, -0.5, 0.8660254, -1.0, 0.0, 0.0, 0.0, -0.8660254, -0.5});
  camera.translation_base_from_camera = Eigen::Vector3d(3.694, 0.0, 0.89);
  camera.ground_min_distance_m = 0.6;
  camera.k << 1051.81, 0.0, 1279.5, 0.0, 1051.81, 719.5, 0.0, 0.0, 1.0;
  camera.distortion = Eigen::VectorXd::Zero(4);
  camera.lens = Lens::Equidistant;
  camera.calibration_width = 2560;
  camera.calibration_height = 1440;
  return camera;
}

CameraSettings rear_camera()
{
  CameraSettings camera = front_camera();
  camera.name = "rear";
  camera.rotation_base_from_camera =
    rotation_of({0.0, 0.5, -0.8660254, 1.0, 0.0, 0.0, 0.0, -0.8660254, -0.5});
  camera.translation_base_from_camera = Eigen::Vector3d(-0.82, 0.0, 1.26);
  camera.ground_min_distance_m = 0.0;
  return camera;
}

// The intrinsics as they apply to the frame the tracker measured pixels in.
Eigen::Matrix3d track_intrinsics(const CameraSettings & camera)
{
  Eigen::Matrix3d k = camera.k;
  k.row(0) *= static_cast<double>(kTrackWidth) / camera.calibration_width;
  k.row(1) *= static_cast<double>(kTrackHeight) / camera.calibration_height;
  return k;
}

// A ground point in base_link, as this camera would report it. Returns false
// when the point is behind the lens or outside the frame.
bool project(
  const CameraSettings & camera, const Eigen::Matrix3d & k,
  const Eigen::Vector3d & point_base, Eigen::Vector2d & pixel)
{
  const Eigen::Vector3d in_camera = camera.rotation_base_from_camera.transpose() *
    (point_base - camera.translation_base_from_camera);
  if (in_camera.z() <= 1e-6) {
    return false;
  }
  // Pinhole first, then bend it the way an equidistant lens does: the pinhole
  // radius is f*tan(theta), the fisheye radius is f*theta.
  const double nx = in_camera.x() / in_camera.z();
  const double ny = in_camera.y() / in_camera.z();
  const double radius = std::hypot(nx, ny);
  double scale = 1.0;
  if (radius > 1e-12) {
    scale = std::atan(radius) / radius;
  }
  pixel.x() = k(0, 0) * nx * scale + k(0, 2);
  pixel.y() = k(1, 1) * ny * scale + k(1, 2);
  return pixel.x() >= 0.0 && pixel.x() < kTrackWidth && pixel.y() >= 0.0 &&
         pixel.y() < kTrackHeight;
}

Eigen::Vector4d yaw_quaternion(double yaw)
{
  return Eigen::Vector4d(0.0, 0.0, std::sin(0.5 * yaw), std::cos(0.5 * yaw));
}

// Points spread over the road, in the world frame. The third coordinate is a
// height: everything is on the road unless a test asks otherwise.
std::vector<Eigen::Vector3d> road_points()
{
  std::vector<Eigen::Vector3d> points;
  for (int i = -40; i <= 120; ++i) {
    for (int j = -12; j <= 12; ++j) {
      points.emplace_back(0.25 * i, 0.25 * j, 0.0);
    }
  }
  return points;
}

// The same road with something standing on it, off to one side of the path so
// the vehicle drives past rather than through.
std::vector<Eigen::Vector3d> road_with_an_obstacle(double height)
{
  std::vector<Eigen::Vector3d> points = road_points();
  for (int i = 0; i < 24; ++i) {
    for (int j = 0; j < 4; ++j) {
      points.emplace_back(6.0 + 0.1 * i, 1.6 + 0.1 * j, height);
    }
  }
  return points;
}

EstimatorSettings deployment_settings()
{
  EstimatorSettings settings;
  settings.cameras = {front_camera(), rear_camera()};
  settings.ground_max_distance_m = 8.0;
  settings.ground_ransac_threshold_m = 0.12;
  settings.ground_min_inliers = 24;
  settings.solve_trigger_disparity_px = 3.0;
  settings.mapping_min_period_sec = 1e9;
  settings.sync_tolerance_sec = 0.02;
  return settings;
}

struct Drive
{
  Estimator estimator;
  std::vector<Eigen::Vector3d> world;
  double stamp = 1.0;
  double travelled = 0.0;
  Pose2 truth;
  // How many cameras to feed. The estimator fuses a list rather than a pair,
  // so one is a supported rig and not a degenerate case.
  size_t cameras = 2;

  explicit Drive(const EstimatorSettings & settings)
  : estimator(settings), world(road_points()), cameras(settings.cameras.size()) {}

  // One camera frame at the current truth pose, for every camera.
  void step(double forward, double yaw_rate, double dt)
  {
    // The IMU runs at twice the camera rate, as the release kit does.
    for (int i = 0; i < 2; ++i) {
      ImuSample imu;
      imu.stamp = stamp + 0.5 * dt * i;
      imu.orientation = yaw_quaternion(truth.yaw + yaw_rate * 0.5 * dt * i);
      imu.angular_velocity = Eigen::Vector3d(0.0, 0.0, yaw_rate);
      imu.linear_acceleration = Eigen::Vector3d(0.0, 0.0, kGravity);
      estimator.ingest_imu(imu);
    }

    truth = truth.compose(
      monoscale::motion_from_twist(forward, 0.0, yaw_rate, dt));
    travelled += forward * dt;
    stamp += dt;

    for (size_t index = 0; index < cameras; ++index) {
      const CameraSettings camera = index == 0 ? front_camera() : rear_camera();
      const Eigen::Matrix3d k = track_intrinsics(camera);
      TrackFrame frame;
      frame.stamp = stamp;
      frame.width = kTrackWidth;
      frame.height = kTrackHeight;

      std::vector<int64_t> ids;
      std::vector<Eigen::Vector2d> pixels;
      const double c = std::cos(truth.yaw);
      const double s = std::sin(truth.yaw);
      for (size_t n = 0; n < world.size(); ++n) {
        // World into base_link at the truth pose.
        const double dx = world[n].x() - truth.x;
        const double dy = world[n].y() - truth.y;
        const Eigen::Vector3d in_base(
          c * dx + s * dy, -s * dx + c * dy, world[n].z());
        Eigen::Vector2d pixel;
        if (!project(camera, k, in_base, pixel)) {
          continue;
        }
        ids.push_back(static_cast<int64_t>(n));
        pixels.push_back(pixel);
      }

      const Eigen::Index count = static_cast<Eigen::Index>(ids.size());
      frame.ids.resize(count);
      frame.pixels.resize(count, 2);
      frame.previous_pixels.resize(count, 2);
      for (Eigen::Index n = 0; n < count; ++n) {
        frame.ids(n) = ids[static_cast<size_t>(n)];
        frame.pixels(n, 0) = pixels[static_cast<size_t>(n)].x();
        frame.pixels(n, 1) = pixels[static_cast<size_t>(n)].y();
        // The tracker reports where each feature was on the hop before. A
        // rough stand-in is enough: only the 90th percentile of the shift is
        // read, and only to decide when to solve.
        frame.previous_pixels(n, 0) = pixels[static_cast<size_t>(n)].x() - 4.0;
        frame.previous_pixels(n, 1) = pixels[static_cast<size_t>(n)].y();
      }
      estimator.ingest_tracks(index, frame);
    }
  }
};

}  // namespace

TEST(EstimatorDrive, RecoversAStraightDriveInMetres)
{
  Drive drive(deployment_settings());

  // 2 m/s for four seconds at 30 Hz.
  for (int i = 0; i < 120; ++i) {
    drive.step(2.0, 0.0, 1.0 / 30.0);
  }
  const auto updates = drive.estimator.take_updates();

  ASSERT_FALSE(updates.empty());
  const Pose2 & pose = drive.estimator.pose();
  ASSERT_GT(drive.estimator.diagnostics().map_aligned_frames, 0)
    << "the anchor map never answered, so this measured the fallback instead";

  // The pose is held at the origin until the map can answer, so what it has
  // travelled is short of the truth by that warm-up. What must not happen is a
  // scale error: whatever it did cover, it covered in metres.
  EXPECT_GT(pose.x, 0.5 * drive.travelled);
  EXPECT_LT(pose.x, drive.travelled + 0.5);
  EXPECT_NEAR(pose.y, 0.0, 0.3);
  EXPECT_NEAR(pose.yaw, 0.0, 0.02);
}

TEST(EstimatorDrive, ACurvedDriveDoesNotInventLateralVelocity)
{
  EstimatorSettings settings = deployment_settings();
  settings.twist_lowpass_tau = 0.0;
  Drive drive(settings);

  for (int i = 0; i < 180; ++i) {
    drive.step(2.0, 0.35, 1.0 / 30.0);
  }
  const auto updates = drive.estimator.take_updates();

  ASSERT_FALSE(updates.empty());
  const Eigen::Vector3d & twist = updates.back().twist;
  EXPECT_GT(twist.x(), 1.0);
  EXPECT_NEAR(twist.y(), 0.0, 0.05);
  EXPECT_NEAR(twist.z(), 0.35, 0.05);
}

TEST(EstimatorDrive, ARejectedMapAlignmentDoesNotMakeThePoseValid)
{
  EstimatorSettings settings = deployment_settings();
  settings.max_translation_per_frame_m = 1e-6;
  Drive drive(settings);

  for (int i = 0; i < 180; ++i) {
    drive.step(2.0, 0.0, 1.0 / 30.0);
  }
  const auto updates = drive.estimator.take_updates();

  ASSERT_GT(drive.estimator.diagnostics().map_aligned_frames, 0);
  ASSERT_FALSE(updates.empty());
  for (const auto & update : updates) {
    EXPECT_FALSE(update.pose_valid);
  }
}

TEST(EstimatorDrive, TheScaleComesOutOfTheGroundPlaneNotAFit)
{
  // The claim the whole stack rests on: distance is metric because the camera
  // height is. Halving the mounting height halves every recovered range, so a
  // drive solved with a wrong height comes out proportionally wrong -- which is
  // what makes the height the thing to get right rather than a free parameter.
  EstimatorSettings honest = deployment_settings();
  EstimatorSettings shortened = deployment_settings();
  for (auto & camera : shortened.cameras) {
    camera.range_scale = 2.0;
  }

  Drive truthful(honest);
  Drive shrunk(shortened);
  for (int i = 0; i < 120; ++i) {
    truthful.step(2.0, 0.0, 1.0 / 30.0);
    shrunk.step(2.0, 0.0, 1.0 / 30.0);
  }

  const double honest_x = truthful.estimator.pose().x;
  const double shrunk_x = shrunk.estimator.pose().x;
  ASSERT_GT(honest_x, 1.0);
  // Dividing every range by two halves the measured travel.
  EXPECT_NEAR(shrunk_x / honest_x, 0.5, 0.1);
}

TEST(EstimatorDrive, AFrameWithNoPeerDoesNotStallTheRun)
{
  // Online the two cameras do not always deliver together. A frame with no
  // partner inside the tolerance has to be dropped rather than held, or the
  // queues drift apart and no pair ever forms again.
  EstimatorSettings settings = deployment_settings();
  Drive drive(settings);

  for (int i = 0; i < 60; ++i) {
    drive.step(2.0, 0.0, 1.0 / 30.0);
  }
  const int64_t before = drive.estimator.diagnostics().pairs_seen;

  // A lone front frame, far from anything the rear will produce.
  TrackFrame orphan;
  orphan.stamp = drive.stamp + 5.0;
  orphan.width = kTrackWidth;
  orphan.height = kTrackHeight;
  orphan.ids.resize(0);
  orphan.pixels.resize(0, 2);
  orphan.previous_pixels.resize(0, 2);
  drive.estimator.ingest_tracks(0, orphan);

  for (int i = 0; i < 60; ++i) {
    drive.step(2.0, 0.0, 1.0 / 30.0);
  }

  EXPECT_GT(drive.estimator.diagnostics().pairs_seen, before + 30);
}

TEST(EstimatorDrive, OneImuSampleDrainsEveryReadyPair)
{
  EstimatorSettings settings = deployment_settings();
  settings.adaptive_solve_interval = false;
  settings.frame_decimation = 100;
  Estimator estimator(settings);

  ImuSample imu;
  imu.stamp = 1.0;
  imu.orientation = yaw_quaternion(0.0);
  imu.linear_acceleration = Eigen::Vector3d(0.0, 0.0, kGravity);
  estimator.ingest_imu(imu);

  for (double stamp : {1.1, 1.2}) {
    TrackFrame frame;
    frame.stamp = stamp;
    frame.width = kTrackWidth;
    frame.height = kTrackHeight;
    frame.ids.resize(0);
    frame.pixels.resize(0, 2);
    frame.previous_pixels.resize(0, 2);
    estimator.ingest_tracks(0, frame);
    estimator.ingest_tracks(1, frame);
  }
  EXPECT_EQ(estimator.diagnostics().pairs_seen, 0);

  imu.stamp = 1.3;
  estimator.ingest_imu(imu);

  EXPECT_EQ(estimator.diagnostics().pairs_seen, 2);
}

TEST(EstimatorDrive, ObstaclesAreSilentRatherThanFatalWithoutImages)
{
  // The Python reached for keyframe triangulation here and dereferenced a
  // frame that has no picture in it, which is a crash on the deployed path as
  // soon as mapping is enabled. Producing no obstacles is the honest answer.
  EstimatorSettings settings = deployment_settings();
  settings.mapping_min_period_sec = 0.0;
  settings.obstacle_slip_baseline_m = 0.0;
  Drive drive(settings);

  for (int i = 0; i < 90; ++i) {
    drive.step(2.0, 0.0, 1.0 / 30.0);
  }
  const auto updates = drive.estimator.take_updates();

  size_t ground = 0;
  size_t obstacles = 0;
  for (const auto & update : updates) {
    for (const auto & point : update.points) {
      if (point.label == monoscale::kObstacleLabel) {
        ++obstacles;
      } else {
        ++ground;
      }
    }
  }
  EXPECT_GT(ground, 0u);
  EXPECT_EQ(obstacles, 0u);
  EXPECT_GT(drive.estimator.diagnostics().obstacles_unavailable, 0);
}

TEST(EstimatorDrive, OneCameraIsARigNotADegenerateCase)
{
  // With a single camera the disagreement between two of them -- the one
  // signal neither can produce alone -- is gone, and `single_camera_variance`
  // carries that weight instead. It runs, and it is measurably worse; what
  // must not happen is that it silently stops solving.
  EstimatorSettings settings = deployment_settings();
  settings.cameras = {front_camera()};
  Drive drive(settings);

  for (int i = 0; i < 120; ++i) {
    drive.step(2.0, 0.0, 1.0 / 30.0);
  }

  ASSERT_GT(drive.estimator.diagnostics().map_aligned_frames, 0)
    << "a single camera never reached the anchor map";
  const Pose2 & pose = drive.estimator.pose();
  EXPECT_GT(pose.x, 0.5 * drive.travelled);
  EXPECT_LT(pose.x, drive.travelled + 0.5);
  EXPECT_NEAR(pose.y, 0.0, 0.5);
}

TEST(EstimatorDrive, AnObstacleIsReadOffHowItsProjectionSlides)
{
  // The default way of placing an obstacle on this path, and the only one it
  // has: a feature standing above the road projects to a ground point that
  // slides against the travel, and how fast it slides is its height. Nothing
  // is triangulated -- this is the residual the ground registration already
  // computes and throws away as an outlier.
  const double height = 0.5;
  EstimatorSettings settings = deployment_settings();
  settings.mapping_min_period_sec = 0.0;
  Drive drive(settings);
  drive.world = road_with_an_obstacle(height);

  for (int i = 0; i < 150; ++i) {
    drive.step(2.0, 0.0, 1.0 / 30.0);
  }

  std::vector<float> heights;
  for (const auto & update : drive.estimator.take_updates()) {
    for (const auto & point : update.points) {
      if (point.label == monoscale::kObstacleLabel) {
        heights.push_back(point.z);
      }
    }
  }

  ASSERT_FALSE(heights.empty()) << "the obstacle never came out of the slip";
  // The method reads a height, not a point cloud: what has to hold is that the
  // heights land near the real one rather than at the road or at the gate.
  std::sort(heights.begin(), heights.end());
  const float middle = heights[heights.size() / 2];
  EXPECT_NEAR(middle, height, 0.2) << "median of " << heights.size() << " readings";
}

TEST(EstimatorDrive, FlatRoadProducesNoObstacles)
{
  // The other half of the same claim. A slip method that finds obstacles on an
  // empty road is worse than none: the grid ships at a threshold where one
  // reading marks a cell.
  EstimatorSettings settings = deployment_settings();
  settings.mapping_min_period_sec = 0.0;
  Drive drive(settings);

  for (int i = 0; i < 150; ++i) {
    drive.step(2.0, 0.0, 1.0 / 30.0);
  }

  size_t obstacles = 0;
  size_t ground = 0;
  for (const auto & update : drive.estimator.take_updates()) {
    for (const auto & point : update.points) {
      (point.label == monoscale::kObstacleLabel ? obstacles : ground) += 1;
    }
  }
  EXPECT_GT(ground, 0u);
  EXPECT_EQ(obstacles, 0u);
}
