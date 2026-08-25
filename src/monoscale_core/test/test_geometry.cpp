// The ground geometry, held to what the Python it replaces was held to.
//
// These are the cases from src/monoscale_odometry/test/test_geometry.py. Where
// the original leaned on numpy's generator the input is built from a small
// deterministic sequence instead, since the claim being made is about outlier
// rejection rather than about a particular draw.

#include <cmath>

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>

#include "monoscale_core/geometry.hpp"

using monoscale::CameraModel;
using monoscale::Lens;
using monoscale::Mask;
using monoscale::PlanarMotion;
using monoscale::Points2;
using monoscale::Points3;
using monoscale::Pose2;

namespace
{

Eigen::Matrix3d intrinsics(double focal, double cx, double cy)
{
  Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
  k(0, 0) = focal;
  k(1, 1) = focal;
  k(0, 2) = cx;
  k(1, 2) = cy;
  return k;
}

// The optical frame of a camera looking straight down the vehicle's x axis:
// image x to the right, image y down, image z forward.
Eigen::Matrix3d forward_optical()
{
  Eigen::Matrix3d rotation;
  rotation << 0.0, 0.0, 1.0,
    -1.0, 0.0, 0.0,
    0.0, -1.0, 0.0;
  return rotation;
}

Eigen::Matrix2d rotation_2d(double yaw)
{
  Eigen::Matrix2d rotation;
  rotation << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
  return rotation;
}

}  // namespace

TEST(Geometry, PixelsToGroundForForwardOpticalCamera)
{
  const CameraModel model = monoscale::make_camera_model(
    intrinsics(100.0, 50.0, 50.0), forward_optical(), Eigen::Vector3d(0.0, 0.0, 1.0));

  Points2 pixels(1, 2);
  pixels << 50.0, 100.0;

  const auto [ground, valid] = monoscale::pixels_to_ground(pixels, model, 20.0);

  ASSERT_EQ(valid.size(), 1);
  EXPECT_TRUE(valid(0));
  EXPECT_NEAR(ground(0, 0), 2.0, 1e-9);
  EXPECT_NEAR(ground(0, 1), 0.0, 1e-9);
}

TEST(Geometry, PlanarMotionMapsCurrentPointsIntoPreviousFrame)
{
  Points2 current(4, 2);
  current << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, -1.0;
  const double yaw = 0.1;
  const Eigen::Matrix2d rotation = rotation_2d(yaw);

  Points2 previous(4, 2);
  for (Eigen::Index i = 0; i < current.rows(); ++i) {
    const Eigen::Vector2d turned = rotation * Eigen::Vector2d(current(i, 0), current(i, 1));
    previous(i, 0) = turned.x() + 0.4;
    previous(i, 1) = turned.y() - 0.2;
  }

  const auto result = monoscale::estimate_planar_motion(previous, current, 0.01, 3, 0.02);

  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->motion.x, 0.4, 1e-5);
  EXPECT_NEAR(result->motion.y, -0.2, 1e-5);
  EXPECT_NEAR(result->motion.yaw, yaw, 1e-5);
}

TEST(Geometry, PlanarMotionWithImuYawRejectsTranslationOutliers)
{
  // A fixed sequence rather than a generator: what is being tested is that the
  // fifteen corrupted correspondences are thrown out and the other sixty-five
  // decide the answer.
  const int count = 80;
  const int corrupted = 15;
  Points2 current(count, 2);
  unsigned int state = 7u;
  const auto next = [&state]() {
      state = state * 1103515245u + 12345u;
      return static_cast<double>((state >> 16) & 0x7fffu) / 32767.0;
    };
  for (int i = 0; i < count; ++i) {
    current(i, 0) = -5.0 + 10.0 * next();
    current(i, 1) = -5.0 + 10.0 * next();
  }

  const double yaw = -0.08;
  const Eigen::Matrix2d rotation = rotation_2d(yaw);
  Points2 previous(count, 2);
  for (int i = 0; i < count; ++i) {
    const Eigen::Vector2d turned = rotation * Eigen::Vector2d(current(i, 0), current(i, 1));
    previous(i, 0) = turned.x() + 0.25;
    previous(i, 1) = turned.y() + 0.04;
  }
  for (int i = 0; i < corrupted; ++i) {
    previous(i, 0) += 0.5 + 1.5 * next();
    previous(i, 1) += 0.5 + 1.5 * next();
  }

  const auto result =
    monoscale::estimate_planar_motion_with_yaw(previous, current, yaw, 0.05, 40);

  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->motion.x, 0.25, 1e-9);
  EXPECT_NEAR(result->motion.y, 0.04, 1e-9);
  EXPECT_DOUBLE_EQ(result->motion.yaw, yaw);
  EXPECT_EQ(result->inliers.count(), count - corrupted);
}

TEST(Geometry, TemporalTriangulationRecoversMetricObstaclePoint)
{
  const Eigen::Matrix3d k = intrinsics(300.0, 320.0, 240.0);
  const Eigen::Matrix3d rotation = forward_optical();
  const Eigen::Vector3d translation(1.0, 0.0, 1.0);
  const CameraModel model = monoscale::make_camera_model(k, rotation, translation);

  const Eigen::Vector3d point_previous_base(8.0, 1.0, 1.5);
  const PlanarMotion motion{0.5, 0.0, 0.0, 100, 1.0};

  const Eigen::Vector3d previous_camera =
    rotation.transpose() * (point_previous_base - translation);
  const Eigen::Vector3d current_base =
    point_previous_base - Eigen::Vector3d(motion.x, motion.y, 0.0);
  const Eigen::Vector3d current_camera = rotation.transpose() * (current_base - translation);

  const auto project = [&k](const Eigen::Vector3d & point) {
      const Eigen::Vector3d pixel = k * point;
      return Eigen::Vector2d(pixel.x() / pixel.z(), pixel.y() / pixel.z());
    };

  Points2 previous_pixels(1, 2);
  Points2 current_pixels(1, 2);
  previous_pixels.row(0) = project(previous_camera).transpose();
  current_pixels.row(0) = project(current_camera).transpose();

  Points3 points;
  Mask valid;
  monoscale::triangulate_temporal_points(
    previous_pixels, current_pixels, model, motion, 0.1, 0.1, 0.1, 30.0, points, valid);

  ASSERT_EQ(valid.size(), 1);
  EXPECT_TRUE(valid(0));
  EXPECT_NEAR(points(0, 0), point_previous_base.x(), 1e-5);
  EXPECT_NEAR(points(0, 1), point_previous_base.y(), 1e-5);
  EXPECT_NEAR(points(0, 2), point_previous_base.z(), 1e-5);
}

TEST(Geometry, EquidistantPixelsAreUndistortedBeforeGeometry)
{
  const Eigen::Matrix3d k = intrinsics(300.0, 320.0, 240.0);
  Eigen::VectorXd distortion(4);
  distortion << 0.1, -0.02, 0.001, 0.0;

  // Start from the pinhole pixel we expect back, push it through the lens, and
  // check the round trip.
  cv::Mat k_cv(3, 3, CV_64F, cv::Scalar(0.0));
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      k_cv.at<double>(r, c) = k(r, c);
    }
  }
  cv::Mat d_cv(4, 1, CV_64F);
  for (int i = 0; i < 4; ++i) {
    d_cv.at<double>(i, 0) = distortion(i);
  }
  cv::Mat normalized(1, 1, CV_64FC2);
  normalized.at<cv::Vec2d>(0, 0) = cv::Vec2d(0.4, -0.2);
  cv::Mat distorted;
  cv::fisheye::distortPoints(normalized, distorted, k_cv, d_cv);

  Points2 pixels(1, 2);
  pixels << distorted.at<cv::Vec2d>(0, 0)[0], distorted.at<cv::Vec2d>(0, 0)[1];

  const CameraModel model = monoscale::make_camera_model(
    k, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), distortion, Lens::Equidistant);
  const Points2 corrected = monoscale::undistort_pixels(pixels, model);

  EXPECT_NEAR(corrected(0, 0), 440.0, 1e-6);
  EXPECT_NEAR(corrected(0, 1), 180.0, 1e-6);
}

TEST(Geometry, EquidistantWithoutCoefficientsMatchesOpenCv)
{
  // The closed form this takes when the coefficients are zero has to land where
  // the iterative solver lands, or the deployed fisheye reads a different
  // ground from the one every recorded measurement was taken on.
  const Eigen::Matrix3d k = intrinsics(1051.81, 1279.5, 719.5);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(4);

  Points2 pixels(5, 2);
  pixels << 1279.5, 719.5,
    1279.5 + 400.0, 719.5,
    1279.5, 719.5 - 300.0,
    1279.5 - 900.0, 719.5 + 500.0,
    1279.5 + 1100.0, 719.5 + 700.0;

  const CameraModel model = monoscale::make_camera_model(
    k, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), zero, Lens::Equidistant);
  const Points2 fast = monoscale::undistort_pixels(pixels, model);

  cv::Mat k_cv(3, 3, CV_64F, cv::Scalar(0.0));
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      k_cv.at<double>(r, c) = k(r, c);
    }
  }
  const cv::Mat d_cv = cv::Mat::zeros(4, 1, CV_64F);
  cv::Mat source(static_cast<int>(pixels.rows()), 1, CV_64FC2);
  for (Eigen::Index i = 0; i < pixels.rows(); ++i) {
    source.at<cv::Vec2d>(static_cast<int>(i), 0) = cv::Vec2d(pixels(i, 0), pixels(i, 1));
  }
  cv::Mat reference;
  cv::fisheye::undistortPoints(source, reference, k_cv, d_cv, cv::noArray(), k_cv);

  for (Eigen::Index i = 0; i < pixels.rows(); ++i) {
    const cv::Vec2d expected = reference.at<cv::Vec2d>(static_cast<int>(i), 0);
    EXPECT_NEAR(fast(i, 0), expected[0], 1e-6) << "row " << i;
    EXPECT_NEAR(fast(i, 1), expected[1], 1e-6) << "row " << i;
  }
}

TEST(Geometry, ZeroDistortionLeavesPixelsUntouched)
{
  const CameraModel model = monoscale::make_camera_model(
    intrinsics(900.0, 960.0, 540.0), Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
    Eigen::VectorXd::Zero(5), Lens::Pinhole);

  Points2 pixels(2, 2);
  pixels << 10.0, 20.0, 1900.0, 1000.0;

  const Points2 corrected = monoscale::undistort_pixels(pixels, model);

  EXPECT_DOUBLE_EQ(corrected(0, 0), 10.0);
  EXPECT_DOUBLE_EQ(corrected(0, 1), 20.0);
  EXPECT_DOUBLE_EQ(corrected(1, 0), 1900.0);
  EXPECT_DOUBLE_EQ(corrected(1, 1), 1000.0);
}

TEST(Geometry, PlumbBobCoefficientsAreStillApplied)
{
  Eigen::VectorXd distortion(5);
  distortion << 0.1, -0.02, 0.0, 0.0, 0.0;
  const CameraModel model = monoscale::make_camera_model(
    intrinsics(300.0, 320.0, 240.0), Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
    distortion, Lens::Pinhole);

  Points2 pixels(1, 2);
  pixels << 440.0, 180.0;

  const Points2 corrected = monoscale::undistort_pixels(pixels, model);

  EXPECT_GT(std::hypot(corrected(0, 0) - 440.0, corrected(0, 1) - 180.0), 1e-3);
}

TEST(Geometry, RelativeMotionIsExpressedInTheOriginFrame)
{
  const Pose2 origin{2.0, 1.0, M_PI / 2.0};
  const Pose2 pose{2.0, 4.0, M_PI / 2.0 + 0.3};

  const PlanarMotion motion = monoscale::relative_motion(origin, pose);

  // Moving along world +y while facing +y is straight ahead.
  EXPECT_NEAR(motion.x, 3.0, 1e-9);
  EXPECT_NEAR(motion.y, 0.0, 1e-9);
  EXPECT_NEAR(motion.yaw, 0.3, 1e-9);
}

TEST(Geometry, RelativeMotionRoundTripsThroughCompose)
{
  const Pose2 origin{-1.0, 5.0, -0.8};
  const Pose2 pose{3.0, 2.0, 1.1};

  const Pose2 restored = origin.compose(monoscale::relative_motion(origin, pose));

  EXPECT_NEAR(restored.x, pose.x, 1e-9);
  EXPECT_NEAR(restored.y, pose.y, 1e-9);
  EXPECT_NEAR(restored.yaw, pose.yaw, 1e-9);
}

TEST(Geometry, TwistRoundTripsThroughAnArc)
{
  // Stretching a measured hop as a straight line is only right with the wheel
  // centred; the twist form has to give the same motion back when the ratio is
  // one, and follow the arc when it is not.
  const PlanarMotion motion{0.31, 0.02, 0.09, 120, 1.0};

  const PlanarMotion same = monoscale::rescale_motion(motion, 1.0);
  EXPECT_DOUBLE_EQ(same.x, motion.x);
  EXPECT_DOUBLE_EQ(same.y, motion.y);
  EXPECT_DOUBLE_EQ(same.yaw, motion.yaw);

  double vx = 0.0;
  double vy = 0.0;
  double omega = 0.0;
  monoscale::twist_from_motion(motion, vx, vy, omega);
  const PlanarMotion restored = monoscale::motion_from_twist(vx, vy, omega, 1.0);
  EXPECT_NEAR(restored.x, motion.x, 1e-12);
  EXPECT_NEAR(restored.y, motion.y, 1e-12);
  EXPECT_NEAR(restored.yaw, motion.yaw, 1e-12);

  // Two half steps compose into the whole one.
  const PlanarMotion half = monoscale::rescale_motion(motion, 0.5);
  const Pose2 walked = Pose2{}.compose(half).compose(half);
  const Pose2 direct = Pose2{}.compose(motion);
  EXPECT_NEAR(walked.x, direct.x, 1e-12);
  EXPECT_NEAR(walked.y, direct.y, 1e-12);
  EXPECT_NEAR(walked.yaw, direct.yaw, 1e-12);
}

// Rescaling a hop has to follow the arc the vehicle is actually on.
//
// Two places stretch a measured motion to a different interval: the pair,
// whose two cameras are stamped a little apart, and the coast that carries the
// pose across a rejected solve. Both used to multiply the translation and be
// done with it, which is right only while the wheel is centred.
// From src/monoscale_odometry/test/test_twist.py.

TEST(Geometry, AStraightHopIsJustStretched)
{
  const PlanarMotion straight{0.4, 0.0, 0.0, 100, 1.0};

  const PlanarMotion doubled = monoscale::rescale_motion(straight, 2.0);

  EXPECT_NEAR(doubled.x, 0.8, 1e-12);
  EXPECT_NEAR(doubled.y, 0.0, 1e-12);
  EXPECT_NEAR(doubled.yaw, 0.0, 1e-12);
}

TEST(Geometry, TheStraightLineAnswerIsWrongByAMeasurableAmount)
{
  // And by how much, so the reason for the change is on the record. A tenth of
  // a second at 4 m/s through a 16 deg/s turn, carried across twice its own
  // length: stretching the straight line puts the vehicle 11 mm off sideways.
  // The pair's two cameras are only a few milliseconds apart and see a
  // fraction of that, but a coast across a rejected solve spans exactly this
  // sort of interval, and there are tens of them in a drive.
  const double speed = 4.0;
  const double rate = 16.0 * M_PI / 180.0;
  const double interval = 0.1;
  const PlanarMotion hop = monoscale::motion_from_twist(speed, 0.0, rate, interval);

  const PlanarMotion stretched{hop.x * 2.0, hop.y * 2.0, hop.yaw * 2.0, 0, 1.0};
  const PlanarMotion arc = monoscale::rescale_motion(hop, 2.0);

  EXPECT_NEAR(std::abs(arc.y - stretched.y), 0.0112, 1e-3);
  // Forward it barely matters: the arc and the chord are the same length to
  // within a part in a thousand at this angle.
  EXPECT_LT(std::abs(arc.x - stretched.x), 1e-3);
}

TEST(Geometry, RescalingToNothingLeavesNothing)
{
  const PlanarMotion motion{0.3, 0.1, 0.2, 7, 1.0};

  const PlanarMotion still = monoscale::rescale_motion(motion, 0.0);

  EXPECT_NEAR(still.x, 0.0, 1e-12);
  EXPECT_NEAR(still.y, 0.0, 1e-12);
  EXPECT_NEAR(still.yaw, 0.0, 1e-12);
}

TEST(Geometry, TheInlierCountSurvivesTheRescale)
{
  // It is the weight the fusion uses, and it is not a property of time.
  const PlanarMotion motion{0.3, 0.1, 0.2, 137, 0.5};

  EXPECT_EQ(monoscale::rescale_motion(motion, 1.7).inliers, 137);
  EXPECT_DOUBLE_EQ(monoscale::rescale_motion(motion, 1.7).scale, 0.5);
}

namespace
{

// Project a point given in the earlier body frame, after the body has moved by
// (`shift`, `yaw`). The earlier frame is the case shift = 0, yaw = 0.
Eigen::Vector2d project_from(
  const Eigen::Vector3d & point, const CameraModel & model, const Eigen::Vector2d & shift,
  double yaw)
{
  Eigen::Matrix3d yawed = Eigen::Matrix3d::Identity();
  yawed.topLeftCorner<2, 2>() = rotation_2d(-yaw);
  const Eigen::Vector3d moved =
    yawed * (point - Eigen::Vector3d(shift.x(), shift.y(), 0.0));
  const Eigen::Vector3d camera =
    model.rotation_base_from_camera.transpose() * (moved - model.translation_base_from_camera);
  const Eigen::Vector3d pixel = model.k * camera;
  return Eigen::Vector2d(pixel.x() / pixel.z(), pixel.y() / pixel.z());
}

}  // namespace

TEST(Geometry, EpipolarBearingRecoversTranslationDirectionFromOffGroundPoints)
{
  // Mounted forward of the axle and above it, so the lever arm contributes a
  // real term the moment the vehicle turns.
  const CameraModel model = monoscale::make_camera_model(
    intrinsics(300.0, 320.0, 240.0), forward_optical(), Eigen::Vector3d(1.5, 0.0, 1.2));

  // Points at assorted heights and ranges. A single plane would still be
  // solvable here -- the rotation is known -- but the point of the constraint
  // is the structure the ground solve cannot use.
  std::vector<Eigen::Vector3d> world;
  for (int i = 0; i < 24; ++i) {
    const double along = 6.0 + 0.9 * (i % 7);
    const double across = -4.0 + 0.7 * (i % 11);
    const double up = 0.2 + 0.45 * (i % 5);
    world.emplace_back(along, across, up);
  }

  for (const double yaw : {0.0, 0.05, -0.08}) {
    const Eigen::Vector2d shift(0.42, 0.09);

    Points2 previous(static_cast<Eigen::Index>(world.size()), 2);
    Points2 current(static_cast<Eigen::Index>(world.size()), 2);
    for (size_t i = 0; i < world.size(); ++i) {
      const Eigen::Index row = static_cast<Eigen::Index>(i);
      previous.row(row) = project_from(world[i], model, Eigen::Vector2d::Zero(), 0.0);
      current.row(row) = project_from(world[i], model, shift, yaw);
    }
    const Mask use = Mask::Constant(static_cast<Eigen::Index>(world.size()), true);

    const auto bearing = monoscale::epipolar_bearing(previous, current, use, model, yaw, 0.02, 6);
    ASSERT_TRUE(bearing.has_value()) << "yaw " << yaw;

    // What the constraint sees is the camera's own displacement, which is the
    // body's plus what the mount swings through.
    const Eigen::Vector2d lever =
      (rotation_2d(yaw) - Eigen::Matrix2d::Identity()) * model.translation_base_from_camera.head<2>();
    const Eigen::Vector2d expected = (shift + lever).normalized();
    const double aligned = bearing->dot(expected);
    EXPECT_NEAR(std::abs(aligned), 1.0, 1e-6) << "yaw " << yaw;
  }
}

TEST(Geometry, EpipolarBearingNeedsEnoughPoints)
{
  const CameraModel model = monoscale::make_camera_model(
    intrinsics(300.0, 320.0, 240.0), forward_optical(), Eigen::Vector3d(1.5, 0.0, 1.2));
  Points2 previous(3, 2);
  Points2 current(3, 2);
  previous << 300.0, 200.0, 340.0, 210.0, 320.0, 260.0;
  current << 302.0, 201.0, 343.0, 212.0, 323.0, 263.0;
  const Mask use = Mask::Constant(3, true);
  EXPECT_FALSE(
    monoscale::epipolar_bearing(previous, current, use, model, 0.0, 0.02, 6).has_value());
}
