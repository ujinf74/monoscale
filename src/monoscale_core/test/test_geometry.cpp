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
