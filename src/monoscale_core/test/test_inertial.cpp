// The inertial propagator, from src/monoscale_odometry/test/test_inertial.py.

#include <cmath>

#include <gtest/gtest.h>

#include "monoscale_core/inertial.hpp"

using monoscale::agrees_with_prediction;
using monoscale::Integration;
using monoscale::PlanarInertialPropagator;

namespace
{

const Eigen::Vector4d kLevel(0.0, 0.0, 0.0, 1.0);
constexpr double kGravity = 9.81;

double feed(
  PlanarInertialPropagator & propagator, double duration, double accel_x,
  double rate = 100.0, double start = 0.0,
  const Eigen::Vector4d & orientation = kLevel)
{
  const int steps = static_cast<int>(duration * rate);
  for (int index = 0; index <= steps; ++index) {
    propagator.add_sample(
      start + index / rate, orientation, Eigen::Vector3d(accel_x, 0.0, kGravity));
  }
  return start + steps / rate;
}

}  // namespace

TEST(Inertial, LevelAndStillProducesNoMotion)
{
  PlanarInertialPropagator propagator;

  feed(propagator, 1.0, 0.0);

  EXPECT_NEAR(propagator.velocity().norm(), 0.0, 1e-9);
  EXPECT_NEAR(propagator.position().norm(), 0.0, 1e-9);
}

TEST(Inertial, ConstantAccelerationIntegratesToTheTextbookAnswer)
{
  PlanarInertialPropagator propagator;
  // Start from a known standstill, as vision would report it.
  propagator.correct_velocity(Eigen::Vector2d::Zero());

  const double end = feed(propagator, 2.0, 1.5);

  // v = at, s = at^2/2
  EXPECT_NEAR(propagator.velocity().x(), 3.0, 1e-6);
  EXPECT_NEAR(propagator.position().x(), 3.0, 3.0 * 1e-3);
  const auto predicted = propagator.predicted_translation(0.0, end, 0.0);
  ASSERT_TRUE(predicted.has_value());
  EXPECT_NEAR(predicted->x(), 3.0, 3.0 * 1e-3);
}

TEST(Inertial, PredictionIsExpressedInTheBodyFrame)
{
  PlanarInertialPropagator propagator;
  propagator.correct_velocity(Eigen::Vector2d::Zero());
  const double end = feed(propagator, 1.0, 2.0);

  const auto forward = propagator.predicted_translation(0.0, end, 0.0);
  const auto sideways = propagator.predicted_translation(0.0, end, M_PI / 2.0);

  ASSERT_TRUE(forward.has_value());
  ASSERT_TRUE(sideways.has_value());
  EXPECT_NEAR(forward->x(), 1.0, 1e-3);
  EXPECT_NEAR(forward->y(), 0.0, 1e-9);
  // Facing +y, the same world motion is now to the vehicle's right.
  EXPECT_NEAR(sideways->x(), 0.0, 1e-6);
  EXPECT_NEAR(sideways->y(), -1.0, 1e-3);
}

TEST(Inertial, GravityIsRemovedUsingTheReportedOrientation)
{
  // Rolled 90 degrees: gravity now reads along the body y axis.
  const double roll = M_PI / 2.0;
  const Eigen::Vector4d orientation(std::sin(roll / 2.0), 0.0, 0.0, std::cos(roll / 2.0));
  PlanarInertialPropagator propagator;

  for (int index = 0; index <= 100; ++index) {
    propagator.add_sample(
      index / 100.0, orientation, Eigen::Vector3d(0.0, -kGravity, 0.0));
  }

  EXPECT_NEAR(propagator.velocity().norm(), 0.0, 1e-6);
}

TEST(Inertial, VisionCorrectionReplacesTheDriftedVelocity)
{
  PlanarInertialPropagator propagator;
  propagator.correct_velocity(Eigen::Vector2d::Zero());
  feed(propagator, 1.0, 1.0);
  ASSERT_NEAR(propagator.velocity().x(), 1.0, 1e-6);

  propagator.correct_velocity(Eigen::Vector2d(0.2, 0.0));

  EXPECT_NEAR(propagator.velocity().x(), 0.2, 1e-9);
}

TEST(Inertial, ALongGapResetsTheVelocityRatherThanExtrapolating)
{
  PlanarInertialPropagator::Settings settings;
  settings.max_gap_sec = 0.1;
  PlanarInertialPropagator propagator(settings);
  propagator.correct_velocity(Eigen::Vector2d::Zero());
  feed(propagator, 1.0, 2.0);

  propagator.add_sample(5.0, kLevel, Eigen::Vector3d(0.0, 0.0, kGravity));

  EXPECT_NEAR(propagator.velocity().norm(), 0.0, 1e-9);
}

TEST(Inertial, GateAcceptsAReasonableVisionAnswer)
{
  EXPECT_TRUE(
    agrees_with_prediction(
      Eigen::Vector2d(0.42, 0.01), Eigen::Vector2d(0.43, 0.0), 0.05, 0.3));
}

TEST(Inertial, GateRejectsTheCollapseToNearZero)
{
  // The measured failure: vision says the vehicle barely moved while the
  // accelerometer says it covered 0.43 m.
  EXPECT_FALSE(
    agrees_with_prediction(
      Eigen::Vector2d(0.05, 0.0), Eigen::Vector2d(0.43, 0.0), 0.05, 0.3));
}

TEST(Inertial, GatePassesEverythingWithoutAPrediction)
{
  EXPECT_TRUE(agrees_with_prediction(Eigen::Vector2d(9.0, 9.0), std::nullopt, 0.05, 0.3));
}

TEST(Inertial, GateStaysOutOfTheWayWhenThePredictionIsTiny)
{
  // Constant velocity means no horizontal acceleration, so a propagator that
  // has never been corrected predicts nothing. It must not veto vision.
  EXPECT_TRUE(
    agrees_with_prediction(
      Eigen::Vector2d(0.12, 0.0), Eigen::Vector2d(0.0, 0.0), 0.08, 0.35));
}

TEST(Inertial, GateStillCatchesTheCollapseOnceThePredictionIsReal)
{
  EXPECT_FALSE(
    agrees_with_prediction(
      Eigen::Vector2d(0.05, 0.0), Eigen::Vector2d(0.43, 0.0), 0.08, 0.35));
}

TEST(Inertial, PredictionIsWithheldUntilVisionAnchorsTheVelocity)
{
  PlanarInertialPropagator propagator;
  // A spawn drop, integrated before anything has been corrected.
  feed(propagator, 0.5, 4.0);

  EXPECT_FALSE(propagator.predicted_translation(0.0, 0.5, 0.0).has_value());

  propagator.correct_velocity(Eigen::Vector2d::Zero());
  feed(propagator, 0.5, 1.0, 100.0, 0.5);

  EXPECT_TRUE(propagator.predicted_translation(0.5, 1.0, 0.0).has_value());
}

TEST(Inertial, AddSampleReportsTheAccelerationAFilterShouldPropagateOn)
{
  PlanarInertialPropagator::Settings settings;
  settings.median_window = 1;
  settings.integration = Integration::Zoh;
  PlanarInertialPropagator propagator(settings);
  propagator.correct_velocity(Eigen::Vector2d::Zero());
  propagator.add_sample(0.0, kLevel, Eigen::Vector3d(0.0, 0.0, kGravity));

  const auto step = propagator.add_sample(0.01, kLevel, Eigen::Vector3d(2.0, 0.0, kGravity));

  EXPECT_NEAR(step.acceleration.x(), 2.0, 1e-9);
  EXPECT_NEAR(step.acceleration.y(), 0.0, 1e-9);
  EXPECT_NEAR(step.dt, 0.01, 1e-12);
}

TEST(Inertial, Rk4ReportsTheIntervalAverageToTheVelocityFilter)
{
  PlanarInertialPropagator::Settings settings;
  settings.max_gap_sec = 2.0;
  settings.median_window = 1;
  settings.integration = Integration::Rk4;
  PlanarInertialPropagator propagator(settings);
  propagator.correct_velocity(Eigen::Vector2d::Zero());
  propagator.add_sample(0.0, kLevel, Eigen::Vector3d(0.0, 0.0, kGravity));

  const auto step = propagator.add_sample(1.0, kLevel, Eigen::Vector3d(2.0, 0.0, kGravity));

  EXPECT_NEAR(step.acceleration.x(), 1.0, 1e-9);
  EXPECT_NEAR(step.dt, 1.0, 1e-12);
}

TEST(Inertial, SpawnAccelerationIsNotIntegratedBeforeVisionAnchor)
{
  PlanarInertialPropagator::Settings settings;
  settings.median_window = 1;
  PlanarInertialPropagator propagator(settings);

  feed(propagator, 0.5, 8.0);

  EXPECT_NEAR(propagator.velocity().norm(), 0.0, 1e-9);
  EXPECT_NEAR(propagator.position().norm(), 0.0, 1e-9);
}

TEST(Inertial, PhysicsImpulseIsRejectedWithoutPoisoningVelocity)
{
  PlanarInertialPropagator::Settings settings;
  settings.max_horizontal_acceleration = 12.0;
  settings.median_window = 3;
  PlanarInertialPropagator propagator(settings);
  propagator.correct_velocity(Eigen::Vector2d(1.0, 0.0));
  propagator.add_sample(0.0, kLevel, Eigen::Vector3d(0.0, 0.0, kGravity));

  const auto step =
    propagator.add_sample(0.01, kLevel, Eigen::Vector3d(900.0, 20.0, kGravity));

  EXPECT_NEAR(step.acceleration.norm(), 0.0, 1e-9);
  EXPECT_NEAR(propagator.velocity().x(), 1.0, 1e-9);
  EXPECT_EQ(propagator.rejected_samples(), 1);
}

TEST(Inertial, Rk4IntegratesLinearlyChangingAcceleration)
{
  PlanarInertialPropagator::Settings settings;
  settings.max_gap_sec = 2.0;
  settings.median_window = 1;
  settings.integration = Integration::Rk4;
  PlanarInertialPropagator propagator(settings);
  propagator.correct_velocity(Eigen::Vector2d::Zero());
  propagator.add_sample(0.0, kLevel, Eigen::Vector3d(0.0, 0.0, kGravity));
  propagator.add_sample(1.0, kLevel, Eigen::Vector3d(2.0, 0.0, kGravity));

  // a(t)=2t: v(1)=1 and x(1)=1/3.
  EXPECT_NEAR(propagator.velocity().x(), 1.0, 1e-9);
  EXPECT_NEAR(propagator.position().x(), 1.0 / 3.0, 1e-9);
}
