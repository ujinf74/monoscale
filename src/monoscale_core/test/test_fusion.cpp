// The velocity filter, from src/monoscale_odometry/test/test_fusion.py.

#include <cmath>

#include <gtest/gtest.h>

#include "monoscale_core/fusion.hpp"

using monoscale::PlanarVelocityFilter;

namespace
{

PlanarVelocityFilter settled(const PlanarVelocityFilter::Settings & settings = {})
{
  PlanarVelocityFilter filter(settings);
  filter.update(Eigen::Vector2d::Zero(), 500);
  return filter;
}

}  // namespace

TEST(Fusion, PredictionFollowsTheAccelerometer)
{
  PlanarVelocityFilter filter = settled();

  for (int i = 0; i < 100; ++i) {
    filter.predict(Eigen::Vector2d(2.0, 0.0), 0.01);
  }

  EXPECT_NEAR(filter.velocity().x(), 2.0, 1e-9);
}

TEST(Fusion, UncertaintyGrowsWhileOnlyPredicting)
{
  PlanarVelocityFilter filter = settled();
  const double before = filter.uncertainty();

  for (int i = 0; i < 50; ++i) {
    filter.predict(Eigen::Vector2d::Zero(), 0.01);
  }

  EXPECT_GT(filter.uncertainty(), before);
}

TEST(Fusion, AWellSupportedMeasurementMovesTheEstimateMore)
{
  PlanarVelocityFilter strong;
  PlanarVelocityFilter weak;
  for (PlanarVelocityFilter * filter : {&strong, &weak}) {
    filter->update(Eigen::Vector2d::Zero(), 300);
    for (int i = 0; i < 10; ++i) {
      filter->predict(Eigen::Vector2d::Zero(), 0.05);
    }
  }

  // Same small disagreement, different support behind it.
  strong.update(Eigen::Vector2d(0.1, 0.0), 3000);
  weak.update(Eigen::Vector2d(0.1, 0.0), 10);

  EXPECT_GT(strong.velocity().x(), weak.velocity().x());
  EXPECT_LT(weak.velocity().x(), 0.5 * strong.velocity().x());
}

TEST(Fusion, MeasurementNoiseFallsAsInliersRise)
{
  PlanarVelocityFilter::Settings settings;
  settings.vision_reference_inliers = 300.0;
  settings.vision_noise = 0.25;
  PlanarVelocityFilter filter(settings);

  EXPECT_LT(filter.measurement_variance(600), filter.measurement_variance(300));
  EXPECT_LT(filter.measurement_variance(300), filter.measurement_variance(50));
}

TEST(Fusion, TheCollapseToZeroIsGatedOutOnceTheFilterIsConfident)
{
  PlanarVelocityFilter::Settings settings;
  settings.acceleration_noise = 0.5;
  PlanarVelocityFilter filter(settings);
  // Settle on 6 m/s with well supported vision.
  for (int i = 0; i < 30; ++i) {
    filter.update(Eigen::Vector2d(6.0, 0.0), 600);
    filter.predict(Eigen::Vector2d::Zero(), 0.05);
  }

  // The failure we measured: vision claims the vehicle nearly stopped.
  const bool accepted = filter.update(Eigen::Vector2d(0.3, 0.0), 40);

  EXPECT_FALSE(accepted);
  EXPECT_NEAR(filter.velocity().x(), 6.0, 0.5);
}

TEST(Fusion, AGenuineDecelerationStillGetsThrough)
{
  PlanarVelocityFilter::Settings settings;
  settings.acceleration_noise = 1.5;
  PlanarVelocityFilter filter(settings);
  for (int i = 0; i < 30; ++i) {
    filter.update(Eigen::Vector2d(2.0, 0.0), 600);
    filter.predict(Eigen::Vector2d::Zero(), 0.05);
  }
  ASSERT_NEAR(filter.velocity().x(), 2.0, 0.1);

  // Braking: the accelerometer reports it and vision agrees frame by frame.
  double speed = 2.0;
  for (int i = 0; i < 10; ++i) {
    filter.predict(Eigen::Vector2d(-4.0, 0.0), 0.05);
    speed = std::max(speed - 4.0 * 0.05, 0.0);
    filter.update(Eigen::Vector2d(speed, 0.0), 600);
  }

  EXPECT_NEAR(filter.velocity().x(), speed, 0.2);
  EXPECT_LT(filter.velocity().x(), 0.5);
}

TEST(Fusion, NothingIsGatedBeforeTheFirstUpdate)
{
  PlanarVelocityFilter filter;

  EXPECT_TRUE(filter.update(Eigen::Vector2d(50.0, 0.0), 5));
}

TEST(Fusion, BodyTranslationUsesTheFusedVelocity)
{
  PlanarVelocityFilter filter;
  // Let the filter converge on 4 m/s along world +y.
  for (int i = 0; i < 30; ++i) {
    filter.predict(Eigen::Vector2d::Zero(), 0.05);
    filter.update(Eigen::Vector2d(0.0, 4.0), 600);
  }

  const Eigen::Vector2d translation = filter.body_translation(0.5, M_PI / 2.0);

  // Facing +y at 4 m/s, half a second is 2 m straight ahead.
  EXPECT_NEAR(translation.x(), 2.0, 0.4);
  EXPECT_NEAR(translation.y(), 0.0, 0.2);
}

TEST(Fusion, MotionToWorldVelocityRoundTrip)
{
  const auto velocity = monoscale::world_velocity_from_motion(0.4, 0.0, 0.1, M_PI / 2.0);

  ASSERT_TRUE(velocity.has_value());
  EXPECT_NEAR(velocity->x(), 0.0, 1e-9);
  EXPECT_NEAR(velocity->y(), 4.0, 1e-9);
}

TEST(Fusion, AnOutlierStillTethersTheFilterInsteadOfBeingDropped)
{
  PlanarVelocityFilter::Settings settings;
  settings.acceleration_noise = 0.5;
  PlanarVelocityFilter filter(settings);
  for (int i = 0; i < 30; ++i) {
    filter.update(Eigen::Vector2d(6.0, 0.0), 600);
    filter.predict(Eigen::Vector2d::Zero(), 0.05);
  }

  const double before = filter.velocity().x();
  const bool accepted = filter.update(Eigen::Vector2d(0.3, 0.0), 40);

  // Reported as an outlier, but it still pulls the estimate a little rather
  // than leaving the filter to integrate alone.
  EXPECT_FALSE(accepted);
  EXPECT_LT(filter.velocity().x(), before);
  EXPECT_GT(filter.velocity().x(), 0.8 * before);
}

TEST(Fusion, RepeatedOutliersEventuallyWinIfTheWorldReallyChanged)
{
  PlanarVelocityFilter::Settings settings;
  settings.acceleration_noise = 0.5;
  PlanarVelocityFilter filter(settings);
  for (int i = 0; i < 30; ++i) {
    filter.update(Eigen::Vector2d(6.0, 0.0), 600);
    filter.predict(Eigen::Vector2d::Zero(), 0.05);
  }

  // About six seconds of consistent disagreement at 20 Hz.
  for (int i = 0; i < 120; ++i) {
    filter.predict(Eigen::Vector2d::Zero(), 0.05);
    filter.update(Eigen::Vector2d(0.3, 0.0), 600);
  }

  EXPECT_NEAR(filter.velocity().x(), 0.3, 0.2);
}

TEST(Fusion, CameraDisagreementDilutesASolveTheInliersCallConfident)
{
  PlanarVelocityFilter trusting;
  PlanarVelocityFilter doubting;
  for (PlanarVelocityFilter * filter : {&trusting, &doubting}) {
    filter->update(Eigen::Vector2d::Zero(), 300);
    for (int i = 0; i < 10; ++i) {
      filter->predict(Eigen::Vector2d::Zero(), 0.05);
    }
  }

  // Same measurement, same inlier count. Only the cross-camera evidence
  // differs, which is exactly what inliers cannot reveal.
  trusting.update(Eigen::Vector2d(0.5, 0.0), 800, 0.0);
  doubting.update(Eigen::Vector2d(0.5, 0.0), 800, 4.0);

  EXPECT_GT(trusting.velocity().x(), doubting.velocity().x());
  EXPECT_LT(doubting.velocity().x(), 0.3 * trusting.velocity().x());
}

TEST(Fusion, DisplacementFilterMeasuresTheHopNotAQuotient)
{
  // The displacement model's whole point: vision reports metres between two
  // image times, and the filter takes them as metres rather than dividing by
  // an interval whose own error would then ride along.
  monoscale::PlanarDisplacementFilter filter;

  filter.open_hop();
  for (int i = 0; i < 10; ++i) {
    filter.predict(Eigen::Vector2d::Zero(), 0.01, 0.0);
  }
  EXPECT_TRUE(filter.update(Eigen::Vector2d(0.20, 0.0), 500));

  const Eigen::Vector2d hop = filter.body_translation(0.0);
  EXPECT_NEAR(hop.x(), 0.20, 0.02);
  EXPECT_NEAR(hop.y(), 0.0, 0.01);
}

TEST(Fusion, DisplacementSpreadTightensAConfidentSolve)
{
  // The inlier count says how many votes there were, not how far apart they
  // landed. Through a turn the votes can be many and still disagree.
  monoscale::PlanarDisplacementFilter filter;

  EXPECT_LT(filter.measurement_variance(500, 0.001), filter.measurement_variance(500, 0.5));
  // The configured noise stays a floor: agreeing closely is not being exact.
  EXPECT_DOUBLE_EQ(
    filter.measurement_variance(500, 1e-9), filter.measurement_variance(500, 0.0));
}

TEST(Fusion, ZeroVelocityUpdateIsWhereTheBiasBecomesObservable)
{
  monoscale::PlanarDisplacementFilter filter;

  // An instrument reporting a steady offset while the vehicle stands still.
  for (int i = 0; i < 200; ++i) {
    filter.predict(Eigen::Vector2d(0.3, 0.0), 0.01, 0.0);
    filter.update_zero_velocity(0.01);
  }

  EXPECT_NEAR(filter.velocity().x(), 0.0, 0.01);
  // Whatever the velocity would have integrated into is the bias, and it has
  // to be found in the bias state rather than absorbed silently.
  EXPECT_GT(filter.bias().x(), 0.05);
}

TEST(Fusion, DisplacementCovarianceStaysSymmetricAndFinite)
{
  monoscale::PlanarDisplacementFilter filter;
  for (int hop = 0; hop < 300; ++hop) {
    filter.open_hop();
    for (int i = 0; i < 5; ++i) {
      filter.predict(Eigen::Vector2d(0.1, -0.05), 0.01, 0.02 * hop);
    }
    filter.update(Eigen::Vector2d(0.01, 0.0), 400, 0.0, 0.02);
  }

  ASSERT_TRUE(filter.last_update().has_value());
  EXPECT_TRUE(std::isfinite(filter.last_update()->nis));
  EXPECT_TRUE(std::isfinite(filter.velocity().x()));
  EXPECT_TRUE(std::isfinite(filter.bias().x()));
}
